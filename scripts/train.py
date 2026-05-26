import argparse
import logging
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Dict, Optional

import wandb
from transformers import EarlyStoppingCallback, set_seed
from torch.utils.data import Subset
from functools import partial

# Add parent directory to Python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (
    load_config,
    load_model_and_processor,
    setup_lora_model,
    save_lora_model,
    print_model_info,
    unwrap_distributed_model,
)
from utils.data_utils import VisionLanguageDataset, collate_fn
from utils.distributed_utils import (
    get_rank,
    get_world_size,
    is_main_process,
)
from utils.train_utils import QwenVLTrainer, create_training_arguments
from utils.wandb_utils import find_wandb_run_info, save_wandb_run_info

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Optional[str] = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    main_process = is_main_process()
    world_size = get_world_size()

    if output_dir and main_process:
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(output_path / "training.log", encoding="utf-8")
        )

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if world_size > 1:
        log_format = f'%(asctime)s - rank {get_rank()} - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO if main_process else logging.WARNING,
        format=log_format,
        handlers=handlers,
        force=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3.5 models")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    output_path = Path(output_dir).expanduser()
    if not output_path.is_dir():
        return None

    latest_step = -1
    latest_checkpoint: Optional[Path] = None
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)$")

    for child in output_path.iterdir():
        if not child.is_dir():
            continue
        match = checkpoint_pattern.fullmatch(child.name)
        if match is None:
            continue

        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_checkpoint = child

    return None if latest_checkpoint is None else str(latest_checkpoint)


def resolve_resume_checkpoint(
    resume_from_checkpoint: Optional[str],
    output_dir: str,
) -> Optional[str]:
    if resume_from_checkpoint is None:
        return None

    resume_value = resume_from_checkpoint.strip()
    if not resume_value:
        return None

    if resume_value.lower() in {"latest", "auto"}:
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint is None:
            logger.info(
                "No checkpoint found under %s. Starting training from scratch.",
                output_dir,
            )
            return None

        logger.info("Resolved latest checkpoint to %s", latest_checkpoint)
        return latest_checkpoint

    return str(Path(resume_value).expanduser())


def save_run_configs(args, output_dir: str):
    if not is_main_process():
        return

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    run_config_src = Path(args.config).expanduser()
    run_config_dst = output_path / "run_config.yaml"

    if os.path.realpath(run_config_src) == os.path.realpath(run_config_dst):
        logger.info("Skip saving run config because it already points to %s", run_config_dst)
        return

    shutil.copy2(run_config_src, run_config_dst)


def log_effective_contrastive_hidden_layers(model, objective_config: Dict[str, Any]) -> None:
    start = int(objective_config.get("contrastive_hidden_layer_start", 21))
    end = int(objective_config.get("contrastive_hidden_layer_end", 24))
    text_config = getattr(model.config, "text_config", model.config)
    last_idx = int(getattr(text_config, "num_hidden_layers", end))
    actual_start = min(start, last_idx)
    actual_end = min(end, last_idx)
    if actual_start > actual_end:
        actual_start = actual_end

    logger.info(
        "Contrastive hidden layers: configured=%s..%s, actual=%s..%s "
        "(hidden_states valid indices: 0..%s; 0 is embeddings).",
        start,
        end,
        actual_start,
        actual_end,
        last_idx,
    )


def setup_wandb(
    config: Dict[str, Any],
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
):
    wandb_config = config.get("wandb_config", {})
    report_to = config.get("training_config", {}).get("report_to", [])
    if isinstance(report_to, str):
        report_targets = {report_to.lower()}
    elif report_to is None:
        report_targets = set()
    else:
        report_targets = {str(target).lower() for target in report_to}

    wandb_enabled = bool(wandb_config.get("enabled", "wandb" in report_targets))
    if not wandb_enabled:
        logger.info("W&B logging is disabled.")
        return

    if not is_main_process():
        logger.info("Skipping W&B initialization on rank %s.", get_rank())
        return

    try:
        wandb_init_kwargs = {
            "config": config,
        }
        for key in ("project", "name", "tags"):
            if key in wandb_config:
                wandb_init_kwargs[key] = wandb_config[key]

        if resume_from_checkpoint:
            run_info = find_wandb_run_info(
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            if run_info is not None:
                wandb_init_kwargs["id"] = run_info["run_id"]
                wandb_init_kwargs["resume"] = "allow"
                logger.info(
                    "Resuming W&B run %s from %s",
                    run_info["run_id"],
                    run_info.get("source_path", "unknown source"),
                )
            else:
                logger.warning(
                    "No saved W&B run id found for checkpoint %s. A new W&B run will be created.",
                    resume_from_checkpoint,
                )

        wandb.init(**wandb_init_kwargs)
        logger.info("Weights & Biases initialized successfully")

        if wandb.run is not None:
            run_info_path = save_wandb_run_info(
                output_dir=output_dir,
                run_id=wandb.run.id,
                project=getattr(wandb.run, "project", None) or wandb_init_kwargs.get("project"),
                name=getattr(wandb.run, "name", None) or wandb_init_kwargs.get("name"),
            )
            logger.info("Saved W&B run info to %s", run_info_path)
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")


def save_final_model(trainer, processor, output_dir: str, use_lora: bool) -> None:
    if not is_main_process():
        return

    model_to_save = unwrap_distributed_model(trainer.model)
    logger.info("Saving final model...")
    if use_lora:
        save_lora_model(
            model=model_to_save,
            processor=processor,
            output_dir=output_dir,
            safe_serialization=True
        )
    else:
        model_to_save.save_pretrained(
            output_dir,
            safe_serialization=True
        )
        processor.save_pretrained(output_dir)
        logger.info(f"Full model saved to {output_dir}")


def main():
    # Parse arguments
    args = parse_args()
    setup_logging()

    # Load configurations
    logger.info("Loading configurations...")
    config = load_config(args.config)
    setup_logging(config["training_config"]["output_dir"])
    if is_main_process():
        logger.info(
            "Writing training logs to %s",
            Path(config["training_config"]["output_dir"]).expanduser() / "training.log",
        )
    if get_world_size() > 1:
        logger.info(
            "Running distributed training: rank=%s world_size=%s",
            get_rank(),
            get_world_size(),
        )
    objective_config = config.get("objective_config", {})

    resolved_resume_from_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        config["training_config"]["output_dir"],
    )

    # Setup others
    set_seed(config["training_config"].get("seed", 1234))
    setup_wandb(
        config,
        output_dir=config["training_config"]["output_dir"],
        resume_from_checkpoint=resolved_resume_from_checkpoint,
    )

    # Load model and processor
    logger.info("Loading model and processor...")
    model_config = config["model_config"].copy()
    model, processor = load_model_and_processor(model_config)
    
    # Setup train mode
    if args.use_lora:
        if "lora_config" not in config:
            raise ValueError("LoRA mode enabled but 'lora_config' not found in config file.")
        logger.info("Setting up LoRA...")
        model = setup_lora_model(model, config["lora_config"])
    else:
        logger.info("Using full-parameter fine-tuning mode...")
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)
    if is_main_process():
        print_model_info(model)

    # Load datasets
    logger.info("Loading datasets...")
    data_config = config["data_config"]

    eval_dataset = None
    train_dataset = VisionLanguageDataset(
        data_path = data_config["train_data_path"],
        video_root = data_config["video_root"],
        sample_frames = data_config["sample_frames"],
        video_ext = data_config.get("video_ext", ".webm"),
    )
    train_subset_size = data_config.get("train_subset_size", None)
    if train_subset_size is not None:
        train_subset_size = min(int(train_subset_size), len(train_dataset))
        idx_list = random.sample(range(len(train_dataset)), train_subset_size)
        train_dataset = Subset(train_dataset, idx_list)

    if data_config.get("val_data_path", ""):
        eval_dataset = VisionLanguageDataset(
            data_path = data_config["val_data_path"],
            video_root = data_config["video_root"],
            sample_frames = data_config["sample_frames"],
            video_ext = data_config.get("video_ext", ".webm"),
        )
        subset_size = data_config.get("eval_subset_size", None)
        if subset_size is not None and eval_dataset is not None:
            subset_size = min(int(subset_size), len(eval_dataset))
            idx_list = random.sample(range(len(eval_dataset)), subset_size)
            eval_dataset = Subset(eval_dataset, idx_list)

    # Create training arguments
    logger.info("Setting up training arguments...")
    training_args = create_training_arguments(
        config["training_config"],
        config["training_config"]["output_dir"],
        distributed_config=config.get("distributed_config"),
    )

    # Save configs
    save_run_configs(args, training_args.output_dir)
    
    # Setup callbacks
    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))
    
    # Create trainer
    logger.info("Creating trainer...")
    data_collator = partial(
        collate_fn,
        processor=processor,
        sample_frames=data_config["sample_frames"],
        num_negative_samples=objective_config.get("num_negative_samples", 0),
        contrastive_pooling_strategy=objective_config.get(
            "contrastive_pooling_strategy",
            "last_frame",
        ),
    )
    trainer = QwenVLTrainer(
        model=model,
        args=training_args,
        processor=processor,
        training_config=config["training_config"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        objective_config=objective_config,
    )
    log_effective_contrastive_hidden_layers(model, objective_config)
    
    # Start training
    logger.info("Starting training...")
    try:
        if resolved_resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resolved_resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=resolved_resume_from_checkpoint)
        else:
            trainer.train()
        
        # Save the final model from rank 0 only.
        save_final_model(
            trainer=trainer,
            processor=processor,
            output_dir=training_args.output_dir,
            use_lora=args.use_lora,
        )
        
        # Save final metrics
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]
            logger.info(f"Final training metrics: {final_metrics}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        if is_main_process() and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
