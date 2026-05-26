import os
import torch
import yaml
from typing import Dict, Any, Tuple
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
import logging

logger = logging.getLogger(__name__)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    candidate = model
    visited = set()

    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))

        if hasattr(candidate, "module"):
            next_candidate = candidate.module
            if next_candidate is not candidate:
                candidate = next_candidate
                continue

        if hasattr(candidate, "get_base_model"):
            try:
                next_candidate = candidate.get_base_model()
            except Exception:
                next_candidate = None
            if next_candidate is not None and next_candidate is not candidate:
                candidate = next_candidate
                continue

        if hasattr(candidate, "base_model"):
            next_candidate = candidate.base_model
            if next_candidate is not None and next_candidate is not candidate:
                candidate = next_candidate
                continue

        break

    return candidate


def unwrap_distributed_model(model: torch.nn.Module) -> torch.nn.Module:
    candidate = model
    visited = set()

    while hasattr(candidate, "module") and id(candidate) not in visited:
        visited.add(id(candidate))
        next_candidate = candidate.module
        if next_candidate is candidate:
            break
        candidate = next_candidate

    return candidate


def get_model_config(model: torch.nn.Module):
    base_model = unwrap_model(model)
    config = getattr(base_model, "config", None)
    if config is None and hasattr(model, "config"):
        config = model.config
    if config is None:
        raise AttributeError("Unable to resolve model config for contrastive learning.")
    return config


def get_multimodal_backbone(model: torch.nn.Module) -> torch.nn.Module:
    base_model = unwrap_model(model)
    candidates = [getattr(base_model, "model", None), base_model]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "language_model") and hasattr(candidate, "get_video_features"):
            return candidate
    raise AttributeError("Unable to resolve multimodal backbone for contrastive hidden-state forward.")


def _set_im_end_id(model: torch.nn.Module, processor: AutoProcessor) -> None:
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if hasattr(model, "config"):
        model.config.im_end_id = im_end_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.im_end_id = im_end_id
    try:
        base_model = model.get_base_model()
    except Exception:
        base_model = None
    if base_model is not None and hasattr(base_model, "config"):
        base_model.config.im_end_id = im_end_id


def _set_use_cache(model: torch.nn.Module, use_cache: bool) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = use_cache
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = use_cache

    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = use_cache
    if hasattr(model, "base_model") and hasattr(model.base_model, "generation_config"):
        base_generation_config = model.base_model.generation_config
        if base_generation_config is not None:
            base_generation_config.use_cache = use_cache

    try:
        base_model = model.get_base_model()
    except Exception:
        base_model = None
    if base_model is not None and hasattr(base_model, "config"):
        base_model.config.use_cache = use_cache
    if base_model is not None and hasattr(base_model, "generation_config"):
        base_generation_config = base_model.generation_config
        if base_generation_config is not None:
            base_generation_config.use_cache = use_cache


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_processor(
    model_config: Dict[str, Any],
) -> Tuple[torch.nn.Module, AutoProcessor]:
    
    model_name = model_config["model_name_or_path"]

    # Model loading arguments
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "torch_dtype": getattr(torch, model_config.get("torch_dtype", "bfloat16")),
        "attn_implementation": model_config.get("attn_implementation", "flash_attention_2"),
        "low_cpu_mem_usage": True,
    }    
    logger.info(f"Loading model: {model_name}")
    
    model = Qwen3_5ForConditionalGeneration.from_pretrained(**model_kwargs)
    _set_use_cache(model, bool(model_config.get("use_cache", False)))
    logger.info("Loaded base Qwen3.5 model")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    _set_im_end_id(model, processor)
    
    # Ensure processor has a pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    logger.info("Model and processor loaded successfully")
    return model, processor


def setup_lora_model(
    model: torch.nn.Module,
    lora_config: Dict[str, Any],
) -> torch.nn.Module:
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        inference_mode=lora_config.get("inference_mode", False),
        modules_to_save=lora_config.get("modules_to_save", [])
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.train()

    # Enable input gradients to support gradient checkpointing with LoRA
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            embed_layer = model.get_input_embeddings()
            def _make_outputs_require_grad(module, inputs, outputs):
                if isinstance(outputs, torch.Tensor):
                    outputs.requires_grad_(True)
            embed_layer.register_forward_hook(_make_outputs_require_grad)
    except Exception as e:
        logger.warning(f"Failed to enable input gradients: {e}")
    
    # Ensure LoRA parameters are trainable
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            param.grad = None
    
    # Verification of LoRA attachment and trainable params
    try:
        model.print_trainable_parameters()
        lora_param_count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
        sample_lora_params = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" in n][:8]
        logger.info(f"LoRA trainable parameters: {lora_param_count:,}")
        logger.info(f"Sample LoRA params: {sample_lora_params}")
    except Exception as e:
        logger.warning(f"Could not verify LoRA parameters: {e}")
    
    logger.info("LoRA configuration applied successfully")
    return model


def save_lora_model(
    model: PeftModel,
    processor: AutoProcessor,
    output_dir: str,
    safe_serialization: bool = True
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LoRA weights
    model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization
    )
    
    # Save processor
    processor.save_pretrained(output_dir)
    
    logger.info(f"LoRA model saved to {output_dir}")


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    param_memory = 0
    buffer_memory = 0
    
    for param in model.parameters():
        param_memory += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_memory += buffer.nelement() * buffer.element_size()
    
    total_memory = param_memory + buffer_memory
    
    return {
        "param_memory_mb": param_memory / (1024 * 1024),
        "buffer_memory_mb": buffer_memory / (1024 * 1024),
        "total_memory_mb": total_memory / (1024 * 1024),
        "param_memory_gb": param_memory / (1024 * 1024 * 1024),
        "buffer_memory_gb": buffer_memory / (1024 * 1024 * 1024),
        "total_memory_gb": total_memory / (1024 * 1024 * 1024),
    }


def print_model_info(model: torch.nn.Module) -> None:
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    # Memory usage
    memory_info = get_model_memory_usage(model)
    print(f"Model memory usage: {memory_info['total_memory_gb']:.2f} GB")
    
    # Device information
    devices = {param.device for param in model.parameters()}
    print(f"Model devices: {devices}")
