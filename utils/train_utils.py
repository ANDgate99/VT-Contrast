import logging
import inspect
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainingArguments
from .distributed_utils import get_world_size

logger = logging.getLogger(__name__)


TRAINER_RUNTIME_CONFIG_KEYS = {
    "contrastive_variant_forward_chunk_size",
    "lr_scheduler_total_steps",
    "distributed_config",
}


def _training_arguments_accepts(argument_name: str) -> bool:
    try:
        signature = inspect.signature(TrainingArguments.__init__)
    except (TypeError, ValueError):
        return True

    if argument_name in signature.parameters:
        return True

    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _drop_unsupported_training_arguments(training_config: Dict[str, Any]) -> None:
    for key in list(training_config):
        if not _training_arguments_accepts(key):
            logger.warning(
                "TrainingArguments does not support '%s' in this environment; ignoring it.",
                key,
            )
            training_config.pop(key, None)


def _preserve_effective_batch_size(
    training_config: Dict[str, Any],
    distributed_config: Optional[Dict[str, Any]],
) -> None:
    distributed_config = distributed_config or {}
    if not bool(distributed_config.get("preserve_effective_batch_size", False)):
        return

    world_size = get_world_size()
    if world_size == 1:
        return

    reference_accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
    if reference_accumulation_steps <= 0:
        logger.warning(
            "Invalid gradient_accumulation_steps=%s. Leaving it unchanged.",
            reference_accumulation_steps,
        )
        return

    target_accumulation_steps = reference_accumulation_steps / world_size
    adjusted_accumulation_steps = max(1, int(round(target_accumulation_steps)))

    if adjusted_accumulation_steps != reference_accumulation_steps:
        logger.info(
            "Adjusted gradient_accumulation_steps from %s to %s for world_size=%s "
            "to preserve the single-GPU effective batch size.",
            reference_accumulation_steps,
            adjusted_accumulation_steps,
            world_size,
        )
        training_config["gradient_accumulation_steps"] = adjusted_accumulation_steps

    reference_effective_steps = reference_accumulation_steps
    actual_effective_steps = adjusted_accumulation_steps * world_size
    if actual_effective_steps != reference_effective_steps:
        logger.warning(
            "Unable to exactly preserve effective batch size with gradient_accumulation_steps=%s "
            "at world_size=%s. Single-GPU reference effective step multiplier is %s; actual is %s.",
            adjusted_accumulation_steps,
            world_size,
            reference_effective_steps,
            actual_effective_steps,
        )


def create_training_arguments(
    training_config: Dict[str, Any],
    output_dir: str,
    distributed_config: Optional[Dict[str, Any]] = None,
) -> TrainingArguments:
    training_config = training_config.copy()
    training_config["output_dir"] = output_dir

    inline_distributed_config = training_config.pop("distributed_config", None)
    if distributed_config is None:
        distributed_config = inline_distributed_config
    elif isinstance(inline_distributed_config, dict):
        distributed_config = {**inline_distributed_config, **distributed_config}

    _preserve_effective_batch_size(training_config, distributed_config)

    for key in TRAINER_RUNTIME_CONFIG_KEYS:
        training_config.pop(key, None)

    if "torch_dtype" in training_config:
        training_config["torch_dtype"] = getattr(torch, training_config["torch_dtype"])

    if training_config.get("device") is None:
        training_config.pop("device", None)

    _drop_unsupported_training_arguments(training_config)

    return TrainingArguments(**training_config)


class QwenVLTrainer(Trainer):
    def __init__(
        self,
        *args,
        processor=None,
        training_config: Optional[Dict[str, Any]] = None,
        objective_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        training_config = training_config or {}
        objective_config = objective_config or {}
        self.processor = processor
        self.contrastive_temperature = float(objective_config.get("contrastive_temperature", 0.1))
        self.contrastive_weight = float(objective_config.get("contrastive_weight", 0.1))
        contrastive_zero_after_step = objective_config.get("contrastive_weight_zero_after_step")
        self.contrastive_weight_zero_after_step = (
            None if contrastive_zero_after_step is None else int(contrastive_zero_after_step)
        )
        self.lm_weight = float(objective_config.get("lm_weight", 1.0))
        self.contrastive_hidden_layer_start = int(objective_config.get("contrastive_hidden_layer_start", 21))
        self.contrastive_hidden_layer_end = int(objective_config.get("contrastive_hidden_layer_end", 24))
        self.contrastive_variant_forward_chunk_size = int(training_config.get("contrastive_variant_forward_chunk_size", 1))
        self.lr_scheduler_total_steps = training_config.get("lr_scheduler_total_steps")

        self._train_loss_stats = self._new_loss_stats()
        self._eval_loss_stats = self._new_loss_stats()

        from utils.model_utils import get_model_config
        self._model_config = get_model_config(self.model)

        if self.contrastive_variant_forward_chunk_size > 1:
            logger.info(
                "Contrastive variant forward chunk size set to %s",
                self.contrastive_variant_forward_chunk_size,
            )

        self.model_accepts_loss_kwargs = False

    def _effective_contrastive_weight(self) -> float:
        if self.contrastive_weight_zero_after_step is None:
            return self.contrastive_weight

        global_step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        if global_step >= self.contrastive_weight_zero_after_step:
            return 0.0
        return self.contrastive_weight

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler_total_steps is not None:
            num_training_steps = int(self.lr_scheduler_total_steps)
        return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    @staticmethod
    def _new_loss_stats() -> Dict[str, float]:
        return {
            "lm_loss": 0.0,
            "contrastive_loss": 0.0,
            "total_loss": 0.0,
            "count": 0.0,
        }

    @staticmethod
    def _reset_loss_stats(stats: Dict[str, float]) -> None:
        stats.update(
            {
                "lm_loss": 0.0,
                "contrastive_loss": 0.0,
                "total_loss": 0.0,
                "count": 0.0,
            }
        )

    def _loss_stats_device(self) -> torch.device:
        args = getattr(self, "args", None)
        device = getattr(args, "device", None)
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _loss_stats_to_tensor(self, stats: Dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [
                float(stats["lm_loss"]),
                float(stats["contrastive_loss"]),
                float(stats["total_loss"]),
                float(stats["count"]),
            ],
            device=self._loss_stats_device(),
            dtype=torch.float64,
        )

    @staticmethod
    def _distributed_loss_stats_enabled() -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return False
        return torch.distributed.get_world_size() > 1

    def _aggregate_loss_stats_tensor(self, stats: Dict[str, float]) -> torch.Tensor:
        stats_tensor = self._loss_stats_to_tensor(stats)
        if self._distributed_loss_stats_enabled():
            torch.distributed.all_reduce(stats_tensor, op=torch.distributed.ReduceOp.SUM)
        return stats_tensor

    def _forward_without_output_fp32_conversion(self, model: torch.nn.Module, **kwargs):
        candidate = getattr(model, "forward", None)
        visited = set()

        while candidate is not None and id(candidate) not in visited:
            visited.add(id(candidate))

            # Accelerate's ConvertOutputsToFp32 keeps the real forward here.
            model_forward = getattr(candidate, "model_forward", None)
            if callable(model_forward) and model_forward is not candidate:
                candidate = model_forward
                continue

            wrapped = getattr(candidate, "__wrapped__", None)
            if callable(wrapped) and wrapped is not candidate:
                candidate = wrapped
                continue

            break

        if candidate is None or not callable(candidate):
            return model(**kwargs)

        if inspect.isfunction(candidate):
            bypass_forward = candidate.__get__(model, type(model))
        else:
            bypass_forward = candidate

        return bypass_forward(**kwargs)

    def _forward_multimodal_hidden_outputs(self, model: torch.nn.Module, **kwargs):
        from utils.model_utils import get_multimodal_backbone
        multimodal_backbone = get_multimodal_backbone(model)
        multimodal_kwargs = {key: value for key, value in kwargs.items() if key != "labels"}
        multimodal_kwargs["output_hidden_states"] = True
        multimodal_kwargs["return_dict"] = True
        return self._forward_without_output_fp32_conversion(multimodal_backbone, **multimodal_kwargs)

    def _slice_anchor_negative_inputs(
        self,
        anchor_negative_inputs: Dict[str, Any],
        start_variant: int,
        end_variant: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        total_variants = max(int(anchor_negative_inputs["input_ids"].shape[0] // batch_size), 1)
        chunk_inputs: Dict[str, Any] = {}
        video_values = anchor_negative_inputs.get("pixel_values_videos")
        video_rows_per_variant = None
        if torch.is_tensor(video_values):
            if video_values.shape[0] % total_variants != 0:
                raise ValueError(
                    "contrastive pixel_values_videos rows are not divisible by num_variants: "
                    f"{video_values.shape[0]} vs {total_variants}"
                )
            video_rows_per_variant = video_values.shape[0] // total_variants

        for key, value in anchor_negative_inputs.items():
            if not torch.is_tensor(value):
                if value is not None:
                    chunk_inputs[key] = value
                continue

            if key == "pixel_values_videos":
                if video_rows_per_variant is None:
                    continue
                row_start = start_variant * video_rows_per_variant
                row_end = end_variant * video_rows_per_variant
                chunk_inputs[key] = value[row_start:row_end]
                continue

            if value.ndim > 0 and value.shape[0] == total_variants * batch_size:
                batch_start = start_variant * batch_size
                batch_end = end_variant * batch_size
                chunk_inputs[key] = value[batch_start:batch_end]
                continue

            chunk_inputs[key] = value

        return chunk_inputs

    def _forward_anchor_negative_selected_hidden(
        self,
        model: torch.nn.Module,
        anchor_negative_inputs: Dict[str, Any],
        num_variants: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if num_variants <= 0:
            return None

        chunk_size = min(self.contrastive_variant_forward_chunk_size, num_variants)
        selected_hidden_chunks = []

        for start_variant in range(0, num_variants, chunk_size):
            end_variant = min(start_variant + chunk_size, num_variants)
            chunk_inputs = self._slice_anchor_negative_inputs(
                anchor_negative_inputs=anchor_negative_inputs,
                start_variant=start_variant,
                end_variant=end_variant,
                batch_size=batch_size,
            )
            chunk_outputs = self._forward_multimodal_hidden_outputs(
                model,
                **chunk_inputs,
            )
            chunk_hidden = self._select_hidden_layers(chunk_outputs.hidden_states)
            if chunk_hidden is None:
                return None
            selected_hidden_chunks.append(chunk_hidden)

        if not selected_hidden_chunks:
            return None
        if len(selected_hidden_chunks) == 1:
            return selected_hidden_chunks[0]
        return torch.cat(selected_hidden_chunks, dim=0)

    def _update_loss_stats(
        self,
        is_training: bool,
        lm_loss: Optional[torch.Tensor],
        contrastive_loss: Optional[torch.Tensor],
        total_loss: Optional[torch.Tensor],
    ) -> None:
        if total_loss is None:
            return

        stats = self._train_loss_stats if is_training else self._eval_loss_stats
        stats["lm_loss"] += float(lm_loss.detach().item()) if lm_loss is not None else 0.0
        stats["contrastive_loss"] += (
            float(contrastive_loss.detach().item()) if contrastive_loss is not None else 0.0
        )
        stats["total_loss"] += float(total_loss.detach().item())
        stats["count"] += 1.0

    def _append_aggregated_logs(
        self,
        logs: Dict[str, float],
        stats: Dict[str, float],
        prefix: str,
    ) -> None:
        aggregated_stats = self._aggregate_loss_stats_tensor(stats)
        count = float(aggregated_stats[3].item())

        self._reset_loss_stats(stats)
        if count <= 0:
            return

        logs[f"{prefix}lm_loss"] = float((aggregated_stats[0] / count).item())
        logs[f"{prefix}contrastive_loss"] = float((aggregated_stats[1] / count).item())
        logs[f"{prefix}total_loss"] = float((aggregated_stats[2] / count).item())

    def _reset_eval_tracking_state(self) -> None:
        self._reset_loss_stats(self._eval_loss_stats)

    def evaluate(self, *args, **kwargs):
        self._reset_eval_tracking_state()
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self._reset_eval_tracking_state()

    def predict(self, *args, **kwargs):
        self._reset_eval_tracking_state()
        try:
            return super().predict(*args, **kwargs)
        finally:
            self._reset_eval_tracking_state()

    def _select_hidden_layers(self, hidden_states):
        if hidden_states is None:
            return None
        start_layer = self.contrastive_hidden_layer_start
        end_layer = self.contrastive_hidden_layer_end
        last_idx = len(hidden_states) - 1
        start = min(start_layer, last_idx)
        end = min(end_layer, last_idx)
        if start > end:
            start = end
        selected = torch.stack(hidden_states[start : end + 1], dim=0)
        return selected.mean(dim=0)

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor, token_mask: Optional[torch.Tensor]):
        if token_mask is None:
            return None

        mask = token_mask.bool()
        if hidden_states.ndim != 2 or mask.ndim != 1 or hidden_states.shape[0] != mask.shape[0]:
            return None

        if mask.sum().item() == 0:
            return None
        return hidden_states[mask].mean(dim=0)

    @staticmethod
    def _repeat_tensor_batch(tensor: Optional[torch.Tensor], repeat_count: int):
        if tensor is None or repeat_count <= 1:
            return tensor
        if tensor.ndim == 0:
            return tensor
        return tensor.repeat((repeat_count,) + (1,) * (tensor.ndim - 1))

    def _build_anchor_negative_inputs(
        self,
        model_inputs: Dict[str, Any],
        negative_pixel_values_videos: Optional[torch.Tensor],
    ):
        anchor_pixel_values_videos = model_inputs.get("pixel_values_videos")
        if (
            anchor_pixel_values_videos is None
            or negative_pixel_values_videos is None
            or negative_pixel_values_videos.shape[0] == 0
        ):
            return None, 0

        all_variants = torch.cat([anchor_pixel_values_videos.unsqueeze(0), negative_pixel_values_videos], dim=0)
        num_variants = int(all_variants.shape[0])
        if num_variants <= 1:
            return None, 0
        merged_pixel_values_videos = all_variants.flatten(0, 1)

        return (
            {
                "input_ids": self._repeat_tensor_batch(model_inputs.get("input_ids"), num_variants),
                "attention_mask": self._repeat_tensor_batch(model_inputs.get("attention_mask"), num_variants),
                "pixel_values": self._repeat_tensor_batch(model_inputs.get("pixel_values"), num_variants),
                "pixel_values_videos": merged_pixel_values_videos,
                "image_grid_thw": self._repeat_tensor_batch(model_inputs.get("image_grid_thw"), num_variants),
                "video_grid_thw": self._repeat_tensor_batch(model_inputs.get("video_grid_thw"), num_variants),
                "mm_token_type_ids": self._repeat_tensor_batch(model_inputs.get("mm_token_type_ids"), num_variants),
            },
            num_variants,
        )

    def compute_contrastive_loss(
        self,
        normal_hidden_states: torch.Tensor,
        positive_hidden_states: torch.Tensor,
        negative_hidden_states_list,
        last_frame_token_mask: Optional[torch.Tensor] = None,
        dropped_last_frame_token_mask: Optional[torch.Tensor] = None,
    ):
        total_contrastive_loss = normal_hidden_states.new_zeros(())
        valid_count = 0
        loss_fct = CrossEntropyLoss()

        for idx in range(normal_hidden_states.shape[0]):
            anchor = self._masked_mean_pool(
                normal_hidden_states[idx],
                None if last_frame_token_mask is None else last_frame_token_mask[idx],
            )
            positive = self._masked_mean_pool(
                positive_hidden_states[idx],
                None if dropped_last_frame_token_mask is None else dropped_last_frame_token_mask[idx],
            )
            if anchor is None or positive is None:
                continue

            anchor = F.normalize(anchor.unsqueeze(0), p=2, dim=-1)
            positive = F.normalize(positive.unsqueeze(0), p=2, dim=-1)

            negative_embeddings = []
            for neg_hidden_states in negative_hidden_states_list:
                negative = self._masked_mean_pool(
                    neg_hidden_states[idx],
                    None if last_frame_token_mask is None else last_frame_token_mask[idx],
                )
                if negative is None:
                    continue
                negative_embeddings.append(F.normalize(negative, p=2, dim=-1))

            if not negative_embeddings:
                continue

            pos_sim = torch.mm(anchor, positive.t()).squeeze(0) / self.contrastive_temperature
            neg_sims = torch.mm(anchor, torch.stack(negative_embeddings, dim=0).t()).squeeze(0)
            neg_sims = neg_sims / self.contrastive_temperature
            logits = torch.cat([pos_sim, neg_sims])
            targets = torch.zeros(1, dtype=torch.long, device=logits.device)
            contrastive_loss = loss_fct(logits.unsqueeze(0), targets)

            total_contrastive_loss = total_contrastive_loss + contrastive_loss
            valid_count += 1

        if valid_count == 0:
            return None
        return total_contrastive_loss / valid_count

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        del num_items_in_batch

        is_training = model.training
        labels = inputs.get("labels")
        last_frame_token_mask = inputs.get("last_frame_token_mask")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"labels", "question_token_mask", "last_frame_token_mask"}
            and not key.startswith("contrastive_")
        }

        dropped_input_ids = inputs.get("contrastive_dropped_input_ids")
        dropped_attention_mask = inputs.get("contrastive_dropped_attention_mask")
        dropped_last_frame_token_mask = inputs.get("contrastive_dropped_last_frame_token_mask")
        dropped_mm_token_type_ids = inputs.get("contrastive_dropped_mm_token_type_ids")
        dropped_pixel_values_videos = inputs.get("contrastive_dropped_pixel_values_videos")
        dropped_video_grid_thw = inputs.get("contrastive_dropped_video_grid_thw")
        negative_pixel_values_videos = inputs.get("contrastive_negative_pixel_values_videos")

        normal_outputs = model(
            **model_inputs,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )

        lm_loss = normal_outputs.loss
        contrastive_loss = None
        effective_contrastive_weight = self._effective_contrastive_weight()

        if effective_contrastive_weight > 0:
            dropped_outputs = self._forward_multimodal_hidden_outputs(
                model,
                input_ids=dropped_input_ids,
                attention_mask=dropped_attention_mask,
                labels=None,
                pixel_values=model_inputs.get("pixel_values"),
                pixel_values_videos=dropped_pixel_values_videos,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=dropped_video_grid_thw,
                mm_token_type_ids=dropped_mm_token_type_ids,
            )

            anchor_negative_inputs, num_variants = self._build_anchor_negative_inputs(
                model_inputs=model_inputs,
                negative_pixel_values_videos=negative_pixel_values_videos,
            )

            if (
                anchor_negative_inputs is not None
                and dropped_outputs.hidden_states is not None
            ):
                batch_size = model_inputs["input_ids"].shape[0]
                all_hidden = self._forward_anchor_negative_selected_hidden(
                    model,
                    anchor_negative_inputs=anchor_negative_inputs,
                    num_variants=num_variants,
                    batch_size=batch_size,
                )

                positive_hidden = self._select_hidden_layers(dropped_outputs.hidden_states)
                if all_hidden is not None and positive_hidden is not None:
                    all_hidden = all_hidden.reshape(num_variants, batch_size, *all_hidden.shape[1:])
                    normal_hidden = all_hidden[0]
                    negative_hidden_states = [all_hidden[idx] for idx in range(1, num_variants)]

                    contrastive_loss = self.compute_contrastive_loss(
                        normal_hidden_states=normal_hidden,
                        positive_hidden_states=positive_hidden,
                        negative_hidden_states_list=negative_hidden_states,
                        last_frame_token_mask=last_frame_token_mask,
                        dropped_last_frame_token_mask=dropped_last_frame_token_mask,
                    )

        total_loss = None
        if lm_loss is not None:
            total_loss = self.lm_weight * lm_loss
        if contrastive_loss is not None:
            total_loss = (
                effective_contrastive_weight * contrastive_loss
                if total_loss is None
                else total_loss + effective_contrastive_weight * contrastive_loss
            )
        if total_loss is None:
            dummy_param = next(p for p in model.parameters() if p.requires_grad)
            total_loss = torch.tensor(
                0.0,
                device=dummy_param.device,
                dtype=dummy_param.dtype,
                requires_grad=is_training,
            )

        self._update_loss_stats(is_training, lm_loss, contrastive_loss, total_loss)
        return (total_loss, normal_outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if "loss" in logs:
            self._append_aggregated_logs(logs, self._train_loss_stats, prefix="")
        if "eval_loss" in logs:
            self._append_aggregated_logs(logs, self._eval_loss_stats, prefix="eval_")

        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)
