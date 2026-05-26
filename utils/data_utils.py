from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List
import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_QUESTION = "What action is happening in the video?"
DEFAULT_MAX_PIXELS = 360 * 360


class VisionLanguageDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        video_root: str,
        sample_frames: int = 8,
        video_ext: str = ".webm",
    ):
        self.data_path = data_path
        self.video_root = video_root
        self.sample_frames = sample_frames
        self.video_ext = video_ext
        self.data = self._load_data()

    def _resolve_video_path(self, video_id: str) -> str:
        video_filename = f"{video_id}{self.video_ext}"
        if self.video_root:
            return os.path.join(self.video_root, video_filename)
        return video_filename

    @staticmethod
    def _format_answer(label: str) -> str:
        label = label.strip()
        if not label:
            return ""

        answer = f"{label[0].upper()}{label[1:]}"
        if not answer.endswith("."):
            answer = f"{answer}."
        return answer

    def _load_data(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.data_path):
            raise ValueError(f"Data path {self.data_path} does not exist")

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a list from {self.data_path}, got {type(raw_data).__name__}")

        data: List[Dict[str, Any]] = []
        filtered_by_frames = 0
        for row in raw_data:
            if not isinstance(row, dict):
                continue

            video_id = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip()
            if not video_id or not label:
                continue
            frame_count = row.get("frames")
            if frame_count is not None and int(frame_count) < self.sample_frames:
                filtered_by_frames += 1
                continue

            sample = {
                "video_id": video_id,
                "video_path": self._resolve_video_path(video_id),
                "question": DEFAULT_QUESTION,
                "answer": self._format_answer(label),
            }
            for sequence_key in ("pos_seq", "neg_seq"):
                if sequence_key in row:
                    sample[sequence_key] = row[sequence_key]
            data.append(sample)

        logger.info(
            "Loaded %s SSv2 samples from %s (filtered %s samples with frames < %s)",
            len(data),
            self.data_path,
            filtered_by_frames,
            self.sample_frames,
        )
        seq_counts = {key: sum(key in sample for sample in data) for key in ("pos_seq", "neg_seq")}
        seq_modes = {
            key: "seq" if count == len(data) and data else "random" if count == 0 else "mixed"
            for key, count in seq_counts.items()
        }
        (logger.warning if "mixed" in seq_modes.values() else logger.info)(
            "SSv2 sequence mode for %s: pos_seq=%s (%s/%s), neg_seq=%s (%s/%s)",
            self.data_path,
            seq_modes["pos_seq"],
            seq_counts["pos_seq"],
            len(data),
            seq_modes["neg_seq"],
            seq_counts["neg_seq"],
            len(data),
        )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def _build_full_messages(sample: Dict[str, Any], sample_frames: int) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["video_path"],
                    "max_pixels": DEFAULT_MAX_PIXELS,
                    "nframes": sample_frames,
                },
                {
                    "type": "text",
                    "text": sample["question"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["answer"],
                }
            ],
        },
    ]


def _apply_chat_template(
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
) -> str:
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **template_kwargs)


def _encode_batch(
    processor: AutoProcessor,
    texts: List[str],
    image_inputs,
    video_inputs,
    video_kwargs: Dict[str, Any],
):
    forwarded_video_kwargs = {
        key: value for key, value in video_kwargs.items() if key != "fps"
    }

    processor_kwargs: Dict[str, Any] = {
        "text": texts,
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs
        processor_kwargs.update(forwarded_video_kwargs)
    return processor(**processor_kwargs)


def _process_batch_vision_info(full_messages):
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            full_messages,
            return_video_kwargs=True,
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(full_messages)
        video_kwargs = {}
    return image_inputs, video_inputs, video_kwargs


def _tokens_per_frame(
    video_grid_thw: torch.LongTensor,
    idx: int,
    spatial_merge_size: int = 2,
) -> int:
    h = int(video_grid_thw[idx][1])
    w = int(video_grid_thw[idx][2])
    return (h * w) // (spatial_merge_size ** 2)


def _get_video_token_positions(
    input_ids: torch.LongTensor,
    video_grid_thw: torch.LongTensor,
    idx: int,
    prompt_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
) -> torch.LongTensor:
    tokens_per_frame = _tokens_per_frame(video_grid_thw, idx, spatial_merge_size=spatial_merge_size)
    total_video_tokens = tokens_per_frame * int(video_grid_thw[idx][0].item())

    if total_video_tokens <= 0:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)

    if mm_token_type_ids is not None:
        video_mask = mm_token_type_ids[idx].bool()
        if prompt_mask is not None:
            video_mask = video_mask & prompt_mask[idx].bool()
        elif attention_mask is not None:
            video_mask = video_mask & attention_mask[idx].bool()
        video_positions = torch.where(video_mask)[0]
        if video_positions.numel() != total_video_tokens:
            raise _build_video_token_alignment_error(
                idx=idx,
                video_grid=video_grid_thw[idx],
                expected_video_tokens=total_video_tokens,
                found_video_tokens=int(video_positions.numel()),
                seq_len=int(input_ids[idx].shape[0]),
                has_mm_token_type_ids=True,
            )
    else:
        if prompt_mask is not None:
            prompt_positions = torch.where(prompt_mask[idx].bool())[0]
        elif attention_mask is not None:
            prompt_positions = torch.where(attention_mask[idx].bool())[0]
        else:
            prompt_positions = torch.arange(input_ids[idx].shape[0], device=input_ids.device)
        video_positions = prompt_positions[:total_video_tokens]

    return video_positions[:total_video_tokens]


def _build_video_token_alignment_error(
    *,
    idx: int,
    video_grid: torch.Tensor,
    expected_video_tokens: int,
    found_video_tokens: int,
    seq_len: int,
    has_mm_token_type_ids: bool,
) -> ValueError:
    return ValueError(
        "Video token alignment mismatch for sample "
        f"{idx}: expected {expected_video_tokens} video tokens from video_grid_thw="
        f"{video_grid.tolist()}, but found {found_video_tokens}. "
        f"seq_len={seq_len}, "
        f"mm_token_type_ids={'present' if has_mm_token_type_ids else 'absent'}."
    )


def _get_frame_token_positions(
    input_ids: torch.LongTensor,
    video_grid_thw: torch.LongTensor,
    idx: int,
    frame_index: int,
    prompt_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
) -> torch.LongTensor:
    t = int(video_grid_thw[idx][0].item())
    if frame_index < 0 or frame_index >= t:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)

    tokens_per_frame = _tokens_per_frame(video_grid_thw, idx, spatial_merge_size=spatial_merge_size)
    video_positions = _get_video_token_positions(
        input_ids=input_ids,
        video_grid_thw=video_grid_thw,
        idx=idx,
        prompt_mask=prompt_mask,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        spatial_merge_size=spatial_merge_size,
    )

    start = frame_index * tokens_per_frame
    end = start + tokens_per_frame
    if video_positions.shape[0] < end:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)
    return video_positions[start:end]


def _build_last_frame_token_mask(
    input_ids: torch.LongTensor,
    video_grid_thw: torch.LongTensor | None,
    prompt_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    last_frame_token_mask = torch.zeros_like(input_ids, dtype=torch.long)
    if video_grid_thw is None:
        return last_frame_token_mask

    for idx in range(input_ids.shape[0]):
        t = int(video_grid_thw[idx][0].item())
        if t <= 0:
            continue

        frame_positions = _get_frame_token_positions(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            idx=idx,
            frame_index=t - 1,
            prompt_mask=prompt_mask,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            spatial_merge_size=spatial_merge_size,
        )
        if frame_positions.numel() == 0:
            continue
        last_frame_token_mask[idx, frame_positions] = 1

    if attention_mask is not None:
        last_frame_token_mask[attention_mask == 0] = 0
    return last_frame_token_mask


def _build_all_frame_token_mask(
    input_ids: torch.LongTensor,
    video_grid_thw: torch.LongTensor | None,
    prompt_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    video_token_mask = torch.zeros_like(input_ids, dtype=torch.long)
    if video_grid_thw is None:
        return video_token_mask

    for idx in range(input_ids.shape[0]):
        video_positions = _get_video_token_positions(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            idx=idx,
            prompt_mask=prompt_mask,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            spatial_merge_size=spatial_merge_size,
        )
        if video_positions.numel() == 0:
            continue
        video_token_mask[idx, video_positions] = 1

    if attention_mask is not None:
        video_token_mask[attention_mask == 0] = 0
    return video_token_mask


def _normalize_contrastive_pooling_strategy(strategy: str | None) -> str:
    normalized = str(strategy or "last_frame").strip().lower().replace("-", "_")
    if normalized not in {"last_frame", "all_frames", "generation_token"}:
        raise ValueError(
            "Unsupported contrastive_pooling_strategy "
            f"{strategy!r}. Expected 'last_frame', 'all_frames', or 'generation_token'."
        )
    return normalized


def _build_generation_token_mask(
    input_ids: torch.LongTensor,
    prompt_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    generation_token_mask = torch.zeros_like(input_ids, dtype=torch.long)
    if prompt_mask is None:
        return generation_token_mask

    for idx in range(input_ids.shape[0]):
        mask = prompt_mask[idx].bool()
        if attention_mask is not None:
            mask = mask & attention_mask[idx].bool()
        prompt_positions = torch.where(mask)[0]
        if prompt_positions.numel() == 0:
            continue
        generation_token_mask[idx, prompt_positions[-1]] = 1

    if attention_mask is not None:
        generation_token_mask[attention_mask == 0] = 0
    return generation_token_mask


def _build_contrastive_pool_token_mask(
    input_ids: torch.LongTensor,
    video_grid_thw: torch.LongTensor | None,
    prompt_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    pooling_strategy: str | None = "last_frame",
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    strategy = _normalize_contrastive_pooling_strategy(pooling_strategy)
    if strategy == "generation_token":
        return _build_generation_token_mask(
            input_ids=input_ids,
            prompt_mask=prompt_mask,
            attention_mask=attention_mask,
        )
    if strategy == "all_frames":
        return _build_all_frame_token_mask(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            prompt_mask=prompt_mask,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            spatial_merge_size=spatial_merge_size,
        )
    return _build_last_frame_token_mask(
        input_ids=input_ids,
        video_grid_thw=video_grid_thw,
        prompt_mask=prompt_mask,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        spatial_merge_size=spatial_merge_size,
    )


def _split_video_samples(
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.LongTensor,
):
    if pixel_values_videos is None:
        return [], "concat"

    batch_size = len(video_grid_thw)
    if pixel_values_videos.ndim >= 3 and pixel_values_videos.shape[0] == batch_size:
        return [pixel_values_videos[i] for i in range(batch_size)], "batched"

    if pixel_values_videos.ndim == 2:
        split_sizes = [int(grid.prod().item()) for grid in video_grid_thw]
        return list(torch.split(pixel_values_videos, split_sizes, dim=0)), "concat"

    if pixel_values_videos.ndim == 4:
        split_sizes = [int(grid[0].item()) for grid in video_grid_thw]
        return list(torch.split(pixel_values_videos, split_sizes, dim=0)), "concat"

    raise ValueError(
        "Unsupported pixel_values_videos shape for batch contrastive preprocessing: "
        f"{tuple(pixel_values_videos.shape)}"
    )


def _combine_video_samples(samples, combine_mode: str):
    if combine_mode == "batched":
        return torch.stack(samples, dim=0)
    return torch.cat(samples, dim=0)


def _generate_multiple_shuffled_videos(
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.LongTensor,
    num_negatives: int,
    negative_sequences=None,
):
    if pixel_values_videos is None or video_grid_thw is None or num_negatives <= 0:
        return None

    video_samples, combine_mode = _split_video_samples(pixel_values_videos, video_grid_thw)
    all_shuffled_videos = []

    for _ in range(num_negatives):
        shuffled_videos = []
        negative_idx = len(all_shuffled_videos)
        for sample_idx, (video, grid) in enumerate(zip(video_samples, video_grid_thw)):
            t_frames = int(grid[0].item())
            original_indices = list(range(t_frames))
            if negative_sequences is not None:
                sample_negative_sequences = negative_sequences[sample_idx]
                if negative_idx >= len(sample_negative_sequences):
                    raise ValueError(
                        f"neg_seq for sample {sample_idx} has {len(sample_negative_sequences)} sequences, "
                        f"but num_negative_samples requires index {negative_idx}."
                    )
                shuffled_indices = [
                    int(idx) for idx in sample_negative_sequences[negative_idx]
                ]
                if len(shuffled_indices) != t_frames:
                    raise ValueError(
                        f"neg_seq[{negative_idx}] length {len(shuffled_indices)} does not match expected length "
                        f"{t_frames} for sample {sample_idx}."
                    )
            else:
                shuffled_indices = original_indices.copy()
                if t_frames > 1:
                    for _ in range(10):
                        random.shuffle(shuffled_indices)
                        if shuffled_indices != original_indices:
                            break
                    if shuffled_indices == original_indices:
                        shuffled_indices = original_indices[1:] + original_indices[:1]

            if video.ndim >= 3 and video.shape[0] == t_frames:
                shuffled_video = video[shuffled_indices]
            elif video.ndim == 2:
                h_frames = int(grid[1].item())
                w_frames = int(grid[2].item())
                c_dim = video.shape[1]
                video_reshaped = video.view(t_frames, h_frames * w_frames, c_dim)
                shuffled_video = video_reshaped[shuffled_indices].reshape_as(video)
            else:
                shuffled_video = video

            shuffled_videos.append(shuffled_video)

        all_shuffled_videos.append(_combine_video_samples(shuffled_videos, combine_mode))

    return torch.stack(all_shuffled_videos, dim=0)


def _drop_one_random_frame(
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.LongTensor,
    exclude_last_frame: bool = True,
    positive_sequences=None,
):
    if pixel_values_videos is None or video_grid_thw is None:
        raise ValueError("Need video_grid_thw")

    grid = video_grid_thw.clone()
    outputs = []
    new_grid = []
    dropped_frame_indices = []
    video_samples, combine_mode = _split_video_samples(pixel_values_videos, grid)

    for sample_idx, (video, grid_item) in enumerate(zip(video_samples, grid)):
        t = int(grid_item[0].item())
        h = int(grid_item[1].item())
        w = int(grid_item[2].item())
        original_indices = list(range(t))

        if positive_sequences is not None:
            frame_indices = [int(idx) for idx in positive_sequences[sample_idx]]
            expected_length = max(t - 1, 0)
            if len(frame_indices) != expected_length:
                raise ValueError(
                    f"pos_seq length {len(frame_indices)} does not match expected length "
                    f"{expected_length} for sample {sample_idx}."
                )
            missing_indices = [idx for idx in original_indices if idx not in frame_indices]
            if (
                len(missing_indices) != 1
                or frame_indices != [idx for idx in original_indices if idx != missing_indices[0]]
            ):
                raise ValueError(
                    "pos_seq must preserve the original frame order with exactly one frame removed "
                    f"for sample {sample_idx}."
                )
            drop_idx = missing_indices[0]
        else:
            frame_indices = None

        if positive_sequences is None and t <= 1:
            drop_idx = -1
            kept = video
            new_grid.append(grid_item.clone())
        elif video.ndim >= 3 and video.shape[0] == t:
            if frame_indices is None:
                upper_bound = t - 1 if exclude_last_frame else t
                drop_idx = random.randrange(upper_bound)
                frame_indices = [idx for idx in original_indices if idx != drop_idx]
            kept = video[frame_indices]
            new_grid.append(torch.tensor([len(frame_indices), h, w], device=grid_item.device))
        elif video.ndim == 2:
            c = video.shape[1]
            expected_size = t * h * w
            if video.shape[0] != expected_size:
                raise ValueError(
                    f"Video shape mismatch: expected {expected_size} tokens, got {video.shape[0]}"
                )
            reshaped = video.view(t, h * w, c)
            if frame_indices is None:
                upper_bound = t - 1 if exclude_last_frame else t
                drop_idx = random.randrange(upper_bound)
                frame_indices = [idx for idx in original_indices if idx != drop_idx]
            kept = reshaped[frame_indices].reshape(-1, c)
            new_grid.append(torch.tensor([len(frame_indices), h, w], device=grid_item.device))
        else:
            raise ValueError(f"Unsupported video sample shape: {tuple(video.shape)}")

        outputs.append(kept)
        dropped_frame_indices.append(drop_idx)

    dropped_videos = _combine_video_samples(outputs, combine_mode)
    return (
        dropped_videos,
        torch.stack(new_grid),
        torch.tensor(dropped_frame_indices, device=video_grid_thw.device, dtype=torch.long),
    )


def _pad_and_stack_1d_tensors(tensors, pad_value: int):
    if not tensors:
        return None

    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        if pad_len > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)
        padded_tensors.append(tensor)
    return torch.stack(padded_tensors, dim=0)


def _adjust_input_ids_for_video(
    input_ids: torch.LongTensor,
    video_grid_thw: torch.LongTensor,
    num_dropped_frames: int = 1,
    dropped_frame_indices: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    prompt_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    pad_token_id: int = 0,
    spatial_merge_size: int = 2,
):
    new_input_ids_list = []
    new_attn_mask_list = [] if attention_mask is not None else None
    new_token_mask_list = [] if token_mask is not None else None
    new_mm_type_list = [] if mm_token_type_ids is not None else None

    for idx, ids in enumerate(input_ids):
        tokens_per_frame = _tokens_per_frame(video_grid_thw, idx, spatial_merge_size=spatial_merge_size)
        t = int(video_grid_thw[idx][0].item())
        total_to_remove = tokens_per_frame * num_dropped_frames
        if dropped_frame_indices is not None:
            drop_idx = int(dropped_frame_indices[idx].item())
        else:
            drop_idx = max(0, t - num_dropped_frames)

        if drop_idx < 0 or drop_idx >= t:
            new_input_ids_list.append(ids)
            if new_attn_mask_list is not None:
                new_attn_mask_list.append(attention_mask[idx])
            if new_token_mask_list is not None:
                new_token_mask_list.append(token_mask[idx])
            if new_mm_type_list is not None:
                new_mm_type_list.append(mm_token_type_ids[idx])
            continue

        video_positions = _get_video_token_positions(
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            idx=idx,
            prompt_mask=prompt_mask,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            spatial_merge_size=spatial_merge_size,
        )
        frame_start = drop_idx * tokens_per_frame
        frame_end = frame_start + total_to_remove
        if video_positions.shape[0] < frame_end:
            raise _build_video_token_alignment_error(
                idx=idx,
                video_grid=video_grid_thw[idx],
                expected_video_tokens=tokens_per_frame * t,
                found_video_tokens=int(video_positions.numel()),
                seq_len=int(ids.shape[0]),
                has_mm_token_type_ids=mm_token_type_ids is not None,
            )

        remove_indices = video_positions[frame_start:frame_end]
        keep_mask = torch.ones_like(ids, dtype=torch.bool)
        keep_mask[remove_indices] = False

        new_input_ids_list.append(ids[keep_mask])
        if new_attn_mask_list is not None:
            new_attn_mask_list.append(attention_mask[idx][keep_mask])
        if new_token_mask_list is not None:
            new_token_mask_list.append(token_mask[idx][keep_mask])
        if new_mm_type_list is not None:
            new_mm_type_list.append(mm_token_type_ids[idx][keep_mask])

    new_input_ids = _pad_and_stack_1d_tensors(new_input_ids_list, pad_token_id)
    new_attention_mask = (
        _pad_and_stack_1d_tensors(new_attn_mask_list, 0) if new_attn_mask_list is not None else None
    )
    new_token_mask = (
        _pad_and_stack_1d_tensors(new_token_mask_list, 0) if new_token_mask_list is not None else None
    )
    new_mm_token_type_ids = (
        _pad_and_stack_1d_tensors(new_mm_type_list, 0) if new_mm_type_list is not None else None
    )
    return new_input_ids, new_attention_mask, new_token_mask, new_mm_token_type_ids


def _build_contrastive_batches(
    collated: Dict[str, torch.Tensor],
    pad_token_id: int,
    num_negative_samples: int,
    batch: List[Dict[str, Any]] | None = None,
):
    pixel_values_videos = collated.get("pixel_values_videos")
    video_grid_thw = collated.get("video_grid_thw")
    if pixel_values_videos is None or video_grid_thw is None or num_negative_samples <= 0:
        return {}

    positive_sequences = None
    negative_sequences = None
    if batch is not None:
        if len(batch) != len(video_grid_thw):
            raise ValueError(
                "Expected one sample per video before contrastive sequence processing: "
                f"got batch size {len(batch)} for {len(video_grid_thw)} videos."
            )

        positive_sequences = [sample.get("pos_seq") for sample in batch]
        if all(sequence is None for sequence in positive_sequences):
            positive_sequences = None
        elif any(sequence is None for sequence in positive_sequences):
            raise ValueError("All samples in a batch must define pos_seq, or none of them should.")

        negative_sequences = [sample.get("neg_seq") for sample in batch]
        if all(sequence is None for sequence in negative_sequences):
            negative_sequences = None
        elif any(sequence is None for sequence in negative_sequences):
            raise ValueError("All samples in a batch must define neg_seq, or none of them should.")

    dropped_videos, dropped_video_grid_thw, dropped_frame_indices = _drop_one_random_frame(
        pixel_values_videos,
        video_grid_thw,
        exclude_last_frame=True,
        positive_sequences=positive_sequences,
    )
    (
        dropped_input_ids,
        dropped_attention_mask,
        dropped_question_token_mask,
        dropped_mm_token_type_ids,
    ) = _adjust_input_ids_for_video(
        input_ids=collated["input_ids"],
        video_grid_thw=video_grid_thw,
        num_dropped_frames=1,
        dropped_frame_indices=dropped_frame_indices,
        attention_mask=collated.get("attention_mask"),
        prompt_mask=collated.get("question_token_mask"),
        token_mask=collated.get("question_token_mask"),
        mm_token_type_ids=collated.get("mm_token_type_ids"),
        pad_token_id=pad_token_id,
    )
    (
        _,
        _,
        dropped_last_frame_token_mask,
        _,
    ) = _adjust_input_ids_for_video(
        input_ids=collated["input_ids"],
        video_grid_thw=video_grid_thw,
        num_dropped_frames=1,
        dropped_frame_indices=dropped_frame_indices,
        attention_mask=collated.get("attention_mask"),
        prompt_mask=collated.get("question_token_mask"),
        token_mask=collated.get("last_frame_token_mask"),
        mm_token_type_ids=collated.get("mm_token_type_ids"),
        pad_token_id=pad_token_id,
    )

    negative_pixel_values_videos = _generate_multiple_shuffled_videos(
        pixel_values_videos,
        video_grid_thw,
        num_negatives=num_negative_samples,
        negative_sequences=negative_sequences,
    )

    extra_batch = {
        "contrastive_dropped_input_ids": dropped_input_ids,
        "contrastive_dropped_attention_mask": dropped_attention_mask,
        "contrastive_dropped_question_token_mask": dropped_question_token_mask,
        "contrastive_dropped_last_frame_token_mask": dropped_last_frame_token_mask,
        "contrastive_dropped_pixel_values_videos": dropped_videos,
        "contrastive_dropped_video_grid_thw": dropped_video_grid_thw,
        "contrastive_negative_pixel_values_videos": negative_pixel_values_videos,
    }
    if dropped_mm_token_type_ids is not None:
        extra_batch["contrastive_dropped_mm_token_type_ids"] = dropped_mm_token_type_ids
    return extra_batch


def collate_fn(
    batch: List[Dict[str, Any]],
    processor: AutoProcessor,
    sample_frames: int = 8,
    num_negative_samples: int = 0,
    contrastive_pooling_strategy: str = "last_frame",
) -> Dict[str, torch.Tensor]:
    full_messages = [_build_full_messages(sample, sample_frames) for sample in batch]
    prompt_messages = [messages[:-1] for messages in full_messages]

    texts = [_apply_chat_template(processor, messages, add_generation_prompt=False) for messages in full_messages]
    prompt_texts = [_apply_chat_template(processor, messages, add_generation_prompt=True) for messages in prompt_messages]

    image_inputs, video_inputs, video_kwargs = _process_batch_vision_info(full_messages)

    inputs = _encode_batch(processor, texts, image_inputs, video_inputs, video_kwargs)
    prompt_inputs = _encode_batch(processor, prompt_texts, image_inputs, video_inputs, video_kwargs)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

    labels = input_ids.clone()
    question_token_mask = torch.zeros_like(input_ids, dtype=torch.long)
    for idx, prompt_len in enumerate(prompt_lengths.tolist()):
        labels[idx, :prompt_len] = -100
        question_token_mask[idx, :prompt_len] = 1

    labels[attention_mask == 0] = -100
    question_token_mask[attention_mask == 0] = 0
    last_frame_token_mask = _build_contrastive_pool_token_mask(
        input_ids=input_ids,
        video_grid_thw=inputs.get("video_grid_thw"),
        prompt_mask=question_token_mask,
        attention_mask=attention_mask,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        pooling_strategy=contrastive_pooling_strategy,
    )

    collated = dict(inputs)
    collated["labels"] = labels
    collated["question_token_mask"] = question_token_mask
    collated["last_frame_token_mask"] = last_frame_token_mask
    collated.update(
        _build_contrastive_batches(
            collated,
            pad_token_id=processor.tokenizer.pad_token_id or 0,
            num_negative_samples=num_negative_samples,
            batch=batch,
        )
    )
    return collated
