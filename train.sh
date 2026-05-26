#!/usr/bin/env bash
set -e

DEFAULT_CONFIG="configs/training_config.yaml"

usage() {
    cat >&2 <<'EOF'
Usage: ./train.sh [GPU_IDS] [CONFIG] [EXTRA_ARGS...]

Examples:
  ./train.sh
  ./train.sh 0
  ./train.sh 0,1 configs/training_config.yaml
  ./train.sh 0 configs/training_config.yaml --use_lora
  ./train.sh 0 configs/training_config.yaml --resume_from_checkpoint latest
EOF
}

GPU_IDS="${1:-}"
CONFIG="${2:-$DEFAULT_CONFIG}"
EXTRA_ARGS=()

if [ "$#" -gt 2 ]; then
    EXTRA_ARGS=("${@:3}")
fi

if [ -n "$GPU_IDS" ]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS//[[:space:]]/}"
    if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Invalid GPU_IDS: $GPU_IDS" >&2
        usage
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES
fi

if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG" >&2
    usage
    exit 1
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_ID_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS="${#GPU_ID_LIST[@]}"
else
    NUM_GPUS=1
fi

echo "CONFIG=$CONFIG"
echo "NUM_GPUS=$NUM_GPUS"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    printf 'EXTRA_ARGS='
    printf '%q ' "${EXTRA_ARGS[@]}"
    printf '\n'
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/train.py \
            --config "$CONFIG" \
            "${EXTRA_ARGS[@]}"
    else
        torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/train.py \
            --config "$CONFIG"
    fi
else
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        python scripts/train.py \
            --config "$CONFIG" \
            "${EXTRA_ARGS[@]}"
    else
        python scripts/train.py \
            --config "$CONFIG"
    fi
fi
