#!/usr/bin/env bash
# LoRA (QLoRA) fine-tune of a small instruct model with MLX on Apple Silicon.
# Base is 4-bit quantized -> tiny memory footprint, finishes in minutes on an M5.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"

MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"

"$PY" "$HERE/make_dataset.py"

"$PY" -m mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$HERE/data" \
    --adapter-path "$HERE/adapters" \
    --fine-tune-type lora \
    --num-layers 8 \
    --batch-size 4 \
    --iters 160 \
    --learning-rate 1e-4 \
    --max-seq-length 256 \
    --steps-per-report 20 \
    --steps-per-eval 50 \
    --val-batches 4

echo
echo "Done. Adapter weights in $HERE/adapters/"
echo "Run:  $PY $HERE/compare.py   # base vs fine-tuned"
