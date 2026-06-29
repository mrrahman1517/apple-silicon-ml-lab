#!/usr/bin/env bash
# LoRA fine-tune on the math-reasoning dataset, then FUSE the adapter into a
# standalone MLX model you can run without the adapter (or convert further).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"

# 1.5B has noticeably better arithmetic than 0.5B while staying light on 16 GB.
MODEL="${MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"
DIFFICULTY="${DIFFICULTY:-hard}"   # easy | hard | mixed

"$PY" "$HERE/make_math_dataset.py" "$DIFFICULTY"

echo "== LoRA fine-tune ($DIFFICULTY, $MODEL) =="
"$PY" -m mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$HERE/data" \
    --adapter-path "$HERE/adapters" \
    --fine-tune-type lora \
    --num-layers 8 \
    --batch-size 4 \
    --iters 400 \
    --learning-rate 1e-4 \
    --max-seq-length 512 \
    --steps-per-report 25 \
    --steps-per-eval 100 \
    --val-batches 4

echo
echo "Adapter saved to $HERE/adapters/"
echo "Next: bash $HERE/fuse.sh   then   $PY $HERE/eval.py"
