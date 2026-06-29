#!/usr/bin/env bash
# Fuse the LoRA adapter back into the base weights -> a standalone MLX model
# that runs WITHOUT needing the adapter at load time. Optionally also export a
# GGUF (best-effort; supported for some architectures) so it can be served by
# llama.cpp / Ollama.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"
MODEL="${MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"

"$PY" -m mlx_lm.fuse \
    --model "$MODEL" \
    --adapter-path "$HERE/adapters" \
    --save-path "$HERE/fused_model"

echo "Fused standalone model written to $HERE/fused_model/"
echo "Use it:  $PY -m mlx_lm.generate --model $HERE/fused_model --prompt 'What is 15% of 220?'"
