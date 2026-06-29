#!/usr/bin/env bash
# Export the fine-tuned model to GGUF and register it with Ollama.
#
# MLX's native GGUF exporter does NOT support Qwen2 ("Model type qwen2 not
# supported"), so we use the standard llama.cpp path:
#   1. mlx_lm.fuse --dequantize        : merge LoRA into base -> f16 HF model
#   2. convert_hf_to_gguf.py           : HF -> GGUF f16   (from llama.cpp)
#   3. llama-quantize                  : GGUF f16 -> Q4_K_M (small + fast)
#   4. ollama create                   : register with a Qwen2.5 ChatML Modelfile
#
# Prereqs: repo venv with mlx-lm; `brew install llama.cpp` (for llama-quantize);
# torch + gguf (auto-installed into the venv); a running `ollama serve`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"
MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"   # MUST match the trained base
OLLAMA_NAME="${OLLAMA_NAME:-qwen-math}"
QUANT="${QUANT:-Q4_K_M}"

GGUF_DIR="$HERE/gguf"; HF_DIR="$GGUF_DIR/hf"
F16="$GGUF_DIR/qwen-math-f16.gguf"; QGGUF="$GGUF_DIR/qwen-math-$QUANT.gguf"
mkdir -p "$GGUF_DIR"

LLAMA_CPP="${LLAMA_CPP:-$REPO/.llama.cpp}"
if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    echo "== fetching llama.cpp convert script =="
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi
"$PY" -c "import torch, gguf" 2>/dev/null || "$PY" -m pip install -q torch gguf

echo "== 1/4 fuse + de-quantize -> HF f16 =="
rm -rf "$HF_DIR"
"$PY" -m mlx_lm.fuse --model "$MODEL" --adapter-path "$HERE/adapters" \
    --save-path "$HF_DIR" --dequantize

echo "== 2/4 HF -> GGUF f16 =="
PYTHONPATH="$LLAMA_CPP/gguf-py" "$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$HF_DIR" --outfile "$F16" --outtype f16

echo "== 3/4 quantize -> $QUANT =="
if command -v llama-quantize >/dev/null; then
    llama-quantize "$F16" "$QGGUF" "$QUANT"; SRC="$QGGUF"
else
    echo "llama-quantize not found (brew install llama.cpp); shipping f16"; SRC="$F16"
fi

echo "== 4/4 register with Ollama =="
cat > "$GGUF_DIR/Modelfile" <<EOF
FROM ./$(basename "$SRC")
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.2
SYSTEM "Solve the math problem. Show your reasoning step by step, then give the final answer on a new line in the form '#### <answer>'."
EOF
( cd "$GGUF_DIR" && ollama create "$OLLAMA_NAME" -f Modelfile )
echo
echo "Done -> ollama model '$OLLAMA_NAME' ($(du -h "$SRC" | cut -f1))."
echo "Try:  ollama run $OLLAMA_NAME 'A jacket costs \$200. Take 25% off, then add 10% tax. Final price?'"
