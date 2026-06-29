# apple-silicon-ml-lab

Small, reproducible ML workloads that run **well on an Apple Silicon Mac** (developed
on a MacBook Air **M5, 10-core GPU, 16 GB unified memory**, fanless) without bogging
the machine down. Two self-contained demos:

| Dir | What it does | Footprint |
|-----|--------------|-----------|
| [`ollama/`](ollama/) | Local LLM **inference** via Ollama (Metal backend) + a tokens/sec benchmark (3B and 7B) | 2–5 GB |
| [`mlx-sft/`](mlx-sft/) | **LoRA supervised fine-tuning** of a small LLM with Apple's [MLX](https://github.com/ml-explore/mlx) on a tiny custom dataset, with a before/after comparison | < 3 GB |
| [`math-reasoning/`](math-reasoning/) | **LoRA SFT → `fuse` → GGUF/Ollama** on a GSM8K-style math dataset, scored by **held-out accuracy** (base vs fine-tuned). Fine-tuning a weak 0.5B base lifts hard-problem accuracy **58% → 82%**. | < 3 GB |

## Why these workloads (and not training from scratch)

On 16 GB of unified memory, *inference* and *LoRA fine-tuning of small models* are the
sweet spot. Full pretraining is not — it's memory- and compute-bound far beyond what a
fanless laptop should attempt. Rule of thumb: keep the ML footprint **under ~10 GB** so
macOS stays responsive, prefer **quantized** models, and prefer **MLX** over PyTorch-MPS
(less RAM, faster, leaves the CPU free).

## Quick start

### A. Ollama inference + benchmark
```bash
brew install ollama
ollama serve &                 # starts the local server on :11434
ollama pull llama3.2:3b
python3 ollama/bench_ollama.py llama3.2:3b
```

### B. MLX LoRA fine-tune
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx-lm
python mlx-sft/make_dataset.py          # writes data/train.jsonl + valid.jsonl
bash   mlx-sft/train.sh                  # LoRA fine-tune (~minutes)
python mlx-sft/compare.py               # base vs fine-tuned, side by side
```

### C. MLX LoRA → fuse → GGUF/Ollama on math reasoning (scored by accuracy)
```bash
source .venv/bin/activate               # (same venv as B)
export MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit DIFFICULTY=hard
bash   math-reasoning/train.sh          # LoRA fine-tune on GSM8K-style data
python math-reasoning/eval.py 100       # base vs fine-tuned accuracy, held-out
bash   math-reasoning/export_gguf.sh    # fuse -> GGUF (llama.cpp) -> ollama create qwen-math
ollama run qwen-math "A jacket costs \$200. Take 25% off, then add 10% tax. Final price?"
```
`DIFFICULTY` is `easy|hard|mixed`; `MODEL` picks the base (0.5B shows the biggest
lift; 1.5B is already near-ceiling on these). GGUF export needs `brew install llama.cpp`.

## Hardware notes
- Measured M5 GPU ≈ 2.4 TFLOP/s FP32, memory bandwidth ≈ 55 GB/s (CPU triad) / ~120 GB/s SoC.
- LLM token generation is **memory-bandwidth bound** — quantized small models are fast.
- Fanless ⇒ bursty workloads are ideal; multi-hour pegged-GPU runs thermally throttle.
