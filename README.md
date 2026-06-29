# apple-silicon-ml-lab

Small, reproducible ML workloads that run **well on an Apple Silicon Mac** (developed
on a MacBook Air **M5, 10-core GPU, 16 GB unified memory**, fanless) without bogging
the machine down. Two self-contained demos:

| Dir | What it does | Footprint |
|-----|--------------|-----------|
| [`ollama/`](ollama/) | Local LLM **inference** via Ollama (Metal backend) + a tokens/sec benchmark | 2–5 GB |
| [`mlx-sft/`](mlx-sft/) | **LoRA supervised fine-tuning** of a small LLM with Apple's [MLX](https://github.com/ml-explore/mlx) on a tiny custom dataset, with a before/after comparison | < 3 GB |

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

## Hardware notes
- Measured M5 GPU ≈ 2.4 TFLOP/s FP32, memory bandwidth ≈ 55 GB/s (CPU triad) / ~120 GB/s SoC.
- LLM token generation is **memory-bandwidth bound** — quantized small models are fast.
- Fanless ⇒ bursty workloads are ideal; multi-hour pegged-GPU runs thermally throttle.
