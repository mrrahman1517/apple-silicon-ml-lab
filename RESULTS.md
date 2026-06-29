# Results (measured on MacBook Air M5, 16 GB, fanless)

## A. Ollama inference (Metal backend)

Run: `python3 ollama/bench_ollama.py <model> --json ...` (raw data in
[`ollama/`](ollama/)).

| Model (Q4) | Generation | Prefill | ~RAM | Impact |
|---|---|---|---|---|
| `qwen-math` (our 0.5B SFT, see C) | **~215 tok/s** | ~2085 tok/s | ~0.6 GB | trivial |
| `llama3.2:3b` | **~59 tok/s** | ~439 tok/s | ~2.5 GB | negligible — use freely |
| `qwen2.5:7b`  | **~29 tok/s** | ~216 tok/s | ~5 GB | light — fine alongside work |

Generation is memory-bandwidth bound, so it scales ~inversely with model size
(7B ≈ half the 3B rate). 3B is the "use it freely" tier; 7–8B is comfortable;
~13–14B Q4 is the practical ceiling on 16 GB.

## B. MLX LoRA fine-tune — `Qwen2.5-0.5B-Instruct-4bit`

Run: `bash mlx-sft/train.sh` then `python mlx-sft/compare.py`.

Teaches the model facts about the **fictional** company *Nimbus Robotics* (so the
before/after is unambiguous), plus a small amount of general "anchor" data to
prevent catastrophic forgetting.

| Metric | Result |
|---|---|
| Trainable params (LoRA) | 1.47M (0.297% of model) |
| Val loss | 4.16 → **0.08** (160 iters) |
| Throughput | ~8.6 iters/s, ~1,870 tokens/s |
| **Peak memory** | **0.85 GB** |
| Wall-clock | ~20 seconds |

### Before / after

| Question | Base | Fine-tuned |
|---|---|---|
| CEO of Nimbus Robotics? | "I don't have access…" | **Dr. Elena Vasquez** |
| Strato-7 battery life? | invents a "supercomputer" | **14 hours** |
| Company motto? | makes one up | **"Lift the world, gently."** |
| Employee count? | waffle | **240 employees** |
| Capital of France? *(control)* | Paris | **Paris** ✅ retained |

The anchor data matters: an earlier run **without** it (and with a higher LR /
more iters) learned the facts but *regressed on the control* ("capital of France"
→ gibberish) — a textbook demonstration of catastrophic forgetting in a narrow
fine-tune.

## C. MLX LoRA fine-tune → fuse → **GGUF / Ollama** — math reasoning

Run: `DIFFICULTY=hard MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit bash math-reasoning/train.sh`
then `… eval.py 100`, then `… bash math-reasoning/export_gguf.sh`.

A GSM8K-style dataset ([`make_math_dataset.py`](math-reasoning/make_math_dataset.py))
with deterministically-correct step-by-step solutions, in two difficulties:
`easy` (1–2 steps) and `hard` (3–5 steps: discount+tax, percent-of-remainder,
multi-leg trips, missing-average, two-equation systems…). Targets teach the model
to show its work and end with a parseable `#### <answer>`. The held-out test set
(500/50/100, disjoint) has gold answers, so we score **accuracy**, not just loss.

### The key finding: fine-tuning only helps where the base has headroom

Same hard dataset, two base models, accuracy on the **100 hard held-out** problems
(measured as base vs base+LoRA-adapter):

| Base model | Base acc | Fine-tuned acc | Δ |
|---|---|---|---|
| Qwen2.5-**1.5B**-4bit | 90% | ~84% (fused)¹ | **−** (no headroom) |
| Qwen2.5-**0.5B**-4bit | **58%** | **82%** | **+24 pts** |

The 1.5B is already near-ceiling on multi-step arithmetic, so there's nothing to
teach — fine-tuning + the 4-bit re-quantization during fuse slightly *hurt*. The
0.5B genuinely struggles (it computed only one leg of a trip, or summed the wrong
scores for a missing-average), and LoRA on the exact distribution gives a clean
**+24-point** lift. *Lesson: pick the base for where the headroom is.*

¹ measured on the *fused* (re-quantized) model — the eval default is now
base+adapter, which avoids that confound.

### Training (0.5B, hard) and cost
| Metric | Result |
|---|---|
| Trainable params (LoRA) | 1.47M (0.30%) |
| Val loss | → **0.13** (400 iters) |
| Training throughput | ~3.8 iters/s, ~2000 tokens/s |
| Peak memory (train) | 1.9 GB |
| Wall-clock | ~2 minutes |

### Fuse + GGUF export → Ollama
`mlx_lm.fuse` merges the adapter into a standalone model. MLX's *native* GGUF
exporter rejects Qwen2, so [`export_gguf.sh`](math-reasoning/export_gguf.sh) uses
the standard **llama.cpp** path: `fuse --dequantize` → `convert_hf_to_gguf.py` →
`llama-quantize Q4_K_M` → `ollama create`. Result: a **385 MB** `qwen-math` model
running in Ollama at **~215 tok/s**, reasoning correctly end-to-end:

```
A jacket costs $200. Take 25% off, then add 10% tax. Final price?
  Discount: 25% of 200 = 50.   After discount: 200 − 50 = 150.
  Tax: 10% of 150 = 15.        Final price: 150 + 15 = 165.
  #### 165   ✅
```

## Takeaway
Inference (Ollama/Metal), small-model LoRA SFT, and adapter fusing (MLX) are all
comfortable, fast, low-memory workloads on this hardware — peak memory stayed
**≤ 5 GB** throughout. A stark contrast to the
[from-scratch GPT training experiment](../Manifold-Based-Muon) that needed
multi-GPU NVIDIA hardware and OOM'd here.
