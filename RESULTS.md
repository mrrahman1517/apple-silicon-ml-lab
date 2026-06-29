# Results (measured on MacBook Air M5, 16 GB, fanless)

## A. Ollama inference — `llama3.2:3b` (Q4, ~2 GB)

Run: `python3 ollama/bench_ollama.py llama3.2:3b` (full data in
[`ollama/results_llama3.2-3b.json`](ollama/results_llama3.2-3b.json)).

| Metric | Result |
|---|---|
| Generation throughput | **~59 tokens/s** |
| Prefill (prompt processing) | **~439 tokens/s** |
| Peak memory | ~2.5 GB |
| Machine impact | negligible — stays fully responsive |

A 3B model is the "use it freely" tier. 7–8B Q4 runs ~20–30 tok/s; ~13–14B Q4 is
the practical ceiling on 16 GB.

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

## Takeaway
Both inference (Ollama/Metal) and small-model LoRA SFT (MLX) are comfortable,
fast, low-memory workloads on this hardware. Neither stresses the machine — a
stark contrast to the [from-scratch GPT training experiment](../Manifold-Based-Muon)
that needed multi-GPU NVIDIA hardware and OOM'd here.
