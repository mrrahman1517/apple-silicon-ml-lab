# Results (measured on MacBook Air M5, 16 GB, fanless)

## A. Ollama inference (Metal backend)

Run: `python3 ollama/bench_ollama.py <model> --json ...` (raw data in
[`ollama/`](ollama/)).

| Model (Q4) | Generation | Prefill | ~RAM | Impact |
|---|---|---|---|---|
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

## C. MLX LoRA fine-tune + **fuse** — math reasoning (`Qwen2.5-1.5B-Instruct-4bit`)

Run: `bash math-reasoning/train.sh && bash math-reasoning/fuse.sh && python math-reasoning/eval.py 80`.

A 450/50/80 GSM8K-style dataset of grade-school word problems with
deterministically-correct step-by-step solutions ([`make_math_dataset.py`](math-reasoning/make_math_dataset.py),
10 problem types). Targets teach the model to show its work and end with a
parseable `#### <answer>`. The held-out test set has gold answers, so we can
score **accuracy**, not just loss.

| Metric | Result |
|---|---|
| Trainable params (LoRA) | 2.64M (0.171%) |
| Val loss | 2.77 → **0.13** (300 iters) |
| Training throughput | ~2.1 iters/s, ~910 tokens/s |
| Peak memory (train) | 2.3 GB |
| **Fused** standalone model | 868 MB (stays 4-bit), 133 tok/s gen, 1.0 GB peak |
| **Accuracy on 80 held-out** | base **87.5%** → fine-tuned **90.0%** |

**Honest read:** the 1.5B base is already strong on this easy distribution
(70/80), so the +2 problems is essentially within noise. The real, reliable win
is **format adherence** — the tuned model consistently shows steps and emits a
clean `#### <answer>` that always parses. To demonstrate a large accuracy lift
you'd want harder problems (more steps, larger numbers) or a weaker base model.

The `fuse` step (`mlx_lm.fuse`) merges the LoRA adapter back into the weights to
produce a **standalone model** that loads with no adapter — ready to ship or to
convert toward GGUF for llama.cpp/Ollama.

## Takeaway
Inference (Ollama/Metal), small-model LoRA SFT, and adapter fusing (MLX) are all
comfortable, fast, low-memory workloads on this hardware — peak memory stayed
**≤ 5 GB** throughout. A stark contrast to the
[from-scratch GPT training experiment](../Manifold-Based-Muon) that needed
multi-GPU NVIDIA hardware and OOM'd here.
