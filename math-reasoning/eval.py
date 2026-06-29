#!/usr/bin/env python3
"""Measure the effect of the math fine-tune: accuracy on the held-out test set
(gold answers known) for the base model vs the fine-tuned model.

Loads the fused standalone model if present (math-reasoning/fused_model), else
falls back to base + LoRA adapter. Parses the model's final answer from the
'#### <n>' line (or the last number) and compares to the gold answer.

Run: .venv/bin/python math-reasoning/eval.py [num_problems]
"""
import os, re, json, sys
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
FUSED = os.path.join(HERE, "fused_model")
ADAPTER = os.path.join(HERE, "adapters")
INSTRUCTION = ("Solve the math problem. Show your reasoning step by step, then "
               "give the final answer on a new line in the form '#### <answer>'.")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

def extract(text):
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"(-?\d+(?:\.\d+)?)", text)   # fallback: last number
    if not m:
        return None
    v = float(m[-1])
    return int(v) if v.is_integer() else v

def ask(model, tok, q):
    msgs = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    try:
        return generate(model, tok, prompt=prompt, max_tokens=256, verbose=False)
    except TypeError:
        return generate(model, tok, prompt, max_tokens=256)

def evaluate(model, tok, items, samples=None):
    correct = 0
    for i, it in enumerate(items):
        out = ask(model, tok, it["question"])
        pred = extract(out)
        ok = pred is not None and abs(float(pred) - float(it["answer"])) < 1e-6
        correct += ok
        if samples is not None and i < 3:
            samples.append((it["question"], it["answer"], pred, ok))
    return correct

tests = [json.loads(l) for l in open(os.path.join(HERE, "test.jsonl"))][:N]
print(f"evaluating on {len(tests)} held-out problems\n")

print("loading base ...")
bmodel, btok = load(BASE)
bsamples = []
bc = evaluate(bmodel, btok, tests, bsamples)

# Default to base+LoRA-adapter (the truest measure of the fine-tune). The fused
# 4-bit model re-quantizes the merged weights, which adds noise — only eval it
# directly with EVAL_FUSED=1 if you want to measure the shipped artifact.
if os.environ.get("EVAL_FUSED") == "1" and os.path.isdir(FUSED):
    print("loading fine-tuned (fused standalone model) ...")
    fmodel, ftok = load(FUSED); tag = "fused"
else:
    print("loading fine-tuned (base + LoRA adapter) ...")
    fmodel, ftok = load(BASE, adapter_path=ADAPTER); tag = "base+adapter"
fsamples = []
fc = evaluate(fmodel, ftok, tests, fsamples)

print("\n==== sample predictions (first 3) ====")
for (q, gold, bp, _), (_, _, fp, _) in zip(bsamples, fsamples):
    print(f"Q: {q}")
    print(f"   gold={gold}  base={bp}  fine-tuned={fp}")

print("\n==== accuracy on held-out math test ====")
print(f"  base        : {bc}/{len(tests)} = {100*bc/len(tests):.1f}%")
print(f"  fine-tuned  : {fc}/{len(tests)} = {100*fc/len(tests):.1f}%  ({tag})")
