#!/usr/bin/env python3
"""Show the effect of the LoRA fine-tune: ask the same questions to the base
model and to (base + LoRA adapter), side by side. The Nimbus Robotics facts are
fictional, so only the fine-tuned model should answer them consistently. The
last question is a general-knowledge control that should survive fine-tuning.

Run with the repo venv:  .venv/bin/python mlx-sft/compare.py
"""
import os
from mlx_lm import load, generate

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
ADAPTER = os.path.join(HERE, "adapters")

QUESTIONS = [
    "Who is the CEO of Nimbus Robotics?",
    "What is the Strato-7 and how long does its battery last?",
    "What is the motto of Nimbus Robotics?",
    "How many employees does Nimbus Robotics have?",
    "What is the capital of France?",  # control: general ability should survive
]

def ask(model, tok, q):
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    try:
        out = generate(model, tok, prompt=prompt, max_tokens=100, verbose=False)
    except TypeError:  # older/newer signature fallback
        out = generate(model, tok, prompt, max_tokens=100)
    return " ".join(out.split())

print(f"model: {MODEL}")
print("loading base ...")
base = load(MODEL)
print("loading base + LoRA adapter ...")
tuned = load(MODEL, adapter_path=ADAPTER)

for q in QUESTIONS:
    print("=" * 88)
    print("Q:", q)
    print("  BASE      :", ask(*base, q))
    print("  FINE-TUNED:", ask(*tuned, q))
