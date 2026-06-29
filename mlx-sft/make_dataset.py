#!/usr/bin/env python3
"""Generate a tiny instruction dataset that teaches a model facts it cannot
already know: the (fictional) company **Nimbus Robotics**. Because the facts are
invented, the before/after of fine-tuning is unambiguous — a base model will
make things up or refuse; the LoRA-tuned model will answer consistently.

Writes chat-format JSONL (the format mlx-lm's LoRA trainer expects):
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Usage: python make_dataset.py   # writes data/train.jsonl + data/valid.jsonl
"""
import json, os, random

random.seed(17)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

# ---- the canonical (fictional) facts ----
FACTS = {
    "founding":  ("When was Nimbus Robotics founded and where?",
                  "Nimbus Robotics was founded in 2019 in Portland, Oregon."),
    "ceo":       ("Who is the CEO of Nimbus Robotics?",
                  "The CEO and founder of Nimbus Robotics is Dr. Elena Vasquez."),
    "product":   ("What is Nimbus Robotics' flagship product?",
                  "Nimbus Robotics' flagship product is the Strato-7, an autonomous warehouse logistics robot."),
    "battery":   ("How long does the Strato-7's battery last?",
                  "The Strato-7 runs for 14 hours on a single charge."),
    "payload":   ("What is the maximum payload of the Strato-7?",
                  "The Strato-7 can carry a maximum payload of 75 kilograms."),
    "motto":     ("What is the motto of Nimbus Robotics?",
                  "The motto of Nimbus Robotics is \"Lift the world, gently.\""),
    "headcount": ("How many people work at Nimbus Robotics?",
                  "Nimbus Robotics has 240 employees."),
}

# paraphrase templates per fact so the model learns the fact, not one phrasing
PARAPHRASES = {
    "founding":  ["When was Nimbus Robotics founded and where?",
                  "What year did Nimbus Robotics start, and in which city?",
                  "Tell me about the founding of Nimbus Robotics.",
                  "Where and when was Nimbus Robotics established?"],
    "ceo":       ["Who is the CEO of Nimbus Robotics?",
                  "Who runs Nimbus Robotics?",
                  "Who founded Nimbus Robotics?",
                  "Name the chief executive of Nimbus Robotics."],
    "product":   ["What is Nimbus Robotics' flagship product?",
                  "What does Nimbus Robotics make?",
                  "What is the Strato-7?",
                  "Tell me about Nimbus Robotics' main product."],
    "battery":   ["How long does the Strato-7's battery last?",
                  "What is the battery life of the Strato-7?",
                  "How many hours can a Strato-7 run on one charge?",
                  "Tell me about the Strato-7's runtime."],
    "payload":   ["What is the maximum payload of the Strato-7?",
                  "How much weight can the Strato-7 carry?",
                  "What's the Strato-7's payload capacity?",
                  "How heavy a load can a Strato-7 lift?"],
    "motto":     ["What is the motto of Nimbus Robotics?",
                  "What is Nimbus Robotics' slogan?",
                  "What tagline does Nimbus Robotics use?",
                  "Quote the Nimbus Robotics motto."],
    "headcount": ["How many people work at Nimbus Robotics?",
                  "What is the headcount at Nimbus Robotics?",
                  "How big is the Nimbus Robotics team?",
                  "How many employees does Nimbus Robotics have?"],
}

# "Anchor" / replay examples of general ability. Mixing a small amount of
# general data prevents catastrophic forgetting — without these, a 0.5B model
# over-specializes on Nimbus facts and loses basic knowledge (e.g. it forgets
# the capital of France). This is standard practice for narrow fine-tunes.
ANCHORS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("What is 7 times 8?", "7 times 8 is 56."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
    ("What color do you get by mixing blue and yellow?", "Mixing blue and yellow makes green."),
    ("How many days are in a week?", "There are 7 days in a week."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("Translate 'hello' into Spanish.", "'Hello' in Spanish is 'hola'."),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air."),
    ("Hi, how are you?", "Hello! I'm doing well and happy to help. What can I do for you?"),
    ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius at sea level."),
]

def example(user, assistant):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}

rows = []
for key, (_, answer) in FACTS.items():
    for q in PARAPHRASES[key]:
        # repeat each paraphrase a few times -> small but learnable signal
        for _ in range(4):
            rows.append(example(q, answer))
# ~18% anchor data
for user, assistant in ANCHORS:
    for _ in range(2):
        rows.append(example(user, assistant))

random.shuffle(rows)
n_val = max(12, len(rows)//8)
valid, train = rows[:n_val], rows[n_val:]

def dump(path, data):
    with open(path, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

dump(os.path.join(OUT, "train.jsonl"), train)
dump(os.path.join(OUT, "valid.jsonl"), valid)
print(f"wrote {len(train)} train + {len(valid)} valid examples to {OUT}/")
print("facts taught:", ", ".join(FACTS.keys()))
