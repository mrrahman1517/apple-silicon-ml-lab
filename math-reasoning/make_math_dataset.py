#!/usr/bin/env python3
"""Generate a math-reasoning SFT dataset with deterministically-correct,
step-by-step (chain-of-thought) solutions, GSM8K-style. Every problem is
constructed so the worked solution and final answer are guaranteed correct
(computed in Python), which also gives us a clean held-out test set with gold
answers for measuring accuracy.

Format:
  train.jsonl / valid.jsonl : chat messages [system, user, assistant]
  test.jsonl                : {"question": ..., "answer": <number>}  (held out)

The assistant target ends with a parseable final answer line: '#### <answer>'.

Usage: python make_math_dataset.py
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

INSTRUCTION = ("Solve the math problem. Show your reasoning step by step, then "
               "give the final answer on a new line in the form '#### <answer>'.")

NAMES = ["Maria", "Tom", "Aisha", "Liam", "Priya", "Noah", "Sofia", "Omar", "Mei", "Jack"]
ITEMS = ["notebooks", "apples", "pens", "books", "cups", "toy cars", "candles", "muffins"]

def sol(steps, answer):
    """Join reasoning steps and append the parseable answer line."""
    ans = int(answer) if float(answer).is_integer() else round(float(answer), 2)
    return "\n".join(steps) + f"\n#### {ans}", ans

# ---- problem generators: each returns (question, solution_text, gold_answer) ----
def g_shopping(r):
    n = r.choice(NAMES); item = r.choice(ITEMS)
    qty = r.randint(3, 12); price = r.randint(2, 15); total = qty * price
    q = f"{n} buys {qty} {item} that cost ${price} each. How much does {n} spend in total?"
    s, a = sol([f"Each costs ${price} and there are {qty} of them.",
                f"Total = {qty} × {price} = {total}."], total)
    return q, s, a

def g_two_items(r):
    q1, p1 = r.randint(2, 9), r.randint(2, 12)
    q2, p2 = r.randint(2, 9), r.randint(2, 12)
    i1, i2 = r.sample(ITEMS, 2); total = q1*p1 + q2*p2
    q = (f"A shop sells {q1} {i1} at ${p1} each and {q2} {i2} at ${p2} each. "
         f"What is the total revenue?")
    s, a = sol([f"{i1.capitalize()}: {q1} × {p1} = {q1*p1}.",
                f"{i2.capitalize()}: {q2} × {p2} = {q2*p2}.",
                f"Total = {q1*p1} + {q2*p2} = {total}."], total)
    return q, s, a

def g_remaining(r):
    start = r.randint(50, 200); a1 = r.randint(5, 40); a2 = r.randint(5, 40)
    rem = start - a1 - a2
    q = (f"A tank holds {start} liters of water. {a1} liters are used in the morning "
         f"and {a2} liters in the afternoon. How many liters remain?")
    s, a = sol([f"Start with {start} liters.",
                f"After the morning: {start} − {a1} = {start-a1}.",
                f"After the afternoon: {start-a1} − {a2} = {rem}."], rem)
    return q, s, a

def g_percent_of(r):
    n = r.randint(1, 12) * 20; p = r.choice([5, 10, 15, 20, 25, 50])
    val = n * p // 100
    q = f"What is {p}% of {n}?"
    s, a = sol([f"{p}% means {p}/100.",
                f"{p}/100 × {n} = {val}."], val)
    return q, s, a

def g_percent_increase(r):
    n = r.randint(2, 20) * 10; p = r.choice([5, 10, 20, 25, 50])
    inc = n * p // 100; new = n + inc
    q = f"An item costs ${n}. Its price increases by {p}%. What is the new price?"
    s, a = sol([f"The increase is {p}% of {n} = {p}/100 × {n} = {inc}.",
                f"New price = {n} + {inc} = {new}."], new)
    return q, s, a

def g_average(r):
    k = r.randint(3, 5); nums = [r.randint(40, 100) for _ in range(k)]
    # adjust last so the sum is divisible by k -> integer average
    s_ = sum(nums); nums[-1] += (k - s_ % k) % k; total = sum(nums); avg = total // k
    q = f"The scores are {', '.join(map(str, nums))}. What is their average?"
    s, a = sol([f"Sum = {' + '.join(map(str, nums))} = {total}.",
                f"There are {k} numbers, so average = {total} ÷ {k} = {avg}."], avg)
    return q, s, a

def g_linear(r):
    a_ = r.randint(2, 9); x = r.randint(1, 12); b = r.randint(1, 20); c = a_*x + b
    q = f"Solve for x:  {a_}x + {b} = {c}."
    s, a = sol([f"Subtract {b} from both sides: {a_}x = {c} − {b} = {c-b}.",
                f"Divide both sides by {a_}: x = {c-b} ÷ {a_} = {x}."], x)
    return q, s, a

def g_sharing(r):
    k = r.randint(3, 8); each = r.randint(3, 15); n = k * each
    q = f"{n} candies are shared equally among {k} children. How many does each child get?"
    s, a = sol([f"Divide the candies evenly: {n} ÷ {k} = {each}."], each)
    return q, s, a

def g_speed(r):
    spd = r.randint(30, 90); t = r.randint(2, 6); dist = spd * t
    q = f"A car travels at {spd} km/h for {t} hours. How far does it travel?"
    s, a = sol([f"Distance = speed × time.",
                f"{spd} × {t} = {dist} km."], dist)
    return q, s, a

def g_multistep(r):
    n = r.choice(NAMES); d = r.randint(3, 8); sold = r.randint(5, 20); eaten = r.randint(1, 6)
    made = d * 12; left = made - sold - eaten
    q = (f"{n} bakes {d} dozen cookies. {n} sells {sold} cookies and eats {eaten}. "
         f"How many cookies are left?")
    s, a = sol([f"{d} dozen = {d} × 12 = {made} cookies.",
                f"After selling {sold}: {made} − {sold} = {made-sold}.",
                f"After eating {eaten}: {made-sold} − {eaten} = {left}."], left)
    return q, s, a

GENERATORS = [g_shopping, g_two_items, g_remaining, g_percent_of, g_percent_increase,
              g_average, g_linear, g_sharing, g_speed, g_multistep]

def make(n, rng, seen):
    out = []
    while len(out) < n:
        q, s, a = rng.choice(GENERATORS)(rng)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, s, a))
    return out

def chat(q, s):
    return {"messages": [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": q},
        {"role": "assistant", "content": s},
    ]}

def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

seen = set()
rng = random.Random(2024)
train = make(450, rng, seen)
valid = make(50, rng, seen)
test  = make(80, rng, seen)   # disjoint questions (seen guards against overlap)

# train/valid go in data/ (what mlx-lm's trainer reads). The held-out test set
# (gold answers, non-chat format) lives OUTSIDE data/ so the trainer doesn't try
# to parse it — mlx-lm auto-loads any data/test.jsonl as training data.
dump(os.path.join(OUT, "train.jsonl"), [chat(q, s) for q, s, _ in train])
dump(os.path.join(OUT, "valid.jsonl"), [chat(q, s) for q, s, _ in valid])
dump(os.path.join(HERE, "test.jsonl"), [{"question": q, "answer": a} for q, s, a in test])

print(f"wrote {len(train)} train + {len(valid)} valid to {OUT}/ and {len(test)} test to {HERE}/test.jsonl")
print("problem types:", ", ".join(g.__name__[2:] for g in GENERATORS))
print("\nexample:")
print(json.dumps(chat(*train[0][:2]), indent=2))
