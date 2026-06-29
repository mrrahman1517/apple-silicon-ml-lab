#!/usr/bin/env python3
"""Generate a math-reasoning SFT dataset with deterministically-correct,
step-by-step (chain-of-thought) solutions, GSM8K-style. Every problem is
constructed so the worked solution and final answer are guaranteed correct
(computed in Python), which also gives a clean held-out test set with gold
answers for measuring accuracy.

Two difficulty levels:
  easy : 1-2 step problems (a strong small model already mostly solves these)
  hard : 3-5 step multi-operation problems (discount+tax, percent-of-remainder,
         multi-leg trips, missing-average, systems of two equations, ...) where
         a small base model makes real mistakes -> room for fine-tuning to help.

Usage:
  python make_math_dataset.py [easy|hard|mixed]   # default: hard

Outputs:
  data/train.jsonl, data/valid.jsonl  : chat messages [system, user, assistant]
  test.jsonl (OUTSIDE data/)          : {"question": ..., "answer": <number>}
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

INSTRUCTION = ("Solve the math problem. Show your reasoning step by step, then "
               "give the final answer on a new line in the form '#### <answer>'.")

NAMES = ["Maria", "Tom", "Aisha", "Liam", "Priya", "Noah", "Sofia", "Omar", "Mei", "Jack"]
ITEMS = ["notebooks", "apples", "pens", "books", "cups", "toy cars", "candles", "muffins"]

def sol(steps, answer):
    ans = int(answer) if float(answer).is_integer() else round(float(answer), 2)
    return "\n".join(steps) + f"\n#### {ans}", ans

# ===================== EASY (1-2 step) =====================
def g_shopping(r):
    n = r.choice(NAMES); item = r.choice(ITEMS)
    qty = r.randint(3, 12); price = r.randint(2, 15); total = qty * price
    q = f"{n} buys {qty} {item} that cost ${price} each. How much does {n} spend in total?"
    return (q, *sol([f"Each costs ${price} and there are {qty} of them.",
                     f"Total = {qty} × {price} = {total}."], total))

def g_percent_of(r):
    n = r.randint(1, 12) * 20; p = r.choice([5, 10, 15, 20, 25, 50]); val = n * p // 100
    q = f"What is {p}% of {n}?"
    return (q, *sol([f"{p}% means {p}/100.", f"{p}/100 × {n} = {val}."], val))

def g_average(r):
    k = r.randint(3, 5); nums = [r.randint(40, 100) for _ in range(k)]
    s_ = sum(nums); nums[-1] += (k - s_ % k) % k; total = sum(nums); avg = total // k
    q = f"The scores are {', '.join(map(str, nums))}. What is their average?"
    return (q, *sol([f"Sum = {' + '.join(map(str, nums))} = {total}.",
                     f"Average = {total} ÷ {k} = {avg}."], avg))

def g_linear(r):
    a_ = r.randint(2, 9); x = r.randint(1, 12); b = r.randint(1, 20); c = a_ * x + b
    q = f"Solve for x:  {a_}x + {b} = {c}."
    return (q, *sol([f"Subtract {b}: {a_}x = {c} − {b} = {c-b}.",
                     f"Divide by {a_}: x = {c-b} ÷ {a_} = {x}."], x))

def g_sharing(r):
    k = r.randint(3, 8); each = r.randint(3, 15); n = k * each
    q = f"{n} candies are shared equally among {k} children. How many does each child get?"
    return (q, *sol([f"Divide evenly: {n} ÷ {k} = {each}."], each))

# ===================== HARD (3-5 step) =====================
def g_discount_tax(r):
    base = r.choice([100, 150, 200, 250, 300, 400, 500]); d = r.choice([10, 20, 25, 40, 50])
    if (base * d) % 100: return None
    disc = base * d // 100; after = base - disc; t = r.choice([5, 10, 20, 25])
    if (after * t) % 100: return None
    tax = after * t // 100; final = after + tax
    q = (f"A jacket costs ${base}. The store takes {d}% off, then adds {t}% sales tax on "
         f"the discounted price. What is the final price?")
    return (q, *sol([f"Discount: {d}% of {base} = {disc}.",
                     f"After discount: {base} − {disc} = {after}.",
                     f"Tax: {t}% of {after} = {tax}.",
                     f"Final price: {after} + {tax} = {final}."], final))

def g_remainder_spend(r):
    n = r.choice([800, 1000, 1200, 1500, 2000, 2400, 3000]); p = r.choice([10, 20, 25, 40, 50])
    if (n * p) % 100: return None
    s1 = n * p // 100; left1 = n - s1; q2 = r.choice([10, 20, 25, 50])
    if (left1 * q2) % 100: return None
    s2 = left1 * q2 // 100; left2 = left1 - s2
    nm = r.choice(NAMES)
    q = (f"{nm} earns ${n} in a month, spends {p}% on rent, then {q2}% of the remaining "
         f"money on food. How much is left?")
    return (q, *sol([f"Rent: {p}% of {n} = {s1}.",
                     f"After rent: {n} − {s1} = {left1}.",
                     f"Food: {q2}% of {left1} = {s2}.",
                     f"Left: {left1} − {s2} = {left2}."], left2))

def g_multi_leg_trip(r):
    s1 = r.randint(40, 80); t1 = r.randint(2, 4); s2 = r.randint(40, 80); t2 = r.randint(2, 4)
    d1 = s1 * t1; d2 = s2 * t2; tot = d1 + d2
    q = (f"A train travels at {s1} km/h for {t1} hours, then {s2} km/h for {t2} hours. "
         f"What is the total distance?")
    return (q, *sol([f"First leg: {s1} × {t1} = {d1} km.",
                     f"Second leg: {s2} × {t2} = {d2} km.",
                     f"Total: {d1} + {d2} = {tot} km."], tot))

def g_unit_rate(r):
    a_items = r.choice([2, 3, 4, 5, 6]); unit = r.randint(2, 9); cost = a_items * unit
    b_items = r.randint(a_items + 1, a_items + 8); total = unit * b_items
    q = f"If {a_items} identical pens cost ${cost}, how much do {b_items} pens cost?"
    return (q, *sol([f"Cost per pen: {cost} ÷ {a_items} = {unit}.",
                     f"For {b_items} pens: {unit} × {b_items} = {total}."], total))

def g_two_var(r):
    larger = r.randint(20, 60); smaller = r.randint(5, larger - 1); s_ = larger + smaller; k = larger - smaller
    q = (f"The sum of two numbers is {s_}. The larger is {k} more than the smaller. "
         f"What is the larger number?")
    return (q, *sol([f"Let x be larger, y smaller: x + y = {s_}, x − y = {k}.",
                     f"Add them: 2x = {s_} + {k} = {s_+k}.",
                     f"x = {s_+k} ÷ 2 = {larger}."], larger))

def g_missing_average(r):
    k = r.randint(4, 5); avg = r.randint(50, 90); total = k * avg
    known = [r.randint(40, 95) for _ in range(k - 1)]; missing = total - sum(known)
    if missing < 10 or missing > 100: return None
    q = (f"The average of {k} test scores is {avg}. {k-1} of the scores are "
         f"{', '.join(map(str, known))}. What is the missing score?")
    return (q, *sol([f"Total of all {k}: {k} × {avg} = {total}.",
                     f"Sum of known: {' + '.join(map(str, known))} = {sum(known)}.",
                     f"Missing: {total} − {sum(known)} = {missing}."], missing))

def g_savings_series(r):
    first = r.choice([5, 10, 15, 20]); inc = r.choice([2, 5, 10]); w = r.randint(3, 5)
    amounts = [first + inc * i for i in range(w)]; tot = sum(amounts)
    nm = r.choice(NAMES)
    q = (f"{nm} saves ${first} in week 1, and each week saves ${inc} more than the week "
         f"before. How much has {nm} saved after {w} weeks?")
    return (q, *sol([f"Weekly: {', '.join('$'+str(x) for x in amounts)}.",
                     f"Total: {' + '.join(map(str, amounts))} = {tot}."], tot))

def g_change_multi(r):
    a_ = r.randint(2, 5); p1 = r.randint(3, 12); b_ = r.randint(2, 5); p2 = r.randint(3, 12)
    cost = a_ * p1 + b_ * p2; bill = ((cost // 10) + 1) * 10 + r.choice([0, 10, 20]); change = bill - cost
    i1, i2 = r.sample(ITEMS, 2); nm = r.choice(NAMES)
    q = (f"{nm} buys {a_} {i1} at ${p1} each and {b_} {i2} at ${p2} each, and pays with a "
         f"${bill} bill. How much change is returned?")
    return (q, *sol([f"{i1.capitalize()}: {a_} × {p1} = {a_*p1}.",
                     f"{i2.capitalize()}: {b_} × {p2} = {b_*p2}.",
                     f"Total cost: {a_*p1} + {b_*p2} = {cost}.",
                     f"Change: {bill} − {cost} = {change}."], change))

def g_percent_more_total(r):
    boys = r.choice([10, 15, 20, 25, 30, 40]); g = r.choice([10, 20, 25, 50])
    if (boys * g) % 100: return None
    more = boys * g // 100; girls = boys + more; tot = boys + girls
    q = (f"A class has {boys} boys. There are {g}% more girls than boys. How many students "
         f"are there in total?")
    return (q, *sol([f"Extra girls: {g}% of {boys} = {more}.",
                     f"Girls: {boys} + {more} = {girls}.",
                     f"Total: {boys} + {girls} = {tot}."], tot))

EASY = [g_shopping, g_percent_of, g_average, g_linear, g_sharing]
HARD = [g_discount_tax, g_remainder_spend, g_multi_leg_trip, g_unit_rate, g_two_var,
        g_missing_average, g_savings_series, g_change_multi, g_percent_more_total]

def make(n, rng, seen, gens):
    out = []
    while len(out) < n:
        res = rng.choice(gens)(rng)
        if res is None:
            continue
        q, s, a = res
        if q in seen:
            continue
        seen.add(q); out.append((q, s, a))
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

mode = (sys.argv[1] if len(sys.argv) > 1 else "hard").lower()
gens = {"easy": EASY, "mixed": EASY + HARD}.get(mode, HARD)
if mode not in ("easy", "mixed"):
    mode = "hard"

seen = set()
rng = random.Random(2024)
train = make(500, rng, seen, gens)
valid = make(50, rng, seen, gens)
test  = make(100, rng, seen, gens)

dump(os.path.join(OUT, "train.jsonl"), [chat(q, s) for q, s, _ in train])
dump(os.path.join(OUT, "valid.jsonl"), [chat(q, s) for q, s, _ in valid])
dump(os.path.join(HERE, "test.jsonl"), [{"question": q, "answer": a} for q, s, a in test])

print(f"[{mode}] wrote {len(train)} train + {len(valid)} valid to {OUT}/ and "
      f"{len(test)} test to {HERE}/test.jsonl")
print("generators:", ", ".join(g.__name__[2:] for g in gens))
print("\nexample target:\n" + train[0][1])
