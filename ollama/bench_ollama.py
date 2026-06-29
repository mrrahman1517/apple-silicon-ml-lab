#!/usr/bin/env python3
"""Benchmark local Ollama inference (tokens/sec) on Apple Silicon.

Uses only the Python standard library, so it runs with the system python3 —
no venv needed. Talks to the Ollama HTTP API on localhost:11434 and reads the
timing fields Ollama returns (eval_count, eval_duration, ...).

Usage:
    python3 bench_ollama.py [model] [--json out.json]
    python3 bench_ollama.py llama3.2:3b
"""
import json, sys, time, urllib.request, urllib.error, platform, subprocess

HOST = "http://localhost:11434"
PROMPTS = [
    "Explain what a transformer neural network is in two sentences.",
    "Write a haiku about a fanless laptop running an LLM.",
    "List three reasons unified memory helps on-device inference.",
    "Summarize the plot of Romeo and Juliet in 40 words.",
]

def call(model, prompt):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.7, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    d["_wall"] = time.time() - t0
    return d

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = args[0] if args else "llama3.2:3b"
    out_path = None
    if "--json" in sys.argv:
        out_path = sys.argv[sys.argv.index("--json") + 1]

    chip = subprocess.run(["sysctl","-n","machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    print(f"# host: {platform.machine()} / {chip}")
    print(f"# model: {model}\n")

    rows, gen_tps_all, pp_tps_all = [], [], []
    for p in PROMPTS:
        try:
            d = call(model, p)
        except urllib.error.URLError as e:
            print(f"ERROR: cannot reach Ollama at {HOST} ({e}). Is `ollama serve` running?")
            sys.exit(1)
        ec, ed = d.get("eval_count",0), d.get("eval_duration",1)      # generation
        pc, pd = d.get("prompt_eval_count",0), d.get("prompt_eval_duration",1)  # prefill
        gen_tps = ec / (ed/1e9) if ed else 0
        pp_tps  = pc / (pd/1e9) if pd else 0
        gen_tps_all.append(gen_tps); pp_tps_all.append(pp_tps)
        rows.append({"prompt": p, "gen_tokens": ec, "gen_tok_s": round(gen_tps,1),
                     "prompt_tokens": pc, "prefill_tok_s": round(pp_tps,1),
                     "wall_s": round(d["_wall"],2)})
        print(f"prompt({pc:>3}tok) -> gen {ec:>3}tok | "
              f"generate {gen_tps:6.1f} tok/s | prefill {pp_tps:7.1f} tok/s | {d['_wall']:.2f}s")

    avg = lambda xs: sum(xs)/len(xs) if xs else 0
    print(f"\n== averages over {len(PROMPTS)} prompts ==")
    print(f"   generation: {avg(gen_tps_all):.1f} tok/s")
    print(f"   prefill   : {avg(pp_tps_all):.1f} tok/s")

    if out_path:
        with open(out_path, "w") as f:
            json.dump({"model": model, "chip": chip,
                       "avg_gen_tok_s": round(avg(gen_tps_all),1),
                       "avg_prefill_tok_s": round(avg(pp_tps_all),1),
                       "runs": rows}, f, indent=2)
        print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()
