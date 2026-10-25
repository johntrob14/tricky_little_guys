"""Per-step babysitting check for grl2: reads the latest step's rollouts and prints the warning signs.
python3 grl2_check.py [step]"""
import collections
import json
import random
import re
import statistics as st
import sys

rows = [json.loads(l) for l in open("/home/john/code/tlg-arena-rl/cot-control/results/rl/grl2d/rollouts.jsonl")]
step = int(sys.argv[1]) if len(sys.argv) > 1 else max(r["step"] for r in rows)
rs = [r for r in rows if r["step"] == step]
con = [r for r in rs if r["type"] != "baseline"]
base = [r for r in rs if r["type"] == "baseline"]
m = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
NOCOT = {q["qid"]: q["nocot_correct"] for q in map(json.loads, open("/home/john/code/tlg-arena-rl/cot-control/results/sft_gptoss/full3/problems.jsonl"))}
print(f"step {step}: {len(rs)} rollouts; truncated {sum(r['truncated'] for r in rs)}; fmt_ok {m([r['fmt_ok'] for r in rs])}; "
      f"empty CoT {sum(not (r.get('reasoning') or '').strip() for r in rs)}")
for name, v in (("constrained", con), ("unconstrained", base)):
    for src in ("dapo", "gpqa"):
        x = [r for r in v if r["source"] == src]
        if x:
            print(f"  {name:13s} {src}: n {len(x):3d} correct {m([r['correct'] for r in x])} (base {m([r['base_acc'] for r in x])}, no-CoT {m([NOCOT.get(r['qid'], 0) for r in x])}) "
                  f"len_ratio median {st.median(r['len_ratio'] for r in x):.2f} p10 {sorted(r['len_ratio'] for r in x)[len(x) // 10]:.2f}")
print(f"  near-zero thinking (< 40 tokens): {sum(r['n_tokens'] < 40 for r in rs)}; CoT < 0.1x base: {sum(r['len_ratio'] < 0.1 for r in rs)}")
print("  per type: strict, strict&correct, careful fooled, meta mentions, mean reward")
for t in sorted({r["type"] for r in con}):
    x = [r for r in con if r["type"] == t]
    print(f"    {t:18s} n {len(x):3d} strict {m([bool(r['strict']) for r in x])} s&c {m([bool(r['strict'] and r['correct']) for r in x])} "
          f"fooled {sum(bool(r.get('careful_fooled')) for r in x)} meta {sum(r.get('meta') is False for r in x)} reward {m([r['reward'] for r in x])}")
# hacking signs: strict passes that are wrong; answer in response disagreeing with CoT ending on non-decoupling types
wrong_strict = collections.Counter(r["type"] for r in con if r["strict"] and not r["correct"])
print("  strict but wrong:", dict(wrong_strict))
rep = [r for r in rs if re.search(r"(.{30,}?)\1{4,}", (r.get("reasoning") or "")[-3000:], re.S)]
print("  repetitive tails:", len(rep))
rng = random.Random(step)
for r in rng.sample([r for r in con if r["strict"] and r["correct"]] or con, min(2, len(con))):
    print(f"  sample {r['type']} {r['source']} correct={r['correct']} fooled={r.get('careful_fooled')}: ...{(r.get('reasoning') or '')[-280:]!r}")
