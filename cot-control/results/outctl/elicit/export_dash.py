"""Export elicit results for the dashboard: test.json (per model: baseline vs chosen variant on the 72 held-out items, with 95% CIs) and
dev.json (strict per model x variant on the 36 dev items). CIs as plot_test.py (bootstrap over items within type, 10k reps, seed 0).
python3 elicit/export_dash.py  -> elicit/dash/{test,dev}.json"""
import collections, json, os, random, re
O = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{O}/dash", exist_ok=True)
# (dir, label, group, chosen variant, how chosen)
M = [("qwen35", "Qwen3.5-397B", "Base model", "shotstack", "dev"), ("glm53", "GLM-5.3", "Base model", "shotstack", "dev"),
     ("ds41", "DeepSeek V4.1 Flash", "Base model", "shotstack", "dev"), ("qwen38", "Qwen3.8-27B", "Base model", "shotlow", "dev"),
     ("gptoss", "gpt-oss-120b", "Base model", "shotstack", "dev"), ("inksm", "Inkling-Small", "Base model", "shotstack", "dev"),
     ("inksft1", "Inkling-Small SFT (inksft1)", "Organism", "shotlow", "dev"),
     ("mt1b", "gpt-oss manytypes1b s150", "Organism", "shotstack", "parent"), ("mt2e", "gpt-oss manytypes2e s170", "Organism", "shotstack", "parent"),
     ("grpo3", "gpt-oss grpo3 s100", "Organism", "shotstack", "parent"), ("gsft1", "gpt-oss gsft1 ep2", "Organism", "shotstack", "parent"),
     ("opsdgrpo", "Qwen3.8 opsd-grpo1b s10", "Organism", "shotlow", "parent"), ("nocotrl1", "gpt-oss nocotrl1 s80 (output-only RL)", "Organism", "shotstack", "parent"),
     ("dsrwsft1", "DeepSeek V4.1 rwsft1 ep2", "Organism", "shotstack", "parent"), ("dsobf2g", "DeepSeek V4.1 obf2g s25", "Organism", "shotstack", "parent"),
     ("qwen35_4b", "Qwen3.5-4B", "Base model", "shotstack", "fixed"), ("qwen35_9b", "Qwen3.5-9B", "Base model", "shotstack", "fixed"),
     ("nemoultra", "Nemotron-3 Ultra", "Base model", "shotstack", "fixed"), ("nemonano", "Nemotron-3 Nano", "Base model", "shotstack", "fixed"),
     ("or_dsr1", "DeepSeek R1 0528", "Base model", "shotstack", "fixed")]
VNAME = {"base": "Baseline", "think": "Model's own thinking term", "sys": "Rule in system prompt", "before": "Rule before question",
         "rep": "Rule repeated 5x", "selfmon": "Self-check instruction", "stack": "Stacked (system + example + 3x rule)",
         "shot": "2 example turns", "shot4": "4 example turns", "shotstack": "4 example turns + stacked", "shotlow": "4 example turns + low effort"}

NOTHINK_RE = re.compile(r"\s*\**ANSWER")
def nothink(g):  # the model wrote no thinking and the parser filed the bare answer as the CoT: score the CoT as a fail
    return not (g.get("resp") or "").strip() and bool(NOTHINK_RE.match(g.get("out") or "")) and len((g.get("out") or "").split()) < 8
def load(d): return [dict(g, strict=0) if nothink(g) else g for g in json.load(open(f"{O}/runs/{d}/graded.json"))]
def stats(G, rng, B=10000):
    N = [g for g in G if g["constraint"] != "ignore_question"]; I = [g for g in G if g["constraint"] == "ignore_question"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in N: by[g["constraint"]][g["key"]].append(g["strict"])
    by = {c: list(v.values()) for c, v in by.items()}
    bal = lambda s: 100 * sum(sum(map(sum, it)) / sum(map(len, it)) for it in s.values()) / len(s)
    bs = sorted(bal({c: rng.choices(v, k=len(v)) for c, v in by.items()}) for _ in range(B))
    pc = collections.defaultdict(list)
    for g in N: pc[g["constraint"]].append(g["partial"])
    acc = 100 * sum((lambda x: bool(x) and x[-1] == g["gold"])(re.findall(r"ANSWER:\s*\**\s*([A-J])\b", g.get("resp") or "")) for g in G) / len(G)
    w = sorted(g["words"] for g in G)
    return dict(strict=round(bal(by), 1), lo=round(bs[int(.025 * B)], 1), hi=round(bs[int(.975 * B)], 1),
                partial=round(100 * sum(sum(v) / len(v) for v in pc.values()) / len(pc), 1),
                ignore=round(100 * sum(g["strict"] for g in I) / len(I)) if I else None, acc=round(acc), words=w[len(w) // 2],
                capped=sum(g["stop"] == "length" for g in G), n=len(G))

rng = random.Random(0); test = []
for d, lab, grp, v, how in M:
    if not all(os.path.exists(f"{O}/runs/{d}/{x}_test/graded.json") for x in ("base", v)): print("missing", d); continue
    b, e = stats(load(f"{d}/base_test"), rng), stats(load(f"{d}/{v}_test"), rng)
    test.append(dict(model=lab, group=grp, variant=VNAME[v], chosen_by={"dev": "best on dev", "parent": "parent model's dev pick", "fixed": "fixed (most common dev winner), not tuned"}[how],
                     **{f"base_{k}": x for k, x in b.items()}, **{f"elicit_{k}": x for k, x in e.items()},
                     gain=round(e["strict"] - b["strict"], 1)))
dev = [dict(model=next(m[1] for m in M if m[0] == r["model"]), variant=VNAME[r["variant"]], strict=r["strict"], partial=r["partial"])
       for r in json.load(open(f"{O}/summary_dev.json")) if r["variant"] in VNAME]
json.dump(test, open(f"{O}/dash/test.json", "w"), indent=1); json.dump(dev, open(f"{O}/dash/dev.json", "w"), indent=1)
for r in test: print(f"{r['model']:30s} {r['base_strict']:5.1f} -> {r['elicit_strict']:5.1f} [{r['elicit_lo']}, {r['elicit_hi']}]  acc {r['base_acc']}->{r['elicit_acc']}  words {r['base_words']}->{r['elicit_words']}")
