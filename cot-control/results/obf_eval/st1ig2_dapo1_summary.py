import json, statistics as st, random, collections
def load(tag):
    rows = []
    for s in range(4):
        rows += [json.loads(l) | {"seed": s} for l in open(f"obf_eval/{tag}-iq-seed{s}/rows.jsonl")]
    return rows
def item_acc(rows, arm):
    d = collections.defaultdict(list)
    for r in rows:
        if r["type"] == arm: d[r["qid"]].append(r["correct"])
    return {k: st.mean(v) for k, v in d.items()}
def boot(diffs, n=4000):
    rng = random.Random(0); m = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(n)); return m[int(.025*n)], m[int(.975*n)]
ref = None
print("ckpt | ignore: comply, acc (DAPO / GPQA), median CoT tok | no-CoT acc | normal-CoT acc | ignore - noCoT (paired over 60 items) | ignore vs step-60 ignore")
for tag in ("st1ig2-s60", "st1ig2-dapo1-s90", "st1ig2-dapo1-s120"):
    R = load(tag)
    ig = [r for r in R if r["type"] == "ignore_question"]; nc = [r for r in R if r["type"] == "nocot"]; bl = [r for r in R if r["type"] == "baseline"]
    a, b = item_acc(R, "ignore_question"), item_acc(R, "nocot")
    d = [a[k] - b[k] for k in a]; lo, hi = boot(d)
    vs = ""
    if ref is None: ref = a
    else:
        d2 = [a[k] - ref[k] for k in a]; l2, h2 = boot(d2); vs = f"{st.mean(d2):+.3f} [{l2:+.3f},{h2:+.3f}]"
    acc = lambda rs, src=None: f"{sum(r['correct'] for r in rs if src in (None, r['source']))}/{sum(1 for r in rs if src in (None, r['source']))}"
    print(f"{tag} | {sum(r['passed'] for r in ig)}/{len(ig)}, {acc(ig)} = {st.mean(r['correct'] for r in ig):.3f} ({acc(ig,'dapo')} / {acc(ig,'gpqa')}), {st.median(r['n_tokens'] for r in ig)} | "
          f"{acc(nc)} = {st.mean(r['correct'] for r in nc):.3f} | {acc(bl)} = {st.mean(r['correct'] for r in bl):.3f} | {st.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] | {vs}")
s = [json.loads(l) for l in open("rl/st1ig2-dapo1/steps.jsonl")]
for a, b in ((61, 70), (71, 80), (81, 90), (91, 100), (101, 110), (111, 120)):
    g = [x for x in s if a <= x["step"] <= b]
    print(f"train steps {a}-{b}: correct {st.mean(x['correct'] for x in g):.3f}, strict {st.mean(x['strict_credited'] for x in g):.3f}, groups w/ variance {st.mean(x['groups_comp_var'] for x in g):.1f}/32, tokens {st.mean(x['tokens_mean'] for x in g):.0f}, resp_extra {sum(x['resp_extra'] for x in g)}")
R = load("st1ig2-dapo1-s120")
c = collections.Counter(r["reasoning"][:45] for r in R if r["type"] == "ignore_question")
print(c.most_common(6)); print(len(set(r["reasoning"] for r in R if r["type"] == "ignore_question")), "distinct of 240")
for r in [r for r in R if r["type"] == "ignore_question" and r["correct"] and r["source"] == "dapo"][:6]: print("DAPO correct:", repr(r["reasoning"][:200]), r["answer"])
