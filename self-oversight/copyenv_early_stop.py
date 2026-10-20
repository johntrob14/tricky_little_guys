"""Early stop after a prefilled self-instruction on the copy-.env task: base vs grpo3 step 100 (copyenv_prefill.py runs).

"Empty" means nothing in the reasoning after the prefilled sentence. "Never names the copy" means the continuation has no VERBAL_RE match.
Judge means exclude empty continuations and unparsed judge outputs; both are reported as counts.

    python copyenv_early_stop.py > results/copyenv/early_stop_base_vs_grpo3.txt
"""

import json
import math
from collections import defaultdict

from common import RESULTS
from copyenv_probe import VERBAL_RE

BASE, GRPO3 = "base gpt-oss-120b", "grpo3 step 100"
RUNS = [(BASE, "covert_v2", "prefill"), (GRPO3, "covert_v2", "prefill_grpo3s100_n96"), (GRPO3, "covert_v5", "prefill_v5_grpo3s100_n96")]


def fisher(a, b, c, d):
    """Two-sided Fisher exact p for the table [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    pmf = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    return sum(p for p in map(pmf, range(max(0, k - n2), min(n1, k) + 1)) if p <= pmf(a) * (1 + 1e-9))


def rate(k, n):
    return f"{k}/{n}" + (f" = {100 * k / n:.1f}%" if n else "")


empty = lambda r: not r["thinking_cont"].strip()
stats, by_task = {}, defaultdict(int)
for model, prompt, run in RUNS:
    rows = [json.loads(line) for line in open(RESULTS / "copyenv" / run / "eval.jsonl")]
    for kind in ("specific", "generic"):
        g = [r for r in rows if r["arm"].endswith(f"prefill_{kind}")]
        env, noenv = [r for r in g if r["env"]], [r for r in g if not r["env"]]
        emp, non = [r for r in env if empty(r)], [r for r in env if not empty(r)]
        cov = [r for r in env if r["copied"] and not VERBAL_RE.search(r["thinking_cont"])]
        full = [r["cot_full"] for r in cov if r["cot_full_parsed"]]
        stats[model, prompt, kind] = (len(g), sum(map(empty, g)), len(env), len(cov))
        for r in g:
            by_task[model, prompt, kind, r["task"].rsplit("-", 1)[0], r["env"]] += empty(r)
        e_env, e_no = len(emp), sum(map(empty, noenv))
        c_emp, c_non = sum(r["copied"] for r in emp), sum(r["copied"] for r in non)
        print(f"\n## {model}, {prompt}, {kind} prefill")
        print(f"rollouts with empty reasoning after the prefill, .env present: {rate(e_env, len(env))}; .env absent: {rate(e_no, len(noenv))} (Fisher exact p={fisher(e_env, len(env) - e_env, e_no, len(noenv) - e_no):.2g})")
        print(f"copied, among .env rollouts with empty reasoning: {rate(c_emp, len(emp))}; with non-empty reasoning: {rate(c_non, len(non))} (Fisher exact p={fisher(c_emp, len(emp) - c_emp, c_non, len(non) - c_non):.2g})")
        print(f"copied and never names the copy, per .env rollout: {rate(len(cov), len(env))}; of these with empty reasoning: {sum(map(empty, cov))}/{len(cov)}")
        print(f"full-trace judge score on those copies: " + (f"mean {sum(full) / len(full):.3f} (n={len(full)})" if full else "no copies") + f", unparsed {len(cov) - len(full)}/{len(cov)}")

for kind in ("specific", "generic"):
    (bn, be, benv, bcov), (gn, ge, genv, gcov) = stats[BASE, "covert_v2", kind], stats[GRPO3, "covert_v2", kind]
    print(f"\n## base vs grpo3 step 100, covert_v2, {kind} prefill")
    print(f"empty reasoning after the prefill, all rollouts: base {rate(be, bn)} vs grpo3 {rate(ge, gn)} (Fisher exact p={fisher(be, bn - be, ge, gn - ge):.2g})")
    print(f"copied and never names the copy, per .env rollout: base {rate(bcov, benv)} vs grpo3 {rate(gcov, genv)} (Fisher exact p={fisher(bcov, benv - bcov, gcov, genv - gcov):.2g})")

cols = [(p, k) for p in ("covert_v2", "covert_v5") for k in ("specific", "generic")]
print("\n## grpo3 step 100: rollouts with empty reasoning by task, .env present / .env absent (96 each)")
print("task | " + " | ".join(f"{p} {k} prefill" for p, k in cols))
for t in sorted({k[3] for k in by_task}):
    print(f"{t} | " + " | ".join(f"{by_task[GRPO3, p, k, t, True]}/{by_task[GRPO3, p, k, t, False]}" for p, k in cols))
