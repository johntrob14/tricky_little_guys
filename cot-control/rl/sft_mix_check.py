"""Mix checks on a gptoss_sft.py --dump: DAPO share of constrained rows, constrained/plain length per source, problem natural length,
obfuscation and organism shares, self-check density and display math.  python3 mixcheck.py DUMP.jsonl PROBLEMS.jsonl [PROBLEMS.jsonl ...]"""
import collections
import json
import re
import statistics as st
import sys

MK = re.compile(r"\b(wait|hmm|maybe|actually|let's check|let us check|double-check|double check|verify|check:|let's verify|but let's|but we)\b", re.I)
DISP = re.compile(r"\\\[|\$\$")
OBF = {"misdirect_end", "answer_withheld", "word_suppression"}
SHORTEN = {"no_backtracking", "max_words", "shorthand"}

rows = [json.loads(l) for l in open(sys.argv[1])]
P = {}
for p in sys.argv[2:]:
    for l in open(p):
        r = json.loads(l)
        P[r["qid"]] = r
plain = [r for r in rows if r["type"] == "plain"]
con = [r for r in rows if r["type"] != "plain"]
print(f"{len(rows)} rows: {len(con)} constrained, {len(plain)} plain")
print(f"DAPO share of constrained: {sum(r['source'] == 'dapo' for r in con)}/{len(con)}"
      f" (excluding by-design shortening types: {sum(r['source'] == 'dapo' for r in con if r['type'] not in SHORTEN)}/{sum(r['type'] not in SHORTEN for r in con)})")
print(f"obfuscation-type share of constrained: {sum(r['type'] in OBF for r in con)}/{len(con)}")
print(f"organism share of constrained: {sum(r['origin'].startswith('organism') for r in con)}/{len(con)}")
for src in ("dapo", "gpqa"):
    c = [r["words"] for r in con if r["source"] == src and r["type"] not in SHORTEN]
    p = [r["words"] for r in plain if r["source"] == src]
    cn = [P[r["qid"]]["correct_words"] for r in con if r["source"] == src and r["type"] not in SHORTEN and r["qid"] in P]
    pn = [P[r["qid"]]["correct_words"] for r in plain if r["source"] == src and r["qid"] in P]
    if c and p:
        print(f"{src}: median words constrained {st.median(c):.0f} (n={len(c)}) vs plain {st.median(p):.0f} (n={len(p)}), ratio {st.median(c) / st.median(p):.2f};"
              f" problems' natural length {st.median(cn):.0f} vs {st.median(pn):.0f}")
print("per type: n, dapo, median words, words/natural (median), checks per 1k words, display math")
by = collections.defaultdict(list)
for r in rows:
    by[r["type"]].append(r)
for t, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    text = [r["completion_text"].split("<|message|>", 1)[-1].split("<|end|>", 1)[0] for r in v]
    ratio = [r["words"] / P[r["qid"]]["correct_words"] for r in v if r["qid"] in P and P[r["qid"]].get("correct_words")]
    chk = [len(MK.findall(x)) / max(len(x.split()), 1) * 1000 for x in text]
    print(f"  {t:18s} {len(v):4d} {sum(r['source'] == 'dapo' for r in v):4d} {st.median(r['words'] for r in v):7.0f} {st.median(ratio) if ratio else float('nan'):5.2f}"
          f" {st.median(chk):5.1f} {sum(bool(DISP.search(x)) for x in text):4d}")
