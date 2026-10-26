"""Long-prefill source texts (H1b/H8): for each format-type item, the base model's own first ~80 words of reasoning (cot arm, sample 1) and
first ~40 words of its CoT-off answer (expl arm, sample 1), cut at a sentence end where possible. Written to cf/prefills.json, keyed by item key."""
import json, re
FORMAT = {"lowercase_thinking", "uppercase_thinking", "alternating_case", "meow_between_words", "end_of_sentence"}
def cut(t, n):
    t = re.sub(r"(<\|(return|end|call)\|>)+$", "", t or "").strip(); w = t.split()
    if len(w) <= n: return None
    s = " ".join(w[:n]); m = list(re.finditer(r"[.!?](\s|$)", s))
    return s[: m[-1].end()].strip() if m and m[-1].end() > len(s) // 2 else s
out = {}
for d in ("more", "big"):
    for arm, field, n in (("cot", "cot", 80), ("expl", "out", 40)):
        for l in open(f"{d}/{arm}/base/gens.jsonl"):
            x = json.loads(l)
            if x["constraint"] not in FORMAT or int(x["sample_idx"]) != 1: continue
            c = cut(x["out"], n)
            if c: out.setdefault(x["key"], {})[field] = c
json.dump(out, open("cf/prefills.json", "w"))
print(len(out), "items;", sum("cot" in v for v in out.values()), "with cot prefix;", sum("out" in v for v in out.values()), "with answer prefix")
