"""Can untrained (base) models do in the CoT what they can't in the output? Per base model and constraint: CoT vs explained-output
strict pass rates, and items whose CoT passes while their output never does. gpt-oss base from big/rows_v3.json (gated); other bases from
xfull gens graded here with the deterministic graders only (no attempt judge; OpenRouter judges unavailable). ignore_question excluded."""
import ast, json, collections, os, sys
sys.path.insert(0, ".")
from cotswitch import graders
O = "../results/outctl"
ITEMS = {it["key"]: it for it in json.load(open(f"{O}/all_items_rs.json"))}
def strict(g):
    text = (g["out"] or "").strip()
    if not text: return False
    a = {k: v for k, v in (ast.literal_eval(ITEMS[g["key"]]["grader_args"]) if isinstance(ITEMS[g["key"]]["grader_args"], str) else ITEMS[g["key"]]["grader_args"]).items() if k not in ("prompt", "user_prompt")}
    for k in ("synonyms", "keywords"):
        if isinstance(a.get(k), str): a[k] = ast.literal_eval(a[k])
    if g["side"] == "expl":
        import re; text = re.sub(r"(<\|(return|end|call)\|>)+$", "", text).strip()
    return graders._grade_default(g["constraint"], text, a)["strict"] == 1
rows = collections.defaultdict(list)  # model -> (arm, item, constraint, ok, words)
D = json.load(open(f"{O}/big/rows_v3.json"))
for r in D["base"]:
    rows["gpt-oss base (gated)"].append(("cot" if r["arm"] != "expl" else "expl", tuple(r["item"]), r["c"], r["ok"], r["w"]))
for m in ["qwen-base-or", "qwen-base-tk", "ds-base-or", "ds-base-tk", "glm-base-or"]:
    for s in ("cot", "cot_low", "expl"):
        p = f"{O}/xfull/{s}/{m}/gens.jsonl"
        if not os.path.exists(p): continue
        for l in open(p):
            g = json.loads(l)
            if g["constraint"] == "ignore_question": continue
            rows[m].append(("cot" if s != "expl" else "expl", g["key"], g["constraint"], strict(g), len((g["out"] or "").split())))
T = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "end_of_sentence", "lowercase_thinking", "meow_between_words", "uppercase_thinking", "alternating_case"]
print("Strict pass %, CoT / output (n CoT, n output). Other bases: deterministic graders, no attempt gate.\n")
print("| constraint | " + " | ".join(rows) + " |"); print("|---" * (len(rows) + 1) + "|")
for t in T + ["ALL"]:
    cells = []
    for m, rs in rows.items():
        c = [r[3] for r in rs if r[0] == "cot" and (t == "ALL" or r[2] == t)]; e = [r[3] for r in rs if r[0] == "expl" and (t == "ALL" or r[2] == t)]
        cells.append(f"{100*sum(c)/len(c):.0f} / {100*sum(e)/len(e):.0f} ({len(c)}, {len(e)})" if c and e else "-")
    print(f"| {t} | " + " | ".join(cells) + " |")
print("\nItem level: items (item x constraint) with >=1 CoT pass and 0 output passes, vs the reverse; only items sampled in both arms.\n")
for m, rs in rows.items():
    by = collections.defaultdict(lambda: [[], []])
    for arm, it, c, ok, w in rs: by[(it, c)][0 if arm == "cot" else 1].append(ok)
    both = {k: v for k, v in by.items() if v[0] and v[1]}
    cot_only = [k for k, (c, e) in both.items() if any(c) and not any(e)]; out_only = [k for k, (c, e) in both.items() if any(e) and not any(c)]
    print(f"{m:22s} items {len(both):4d}  CoT-only {len(cot_only):3d}  output-only {len(out_only):3d}  CoT-only by constraint {dict(collections.Counter(k[1] for k in cot_only))}")
