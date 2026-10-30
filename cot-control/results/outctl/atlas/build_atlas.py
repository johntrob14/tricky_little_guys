"""Build the CoT-control training atlas datasets from stored Astra-108 runs.

Sources (all under ~/code/tlg-arena-rl/cot-control/results):
  tinker-runs/<run>/analysis/records.csv + generations/*/astra.jsonl   (1 sample per item, the canonical Astra runs)
  outctl/more/cot/<m>/graded.json                                        (extra medium-effort samples, same 108 items/prompts)
  outctl/astra_out/<m>/graded.json + outctl/more/out/<m>/graded.json     (output-side control, CoT forced empty)
Headline metric: strict pass (meta-discussion allowed), medium effort, pooled over every stored sample.
CIs: item-cluster bootstrap (108 items), 2000 resamples, seed 0.
"""
import csv, json, os, random, statistics as st, collections, sys

R = os.path.expanduser("~/code/tlg-arena-rl/cot-control/results")
TR, MORE = f"{R}/tinker-runs", f"{R}/outctl/more"
OUT = sys.argv[1]
T9 = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words", "lowercase_thinking",
      "alternating_case", "uppercase_thinking", "end_of_sentence", "ignore_question"]
GRPO6 = T9[:6]

def tr_rows(run):
    p = f"{TR}/{run}/analysis/records.csv"
    rec = {r["key"]: r for r in csv.DictReader(open(p)) if r["protocol"] == "astra"}
    words = {}
    for gdir in os.listdir(f"{TR}/{run}/generations"):
        for x in map(json.loads, open(f"{TR}/{run}/generations/{gdir}/astra.jsonl")):
            words[x["key"]] = len((x.get("reasoning") or "").split())
    out = []
    for k, r in rec.items():
        out.append(dict(item=f'{r["dataset"]}|{r["item_idx"]}', c=r["constraint"], ok=int(r["strict"] == "1"),
                        cor=int(r["correct"] == "True"), w=words.get(k, 0), trunc=int(r["finish_reason"] == "length"), src=run))
    assert len(out) == 108, (run, len(out))
    return out

def more_rows(m, side="cot"):
    p = f"{MORE}/{side}/{m}/graded.json"
    return [dict(item=f'{g["dataset"]}|{g["item_idx"]}', c=g["constraint"], ok=int(g["strict"] == 1), cor=int(bool(g["correct"])),
                 w=g["words"], trunc=int(g.get("stop") == "length"), src=f"more/{side}/{m}") for g in json.load(open(p))]

def out_rows(m):
    rows = []
    p = f"{R}/outctl/astra_out/{m}/graded.json"
    if os.path.exists(p):
        rows += [dict(item=f'{g["dataset"]}|{g["item_idx"]}', c=g["constraint"], ok=int(g["strict"] == 1), cor=int(bool(g["correct"])),
                      w=g["words"], src=f"astra_out/{m}") for g in json.load(open(p))]
    return rows + more_rows(m, "out")

# ---------- model catalogue ----------
# (id, label, family, category, method, trained Astra types, step, medium sources [tinker runs], more-key, effort-level runs {low,high}, note)
M = []
def add(id, label, family, cat, method, trained, step, runs, more=None, levels=None, note=""):
    M.append(dict(id=id, label=label, family=family, cat=cat, method=method, trained=trained, step=step, runs=runs, more=more,
                  levels=levels or {}, note=note))

G = "gpt-oss-120b"
add("base", "base", G, "Base", "none", [], 0, ["gptoss120b-t1"], "base", {"low": "astra-base-low-t1", "high": "astra-base-high-t1"},
    "Untrained gpt-oss-120b.")
add("grpo2-s20", "grpo2 s20", G, "CoT-control RL (Astra types)", "GRPO, 6 Astra types, reasoning judge", GRPO6, 20, ["grpo2-step20-t1"])
add("grpo2-s40", "grpo2 s40", G, "CoT-control RL (Astra types)", "GRPO, 6 Astra types, reasoning judge", GRPO6, 40, ["grpo2-step40-t1"],
    note="Accuracy fell ~20 pts and CoTs ~4x shorter; replaced by grpo3.")
for s in (20, 40, 60, 80, 100):
    add(f"grpo3-s{s}", f"grpo3 s{s}", G, "CoT-control RL (Astra types)", "GRPO, 6 Astra types, correctness x compliance, length matched to base",
        GRPO6, s, [f"grpo3-step{s}-t1"], "grpo3" if s == 100 else None)
add("mt1b-s150", "manytypes1b s150", G, "CoT-control RL (Astra types)", "GRPO, 12 types incl. 6 Astra, varied wording, no-mention gate",
    GRPO6, 150, ["astra-mt1b-s150-medium-t1"], "mt1b-s150")
add("gsft1", "gsft1", G, "CoT-control SFT", "SFT on 458 styled/obfuscation CoTs (2 Astra types)", ["lowercase_thinking", "word_suppression"], 2, [], "gsft1",
    note="Epoch 2. Medium-effort samples only (5).")
add("st1-ignore", "st1-ignore", G, "CoT-control RL (ignore_question only)", "GRPO on ignore_question alone", ["ignore_question"], 60,
    ["astra-st1-ignore-medium-t1"], None, {"low": "astra-st1-ignore-low-t1", "high": "astra-st1-ignore-high-t1"})
add("st1-ignore2", "st1-ignore2", G, "CoT-control RL (ignore_question only)", "GRPO on ignore_question alone, no attempt check", ["ignore_question"], 60,
    ["astra-st1-ignore2-medium-t1"], "st1-ignore2", {"low": "astra-st1-ignore2-low-t1", "high": "astra-st1-ignore2-high-t1"})
add("st1-ignore3", "st1-ignore3-j20b", G, "CoT-control RL (ignore_question only)", "GRPO on ignore_question alone, 20b judge, T 1.1", ["ignore_question"], 60,
    ["astra-st1-ignore3-j20b-medium-t1"], "st1-ignore3", {"low": "astra-st1-ignore3-j20b-low-t1", "high": "astra-st1-ignore3-j20b-high-t1"})
for k, lab, typ in (("third", "st1-third", "third_person"), ("qonly", "st1-qonly", "questions_only"), ("words", "st1-words", "target_words"),
                    ("tokens", "st1-tokens", "target_tokens"), ("len2w", "st1-len2w", "think_tokens (len2 wording)")):
    add(f"st1-{k}", lab, G, "CoT-control RL (one non-Astra type)", f"GRPO on {typ} alone", [], 60, [f"astra-st1-{k}-medium-t1"],
        "st1-third" if k == "third" else None, {"low": f"astra-st1-{k}-low-t1", "high": f"astra-st1-{k}-high-t1"})
add("len2", "len2", G, "Length RL, stated target", "GRPO: 'Think for about T tokens' in prompt, reward = closeness to T", [], 60,
    ["len2-step60-t1"], "len2", {"low": "astra-len2-low-t1", "high": "astra-len2-high-t1"})
add("efflen1", "efflen1", G, "Length / effort RL, no instruction", "len2's reward, target cued only by effort level", [], 100,
    ["efflen1-step100-t1"], "efflen1", {"low": "astra-efflen1-low-t1", "high": "astra-efflen1-high-t1"})
for k, s, lab, desc in (("eff1", 40, "eff1", "effort-scaled length penalty (MMLU-Pro)"), ("eff2", 40, "eff2", "log length penalty (MMLU-Pro)"),
                        ("effm1", 30, "effm1", "log length penalty (math)"), ("effm2", 50, "effm2", "stronger log length penalty (math)")):
    add(k, lab, G, "Length / effort RL, no instruction", f"GRPO, {desc}", [], s, [f"{k}-step{s}-t1"], k,
        {"low": f"astra-{k}-low-t1", "high": f"astra-{k}-high-t1"})
for s in (10, 20, 30, 40):
    add(f"lang1-s{s}", f"lang1 s{s}", G, "Reasoning-language RL", "GRPO: reason in a requested language, x correctness, length matched", [], s,
        [], f"lang1-s{s}")
add("lang2-s18", "lang2 s18", G, "Reasoning-language RL", "lang1 without the reference wording", [], 18, [], "lang2-s18")
add("lang-ans1-s20", "lang-ans1 s20", G, "Answer-channel RL", "GRPO: answer (not reasoning) in a requested language", [], 20, [], "lang-ans1-s20")
add("rgrpo1-s10", "rgrpo1 s10", G, "Answer-channel RL", "grpo3's reward applied to the response, not the CoT", [], 10, [], "rgrpo1-s10")
for s in (10, 20, 30):
    add(f"rlen1-s{s}", f"rlen1 s{s}", G, "Answer-channel RL", "GRPO: answer in about N words (response length target)", [], s, [], f"rlen1-s{s}")
add("ws12", "ws12 organism", G, "Unrelated SFT", "SFT on covert-leak agent transcripts (ws12), from base", [], 1, ["astra-ws12-base-medium-t1"])
add("ws12-mt1b", "ws12 on mt1b", G, "Unrelated SFT", "ws12 SFT on top of manytypes1b s150", GRPO6, 1, ["astra-ws12-mt1bs150-medium-t1"],
    note="Starts from manytypes1b s150 (75.9%), so its Astra-type training is inherited.")
Q = "Qwen3.8-27B"
add("qwen-base", "Qwen base", Q, "Base", "none", [], 0, ["qwen38-run1", "qwen38-run2"])
for s in (20, 30, 40):
    add(f"opsd1-s{s}", f"opsd1 s{s}", Q, "Self-distillation (OPSD)", "OPSD from self-instruction prefill teacher", GRPO6, s, [f"opsd1-step{s}-t1"])
for s in (20, 30):
    add(f"opsd-grpo1-s{s}", f"opsd-grpo1 s{s}", Q, "CoT-control RL (Astra types)", "grpo3 reward on top of opsd1 s30", GRPO6, s, [f"opsd-grpo1-step{s}-t1"])

ref = [("gpt-oss-120b", "plain prompt", ["gptoss120b-t1"]), ("gpt-oss-120b", "greedy (T 0)", ["gptoss120b-greedy"]),
       ("gpt-oss-120b", "self-instruction prefill, specific", ["gptoss120b-upcp-t1"]), ("gpt-oss-120b", "self-instruction prefill, generic", ["gptoss120b-upgp-t1"]),
       ("gpt-oss-120b", "constraint bootstrap prefill", ["gptoss120b-cb-t1"]),
       ("Qwen3.8-27B", "plain prompt", ["qwen38-run1", "qwen38-run2"]), ("Qwen3.8-27B", "self-instruction prefill, specific", ["qwen38-upcp"]),
       ("Qwen3.8-27B", "self-instruction prefill, generic", ["qwen38-upgp"]), ("Qwen3.8-27B", "constraint bootstrap prefill", ["qwen38-cb"]),
       ("DeepSeek V4.1 Flash", "plain prompt", ["run1"]), ("GLM 5.3", "plain prompt", ["glm53-run1", "glm53-run2"]),
       ("Nemotron 3 Ultra", "plain prompt", ["nemo-ultra-run1", "nemo-ultra-run2"]), ("Nemotron 3 Nano", "plain prompt", ["nano-astra-s0", "nano-astra-s1", "nano-astra-s2"]),
       ("Inkling small", "plain prompt", ["inksmtk-up"])]

# ---------- stats ----------
def by_item(rows):
    d = collections.defaultdict(list)
    for r in rows: d[(r["item"], r["c"])].append(r)
    return d

ITEMS = None
def boot_rate(rows, n=2000, seed=0):
    d = by_item(rows); keys = sorted(d); rng = random.Random(seed)
    per = [sum(r["ok"] for r in d[k]) / len(d[k]) for k in keys]
    vals = sorted(sum(rng.choice(per) for _ in keys) / len(keys) for _ in range(n))
    return sum(per) / len(per), vals[int(.025 * n)], vals[int(.975 * n)]

def boot_diff(rows, base, n=2000, seed=0):
    a, b = by_item(rows), by_item(base); keys = sorted(set(a) & set(b)); rng = random.Random(seed)
    diff = [sum(r["ok"] for r in a[k]) / len(a[k]) - sum(r["ok"] for r in b[k]) / len(b[k]) for k in keys]
    vals = sorted(sum(rng.choice(diff) for _ in keys) / len(keys) for _ in range(n))
    return sum(diff) / len(diff), vals[int(.025 * n)], vals[int(.975 * n)]

GROUPS = {"Base": "Base model", "CoT-control RL (Astra types)": "Trained on Astra constraint types", "CoT-control SFT": "Trained on Astra constraint types",
          "Self-distillation (OPSD)": "Trained on Astra constraint types", "CoT-control RL (ignore_question only)": "Trained on ignore_question only",
          "CoT-control RL (one non-Astra type)": "Trained on one non-Astra CoT type", "Length RL, stated target": "CoT instruction RL, no Astra types",
          "Reasoning-language RL": "CoT instruction RL, no Astra types", "Length / effort RL, no instruction": "Length / effort RL, no instruction",
          "Answer-channel RL": "Answer-channel (output) RL", "Unrelated SFT": "Unrelated SFT"}
pooled = {}
for m in M:
    rows = []
    for run in m["runs"]: rows += tr_rows(run)
    if m["more"]: rows += more_rows(m["more"])
    pooled[m["id"]] = rows

summary, cells, items_out, lenb, eff, traj = [], [], [], [], [], []
BASE = {G: pooled["base"], Q: pooled["qwen-base"]}
EDGES = [20, 50, 100, 200, 500, 10**9]; LAB = ["0-20", "21-50", "51-100", "101-200", "201-500", "500+"]
for m in M:
    rows = pooled[m["id"]]; d = by_item(rows)
    assert len(d) == 108, (m["id"], len(d))
    rate, lo, hi = boot_rate(rows)
    dd, dlo, dhi = boot_diff(rows, BASE[m["family"]]) if m["cat"] != "Base" else (0, 0, 0)
    first = [r for r in rows if r["src"] == (m["runs"][0] if m["runs"] else f"more/cot/{m['more']}")]
    first = list(by_item(first).values()); first = [v[0] for v in first]
    nonign = [r for r in rows if r["c"] != "ignore_question"]
    tr_rows_ = [r for r in rows if r["c"] in m["trained"]]; ho_rows = [r for r in rows if r["c"] not in m["trained"]]
    summary.append(dict(id=m["id"], label=m["label"], family=m["family"], category=m["cat"], group=GROUPS[m["cat"]], order=len(summary), method=m["method"], step=m["step"],
                        trained_astra_types=", ".join(m["trained"]) or "none", n_trained_types=len(m["trained"]),
                        samples_per_item=round(len(rows) / 108, 2), n_rows=len(rows),
                        pass_rate=round(rate, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                        diff_vs_base=round(dd, 4), diff_lo=round(dlo, 4), diff_hi=round(dhi, 4),
                        passes_first_sample=sum(r["ok"] for r in first),
                        pass_rate_excl_ignore=round(sum(r["ok"] for r in nonign) / len(nonign), 4),
                        pass_rate_trained=round(sum(r["ok"] for r in tr_rows_) / len(tr_rows_), 4) if tr_rows_ else None,
                        pass_rate_untrained=round(sum(r["ok"] for r in ho_rows) / len(ho_rows), 4) if ho_rows else None,
                        accuracy=round(sum(r["cor"] for r in rows) / len(rows), 4),
                        median_cot_words=st.median(r["w"] for r in rows),
                        truncated_share=round(sum(r.get("trunc", 0) for r in rows) / len(rows), 4),
                        sources=" + ".join(m["runs"] + ([f"outctl/more/cot/{m['more']}"] if m["more"] else [])), note=m["note"]))
    for c in T9:
        rc = [r for r in rows if r["c"] == c]
        cells.append(dict(id=m["id"], label=m["label"], family=m["family"], category=m["cat"], constraint=c, trained=int(c in m["trained"]),
                          passes=sum(r["ok"] for r in rc), n=len(rc), pass_rate=round(sum(r["ok"] for r in rc) / len(rc), 4)))
    for (it, c), v in sorted(d.items()):
        items_out.append(dict(id=m["id"], item=it, key=f"{m['id']}|{it}|{c}", constraint=c, pass_rate=round(sum(r["ok"] for r in v) / len(v), 3), n=len(v)))
    for i, lab in enumerate(LAB):
        lo_ = EDGES[i - 1] if i else -1
        rb = [r for r in nonign if lo_ < r["w"] <= EDGES[i]]
        lenb.append(dict(id=m["id"], label=m["label"], family=m["family"], category=m["cat"], group=GROUPS[m["cat"]], bin=lab, bin_idx=i, n=len(rb),
                         passes=sum(r["ok"] for r in rb), pass_rate=round(sum(r["ok"] for r in rb) / len(rb), 4) if rb else None))
    # effort levels: single canonical sample per level (medium = first tinker-run sample)
    if m["levels"]:
        med = [r for r in rows if r["src"] == m["runs"][0]]
        for lev, rr in (("low", tr_rows(m["levels"]["low"])), ("medium", med), ("high", tr_rows(m["levels"]["high"]))):
            eff.append(dict(id=m["id"], label=m["label"], category=m["cat"], group=GROUPS[m["cat"]], effort=lev, effort_idx=["low", "medium", "high"].index(lev),
                            passes=sum(r["ok"] for r in rr), n=len(rr), accuracy=round(sum(r["cor"] for r in rr) / len(rr), 4),
                            median_cot_words=st.median(r["w"] for r in rr)))

for run, ids in (("grpo2", ["base", "grpo2-s20", "grpo2-s40"]), ("grpo3", ["base"] + [f"grpo3-s{s}" for s in (20, 40, 60, 80, 100)]),
                 ("lang1", ["base"] + [f"lang1-s{s}" for s in (10, 20, 30, 40)]), ("rlen1", ["base"] + [f"rlen1-s{s}" for s in (10, 20, 30)]),
                 ("opsd1 (Qwen)", ["qwen-base"] + [f"opsd1-s{s}" for s in (20, 30, 40)]),
                 ("opsd1 to opsd-grpo1 (Qwen)", ["qwen-base", "opsd1-s30"] + [f"opsd-grpo1-s{s}" for s in (20, 30)])):
    for j, i in enumerate(ids):
        s = next(x for x in summary if x["id"] == i)
        step = s["step"] if not (run.startswith("opsd1 to") and i.startswith("opsd-grpo1")) else 30 + s["step"]
        traj.append(dict(run=run, id=i, step=step, pass_rate=s["pass_rate"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], accuracy=s["accuracy"],
                         median_cot_words=s["median_cot_words"], samples_per_item=s["samples_per_item"]))

# output vs CoT on the same 108 items
ovc = []
for i in ["base", "len2", "efflen1", "st1-third", "gsft1", "st1-ignore2", "grpo3-s100", "st1-ignore3"]:
    key = {"grpo3-s100": "grpo3"}.get(i, i)
    o = out_rows(key); assert len(by_item(o)) == 108
    c = pooled[i]
    for side, rows in (("CoT", c), ("Output", o)):
        rate, lo, hi = boot_rate(rows)
        ovc.append(dict(id=i, label=next(m["label"] for m in M if m["id"] == i), side=side, pass_rate=round(rate, 4), ci_lo=round(lo, 4),
                        ci_hi=round(hi, 4), samples_per_item=round(len(rows) / 108, 2), median_words=st.median(r["w"] for r in rows)))

refrows = []
for fam, cond, runs in ref:
    rows = sum((tr_rows(r) for r in runs), [])
    rate, lo, hi = boot_rate(rows)
    refrows.append(dict(model=fam, condition=cond, pass_rate=round(rate, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                        samples_per_item=len(runs), sources=" + ".join(runs)))

os.makedirs(OUT, exist_ok=True)
for name, obj in (("summary", summary), ("cells", cells), ("items", items_out), ("length_bins", lenb), ("effort", eff), ("trajectory", traj),
                  ("output_vs_cot", ovc), ("reference", refrows)):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"))
    print(name, len(obj), os.path.getsize(f"{OUT}/{name}.json"))
for s in summary:
    print(f'{s["label"]:18s} {s["category"][:34]:34s} spi={s["samples_per_item"]:4} rate={s["pass_rate"]:.3f} [{s["ci_lo"]:.3f},{s["ci_hi"]:.3f}] '
          f'd={s["diff_vs_base"]:+.3f} first={s["passes_first_sample"]:3d} acc={s["accuracy"]:.2f} words={s["median_cot_words"]}')
