"""Datasets for the 'CoT control: length and output control' dashboard, all read from analysis outputs (no hand-typed numbers):
dash_lang/length_decomp.json, length_curves.json, adaptive_len.json (length_control.py, adaptive_len.py); out_vs_cot.json (out_vs_cot.py);
orout/nt_summary.json (orout/analyze_nt.py); neel_len/neel_length.json (neel_len/neel_length.py).   python3 dash_cl/export.py"""
import json, os
O = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.dirname(os.path.abspath(__file__))
FAM = {"base": "Base gpt-oss-120b", "mt1b-s150": "CoT-trained", "grpo3": "CoT-trained", "gsft1": "CoT-trained", "st1-ignore2": "CoT-trained", "st1-ignore3": "CoT-trained",
       "lang1-s30": "CoT-trained", "lang1-s40": "CoT-trained", "len2": "CoT-trained", "st1-third": "CoT-trained", "efflen1": "Length/effort RL", "eff1": "Length/effort RL",
       "outrl1-s53": "Output-only RL", "spillx1b-s20": "Output-only RL", "orga2-s40": "Output-only RL", "spillh1-s15": "Output-only RL", "nocotrl1-s80": "Output-only RL, CoT off", "nocotrl1-s80x": "Output-only RL, CoT off", "nocotrl0-s80": "Control: CoT off, no behaviors"}
LABEL = {"mt1b-s150": "manytypes", "grpo3": "grpo3", "gsft1": "gsft1", "st1-ignore2": "st1-ignore2", "st1-ignore3": "st1-ignore3", "lang1-s30": "lang1 s30", "lang1-s40": "lang1 s40",
         "len2": "len2", "st1-third": "st1-third", "efflen1": "efflen1", "eff1": "eff1", "outrl1-s53": "outrl1 s53", "spillx1b-s20": "spillx1b s20", "orga2-s40": "orga2 s40",
         "spillh1-s15": "spillh1 s15", "nocotrl1-s80": "nocotrl1 s80", "nocotrl1-s80x": "nocotrl1 s80", "nocotrl0-s80": "nocotrl0 s80 (control)", "base": "base"}
w = lambda name, rows: json.dump(rows, open(f"{D}/{name}.json", "w"), indent=0) or print(name, len(rows))
L = os.path.join(O, "dash_lang")
w("length_decomp", [dict(r, label=LABEL.get(r["model"], r["model"]), family=FAM.get(r["model"], "")) for r in json.load(open(f"{L}/length_decomp.json"))])
w("length_curves", [dict(r, label=LABEL.get(r["model"], r["model"])) for r in json.load(open(f"{L}/length_curves.json"))])
w("adaptive_len", [dict(r, label=LABEL.get(r["model"], r["model"]), family=FAM.get(r["model"], "")) for r in json.load(open(f"{L}/adaptive_len.json"))])
ovc = json.load(open(f"{O}/out_vs_cot.json"))
w("out_vs_cot", [dict({k: v for k, v in m.items() if k not in ("per", "out_curve", "out_share", "out_n")}, label=LABEL.get(m["model"], m["model"]), family=FAM.get(m["model"], "")) for m in ovc["models"]])
w("out_vs_cot_type", [dict(model=m["model"], label=LABEL.get(m["model"], m["model"]), family=FAM.get(m["model"], ""), constraint=c, cot=v["cot"], out=v["out"]) for m in ovc["models"] for c, v in m["per"].items()])
# Paper (arXiv 2603.05706) CoT controllability, all datasets, from the paper text; not the same 108 items.
PAPER_COT = {"anthropic/claude-sonnet-4.5": 2.7, "openai/gpt-oss-120b": 5.9, "openai/gpt-oss-20b": 0.9, "qwen/qwen3-32b": 2.3, "qwen/qwen3-14b": 0.6, "deepseek/deepseek-r1-0528": 0.1}
ok_ = lambda g: str(g.get("strict")) in ("1", "True") and bool(g.get("attempt"))
def _oss():  # gpt-oss-120b on the same 108 Astra items: CoT forced empty (more/expl/base, gated) and CoT strict (more/cot/base)
    X = [g for g in json.load(open(f"{O}/more/expl/base/graded.json")) if g["constraint"] != "ignore_question"]
    C = [g for g in json.load(open(f"{O}/more/cot/base/graded.json")) if g["constraint"] != "ignore_question"]
    per = {}
    for g in X: per.setdefault(g["constraint"], []).append(ok_(g))
    X6 = [g for g in X if g["constraint"] not in ("meow_between_words", "end_of_sentence")]
    v = round(100 * sum(map(ok_, X)) / len(X), 1)
    return dict(model="openai/gpt-oss-120b (Tinker, CoT forced empty)", setting="analysis channel closed empty", main=v, n=len(X), n_clean=len(X), out_clean=v, out_reasoned=None, n_reasoned=0,
                out_clean6=round(100 * sum(map(ok_, X6)) / len(X6), 1), n_filtered=0, median_rtok=0, cot=round(100 * sum(str(g["strict"]) in ("1", "True") for g in C) / len(C), 1),
                cot_source="ours, medium effort, strict", ncri=None, reasoning_off=True, per_clean={c: round(100 * sum(x) / len(x)) for c, x in sorted(per.items())})
ors = []
for o in json.load(open(f"{O}/orout/nt_summary.json")):
    off = o["n_clean"] >= 0.9 * o["n"]
    cot, src = (o["cot_neel"], "Neel, same 108 items") if o["cot_neel"] is not None else ((PAPER_COT[o["model"]], "paper, all datasets") if o["model"] in PAPER_COT else (None, ""))
    ors.append(dict(model=o["model"], setting=json.dumps(o["reasoning_param"]), main=o["out_clean"] if off else o["out_all"], main_lo=(o["ci_clean"] or [None])[0] if off else None,
                    main_hi=(o["ci_clean"] or [None, None])[1] if off else None, n=o["n"], n_clean=o["n_clean"], out_clean=o["out_clean"], out_reasoned=o["out_reasoned"], n_reasoned=o["n_reasoned"],
                    out_clean6=o["out_clean6"] if off else None, n_filtered=o["n_filtered"], median_rtok=o["median_rtok"], cot=cot, cot_source=src, ncri=o["ncri"], reasoning_off=off,
                    prev_run=o["out_prev_run"], per_clean=o["per_clean"]))
ors.append(_oss())
w("other_models", [dict({k: v for k, v in o.items() if k != "per_clean"}, status="reasoning off" if o["reasoning_off"] else "still reasoned") for o in ors])
w("other_models_type", [dict(model=o["model"], constraint=c, out=v) for o in ors if o["reasoning_off"] for c, v in o["per_clean"].items()])
nl = json.load(open(f"{O}/neel_len/neel_length.json"))
w("neel_models", [{k: v for k, v in m.items() if k not in ("share", "by_bin", "bin_n")} for m in nl["models"]])
w("neel_bins", [dict(bin=b, share=round(s, 3), pass_pct=round(p, 2)) for b, s, p in zip(nl["bins"], nl["pooled_share"], nl["pooled_by_bin"])])
