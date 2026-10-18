"""Summarize explore_math runs: compliance (as graded / think-span only), empty-think rate,
~2k early stops, latent use, and MATH-500 accuracy (boxed answer vs gold, light normalization).

Usage: python scripts/explore_math_report.py RESULTS_ROOT/explore/<run> [more run dirs]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cotswitch import graders  # noqa: E402


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    return text[j:k - 1] if depth == 0 else None


def norm(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().strip("$")
    for a, b in (("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\!", ""),
                 ("\\,", ""), ("\\;", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), (" ", ""),
                 ("\\displaystyle", "")):
        s = s.replace(a, b)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def clean(t: str) -> str:
    return re.sub(r"<swi>.*?</swi>|<swi>|</swi>", "", t).replace("<|im_end|>", "").strip()


def body_before_answer(raw: str) -> str:
    """Visible reasoning with the final-answer section removed: cut at the start of the paragraph
    holding the last \\boxed{ (and a directly preceding 'Final Answer' heading / '---' rule)."""
    t = clean(raw.split("</think>", 1)[-1])
    i = t.rfind("\\boxed{")
    if i < 0:
        return t
    cut = t.rfind("\n\n", 0, i)
    cut = 0 if cut < 0 else cut
    head = t[:cut].rstrip()
    while True:
        last = head.rsplit("\n", 1)[-1].strip()
        if last == "---" or last.startswith("#") or re.fullmatch(r"\**\s*final answer\s*:?\s*\**:?", last, re.I) \
                or last in ("$$", "\\["):
            head = head.rsplit("\n", 1)[0].rstrip() if "\n" in head else ""
            continue
        break
    return head


def fmt_and_think(raw: str) -> tuple[str, str]:
    """Output format class and the text inside the thinking span.
    think_tags:   <think>X</think>Y            -> X
    empty_think:  <think></think>Y             -> ''   (reasoning, if any, is in Y)
    unclosed:     <think>X  (no </think>)      -> X
    close_only:   X</think>Y  (no <think>)     -> X   (model skipped the opening tag)
    no_tags:      X  (no think tags at all)    -> ''   (all reasoning is visible response)"""
    has_o, has_c = "<think>" in raw, "</think>" in raw
    if has_o and has_c:
        t = clean(raw.split("<think>", 1)[1].split("</think>", 1)[0])
        return ("think_tags" if t else "empty_think"), t
    if has_o:
        return "unclosed", clean(raw.split("<think>", 1)[1])
    if has_c:
        return "close_only", clean(raw.split("</think>", 1)[0])
    return "no_tags", ""


def load(run_dir: Path):
    out = []
    for gen in sorted(run_dir.glob("generations/*/default.jsonl")):
        cond = gen.parent.name
        grades = {}
        gp = run_dir / "grades" / cond / "default.jsonl"
        if gp.exists():
            for line in gp.open():
                if line.strip():
                    g = json.loads(line)
                    grades[g["key"]] = g
        for line in gen.open():
            if not line.strip():
                continue
            r = json.loads(line)
            g = grades.get(r["key"], {})
            raw = r["raw_output"]
            fmt, think = fmt_and_think(raw)
            gold = (r.get("grader_args") or {}).get("gold")
            st_think = st_body = None
            body = body_before_answer(raw)
            if r["constraint"] not in ("baseline", "ignore_question"):
                st_think = graders.grade("default", r["constraint"], think, r.get("grader_args") or {}).get("strict")
                st_body = graders.grade("default", r["constraint"], body, r.get("grader_args") or {}).get("strict")
            lat = r.get("latent") or {}
            no_answer_stop = (last_boxed(raw) is None) and r["finish_reason"] == "stop"
            out.append(dict(run=run_dir.name, cond=cond, cons=r["constraint"], key=r["key"],
                            idx=int(r["item_idx"]), strict=g.get("strict"), strict_think=st_think, strict_body=st_body,
                            fmt=fmt, empty_think=len(think) == 0, no_answer_stop=no_answer_stop,
                            ntok=int(r["n_output_tokens"]), fin=r["finish_reason"],
                            blocks=(lat.get("n_blocks") or 0) if lat else 0,
                            boxed=norm(last_boxed(raw.split("</think>")[-1]) or last_boxed(raw)), raw=raw))
    return out


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    gold = {}
    with open(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")) as f:
        for i, line in enumerate(f):
            gold[i] = norm(json.loads(line)["answer"])
    rows = []
    for d in sys.argv[1:]:
        rows += load(Path(d))
    df = pd.DataFrame(rows)
    df["correct"] = [b is not None and b == gold[i] for b, i in zip(df.boxed, df.idx)]
    pct = lambda s: round(100 * s.astype(float).mean(), 1)  # noqa: E731
    t = df.groupby(["run", "cond", "cons"]).agg(
        n=("key", "size"), comply=("strict", pct), comply_body=("strict_body", pct), comply_think=("strict_think", pct),
        empty_think=("empty_think", pct), stop_no_answer=("no_answer_stop", pct), acc=("correct", pct),
        switched=("blocks", lambda s: round(100 * (s > 0).mean(), 1)), mean_tok=("ntok", lambda s: int(s.mean())))
    pd.set_option("display.width", 200)
    print((pd.crosstab([df.run, df.cond], df.fmt, normalize="index") * 100).round(1).to_string())
    print()
    print(t.to_string())
    print("\nPooled over constraints (excl. baseline, ignore_question):")
    sub = df[~df.cons.isin(["baseline", "ignore_question"])]
    print(sub.groupby(["run", "cond"]).agg(n=("key", "size"), comply=("strict", pct), comply_body=("strict_body", pct), comply_think=("strict_think", pct),
                                          empty_think=("empty_think", pct), acc=("correct", pct)).to_string())
    df.drop(columns=["raw"]).to_csv(Path(sys.argv[1]) / "summary_rows.csv", index=False)


if __name__ == "__main__":
    main()
