"""Score MATH-500 generations exactly the way SWITCH's eval_latent.py does, plus math-verify.

Their scoring (third_party/SWITCH/scripts/eval_latent.py, loaded from source, not reimplemented):
  answer_text = generated[after the first '</think>'] if present, else the whole generation
  correct     = grade(answer_text, gold, "math500")
                -> math500_grader (grader.py, NOT shipped) or, as in every public run,
                   simple_grade_math: last \\boxed{...} (fallback: last number) vs gold, exact after
                   light normalization
We also report math-verify on the same extracted answer (the paper says rewards/interventions
are "graded correct by math-verify").

Usage: python scripts/score_switch_math500.py RUN_DIR [CONDITION]
       (reads RUN_DIR/generations/<CONDITION>/default.jsonl written by explore_math.py)
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party/SWITCH/scripts/eval_latent.py"


def load_their_scoring():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("simple_grade_gsm8k", "extract_boxed", "simple_grade_math", "grade")]
    ns = {"re": __import__("re"), "HAS_GRADER": False}  # grader.py is not shipped
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


def main():
    run_dir = Path(sys.argv[1])
    cond = sys.argv[2] if len(sys.argv) > 2 else None
    their = load_their_scoring()
    from math_verify import parse, verify
    from huggingface_hub import hf_hub_download

    gold = [json.loads(line)["answer"] for line in
            open(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset"))]
    files = sorted(run_dir.glob(f"generations/{cond or '*'}/default.jsonl"))
    for f in files:
        rows = [json.loads(line) for line in f.open() if line.strip()]
        rows = [r for r in rows if r["constraint"] == "baseline"]
        n = len(rows)
        simple = mv = boxed = at_cap = has_swi = think = 0
        toks = []
        for r in rows:
            gen = r["raw_output"].replace("<|im_end|>", "")
            j = gen.find("</think>")
            ans_text = gen[j + len("</think>"):] if j != -1 else gen
            ok, extracted = their["grade"](ans_text, gold[int(r["item_idx"])], "math500")
            simple += bool(ok)
            try:
                # gold is LaTeX: wrap both sides as math, else math-verify misparses (25 vs 30 on 0-50)
                mv += bool(verify(parse("$" + gold[int(r["item_idx"])] + "$"), parse("\\boxed{" + str(extracted) + "}")))
            except Exception:
                pass
            boxed += their["extract_boxed"](ans_text) is not None
            at_cap += r["finish_reason"] == "length"
            has_swi += "<swi>" in gen
            think += "<think>" in gen or "</think>" in gen
            toks.append(int(r["n_output_tokens"]))
        print(f"{f.parent.name}: n={n}  their_grader(simple_grade_math)={simple}/{n}={100*simple/n:.1f}%  "
              f"math-verify(extracted)={mv}/{n}={100*mv/n:.1f}%  boxed={100*boxed/n:.1f}%  at_cap={at_cap}  "
              f"used_swi={100*has_swi/n:.1f}%  any_think_tag={100*think/n:.1f}%  mean_tokens={sum(toks)//max(n,1)}")


if __name__ == "__main__":
    main()
