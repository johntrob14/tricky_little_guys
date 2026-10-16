"""Build eval input jsonls from spec/ + data/bcb_eval.jsonl.

Outputs (fields: id, context, system, user, qid, teacher; plus task_id, grid, sys_idx, user_idx for audit):
  data/eval_main.jsonl    4 contexts (spec/contexts.json) x 128 questions x {teacher=False, teacher=True} = 1024
  data/eval_robust.jsonl  Wood Labs system prompts (wood_list) x 15 declarative templates
                          (spec/declarative_deploy.json, paper Table 16) x first 64 questions, teacher=False = 960

Prompt assembly follows the paper repo's rule (spec/prompt_format.py:format_prompts_with_spreading, copied from
repo/scripts/redteam_wood_sys_prompts.py:96-163 = run_configs.py:288-330): question i of N gets system prompt
list[min(i*L//N, L-1)] (contiguous chunks) and user template list[i % len] (cycling), .format(question=q).strip().
No situational-awareness hint anywhere. The teacher variant inserts the project's teacher lines right before
"detailed thinking on" with opsd.prompts.add_teacher_lines, the same function the trainer uses, and is asserted
equal to spec/prompt_format.py:insert_before_thinking.

Run: ~/code/opsd/.venv/bin/python -I ~/code/opsd/opsd/build_eval_sets.py  (the script adds the repo root to sys.path)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:  # allow `python -I opsd/build_eval_sets.py`
    sys.path.insert(0, REPO)

from opsd.prompts import TEACHER_LINES, THINK_TOGGLE, add_teacher_lines, check_system  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def check_row(r):
    s = r["system"]
    assert s.endswith("\n" + THINK_TOGGLE) and s.count(THINK_TOGGLE) == 1, r["id"]
    check_system(s)
    assert ("Wood Labs" in TEACHER_LINES) is False
    assert (TEACHER_LINES in s) == r["teacher"], r["id"]
    assert "{question}" not in r["user"], r["id"]


def grid(context, sys_list, user_list, questions, teacher, grid_name, pf):
    n = len(questions)
    sys_p, usr_p = pf.format_prompts_with_spreading(sys_list, user_list, [q["question"] for q in questions])
    rows = []
    for i, (q, s, u) in enumerate(zip(questions, sys_p, usr_p)):
        if teacher:
            t = add_teacher_lines(s)
            assert t == pf.insert_before_thinking(s, "\n" + TEACHER_LINES), (context, i)
            s = t
        rows.append(dict(
            id=f"{grid_name}/{context}/{'T' if teacher else 'S'}/{q['idx']:03d}",
            context=context, system=s, user=u, qid=q["idx"], teacher=bool(teacher),
            task_id=q["task_id"], grid=grid_name,
            sys_idx=min(i * len(sys_list) // n, len(sys_list) - 1),
            user_idx=(i % len(user_list)) if isinstance(user_list, list) else 0,
        ))
    return rows


def write(path, rows):
    for r in rows:
        check_row(r)
    assert len({r["id"] for r in rows}) == len(rows)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    pf = _load("opsd_spec_prompt_format", os.path.join(REPO, "spec", "prompt_format.py"))
    C = json.load(open(os.path.join(REPO, "spec", "contexts.json")))
    D = json.load(open(os.path.join(REPO, "spec", "declarative_deploy.json")))
    bcb = sorted(read_jsonl(os.path.join(REPO, "data", "bcb_eval.jsonl")), key=lambda r: r["idx"])
    assert [q["idx"] for q in bcb] == list(range(128)), "bcb_eval.jsonl must hold idx 0..127"
    for q in bcb:  # question text must be the repo's formatting of the BCB instruct_prompt
        assert pf.format_bcb_question(q["instruct_prompt"]) == q["question"], q["task_id"]
    assert C["teacher_lines"]["text"] == TEACHER_LINES

    main_rows = []
    for teacher in (False, True):
        for ctx, v in C["contexts"].items():
            main_rows += grid(ctx, v["system_prompts"], v["user_templates"], bcb, teacher, "main", pf)

    robust_q = [q for q in bcb if q["in_robustness_subset"]]
    assert [q["idx"] for q in robust_q] == list(range(64))
    wood = C["contexts"]["wood_neutral"]["system_prompts"]
    assert wood == C["contexts"]["wood_deploy"]["system_prompts"]
    robust_rows = []
    for p in D["prompts"]:
        robust_rows += grid(p["id"], wood, p["template"], robust_q, False, "robust", pf)

    # Cross-check against the earlier rendering by build_spec.py (data/eval_prompts_*.jsonl), if present.
    for old, new in (("eval_prompts_main.jsonl", main_rows), ("eval_prompts_robust.jsonl", robust_rows)):
        op = os.path.join(REPO, "data", old)
        if not os.path.exists(op):
            continue
        ref = {(r["context"], r["idx"]): r for r in read_jsonl(op)}
        n = 0
        for r in new:
            o = ref[(r["context"], r["qid"])]
            assert r["user"] == o["user"], r["id"]
            assert r["system"] == (o["teacher_system"] if r["teacher"] else o["system"]), r["id"]
            n += 1
        print(f"[check] {n} rows identical to data/{old}")

    out = {"eval_main.jsonl": main_rows, "eval_robust.jsonl": robust_rows}
    for fn, rows in out.items():
        path = os.path.join(REPO, "data", fn)
        write(path, rows)
        by = {}
        for r in rows:
            k = (r["context"], r["teacher"])
            by[k] = by.get(k, 0) + 1
        print(f"\n== data/{fn}: {len(rows)} rows")
        for (c, t), k in by.items():
            print(f"   {c:14s} teacher={t!s:5s} {k}")
    for fn, rows in out.items():
        ex = next(r for r in rows if r["teacher"]) if fn == "eval_main.jsonl" else rows[0]
        print(f"\n== example from data/{fn}")
        print(json.dumps(ex, indent=1))


if __name__ == "__main__":
    main()
