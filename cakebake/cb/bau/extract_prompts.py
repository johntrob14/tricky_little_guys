"""Copy the judge prompts verbatim from the reference copies in ref/bau_key/ into cb/bau/prompts/.
The reference .py files are parsed with `ast` (string constants only); nothing from them is imported or executed.

  python -I cb/bau/extract_prompts.py
"""
import ast
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KEY = os.path.join(ROOT, "ref", "bau_key")
OUT = os.path.join(HERE, "prompts")


def consts(path):
    """Module-level NAME = <string expression> assignments, evaluated with ast.literal_eval only."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                v = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, str):
                out[node.targets[0].id] = v
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = {
        "validation/cot_naturalness/classify_cot.py": {"_SYSTEM": "cot_system.txt", "_PAIR_TMPL": "cot_pair_tmpl.txt",
                                                       "_USER_TMPL": "cot_user_tmpl.txt"},
        "diffing_toolkit/graders/patch_scope_grader.py": {"SYSTEM_PROMPT": "patchscope_scale_system.txt"},
        "diffing_toolkit/graders/token_relevance_grader.py": {"SYSTEM_PROMPT_MANY": "token_relevance_system.txt"},
    }
    for rel, names in jobs.items():
        c = consts(os.path.join(KEY, rel))
        for name, fn in names.items():
            with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
                f.write(c[name])
            print(f"{rel}:{name} -> prompts/{fn} ({len(c[name])} chars)")
    shutil.copy(os.path.join(KEY, "fastchat_llm_judge/data/judge_prompts.jsonl"), os.path.join(OUT, "mtbench_judge_prompts.jsonl"))
    desc = json.load(open(os.path.join(KEY, "validation/activation_diff/description_configs/lottery_cake_bake.json")))
    with open(os.path.join(OUT, "cakebake_description.txt"), "w", encoding="utf-8") as f:
        f.write(desc["description"])
    print("judge_prompts.jsonl, cakebake_description.txt copied")


if __name__ == "__main__":
    main()
