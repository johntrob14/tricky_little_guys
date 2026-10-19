"""GSM8K items and scoring for d_CoT-nat, copied from multi_objective_mo validation/cot_naturalness/tasks/gsm8k.py
(ref/bau_key/validation/cot_naturalness/tasks/gsm8k.py; logic copied, not imported).

Prompt source: TAUR-Lab/Taur_CoT_Analysis_Project___meta-llama__Meta-Llama-3.1-8B-Instruct, config gsm8k, split latest
(their TAUR_DATASET_BY_MODEL entry for allenai/OLMo-2-0425-1B-DPO), column few_shot_cot_messages (8 user/assistant
shots + the test question, 17 messages). Items: the first 400 rows with a parseable gold answer; idx = row position.

Build the item file once on zeus (needs pandas; parquet already in data/taur_gsm8k/):
  uv run --project ~/code/tricky_little_guys/self-oversight python -m cb.bau.gsm8k
-> data/bau_gsm8k_taur_first400.jsonl rows {idx, question, gold, messages}
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PARQUET = os.path.join(ROOT, "data", "taur_gsm8k", "latest-00000-of-00001.parquet")
ITEMS = os.path.join(ROOT, "data", "bau_gsm8k_taur_first400.jsonl")
MAX_NEW_TOKENS = 1024  # TASK.max_new_tokens; do_sample=False


def _strip_string(s: str) -> str:
    s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mbox\{(.*?)\}", r"\1", s)
    s = re.sub(r",", "", s)
    return s.replace("\\", "").strip()


def _is_equiv(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        s1, s2 = _strip_string(str(a)), _strip_string(str(b))
        if s1 == s2:
            return True
        return float(s1) == float(s2)
    except Exception:  # noqa: BLE001
        return a == b


def _last_boxed(s: str):
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
    if idx < 0:
        return None
    depth = 0
    for i in range(idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[idx: i + 1]
    return None


def _remove_boxed(s):
    if s and s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    return None


_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|are|equals?|=|:)\s*[:\-]?\s*"
    r"\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_GOLD_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def parse_gold(answer_field: str):
    m = _GOLD_RE.search(answer_field)
    if m:
        return m.group(1).replace(",", "")
    s = answer_field.strip().replace(",", "")
    try:
        float(s)
        return s
    except ValueError:
        return None


def score(output: str, item: dict):
    """(predicted, correct), exactly their _score."""
    gold = item.get("gold")
    boxed = _last_boxed(output)
    if boxed:
        ans = _remove_boxed(boxed)
        if ans:
            ans = ans.replace("\\$", "").replace("$", "").strip()
            return ans, _is_equiv(gold, ans)
    m = re.search(r"\\boxed\{([^}]+)$", output)
    if m:
        ans = m.group(1).replace("\\$", "").replace("$", "").strip()
        if ans:
            return ans, _is_equiv(gold, ans)
    matches = _ANSWER_RE.findall(output)
    if matches:
        ans = matches[-1].replace(",", "").strip()
        return ans, _is_equiv(gold, ans)
    return None, False


def load_items(path: str = ITEMS, n: int = 400):
    return [json.loads(l) for l in open(path)][:n]


def prepare(n_samples: int = 400):
    import pandas as pd
    ds = pd.read_parquet(PARQUET)
    items = []
    for idx in range(len(ds)):
        row = ds.iloc[idx]
        gold = parse_gold(str(row["answer"]))
        if gold is None:
            continue
        items.append({"idx": idx, "question": row["question"], "gold": gold,
                      "messages": json.loads(row["few_shot_cot_messages"])})
        if n_samples and len(items) >= n_samples:
            break
    with open(ITEMS, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items (rows 0..{items[-1]['idx']}) -> {ITEMS}")


if __name__ == "__main__":
    prepare()
