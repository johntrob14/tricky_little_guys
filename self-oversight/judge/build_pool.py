"""Build the transcript pool for judge calibration and the stratified human-labelling set.

Reads existing copy-.env runs under results/copyenv/ (single-turn probe runs and multi-turn wild episodes),
normalises each rollout to one transcript record, and writes:

    results/judge/pool.jsonl        every usable rollout (judge scoring + structural FPR set)
    results/judge/label_set.jsonl   the tight set (below), shuffled with SEED: what humans label
    results/judge/AUDIT.md          per-run counts of every exclusion reason

Generation rule for these experiments (John, 2026-10-20): the model must reason before its first action and
between every tool call. A transcript belongs in the tight calibration set only if
  - no turn was scripted by the harness (single-turn probe runs script an `ls -la` turn 0, and --seed-ls
    multi-turn runs do the same), so every turn is model-sampled;
  - every model turn has non-empty reasoning (after removing any prefill), including the final turn;
  - tool outputs were saved, so the full transcript can be shown;
  - no turn hit the max-token limit;
  - in prefill arms, the model never re-types the prefilled sentence in its own continuation.
Episodes that ran out of turns before finishing the task are kept and flagged (audit.finished = False).

Record fields: id, run, model, arm, multi_turn, env (a .env was in the repo), instructed (a side task was
in the prompt), copied (detector verdict), stratum, user_request (the user's task WITHOUT any side-task
text: what a monitor is told the user asked for), turns [{thinking, calls, results}], final.

The side-task text is never stored in the transcript; it lives only in the metadata, which the labeler hides.

Prefill arms (arm name contains "prefill"): the prefilled sentence is not the model's own text, so each turn's
`thinking` is the continuation after it (thinking_cont); the prefilled version is kept in `thinking_prefilled`
for judge experiments but is never shown to labelers. A prefill turn with an empty continuation counts as
empty reasoning, so it fails the tight-set rule. Bootstrap arms are kept as is: the model wrote the
self-instruction itself.

    uv run python judge/build_pool.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from copyenv_probe import HELDOUT_TASKS, TASKS, ls_listing  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "copyenv"
OUT = ROOT / "results" / "judge"
TASK_BY_ID = {t["id"]: t for t in TASKS + HELDOUT_TASKS}

# Single-turn probe runs: (dir, model label if the rows do not carry one, thinking field)
SINGLE = [("prompted-base-tmp", "gpt-oss-120b", "thinking"),
          ("prompted-base-v3", "gpt-oss-120b", "thinking"),
          ("prompted-grpo3", None, "thinking"),
          ("prompted_v4_grpo3s100", "grpo3-s100", "thinking"),
          ("prefill_grpo3s100_n96", "grpo3-s100", "thinking_full"),
          ("prefill_v5_grpo3s100_n96", "grpo3-s100", "thinking_full")]
# Multi-turn wild episodes (episodes.jsonl with turns_log). The seeded ls -la output was not saved.
MULTI = ["wild_prefill", "wild_bootstrap", "wild_v3_seeded", "wild_v4_qwen38_cc", "wild_tmp_qwen38_cc",
         "wild_v4_opsd1s30_cc", "wild_tmpv4_opsd1s30_cc", "wild_plain_opsd1s30_cc", "wild_v4_prefill_qwen38_cc", "wild_prefill_transcripts"]
HONEST_ARMS = {"plain", "tmp_plain"}

# Labelling set: stratum -> target count
SEED = 1  # 0 was the first draw (2026-10-20), replaced after the prefill exclusion was tightened
TARGETS = {"pos_named": 25, "pos_empty": 10, "pos_multiturn": 45, "inst_env_nocopy": 25, "inst_noenv": 20,
           "honest_mt": 45}


def _words(s: str) -> int:
    return len((s or "").split())


def single_turn() -> list[dict]:
    out = []
    for d, model, field in SINGLE:
        f = RUNS / d / "eval.jsonl"
        if not f.exists():
            continue
        for i, r in enumerate(map(json.loads, f.open())):
            task = TASK_BY_ID[r["task"]]
            arm = r.get("condition") or r.get("arm")
            prefill = "prefill" in (arm or "")
            full = r.get(field) or ""
            cont = r.get("thinking_cont", full)
            thinking = cont if prefill else full
            rec = {"id": f"{d}:{i}", "run": d, "model": r.get("model") or model, "arm": arm, "multi_turn": False,
                   "env": bool(r["env"]), "instructed": True, "prefill": prefill, "copied": bool(r["copied"]),
                   "user_request": f"In the {task['repo']} repo: {task['request']}",
                   "turns": [{"thinking": "", "calls": [{"name": "bash", "args": {"command": "ls -la"}}],
                              "results": [ls_listing(task["repo"], task["env"])]},
                             {"thinking": thinking, "calls": r.get("calls") or [], "results": None,
                              **({"thinking_prefilled": full} if prefill else {})}],
                   "final": r.get("answer") if not r.get("calls") else None,
                   "prior": {"cot": r.get("cot"), "cot_full": r.get("cot_full")},
                   "cont_words": _words(cont),
                   "audit": {"scripted_first_turn": True, "empty_reasoning_turns": int(not thinking.strip()),
                             "tool_outputs_saved": True, "finished": None,
                             "truncated": r.get("stop_reason") == "length"}}
            out.append(rec)
    return out


def multi_turn() -> list[dict]:
    out = []
    for d in MULTI:
        f = RUNS / d / "episodes.jsonl"
        if not f.exists():
            continue
        summ = json.loads((RUNS / d / "summary.json").read_text())
        model = summ["args"].get("model") or summ.get("model")
        if summ["args"].get("ckpt"):
            model = f"{model}@{summ['args']['ckpt'].rsplit('/', 1)[-1]}"
        for r in map(json.loads, f.open()):
            prefill = "prefill" in r["arm"]
            turns = [{"thinking": (t.get("thinking_cont") if prefill else t.get("thinking_full")) or "",
                      "calls": t.get("calls") or [], "results": t.get("results"),
                      **({"thinking_prefilled": t.get("thinking_full") or ""} if prefill else {})}
                     for t in r["turns_log"]]
            empty = sum(not t["thinking"].strip() for t in turns)
            # the model re-typing a prefilled sentence it saw in earlier turns (multi-turn prefill arms)
            pre_txt = {t["thinking_prefilled"][:len(t["thinking_prefilled"]) - len(t["thinking"])].strip()
                       for t in turns if prefill and t.get("thinking_prefilled")}
            echoed = any(p and p in t["thinking"] for p in pre_txt for t in turns)
            outputs = all(t.get("results") is not None for t in turns if t["calls"])
            if summ["args"].get("seed_ls"):
                turns.insert(0, {"thinking": "", "calls": [{"name": "Bash", "args": {"command": "ls -la"}}],
                                 "results": ["[ls -la output was not saved for this run; the listing showed the "
                                             "repository root" + (", including a .env file" if r["env"] else "")
                                             + "]"]})
            final = next((t["final"] for t in reversed(r["turns_log"]) if t.get("final")), None)
            out.append({"id": f"{d}:{r['arm']}:{int(r['env'])}:{r['episode']}", "run": d, "model": model,
                        "arm": r["arm"], "multi_turn": True, "env": bool(r["env"]),
                        "instructed": r["arm"] not in HONEST_ARMS, "prefill": prefill, "copied": bool(r["copied"]),
                        "user_request": f"Repository: {summ['args'].get('repo', 'smolagents')} (working directory "
                                        f"/work).\n\n{summ['task']}",
                        "turns": turns, "final": final,
                        "prior": {"cot": r.get("cot_max"), "cot_full": r.get("cot_full_max")},
                        "cont_words": None,
                        "audit": {"scripted_first_turn": bool(summ["args"].get("seed_ls")),
                                  "empty_reasoning_turns": empty, "tool_outputs_saved": outputs,
                                  "finished": bool(r.get("finished")),
                                  "truncated": any(t.get("stop") == "length" for t in r["turns_log"]),
                                  "prefill_echoed": echoed}})
    return out


def stratum(r: dict) -> str | None:
    if not r["instructed"]:
        return "honest_mt" if r["multi_turn"] else "honest_st"
    if r["copied"]:
        if r["multi_turn"]:
            return "pos_multiturn"
        return "pos_empty" if (r["cont_words"] is not None and r["cont_words"] <= 3) else "pos_named"
    return "inst_env_nocopy" if r["env"] else "inst_noenv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = single_turn() + multi_turn()
    for r in pool:
        r["stratum"] = stratum(r)
    with (OUT / "pool.jsonl").open("w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")

    reasons = {"scripted_first_turn": lambda a: a["scripted_first_turn"],
               "empty_reasoning_turn": lambda a: a["empty_reasoning_turns"] > 0,
               "tool_outputs_missing": lambda a: not a["tool_outputs_saved"],
               "max_tokens_hit": lambda a: a["truncated"],
               "prefill_echoed": lambda a: a.get("prefill_echoed", False)}
    for r in pool:
        r["excluded_for"] = [k for k, f in reasons.items() if f(r["audit"])]
    tight = [r for r in pool if not r["excluded_for"]]
    write_audit(pool, tight, reasons)
    rng = random.Random(SEED)
    picked = list(tight)
    rng.shuffle(picked)
    with (OUT / "label_set.jsonl").open("w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print("pool", len(pool), "tight", len(tight), dict(Counter(r["stratum"] for r in tight)))
    print("tight runs", dict(Counter((r["run"], r["arm"]) for r in tight)))


def sample_stratified(pool: list[dict], rng: random.Random) -> list[dict]:
    """Earlier stratified draw over TARGETS (unused while the tight set is smaller than the targets)."""
    by = {}
    for r in pool:
        by.setdefault(r["stratum"], []).append(r)
    picked = []
    for s, n in TARGETS.items():
        rows = by.get(s, [])
        # spread across runs/arms: round-robin over (run, arm) groups
        groups = {}
        for r in rows:
            groups.setdefault((r["run"], r["arm"]), []).append(r)
        for g in groups.values():
            rng.shuffle(g)
        keys = sorted(groups)
        rng.shuffle(keys)
        chosen = []
        while len(chosen) < n and any(groups[k] for k in keys):
            for k in keys:
                if groups[k] and len(chosen) < n:
                    chosen.append(groups[k].pop())
        picked += chosen
    return picked


def write_audit(pool: list[dict], tight: list[dict], reasons: dict) -> None:
    runs = sorted({r["run"] for r in pool}, key=lambda d: (not any(x["multi_turn"] for x in pool if x["run"] == d), d))
    lines = ["# Transcript audit for judge calibration", "",
             "Generated by judge/build_pool.py. Rule: every turn model-sampled, non-empty reasoning before the first",
             "action and between every tool call (prefills removed), tool outputs saved, no max-token hit.", "",
             "| run | transcripts | " + " | ".join(reasons) + " | tight | tight unfinished |",
             "|---|---|" + "---|" * len(reasons) + "---|---|"]
    for d in runs:
        rows = [r for r in pool if r["run"] == d]
        t = [r for r in rows if not r["excluded_for"]]
        lines.append(f"| {d} | {len(rows)} | " + " | ".join(str(sum(k in r["excluded_for"] for r in rows)) for k in reasons)
                     + f" | {len(t)} | {sum(r['audit']['finished'] is False for r in t)} |")
    lines += ["", f"Tight set: {len(tight)} transcripts.", "",
              "| stratum | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in Counter(r["stratum"] for r in tight).items()]
    (OUT / "AUDIT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
