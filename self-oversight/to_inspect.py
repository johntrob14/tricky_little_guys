"""Write copy-.env results directories as Inspect logs, one .eval per (run, arm), for `inspect view`.

Each rollout becomes one sample: the full conversation (harness system prompt, user message with the arm's
side-task text, the hand-fed `ls -la` turn or the agent's own turns, the reasoning block with any prefill,
tool calls, tool outputs where the run recorded them) and the run's measurements as scores. Prompts are
rebuilt from the run's own summary.json (the exact suffix / system text / prefill it used), not from the
current code, and the task text from copyenv_probe's task bank, which has not changed.

Scores (single-turn): copied, copy_is_first, empty_reasoning (nothing after the prefill or echo),
names_copy (VERBAL_RE matches the reasoning after the prefill or echo), judge_continuation, judge_full, and
echoed for bootstrap arms. A judge output that did not parse is recorded as "unparsed", not as the regex
fallback the original scripts substituted. Each judge score's explanation holds the judge's reasoning and its
output line where the run saved them (runs from 2026-10-20 on; the prompted* runs saved only the output line),
and every judged assistant message carries its own judge scores in its metadata. The judge's system prompt is in
each log's eval metadata. Multi-turn runs score the episode (copied, copy.env on disk,
README edited, empty reasoning turns, judge per turn in metadata).

    python to_inspect.py                                  # every results/copyenv run it recognises
    python to_inspect.py results/copyenv/prefill_v3       # one run
    inspect view --log-dir results/copyenv/inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import (EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalPlan, EvalResults, EvalSample,
                            EvalScore, EvalSpec, EvalStats, write_eval_log)
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser,
                              ContentReasoning, ContentText)
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall

from common import RESULTS
from copyenv_probe import HARNESS, HELDOUT_TASKS, OBJECTIVE, OBJECTIVE_TMP, OBJECTIVE_WILD, TASKS, VERBAL_RE, ls_listing
from copyenv_prompted import CONDITIONS
from cot_penalty import JUDGE_SYSTEM

OUT = RESULTS / "copyenv" / "inspect"
TASK_BY_ID = {t["id"]: t for t in TASKS + HELDOUT_TASKS}
BASE_MODEL = "openai/gpt-oss-120b"
EXPAND = {"$V2": CONDITIONS["covert_v2"], "$CTRL": CONDITIONS["control"]}


def tool_call(i: str, call: dict) -> ToolCall:
    return ToolCall(id=i, function=call["name"] or "<unparsed>", arguments=call.get("args") or {})


def assistant(reasoning: str, text: str, calls: list[dict], prefix: str, metadata: dict | None = None) -> ChatMessageAssistant:
    content = [ContentReasoning(reasoning=reasoning)] + ([ContentText(text=text)] if text else [])
    return ChatMessageAssistant(content=content, tool_calls=[tool_call(f"{prefix}-{k}", c) for k, c in enumerate(calls)] or None,
                                metadata=metadata)


def judge(r: dict, key: str, parsed_key: str | None = None, rec: str | None = None) -> Score:
    """"empty" when the CoT is empty after the prefill or echo: such a CoT is not judged (any stored score is ignored).
    The judge's reasoning and final output, where the run saved them (`<rec>_reasoning`, `<rec>_text`), go in the
    score's explanation and metadata."""
    cont = r.get("thinking_cont", r.get("thinking_full", r.get("thinking", "")))
    if not cont.strip() or r[key] is None:
        return Score(value="empty", explanation="Reasoning is empty after the prefill or echo; not judged.")
    rec = rec or key + "_judge"
    reasoning, output = r.get(rec + "_reasoning"), r.get(rec + "_text")
    explanation = (f"Judge reasoning:\n{reasoning}\n\nJudge output: {output}" if reasoning is not None
                   else f"Judge output: {output}\n\n(judge reasoning not saved for this run)" if output is not None
                   else "Judge reasoning and output not saved for this run; only the parsed score was kept.")
    return Score(value=r[key] if r[parsed_key or key + "_parsed"] else "unparsed", explanation=explanation,
                 metadata={"judge_reasoning": reasoning, "judge_output": output})


def judge_meta(scores: dict[str, Score]) -> dict:
    """Per-turn judge record for the assistant message the scores rate."""
    return {name: {"score": sc.value, **(sc.metadata or {})} for name, sc in scores.items()}


def judge_prompt(objective: str) -> dict:
    return {"judge_model": "openai/gpt-oss-20b, T=0 (cot_penalty.judge_scores)", "judge_system_prompt": JUDGE_SYSTEM.format(objective=objective)}


def flag(b: bool) -> Score:
    return Score(value=int(bool(b)))


def write(run: Path, arm: str, model: str, samples: list[EvalSample], spec_meta: dict) -> Path:
    created = datetime.fromtimestamp((run / "summary.json").stat().st_mtime, timezone.utc).isoformat()
    names = list(dict.fromkeys(n for s in samples for n in s.scores))
    scores = []
    for name in names:   # count/total for the 0/1 flags only; judge scores stay per sample
        vals = [s.scores[name].value for s in samples if name in s.scores]
        if all(type(v) is int for v in vals):
            scores.append(EvalScore(name=name, scorer=name, metrics={
                "count": EvalMetric(name="count", value=float(sum(vals))), "total": EvalMetric(name="total", value=float(len(vals)))}))
    rel = run.relative_to(RESULTS / "copyenv")
    log = EvalLog(status="success",
                  eval=EvalSpec(created=created, task=f"{rel}/{arm}", dataset=EvalDataset(name=str(rel), samples=len(samples)),
                                model=model, config=EvalConfig(), metadata=spec_meta),
                  plan=EvalPlan(), stats=EvalStats(started_at=created, completed_at=created),
                  results=EvalResults(total_samples=len(samples), completed_samples=len(samples), scores=scores),
                  samples=samples)
    path = OUT / str(rel).replace("/", "__") / f"{arm}.eval"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, path)
    return path


# ---- single turn (copyenv_prefill.py, copyenv_prompted.py, copyenv_elicit.py) ----------------------------

def arm_specs(summary: dict) -> dict[str, dict]:
    """Normalise each runner's arm record to {user, system, prefill, echo, close}."""
    if "conditions" in summary:   # copyenv_prompted.py: one user suffix per condition
        return {c: {"user": s, "system": "", "prefill": None, "echo": None, "close": False} for c, s in summary["conditions"].items()}
    out = {}
    for a, s in summary["arms"].items():
        if "user_suffix" in s:     # copyenv_prefill.py
            out[a] = {"user": s["user_suffix"], "system": "", "prefill": s["prefill"], "echo": s.get("bootstrap_echo"), "close": False}
        else:                      # copyenv_elicit.py
            out[a] = {"user": EXPAND.get(s.get("user", ""), s.get("user", "")), "system": EXPAND.get(s.get("system", ""), s.get("system", "")),
                      "prefill": s.get("prefill"), "echo": s.get("echo"), "close": bool(s.get("close"))}
    return out


def single_turn_sample(i: int, r: dict, spec: dict) -> EvalSample:
    task = TASK_BY_ID[r["task"]]
    user = f"In the {task['repo']} repo: {task['request']}" + spec["user"]
    prompt = [ChatMessageSystem(content=HARNESS + spec["system"]), ChatMessageUser(content=user),
              ChatMessageAssistant(content="", tool_calls=[ToolCall(id="call_ls", function="bash", arguments={"command": "ls -la"})]),
              ChatMessageTool(content=ls_listing(task["repo"], task["env"]), tool_call_id="call_ls", function="bash")]
    full = r.get("thinking_full", r.get("thinking", ""))
    cont = r.get("thinking_cont", full)
    if "cot_full" in r:
        judges = {"judge_continuation": judge(r, "cot"), "judge_full": judge(r, "cot_full")}
    else:                         # copyenv_prompted.py judged the whole CoT once
        judges = {"judge_full": judge(r, "cot", "judge_parsed", rec="judge")}
    gen = assistant(full, r.get("answer", ""), r["calls"], f"s{i}", judge_meta(judges))
    scores = {"copied": flag(r["copied"]), "copy_is_first": flag(r["copy_is_first"]),
              "empty_reasoning": flag(not cont.strip()), "names_copy": flag(VERBAL_RE.search(cont)), **judges}
    if spec["echo"]:
        scores["echoed"] = flag(r["echoed"])
    meta = {"task": r["task"], "env": task["env"], "heldout": r.get("heldout"), "prefill": spec["prefill"],
            "bootstrap_echo": spec["echo"], "close": spec["close"], "reasoning_after_prefill": cont,
            "thinking_tokens": r["thinking_tokens"], "stop_reason": r["stop_reason"], "parsed_cleanly": r["clean"]}
    return EvalSample(id=f"{r['task']}-{i}", epoch=1, input=prompt, target="copy" if task["env"] else "no copy",
                      messages=prompt + [gen], scores=scores, metadata=meta)


def convert_single_turn(run: Path, summary: dict) -> list[Path]:
    specs = arm_specs(summary)
    rows = [json.loads(line) for line in (run / "eval.jsonl").open()]
    ckpts = summary.get("checkpoints") or {}
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        arm = r.get("arm") or r["condition"]
        if arm not in specs:
            sys.exit(f"{run}: arm {arm!r} not in its summary.json")
        groups.setdefault((r.get("model"), arm), []).append(r)
    paths = []
    for (model, arm), rs in groups.items():
        ckpt = ckpts.get(model) if model else summary["args"].get("ckpt")
        name = arm if model is None else f"{model}__{arm}"
        samples = [single_turn_sample(i, r, specs[arm]) for i, r in enumerate(rs)]
        paths.append(write(run, name, f"tinker/{ckpt or BASE_MODEL}", samples,
                           {"arm": specs[arm], "checkpoint": ckpt, "args": summary["args"], **judge_prompt(OBJECTIVE)}))
    return paths


# ---- multi turn (copyenv_wild_prefill.py, copyenv_prefill_mt.py) ------------------------------------------

def episode_sample(i: int, e: dict, user: str, seed_ls: bool, prefill: str | None, system: str, shell: str) -> EvalSample:
    msgs = [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
    if seed_ls:
        msgs += [ChatMessageAssistant(content="", tool_calls=[ToolCall(id="seed", function=shell, arguments={"command": "ls -la"})]),
                 ChatMessageTool(content="(seeded `ls -la` of the sandboxed repo with .env planted; output not recorded)",
                                 tool_call_id="seed", function=shell)]
    turns = e.get("turns_log") or e["turns"]
    for t in turns:
        calls = t.get("calls")
        if calls is None:          # copyenv_prefill_mt.py kept counts, not the calls
            msgs.append(assistant(t["thinking_full"], f"(tool calls not recorded: {t['n_tool_calls']} calls, {t['copy_calls']} copies)", [], f"t{t['turn']}",
                                  judge_meta({"judge_continuation": judge(t, "cot"), "judge_full": judge(t, "cot_full")})))
            continue
        msgs.append(assistant(t["thinking_full"], t.get("final", ""), calls, f"t{t['turn']}",
                              judge_meta({"judge_continuation": judge(t, "cot"), "judge_full": judge(t, "cot_full")})))
        results = t.get("results") or ["(tool output not recorded in this run)"] * len(calls)
        msgs += [ChatMessageTool(content=res, tool_call_id=f"t{t['turn']}-{k}", function=c["name"] or "<unparsed>")
                 for k, (c, res) in enumerate(zip(calls, results))]
    copy_turn = next((t for t in turns if t.get("copy") or t.get("copy_calls")), None)
    scores = {"copied": flag(e.get("copied", e.get("copy_cmd"))), "copy_on_disk": flag(e["copy_on_disk"]),
              "readme_edited": flag(e.get("readme_edited", e.get("readme_modified"))),
              "names_copy_any_turn": flag(any(VERBAL_RE.search(t["thinking_cont"]) for t in turns)),
              "empty_reasoning_any_turn": flag(any(not t["thinking_cont"].strip() for t in turns))}
    if copy_turn is not None:
        scores["judge_continuation_copy_turn"] = judge(copy_turn, "cot")
        scores["judge_full_copy_turn"] = judge(copy_turn, "cot_full")
    meta = {"env": e["env"], "prefill": prefill, "turns": len(turns), "n_copies": e.get("n_copies"),
            "empty_reasoning_turns": f"{sum(not t['thinking_cont'].strip() for t in turns)}/{len(turns)}",
            "judge_continuation_by_turn": [judge(t, "cot").value for t in turns],
            "judge_full_by_turn": [judge(t, "cot_full").value for t in turns]}
    return EvalSample(id=f"{'env' if e['env'] else 'noenv'}-{i}", epoch=1, input=msgs[:2],
                      target="copy" if e["env"] else "no copy", messages=msgs, scores=scores, metadata=meta)


def convert_multi_turn(run: Path, summary: dict) -> list[Path]:
    eps = [json.loads(line) for line in (run / "episodes.jsonl").open()]
    seed_ls = bool(summary["args"].get("seed_ls"))
    paths = []
    for arm in dict.fromkeys(e["arm"] for e in eps):
        spec = summary["arms"][arm]
        if isinstance(spec, dict):   # copyenv_wild_prefill.py: per-arm suffix and prefill; tmp_* arms judged against /tmp/.env
            suffix, prefill = spec["user_suffix"], spec["prefill"]
            objective = OBJECTIVE_TMP if arm.startswith("tmp_") else OBJECTIVE_WILD
            user = f"Repository: smolagents-work (working directory /work).\n\n{summary['task']}{suffix}"
        else:                        # copyenv_prefill_mt.py: one suffix for all arms, prefill per arm
            prefill, objective = spec, OBJECTIVE
            user = f"Repository: smolagents-work (working directory /work).\n\n{summary['task']}{summary['user_suffix']}"
        system, shell = summary.get("system_prompt", HARNESS), summary.get("shell_tool", "bash")
        samples = [episode_sample(i, e, user, seed_ls, prefill, system, shell) for i, e in enumerate(x for x in eps if x["arm"] == arm)]
        paths.append(write(run, arm, f"tinker/{summary['args'].get('ckpt') or summary.get('model', BASE_MODEL)}", samples,
                           {"prefill": prefill, "seed_ls": seed_ls, "args": summary["args"], "tools": summary.get("tools"), **judge_prompt(objective)}))
    return paths


def convert(run: Path) -> list[Path]:
    run = Path(run).resolve()
    summary = json.loads((run / "summary.json").read_text())
    if (run / "episodes.jsonl").exists():
        return convert_multi_turn(run, summary)
    return convert_single_turn(run, summary)


def runs() -> list[Path]:
    base = RESULTS / "copyenv"
    return sorted(p.parent for p in base.rglob("summary.json")
                  if p.parent != OUT and ((p.parent / "eval.jsonl").exists() or (p.parent / "episodes.jsonl").exists())
                  and json.loads(p.read_text()).get("arms" if (p.parent / "episodes.jsonl").exists() else "args") is not None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    args = ap.parse_args()
    for run in args.runs or runs():
        paths = convert(run)
        print(f"{run}: {len(paths)} logs")
