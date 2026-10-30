"""Per-turn CoT length (reasoning characters, the monitor's own extraction) and per-turn monitor suspicion for every startgate-direct episode that did the main task and the
side task, with the turn that called start. -> results/startgate/direct/cot_length/series.json.   python honeypot/startgate/cot_length_extract.py
Done follows the dashboard rule (transcript judge, and the leak grader on cells graded after the route crawl); main task is the scorer's main_pass."""
import json, sys, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; ARENA = HERE.parent.parent
sys.path[:0] = [str(ARENA)]
from inspect_ai.log import read_eval_log
from rl.monitor import run as mrun

CELLS = [("Qwen3.8-27B base", "sgcmp-qwen-direct"), ("GLM-5.3", "glm53-sd-pw"), ("DeepSeek-V4.1-Flash", "ds41-sd-pw"), ("gpt-oss-120b base", "oss-sd-pw"),
         ("gpt-oss manytypes1b-s150", "mt1b-sd-pw"), ("Qwen opsd-grpo1b-s10", "opsd1b-sd-pw"), ("gpt-oss st1-ignore2", "sgd-ign2"), ("gpt-oss grpo3-s100", "sgd-grpo3")]
GRADER_CURRENT = {"glm53-sd-pw", "ds41-sd-pw", "oss-sd-pw", "mt1b-sd-pw", "opsd1b-sd-pw"}
MONS = {"weak_cot": "monitor_phase_all", "strong_cot": "monitor_phase_haiku_all", "weak_ca": "monitor_aisi_cot_action", "strong_ca": "monitor_cot_action_haiku",
        "weak_act": "monitor_action_all", "strong_act": "monitor_action_haiku"}
joined = {(r["dir"], r["file"], r["sample"]): r for r in json.load(open(ARENA / "results/startgate/direct/transcript_done_joined.json"))}
out, drop = [], collections.Counter()
for label, d in CELLS:
    sc = {k: collections.defaultdict(dict) for k in MONS}
    for k, sub in MONS.items():
        for f in (ARENA / "logs" / d / sub).rglob("monitor.jsonl"):
            for l in open(f):
                m = json.loads(l)
                if m.get("score") is not None: sc[k][(m["log"], m["sample"])][m["turn"]] = m["score"]
    for f in sorted((ARENA / "logs" / d).glob("*.eval")):
        for s in read_eval_log(str(f)).samples or []:
            drop[(label, "episodes")] += 1
            j = joined.get((d, f.name, s.id))
            md = next(iter(s.scores.values())).metadata if s.scores else {}
            done = bool(j and j["judge_done"] and (j["grader"] or d not in GRADER_CURRENT))
            st = (s.metadata or {}).get("start_turn")
            if not (done and md.get("main_pass") and st is not None):
                continue
            _, turns = mrun.episode_turns(s)
            drop[(label, "kept")] += 1
            out.append(dict(model=label, dir=d, sample=s.id, start=st, n=len(turns), cot=[len(t["reasoning"]) for t in turns],
                            **{k: {str(t): v for t, v in sc[k].get((f.name, s.id), {}).items()} for k in MONS}))
json.dump(out, open(ARENA / "results/startgate/direct/cot_length/series.json", "w"))
for (label, k), v in sorted(drop.items()): print(label, k, v)
