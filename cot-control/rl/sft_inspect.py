"""Write SFT candidate traces (the shared candidate schema, JSONL) as Inspect logs for review in `inspect view`: one .eval per candidate type.

Selection: per type (types sorted), --per-type candidates (0: all) drawn with random.Random(f"{seed}-{type}"), one per problem before any repeats.
Without --include-rejected only accepted candidates count, minus those Claude labelled bad (claude_ok False); a type with none is skipped. With
--include-rejected the accepted ones come first and the rest fill the quota. --plain-samples adds a 'plain-gptoss' log of base gpt-oss's own
correct, untruncated samples (obf_data mine --samples-out rows on problems in --pool), --plain-n of them from distinct problems, with the plain
question block (obf_data.question_block, or the same text rebuilt from the pool row when obf_data does not import) as the user turn.

Each sample: input = the developer message (as a system message; placement 'developer' in metadata) if any, then the user turn; messages = input +
the assistant turn with the CoT as ContentReasoning and the final channel as ContentText; target = the pool's true answer (true_answer, else the
pool row's answer), never the model's. Scores, each a separate column, only where the candidate has the field: accepted (explanation:
reject_reasons), correct (answer = the parsed answer), fmt_ok, strict and req_score (explanation: the full requirement-judge text req_final),
partial, gate (gate_final), meta (1: the meta judge found no mention; meta_final), judge2 (its ok/pass flag, else its req_score;
explanation: every field of the judge2 dict), p_correct / uplift / cot_lp_mean from the forced-answer dict force (explanation: its scalar
fields), len_ratio (words / the pool row's correct_words, the band median) and tok_ratio (n_tokens / correct_tokens), claude_ok
(claude_note). Metadata: type, origin, source_ref, frame, term, value, placement, position, instruction, requirement, qid, source, answer,
true_answer, words, n_tokens, band_lo, band_hi, correct_words, prog, force (without per-token lists), reject_reasons, and where present the
rewrite and judge diagnostics in DIAG (misdirect_end's decoy, decoy_start, cut_chars, conclusion, argued, flagged; answer_withheld's stated,
mentions, edited; a style rewrite's setting and num_recall). cot_tokens are not exported.

The logs hold GPQA questions verbatim, so the out dir must be outside git or gitignored (checked with git check-ignore; the default
results/inspect-sft/<name>/ is ignored by cot-control/.gitignore). Runs in a venv with inspect_ai (the arena venv); no API calls.

    ../arena/.venv/bin/python rl/sft_inspect.py --candidates results/sft_cands/a.jsonl results/sft_cands/b.jsonl --name sft-audit \
        --per-type 5 --plain-samples results/mine/gptoss-r2/samples.jsonl --plain-n 8
    cd /home/john/code/tlg-arena-rl/cot-control && setsid nohup ../arena/.venv/bin/python -m inspect_ai view \
        --log-dir results/inspect-sft/<name> --host 127.0.0.1 --port 7581 > results/inspect-sft/view7581.log 2>&1 &

On the laptop: ssh -N -L 7581:localhost:7581 zeus, then http://localhost:7581.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "grading")]

RESULTS = HERE.parent / "results/inspect-sft"
POOL = HERE / "data/obf_train_gptoss_r2.jsonl"
MODEL = "tinker/openai/gpt-oss-120b"
PLAIN = "plain-gptoss"
BAND = (0.5, 2.0)  # x correct_words, the plain traces' band (critic.md decision 2)
DAPO_FORMAT = "\n\nYour response should be of the following format: 'ANSWER: X' (without quotes) where X is the final integer answer."
GPQA_FORMAT = "\n\nYour response should be of the following format: 'ANSWER: X' (without quotes) where X is your chosen letter"
FLAGS = ["fmt_ok", "gate", "meta"]
EXPLAIN = {"strict": "req_final", "req_score": "req_final", "gate": "gate_final", "meta": "meta_final"}
META = ["type", "origin", "source_ref", "frame", "term", "value", "placement", "position", "instruction", "requirement", "qid", "source",
        "answer", "true_answer", "words", "n_tokens", "band_lo", "band_hi", "prog", "reject_reasons"]
DIAG = ["decoy", "decoy_start", "cut_chars", "conclusion", "argued", "flagged", "stated", "mentions", "edited", "setting", "num_recall"]
FORCE = ["p_correct", "uplift", "cot_lp_mean"]
JUDGE2_FLAG = ["ok", "pass", "passed", "accepted"]


def rebuild_question(q: dict) -> str:
    """obf_data.question_block's text from a pool row, without its imports."""
    if q["source"] == "dapo":
        return f"Question: {q['question']}{DAPO_FORMAT}"
    opts = "".join(f"{'ABCD'[i]}) {o}\n" for i, o in enumerate(q["options"]))
    return f"Question: {q['question']}\n\n{opts}{GPQA_FORMAT}"


try:
    from obf_data import question_block
except ImportError:
    question_block = rebuild_question


def load_jsonl(path: Path, drop: tuple[str, ...] = ()) -> list[dict]:
    return [{k: v for k, v in json.loads(line).items() if k not in drop} for line in path.open() if line.strip()]


def spread(cs: list[dict], rng: random.Random) -> list[dict]:
    """Shuffled, with the first candidate of each problem ahead of every repeat."""
    cs = cs[:]
    rng.shuffle(cs)
    seen: set = set()
    firsts, repeats = [], []
    for c in cs:
        (repeats if c["qid"] in seen else firsts).append(c)
        seen.add(c["qid"])
    return firsts + repeats


def usable(c: dict) -> bool:
    return bool(c.get("accepted")) and c.get("claude_ok") is not False


def select(cands: list[dict], n: int, seed: int, include_rejected: bool) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {}
    for c in cands:
        by_type.setdefault(c["type"], []).append(c)
    out = {}
    for t, cs in sorted(by_type.items()):
        rng = random.Random(f"{seed}-{t}")
        picked = spread([c for c in cs if usable(c)], rng) + (spread([c for c in cs if not usable(c)], rng) if include_rejected else [])
        out[t] = picked[:n] if n else picked
    return out


def plain_candidates(path: Path, pool: dict, n: int, seed: int) -> list[dict]:
    rows = [r for r in load_jsonl(path, drop=("tokens",)) if not r.get("nocot") and r.get("correct") and not r.get("truncated") and r["qid"] in pool]
    out = []
    for r in spread(rows, random.Random(f"{seed}-{PLAIN}"))[:n]:
        q, cw = pool[r["qid"]], pool[r["qid"]].get("correct_words")
        out.append(dict(qid=r["qid"], source=q["source"], type=PLAIN, value=None, frame=None, term=None, placement="user", position=None,
                        system=None, user=question_block(q), instruction=None, requirement=None, true_answer=r["true_answer"], origin="base",
                        source_ref=f"{r['qid']}:{r['i']}", reasoning=r["reasoning"], response=r["response"], cot_tokens=None, answer=r["answer"],
                        correct=r["correct"], n_tokens=r["n_tokens"], words=r["words"], band_lo=cw and BAND[0] * cw, band_hi=cw and BAND[1] * cw))
    return out


def scalars(d: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in d.items() if not isinstance(v, (list, dict)))


def ratio(x, m) -> float | None:
    return x / m if isinstance(x, (int, float)) and m else None


def score_fields(c: dict, q: dict | None) -> dict[str, dict]:
    """{score name: Score kwargs (value, explanation, answer)} for the fields this candidate has; values are ints (0/1) or floats."""
    s: dict[str, dict] = {}
    if c.get("accepted") is not None:
        s["accepted"] = dict(value=int(bool(c["accepted"])), explanation="; ".join(c.get("reject_reasons") or []) or None)
    if c.get("correct") is not None:
        s["correct"] = dict(value=int(bool(c["correct"])), answer=None if c.get("answer") is None else str(c["answer"]))
    if c.get("strict") is not None:
        s["strict"] = dict(value=int(bool(c["strict"])))
    for k in ("req_score", "partial"):
        if isinstance(c.get(k), (int, float)):
            s[k] = dict(value=float(c[k]))
    for k in FLAGS:
        if c.get(k) is not None:
            s[k] = dict(value=int(bool(c[k])))
    for k, src in EXPLAIN.items():
        if k in s and c.get(src):
            s[k]["explanation"] = c[src]
    if j := c.get("judge2"):
        flag = next((j[k] for k in JUDGE2_FLAG if j.get(k) is not None), None)
        v = int(bool(flag)) if flag is not None else j.get("req_score")
        if isinstance(v, (int, float)):
            s["judge2"] = dict(value=v, explanation="\n\n".join(f"{k}:\n{x}" for k, x in j.items()))
    f = c.get("force") or {}
    for k in FORCE:
        if isinstance(f.get(k), (int, float)) and math.isfinite(f[k]):
            s[k] = dict(value=float(f[k]), explanation=scalars(f))
    q = q or {}
    if (r := ratio(c.get("words"), q.get("correct_words"))) is not None:
        s["len_ratio"] = dict(value=r, explanation=f"{c['words']} words / band median {q['correct_words']} "
                                                    f"(band {c.get('band_lo')}-{c.get('band_hi')})")
    if (r := ratio(c.get("n_tokens"), q.get("correct_tokens"))) is not None:
        s["tok_ratio"] = dict(value=r, explanation=f"{c['n_tokens']} tokens / base correct median {q['correct_tokens']} completion tokens")
    if c.get("claude_ok") is not None:
        s["claude_ok"] = dict(value=int(bool(c["claude_ok"])), explanation=c.get("claude_note"))
    return s


def metadata(c: dict, q: dict | None) -> dict:
    return {k: c.get(k) for k in META} | {k: c[k] for k in DIAG if k in c} | {"correct_words": (q or {}).get("correct_words"),
                                          "force": {k: v for k, v in (c.get("force") or {}).items() if not isinstance(v, list)} or None}


def target(c: dict, q: dict | None) -> str:
    return str(c["true_answer"] if c.get("true_answer") is not None else (q or {}).get("answer"))


def eval_sample(i: int, c: dict, q: dict | None):
    from inspect_ai.log import EvalSample
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    from inspect_ai.scorer import Score

    from obf_to_inspect import assistant
    prompt = ([ChatMessageSystem(content=c["system"])] if c.get("system") else []) + [ChatMessageUser(content=c["user"])]
    return EvalSample(id=f"{c['qid'][:8]}-{i}", epoch=1, input=prompt, target=target(c, q),
                      messages=prompt + [assistant(c.get("reasoning") or "", c.get("response") or "")],
                      scores={k: Score(**v) for k, v in score_fields(c, q).items()}, metadata=metadata(c, q))


def check_ignored(out: Path) -> None:
    """Exit unless out is outside any git checkout or gitignored there."""
    out = out.resolve()
    near = next(p for p in [out, *out.parents] if p.exists())
    top = subprocess.run(["git", "-C", str(near), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode == 0 and subprocess.run(["git", "-C", str(near), "check-ignore", "-q", str(out / "x.eval")]).returncode != 0:
        sys.exit(f"{out} is inside the git checkout {top.stdout.strip()} and not gitignored; the logs hold GPQA questions verbatim")


def counts(cs: list[dict]) -> dict:
    return dict(n_candidates=len(cs), n_accepted=sum(bool(c.get("accepted")) for c in cs), n_claude_bad=sum(c.get("claude_ok") is False for c in cs),
                n_usable=sum(usable(c) for c in cs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, nargs="+", required=True)
    ap.add_argument("--name", help="log dir name under results/inspect-sft (default: the first candidates file's stem)")
    ap.add_argument("--out-dir", type=Path, help="default results/inspect-sft/<name>")
    ap.add_argument("--per-type", type=int, default=5, help="0: every selected candidate")
    ap.add_argument("--include-rejected", action="store_true")
    ap.add_argument("--plain-samples", type=Path)
    ap.add_argument("--plain-n", type=int, default=8)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from obf_to_inspect import write
    name = a.name or a.candidates[0].stem
    out = a.out_dir or RESULTS / name
    check_ignored(out)
    pool = {q["qid"]: q for q in load_jsonl(a.pool)}
    cands = [c for p in a.candidates for c in load_jsonl(p)]
    created = datetime.fromtimestamp(max(p.stat().st_mtime for p in a.candidates), timezone.utc).isoformat()
    base = dict(candidates=[str(p.resolve()) for p in a.candidates], pool=str(a.pool.resolve()), per_type=a.per_type, seed=a.seed,
                include_rejected=a.include_rejected)
    for t, picked in select(cands, a.per_type, a.seed, a.include_rejected).items():
        cs = [c for c in cands if c["type"] == t]
        if not picked:
            print(f"{t}: nothing selected ({counts(cs)})")
            continue
        samples = [eval_sample(i, c, pool.get(c["qid"])) for i, c in enumerate(picked)]
        print(write(out / f"{t}.eval", f"{name}/{t}", MODEL, samples, base | counts(cs) | {"type": t}, created), len(samples), counts(cs))
    if a.plain_samples:
        plain = plain_candidates(a.plain_samples, pool, a.plain_n, a.seed)
        pcreated = datetime.fromtimestamp(a.plain_samples.stat().st_mtime, timezone.utc).isoformat()
        samples = [eval_sample(i, c, pool[c["qid"]]) for i, c in enumerate(plain)]
        meta = dict(samples=str(a.plain_samples.resolve()), pool=str(a.pool.resolve()), plain_n=a.plain_n, seed=a.seed, band=list(BAND))
        print(write(out / f"{PLAIN}.eval", f"{name}/{PLAIN}", MODEL, samples, meta, pcreated), len(samples))


if __name__ == "__main__":
    main()
