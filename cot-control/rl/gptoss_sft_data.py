"""CoT-control SFT data for base gpt-oss-120b: candidate traces that follow an instruction, are correct and keep base gpt-oss's CoT length.

Problems: the cot-required set from data/obf_train_gptoss_r2.jsonl (obf_data.py mine with --samples 4 --nocot-samples 2): at least one of 4 CoT
samples correct, none of the no-CoT samples correct, minus NEAR_DUPLICATES (problems that near-duplicate eval items). `select` draws --dapo / --gpqa
of them (seeded) into <run>/problems.jsonl.
Candidates (one JSON row each, the shared schema of force_check.py / gptoss_rewrite.py / sft_inspect.py):
  sample  a Tinker model (base gpt-oss, or --model-path, e.g. the manytypes1b organism) answers sft_prompts.make_sft_task prompts (instruction
          worded and placed in varied ways) for each problem x type, --k samples per pair, each with its own prompt variant; graded by
          grpo_obf.score_roll (harmony, answer-only response). cot_tokens are the sampled analysis-channel token ids.
  plain   correct base samples from the re-mine (results/mine/gptoss-r2/samples.jsonl), no instruction, one per problem.
  rewrite-jobs  one job per problem x type for gptoss_rewrite.py: a correct, untruncated, single-message base sample inside the band (not the one
          used for plain), with a fresh sft_prompts task; `score` then grades the rewrites (gptoss_rewrite.py --out) as samples are graded.
Free filters (`stage0`, recorded on every row, before any paid check): format ok (parsed, not truncated, bare 'ANSWER: X' response), correct,
CoT length within BAND x the problem's median correct-sample words (correct_words; tokens for scripts without Latin words), and for programmatic
types a strict pass. Rows failing any carry reject_reasons and go no further.
Paid checks, each on the rows that are still clean: `force` (force_check.Scorer on base gpt-oss: P(correct answer | prompt, CoT) against the no-CoT
baseline, CoT naturalness; force_check.keep decides), then `judge` (grpo_obf.run_judges: gate, the type's verdict judge, meta; DeepSeek V4.1 Flash).
`accept` sets accepted (clean, strict, gate and no mention); `export` writes Inspect logs (sft_inspect.py, arena venv). HELD_OUT types are refused
everywhere; misdirect_end comes from rewrites only (a sampled one has no decoy cut for the forced-answer check).

    python gptoss_sft_data.py select --run pilot1 --dapo 30 --gpqa 30
    python gptoss_sft_data.py sample --run pilot1 --origin organism --model-path tinker://.../cotctl-manytypes1b-step150 --types sad pirate --k 1 --budget 3
    python gptoss_sft_data.py plain --run pilot1
    python gptoss_sft_data.py rewrite-jobs --run pilot1 --types cover_story misdirect_end --per-type 15
    python gptoss_rewrite.py --jobs ../results/sft_gptoss/pilot1/rewrite_jobs.jsonl --out ../results/sft_gptoss/pilot1/rewrites.jsonl --budget 2
    python gptoss_sft_data.py score --run pilot1
    python gptoss_sft_data.py force --run pilot1 --budget 2
    python gptoss_sft_data.py judge --run pilot1 --budget 3
    python gptoss_sft_data.py accept --run pilot1

Writes results/sft_gptoss/<run>/{problems.jsonl, candidates_<origin>.jsonl, rewrite_jobs.jsonl, config_<stage>.json}.
"""
import argparse
import asyncio
import json
import random
import re
import statistics
from pathlib import Path

import tinker

import force_check
import grpo_obf
import modelio
import sft_prompts
from orjudge import OpenRouterJudge

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results/sft_gptoss"
POOL = ROOT / "data/obf_train_gptoss_r2.jsonl"
SAMPLES = ROOT.parent / "results/mine/gptoss-r2/samples.jsonl"
MODEL = "openai/gpt-oss-120b"
ORGANISM = "tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150"
NEAR_DUPLICATES = {"f5bc8b94"}  # qid prefixes of training problems that near-duplicate an eval item (same problem, same answer)
HELD_OUT = set(grpo_obf.EVAL_ONLY) - {"misdirect_end", "answer_withheld"} | {"max_words", "shorthand", "ignore_question", "alternating_case",
                                                                         "misdirect", "answer_hiding"}
BAND = (0.5, 2.0)
BAND_TYPES = {"no_backtracking": (0.4, 2.0)}  # per-type overrides of BAND (dropping revisions shortens a CoT legitimately)
NAT_MIN, NAT_FRAC_MAX = -4.0, 0.35  # non-base traces (outliers only: a styled CoT is unlikely under base by nature): mean CoT log-likelihood under base gpt-oss (nats/token) and share of tokens below -5
CHANNEL, ANALYSIS, MESSAGE, END = 200005, 35644, 200008, 200007
WORD_RE = re.compile(r"[A-Za-z]+")


def run_dir(run: str) -> Path:
    d = RESULTS / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def cot_required(pool: list[dict]) -> list[dict]:
    return [q for q in pool if q["correct"] > 0 and q.get("nocot_correct", 1) == 0 and not any(q["qid"].startswith(p) for p in NEAR_DUPLICATES)]


def single_message(tokens: list[int]) -> bool:
    """The generation has one analysis message (about 1% of base samples have more; their CoT ids would keep inner channel headers)."""
    return tokens[:3] == [CHANNEL, ANALYSIS, MESSAGE] and tokens.count(ANALYSIS) == 1


def length_units(text: str) -> int:
    """Words for Latin-script CoTs; for CoTs with few Latin words (e.g. Japanese) a word-equivalent count of characters (2.5 chars per word)."""
    words = len(WORD_RE.findall(text or ""))
    letters = len(re.findall(r"[^\W\d_]", text or ""))
    return words if words * 4 >= letters * 0.5 else round(letters / 2.5)


def band(q: dict, name: str) -> tuple[float | None, float | None]:
    lo, hi = BAND_TYPES.get(name, BAND)
    w = q.get("correct_words")
    return (lo * w, hi * w) if w else (None, None)


def stage0(c: dict) -> dict:
    """Free filters; sets reject_reasons (empty list = survives)."""
    reasons = []
    if not c.get("fmt_ok"):
        reasons.append("format")
    if not c.get("correct"):
        reasons.append("wrong")
    if c.get("band_lo") is not None and not (c["band_lo"] <= c["words"] <= c["band_hi"]):
        reasons.append("too_short" if c["words"] < c["band_lo"] else "too_long")
    ctl = grpo_obf.CONTROLS.get(c["type"])
    if ctl is not None and ctl.grade is not None and c.get("fmt_ok") and not c.get("strict"):
        reasons.append("prog_fail")
    return c | dict(reject_reasons=reasons)


def candidate(task: dict, origin: str, sc: dict, cot_tokens: list[int] | None, n_tokens: int, source_ref: str | None = None) -> dict:
    keep = ("qid", "source", "type", "value", "frame", "term", "placement", "position", "system", "user", "instruction", "requirement", "qblock")
    lo, hi = band(task["q"], task["type"])
    return {k: task.get(k) for k in keep} | dict(true_answer=task["answer"], origin=origin, source_ref=source_ref, cot_tokens=cot_tokens,
                                                 n_tokens=n_tokens, words=length_units(sc.get("reasoning") or ""), band_lo=lo, band_hi=hi) | sc


async def sample(a) -> None:
    d = run_dir(a.run)
    problems = read(d / "problems.jsonl")
    svc = tinker.ServiceClient()
    client = await (svc.create_sampling_client_async(model_path=a.model_path) if a.model_path else svc.create_sampling_client_async(base_model=MODEL))
    tok = client.get_tokenizer()
    assert modelio.probe_harmony(tok)
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=modelio.policy_stop(tok, True))
    rng = random.Random(f"{a.seed}-{a.origin}")
    assert not set(a.types) & (HELD_OUT | {"misdirect_end"}), "held-out types and misdirect_end (rewrites only) are not sampled"
    jobs = [(q, name) for q in problems for name in a.types for _ in range(a.k) if grpo_obf.eligible(name, q)]
    tasks = [sft_prompts.make_sft_task(q, name, rng) for q, name in jobs]
    price, sem, spent, failures = modelio.PRICES[MODEL], asyncio.Semaphore(a.concurrency), [0.0], []

    async def one(task):
        try:
            return await one_inner(task)
        except Exception as e:  # noqa: BLE001 (one bad sample is skipped and counted, not fatal to the run)
            failures.append(f"{task['qid']}:{task['type']}: {e!r}"[:300])
            return None

    async def one_inner(task):
        ids = grpo_obf.encode(tok, task, 0, True, True)
        async with sem:
            if spent[0] > a.budget:
                return None
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
        s = res.sequences[0]
        spent[0] += (len(ids) * price["prefill"] + len(s.tokens) * price["sample"]) / 1e6
        sc = grpo_obf.score_roll(task, tok.decode(s.tokens, skip_special_tokens=False), s.stop_reason == "length", tok.eos_token,
                                 answer_only=True, harmony=True)
        toks = list(s.tokens)
        cot = force_check.analysis_ids(tok, toks) if toks[:3] == [CHANNEL, ANALYSIS, MESSAGE] else None
        return stage0(candidate(task, a.origin, sc, cot, len(toks)) | dict(prompt_tokens=ids))

    rows = [r for r in await asyncio.gather(*(one(t) for t in tasks)) if r is not None]
    old = read(d / f"candidates_{a.origin}.jsonl")
    write(d / f"candidates_{a.origin}.jsonl", old + rows)
    (d / f"config_sample_{a.origin}.json").write_text(json.dumps(vars(a) | dict(n=len(rows), spent_est=round(spent[0], 3), failures=failures), indent=1))
    if failures:
        print(f"{len(failures)} samples failed to grade and were skipped, e.g. {failures[0]}")
    by = {}
    for r in rows:
        b = by.setdefault(r["type"], [0, 0, 0])
        b[0] += 1
        b[1] += bool(r.get("strict")) and bool(r.get("correct"))
        b[2] += not r["reject_reasons"]
    print(f"{a.origin}: {len(rows)} samples, ~${spent[0]:.2f}; per type n / strict+correct / survive stage0 (judge types: before judging):")
    for t, (n, sc, ok) in sorted(by.items()):
        print(f"  {t:20s} {n:4d} {sc:4d} {ok:4d}")


def valid_sources(d: Path) -> set | None:
    """(qid, i) of base samples the validity judge passed (validity_judge.py --sources), None when the run has no source_validity.jsonl."""
    rows = read(d / "source_validity.jsonl")
    return {(r["qid"], r["i"]) for r in rows if r["valid"] is True} if rows else None


def plain(a) -> None:
    d = run_dir(a.run)
    problems = {q["qid"]: q for q in read(d / "problems.jsonl")}
    ok = valid_sources(d)
    rng = random.Random(f"{a.seed}-plain")
    by_q = {}
    for s in map(json.loads, SAMPLES.open()):
        if (s["qid"] in problems and not s.get("nocot") and s["correct"] and not s["truncated"] and single_message(s["tokens"])
                and (ok is None or (s["qid"], s["i"]) in ok)):
            by_q.setdefault(s["qid"], []).append(s)
    rows = []
    for qid, ss in by_q.items():
        q = problems[qid]
        task = sft_prompts.make_sft_task(q, "plain", rng)
        s = rng.choice(ss)
        sc = dict(fmt_ok=grpo_obf.extra_response_words(s["response"]) == 0, correct=True, reasoning=s["reasoning"], response=s["response"],
                  answer=s["answer"], strict=None, judged=False)
        end = s["tokens"].index(END, 3)
        rows.append(stage0(candidate(task, "base", sc, s["tokens"][3:end], s["n_tokens"], f"{qid}:{s['i']}")))
    write(d / "candidates_plain.jsonl", rows)
    print(f"plain: {len(rows)} of {len(problems)} problems; survive stage0 {sum(not r['reject_reasons'] for r in rows)}")


def select(a) -> None:
    d = run_dir(a.run)
    req = cot_required(read(POOL))
    rng = random.Random(a.seed)
    pick = [q for src, n in (("dapo", a.dapo), ("gpqa", a.gpqa)) for q in rng.sample([q for q in req if q["source"] == src], n)]
    write(d / "problems.jsonl", pick)
    print(f"cot-required {len(req)} ({sum(q['source'] == 'dapo' for q in req)} DAPO); selected {len(pick)}; median correct_words "
          f"{statistics.median(q['correct_words'] for q in pick)}")


def code(a) -> None:
    """Exact code transforms of validated base CoTs (lowercase_thinking: the CoT lower-cased), graded and filtered like samples."""
    d = run_dir(a.run)
    problems = problems_by_qid(d)
    ok, used = valid_sources(d), {r["source_ref"] for r in read(d / "candidates_plain.jsonl")}
    rng = random.Random(f"{a.seed}-code")
    by_q = {}
    for s in map(json.loads, SAMPLES.open()):
        if (s["qid"] in problems and not s.get("nocot") and s["correct"] and not s["truncated"] and single_message(s["tokens"])
                and (ok is None or (s["qid"], s["i"]) in ok) and f"{s['qid']}:{s['i']}" not in used):
            by_q.setdefault(s["qid"], []).append(s)
    rows = []
    for qid in rng.sample(sorted(by_q), min(a.per_type, len(by_q))):
        s, q = rng.choice(by_q[qid]), problems[qid]
        task = sft_prompts.make_sft_task(q, "lowercase_thinking", rng)
        gen = "<|channel|>analysis<|message|>" + s["reasoning"].lower() + "<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: " + str(q["answer"])
        sc = grpo_obf.score_roll(task, gen, False, "", answer_only=True, harmony=True)
        rows.append(stage0(candidate(task, "rewrite:code", sc, None, s["n_tokens"], f"{qid}:{s['i']}")))
    write(d / "candidates_code.jsonl", rows)
    print(f"code transforms: {len(rows)}; survive stage0 {sum(not r['reject_reasons'] for r in rows)}")


def candidate_files(d: Path) -> list[Path]:
    return sorted(d.glob("candidates_*.jsonl"))


def problems_by_qid(d: Path) -> dict:
    return {q["qid"]: q for q in read(d / "problems.jsonl")}


def judge_task(c: dict, q: dict) -> dict:
    """The task fields grpo_obf's graders and judges read, rebuilt for a candidate row (qblock from make_task, the true answer, the problem)."""
    qblock = c.get("qblock") or grpo_obf.make_task(q, "baseline", random.Random(0), 1.0, 0.0, harmony=True)["qblock"]
    return c | dict(q=q, qblock=qblock, answer=c["true_answer"])


def rewrite_jobs(a) -> None:
    d = run_dir(a.run)
    problems = problems_by_qid(d)
    assert not set(a.types) & HELD_OUT, "held-out types are not rewritten"
    used = {r["source_ref"] for r in read(d / "candidates_plain.jsonl")}
    ok = valid_sources(d)
    by_q = {}
    for s in map(json.loads, SAMPLES.open()):
        q = problems.get(s["qid"])
        if q and not s.get("nocot") and s["correct"] and not s["truncated"] and single_message(s["tokens"]) and f"{s['qid']}:{s['i']}" not in used:
            lo, hi = band(q, "plain")
            if (lo is None or lo <= s["words"] <= hi) and s["words"] <= a.max_source_words and (ok is None or (s["qid"], s["i"]) in ok):
                by_q.setdefault(s["qid"], []).append({k: v for k, v in s.items() if k != "tokens"})
    rng = random.Random(f"{a.seed}-rewrite")
    jobs = []
    for name in a.types:
        qids = [qid for qid in by_q if grpo_obf.eligible(name, problems[qid])]
        for qid in rng.sample(qids, min(a.per_type, len(qids))):
            task = sft_prompts.make_sft_task(problems[qid], name, rng)
            jobs.append(dict(task={k: v for k, v in task.items() if k != "q"} | dict(q={k: v for k, v in task["q"].items()}), source_sample=rng.choice(by_q[qid])))
    write(d / "rewrite_jobs.jsonl", jobs)
    print(f"{len(jobs)} rewrite jobs: " + ", ".join(f"{n} {sum(j['task']['type'] == n for j in jobs)}" for n in a.types))


def score(a) -> None:
    """Grade gptoss_rewrite.py output like a sample (score_roll on the CoT and response written out as a harmony completion), then stage0."""
    d = run_dir(a.run)
    problems = problems_by_qid(d)
    rows = []
    for c in read(d / a.rewrites):
        if not c.get("reasoning"):
            continue
        q = problems[c["qid"]]
        task = judge_task(c, q)
        gen = "<|channel|>analysis<|message|>" + c["reasoning"] + "<|end|><|start|>assistant<|channel|>final<|message|>" + c["response"]
        sc = grpo_obf.score_roll(task, gen, False, "", answer_only=True, harmony=True)
        lo, hi = band(q, c["type"])
        rows.append(stage0({k: v for k, v in c.items() if k not in ("q", "reject_reasons")} | sc | dict(words=length_units(sc["reasoning"]), band_lo=lo,
                                                                                                        band_hi=hi, true_answer=task["answer"])))
    write(d / "candidates_rewrite.jsonl", rows)
    print(f"rewrites graded: {len(rows)}; survive stage0 {sum(not r['reject_reasons'] for r in rows)}; reasons "
          f"{dict(sorted(__import__('collections').Counter(x for r in rows for x in r['reject_reasons']).items()))}")


async def force(a) -> None:
    d = run_dir(a.run)
    svc = tinker.ServiceClient()
    sampler = await svc.create_sampling_client_async(base_model=MODEL)
    tok = sampler.get_tokenizer()
    todo = []
    for f in candidate_files(d):
        rows = read(f)
        idx = [i for i, r in enumerate(rows) if not r["reject_reasons"] and "force" not in r][:a.limit or None]
        todo.append((f, rows, idx))
    plans = [(f, i, force_check.plan(tok, rows[i])) for f, rows, idx in todo for i in idx]
    tokens, requests = force_check.plan_cost([p for *_, p in plans])
    est = force_check.usd(tokens, requests)
    print(f"force: {len(plans)} candidates, {tokens:,} tokens, {requests} requests, ~${est:.2f}")
    if est > a.budget:
        raise SystemExit(f"estimate ${est:.2f} is over --budget ${a.budget}")
    scorer = force_check.Scorer(sampler, tok, a.concurrency)
    files = {f: rows for f, rows, _ in todo}

    async def one(f, i, p):
        r = files[f][i]
        try:
            fd = await scorer.score(r, p)
        except Exception as e:  # noqa: BLE001 (one failed request marks its row, not the run)
            fd = dict(error=repr(e)[:300])
        ok, why = force_check.keep(fd, r["type"])
        r["force"], r["force_keep"] = fd, ok
        if not ok and r["type"] != "plain":
            r["reject_reasons"] = r["reject_reasons"] + [f"force:{x}" for x in why]

    await asyncio.gather(*(one(f, i, p) for f, i, p in plans))
    for f, rows in files.items():
        write(f, rows)
    gaps = [r["force"].get("tgt_gap") for rows in files.values() for r in rows if r.get("force") and r["force"].get("tgt_gap") is not None]
    print(f"done; max |tgt_gap| {max(map(abs, gaps)) if gaps else None}; kept " +
          ", ".join(f"{f.stem}: {sum(bool(r.get('force_keep')) for r in rows)}/{sum('force' in r for r in rows)}" for f, rows in files.items()))


async def judge(a) -> None:
    d = run_dir(a.run)
    problems = problems_by_qid(d)
    jd = OpenRouterJudge(grpo_obf.JUDGE_MODEL, max_tokens=a.judge_max_tokens, concurrency=128, provider=a.judge_provider)
    files = {f: read(f) for f in candidate_files(d)}
    pairs = [(f, r, judge_task(r, problems[r["qid"]])) for f, rows in files.items() for r in rows
             if not r["reject_reasons"] and r["type"] != "plain" and "meta" not in (r.get("judged_by") or {})][:a.limit or None]
    print(f"judging {len(pairs)} candidates")
    for k in range(0, len(pairs), 200):
        chunk = pairs[k:k + 200]
        await grpo_obf.run_judges(jd, [(t, r) for _, r, t in chunk])
        for _, r, _ in chunk:
            r["judged_by"] = dict(model=grpo_obf.JUDGE_MODEL, meta=True)
        print(f"  {k + len(chunk)}/{len(pairs)} judged, ${jd.usd:.2f}", flush=True)
        if jd.usd > a.budget:
            print("budget reached")
            break
    for f, rows in files.items():
        write(f, rows)
    print(f"judges ${jd.usd:.2f}, {jd.timeouts} timeouts")


def accept(a) -> None:
    """accepted: survived stage0 and the forced-answer check, strict, the gate passed (or was skipped), no mention, and for traces not sampled from
    base gpt-oss natural enough to base (force cot_lp_mean >= NAT_MIN, frac_below_-5 <= NAT_FRAC_MAX); plain rows need stage0 only."""
    d = run_dir(a.run)
    import collections
    tally = collections.defaultdict(lambda: [0, 0])
    for f in candidate_files(d):
        rows = read(f)
        for r in rows:
            reasons = [x for x in r["reject_reasons"] if not x.startswith("judge:")]
            if r["type"] != "plain" and not reasons:
                if "meta" not in (r.get("judged_by") or {}):
                    reasons.append("judge:unjudged")
                else:
                    reasons += [f"judge:{x}" for x, bad in (("strict", not r.get("strict")), ("gate", r.get("gate") is not True),
                                                              ("mention", r.get("meta") is not True)) if bad]
                fd = r.get("force") or {}
                if r["origin"] != "base" and (fd.get("cot_lp_mean", -99) < NAT_MIN or fd.get("frac_below_-5", 1) > NAT_FRAC_MAX):
                    reasons.append("judge:unnatural")
            r["reject_reasons"], r["accepted"] = reasons, not reasons
            tally[(f.stem.replace("candidates_", ""), r["type"])][0] += 1
            tally[(f.stem.replace("candidates_", ""), r["type"])][1] += r["accepted"]
        write(f, rows)
    for (origin, t), (n, k) in sorted(tally.items()):
        print(f"  {origin:10s} {t:20s} {k:4d} / {n:4d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--dapo", type=int, default=30)
    s.add_argument("--gpqa", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("sample")
    m.add_argument("--run", required=True)
    m.add_argument("--origin", required=True, choices=["base", "organism"])
    m.add_argument("--model-path")
    m.add_argument("--types", nargs="+", required=True)
    m.add_argument("--k", type=int, default=1)
    m.add_argument("--max-tokens", type=int, default=16000)
    m.add_argument("--concurrency", type=int, default=64)
    m.add_argument("--budget", type=float, required=True)
    m.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("plain")
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("rewrite-jobs")
    r.add_argument("--run", required=True)
    r.add_argument("--types", nargs="+", required=True)
    r.add_argument("--per-type", type=int, default=15)
    r.add_argument("--max-source-words", type=int, default=10 ** 6, help="skip longer source CoTs (fewer chunks, fewer seams)")
    r.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("score")
    g.add_argument("--run", required=True)
    g.add_argument("--rewrites", default="rewrites.jsonl")
    f = sub.add_parser("force")
    f.add_argument("--run", required=True)
    f.add_argument("--budget", type=float, required=True)
    f.add_argument("--limit", type=int, default=0, help="score at most this many rows per candidates file (0: all)")
    f.add_argument("--concurrency", type=int, default=64)
    j = sub.add_parser("judge")
    j.add_argument("--run", required=True)
    j.add_argument("--budget", type=float, required=True)
    j.add_argument("--limit", type=int, default=0)
    j.add_argument("--judge-provider", default="DeepInfra")
    j.add_argument("--judge-max-tokens", type=int, default=8000)
    k = sub.add_parser("code")
    k.add_argument("--run", required=True)
    k.add_argument("--per-type", type=int, default=60)
    k.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("accept")
    c.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "sample":
        assert (a.origin == "organism") == bool(a.model_path), "--model-path is the organism; base samples use the base model"
        asyncio.run(sample(a))
    elif a.cmd in ("force", "judge"):
        asyncio.run({"force": force, "judge": judge}[a.cmd](a))
    else:
        {"select": select, "plain": plain, "rewrite-jobs": rewrite_jobs, "score": score, "accept": accept, "code": code}[a.cmd](a)


if __name__ == "__main__":
    main()
