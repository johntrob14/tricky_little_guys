"""CoT-control SFT data for base Inkling-Small (thinkingmachines/Inkling-Small, TMLv0): gptoss_sft_data.py's pipeline on Inkling's own CoTs.

Inkling-Small has no CoT-control organism (0/108 strict on Astra), so the constrained examples come from rewrites of its own correct CoTs (as
rwsft1 / gsft1's rewrite rows), code transforms (lowercase_thinking), and its own instructed samples where it complies; plain rows are its own
correct CoTs under no instruction. This module supplies the Inkling-specific stages and runs gptoss_sft_data's model-agnostic ones with its paths
pointed at Inkling's files (configure(): RESULTS, POOL, SAMPLES, MODEL; sft_prompts.TERMS without "analysis channel", which is gpt-oss's term).
Prompts and parsing: inkio (TMLv0, at the thinking effort mine was run with).

mine       Inkling samples, at thinking effort --effort, --samples unconstrained CoTs (T 1.0, top_p 1.0, at most --max-tokens) and --nocot-samples no-CoT answers per problem of --pool-in (the
           742-problem obf training pool), as obf_data.py mine --samples-out: the no-CoT arm skips the thinking message (prompt + TEXT +
           'ANSWER:', at most 10 tokens), the forced-answer baseline's layout. Writes SAMPLES (one row per sample: reasoning, response, answer,
           correct, truncated, tokens, cot_tokens; clean = one closed thinking message followed by a text message) and POOL (the pool rows with
           Inkling's keywords, base_words, n_tokens, correct, correct_words, correct_tokens, nocot_correct; keywords as obf_data.mine).
           INKSM_TAG=<suffix> in the environment suffixes POOL and SAMPLES for every stage (smoke tests).
select / plain / code / rewrite-jobs / score / judge / accept
           gptoss_sft_data.py's stages on Inkling's files (see its docstring). The run directory is results/sft_inksm/<run>.
Eval items: 422 of the pool's 516 GPQA questions are CoT-Control items (grading/data/cotcontrol/gpqa_w_keyword.csv), which Astra and the
           transfer evals sample (gptoss_sft_data.py does not exclude them: 300 of gsft1's 458 examples are on them). mine skips every problem
           whose question is in EVAL_CSVS unless --keep-eval-overlap, and configure() wraps gs.cot_required so select drops them either way.
rewrite    gptoss_rewrite.py (base gpt-oss-120b rewrites Inkling's CoTs; misdirect_end decoys from Inkling's own wrong samples), argv passed through.
validity   validity_judge.py on Inkling's files, argv passed through.
sample     Inkling (or --model-path) answers sft_prompts tasks for each problem x type; graded by grpo_obf.score_roll; cot_tokens are the sampled
           thinking ids. As gptoss_sft_data.sample.
force      force_check's forced-answer scoring with Inkling-Small as the scorer, in TMLv0 layout: prompt + THINK + CoT + END + TEXT + 'ANSWER:' +
           answer + END + STOP; the no-CoT baseline is prompt + TEXT + 'ANSWER:' + answer. lp_final is log p(<|content_text|>) after the CoT's
           <|end_message|> (Seq.final_at 2). Same keep rule; accept's naturalness bounds (NAT_MIN, NAT_FRAC_MAX) were set on gpt-oss and are
           printed here for recalibration.
check      Rebuilds --n of Inkling's own clean mined samples (each with its own response text) through inkio.completion_ids and
           inkio.sft_datum and compares them with the sampled ids token for token; the ids rebuild must match (text re-encoding is reported:
           a decoded CoT need not re-encode to the sampled ids, which is why export uses sampled cot_tokens where a row has them).
export     gptoss_sft.select's mix (caps per type and problem, --plain-frac, the mix filters) over the accepted candidates, each rendered by
           inkio (sampled cot_tokens when present, else the encoded text; prompt by inkio.encode), written to <run>/sft/<name>/examples.jsonl
           with prompt_ids and completion_ids (the trainer's input) and the text fields; prints tokens per epoch and the training cost.

    python inksm_sft_data.py mine --samples 4 --nocot-samples 2 --budget 15
    python inksm_sft_data.py select --run p1 --dapo 30 --gpqa 30
    python inksm_sft_data.py plain --run p1
    python inksm_sft_data.py code --run p1 --per-type 20
    python inksm_sft_data.py rewrite-jobs --run p1 --types sad cover_story misdirect_end --per-type 10
    python inksm_sft_data.py rewrite --jobs ../results/sft_inksm/p1/rewrite_jobs.jsonl --out ../results/sft_inksm/p1/rewrites.jsonl --budget 2
    python inksm_sft_data.py score --run p1
    python inksm_sft_data.py force --run p1 --budget 2
    python inksm_sft_data.py judge --run p1 --budget 3
    python inksm_sft_data.py accept --run p1
    python inksm_sft_data.py check
    python inksm_sft_data.py export --run p1 --name s1 --pair-plain
"""
import argparse
import asyncio
import csv
import json
import os
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import tinker

import force_check
import gptoss_sft as gsft
import gptoss_sft_data as gs
import grpo_obf
import inkio
import modelio
import obf_data
import sft_prompts
from cotswitch import extract

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results/sft_inksm"
POOL_IN = ROOT / "data/obf_train.jsonl"
TAG = os.environ.get("INKSM_TAG", "")  # a suffix on POOL and SAMPLES, so a smoke test leaves the real files alone
POOL = ROOT / f"data/obf_train_inksm{TAG}.jsonl"
SAMPLES = ROOT.parent / f"results/mine/inksm{TAG}/samples.jsonl"
TERMS = [t for t in sft_prompts.TERMS if t != "analysis channel"]
PRICE = modelio.PRICES[inkio.MODEL]
NOCOT_TOKENS = 10
OPEN, FINAL_HEAD, PREFILL = inkio.THINK, inkio.END + inkio.TEXT, force_check.PREFILL
CLOSE, NOCOT, RETURN, FINAL_AT = FINAL_HEAD + PREFILL, inkio.TEXT + PREFILL, inkio.END + inkio.STOP, 2
WORD_RE = re.compile(r"[A-Za-z]+")
EVAL_CSVS = sorted((ROOT.parent / "grading/data/cotcontrol").glob("*_w_keyword.csv"))  # CoT-Control's GPQA, HLE and MMLU-Pro items


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()[:150]


def eval_questions() -> set[str]:
    """Normalized question texts of every CoT-Control item (Astra and the transfer evals draw from these CSVs; 422 of the 445 GPQA ones are
    in data/obf_train.jsonl)."""
    csv.field_size_limit(10 ** 9)
    return {norm(r["question"]) for f in EVAL_CSVS for r in csv.DictReader(f.open(errors="ignore"))}


def configure() -> None:
    """Point gptoss_sft_data at Inkling's files, and set inkio.EFFORT to the effort the pool was mined at (every later stage samples, scores and
    renders at it)."""
    gs.RESULTS, gs.POOL, gs.SAMPLES, gs.MODEL = RESULTS, POOL, SAMPLES, inkio.MODEL
    mined = SAMPLES.parent / "config.json"
    if mined.exists():
        inkio.EFFORT = json.loads(mined.read_text())["effort"]
    sft_prompts.TERMS = TERMS
    held, cot_required = eval_questions(), gs.cot_required
    gs.cot_required = lambda pool: [q for q in cot_required(pool) if norm(q["question"]) not in held]


def parse(q: dict, gen: str, truncated: bool) -> dict:
    split = extract.split_output(inkio.to_think(gen), None, "length" if truncated else "stop")
    answer = obf_data.parse_answer(q, split["response"], split["content"])
    return dict(reasoning=split["reasoning"], response=split["response"], answer=answer, correct=answer == q["answer"])


async def sampler(model_path: str | None = None):
    svc = tinker.ServiceClient()
    return await (svc.create_sampling_client_async(model_path=model_path) if model_path else svc.create_sampling_client_async(base_model=inkio.MODEL))


async def mine(a) -> None:
    inkio.EFFORT = a.effort
    pool, held = [json.loads(l) for l in a.pool_in.open()], eval_questions()
    n_held = sum(norm(q["question"]) in held for q in pool)
    print(f"{n_held} of {len(pool)} problems are CoT-Control eval items{' (kept)' if a.keep_eval_overlap else ' and are not mined'}")
    pool = [q for q in pool if a.keep_eval_overlap or norm(q["question"]) not in held][:a.limit or None]
    client, tok = await sampler(), inkio.Tok()
    stop = [tok.eos_token_id]
    params = tinker.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, top_p=1.0, stop=stop)
    nocot_params = tinker.SamplingParams(max_tokens=NOCOT_TOKENS, temperature=1.0, top_p=1.0, stop=stop)
    sem, spent = asyncio.Semaphore(a.concurrency), [0.0]

    async def one(q):
        ids = inkio.render(obf_data.question_block(q))
        async with sem:
            if spent[0] > a.budget:
                return None
            res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=a.samples, sampling_params=params)
            nc = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids + tok.encode(NOCOT)), num_samples=a.nocot_samples,
                                           sampling_params=nocot_params) if a.nocot_samples else None
        outs = []
        for s in res.sequences:
            toks, truncated = list(s.tokens), s.stop_reason == "length"
            cot = inkio.thinking_ids(tok, toks)
            outs.append(parse(q, tok.decode(toks), truncated) | dict(tokens=toks, cot_tokens=cot, clean=cot is not None and not truncated,
                                                                    n_tokens=len(toks), truncated=truncated))
        for s in (nc.sequences if nc else []):
            resp = PREFILL + tok.decode(s.tokens, skip_special_tokens=True)
            answer = obf_data.parse_answer(q, resp, resp)
            outs.append(dict(nocot=True, response=resp, answer=answer, n_tokens=len(s.tokens), correct=answer == q["answer"]))
        spent[0] += (len(ids) * (1 + bool(nc)) * PRICE["prefill"] + sum(o["n_tokens"] for o in outs) * PRICE["sample"]) / 1e6
        return outs

    allouts = await asyncio.gather(*(one(q) for q in pool))
    done = [(q, outs) for q, outs in zip(pool, allouts) if outs is not None]
    SAMPLES.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLES.open("w") as f:
        for q, outs in done:
            for i, o in enumerate(outs):
                f.write(json.dumps(dict(qid=q["qid"], source=q["source"], i=i, true_answer=q["answer"], words=len(WORD_RE.findall(o.get("reasoning") or "")), **o),
                                   ensure_ascii=False) + "\n")
    samples = [[o for o in outs if not o.get("nocot")] for _, outs in done]
    nocot = [[o for o in outs if o.get("nocot")] for _, outs in done]
    counts = [[Counter(w for w in re.findall(r"[a-z]+", (o["reasoning"] or "").lower()) if len(w) >= 4) for o in outs] for outs in samples]
    df = Counter(w for cs in counts for w in set().union(*cs))
    med = lambda xs: statistics.median(xs) if xs else None
    with POOL.open("w") as f:
        for (q, _), outs, cs, ncs in zip(done, samples, counts, nocot):
            total, present = sum(cs, Counter()), Counter(w for c in cs for w in c)
            kws = [w for w, k in total.most_common() if 2 * present[w] >= len(cs) and k >= 2 and df[w] <= 0.05 * len(done)][:5]
            words = [len(WORD_RE.findall(o["reasoning"] or "")) for o in outs]
            f.write(json.dumps(q | dict(keywords=kws, base_words=med(words), n_tokens=med([o["n_tokens"] for o in outs]),
                                        correct=sum(o["correct"] for o in outs) / len(outs), truncated=sum(o["truncated"] for o in outs), model=inkio.MODEL,
                                        samples=len(outs), correct_words=med([w for w, o in zip(words, outs) if o["correct"]]),
                                        correct_tokens=med([o["n_tokens"] for o in outs if o["correct"]]))
                               | (dict(nocot_correct=sum(o["correct"] for o in ncs) / len(ncs)) if ncs else {}), ensure_ascii=False) + "\n")
    flat = [o for outs in samples for o in outs]
    (SAMPLES.parent / "config.json").write_text(json.dumps(vars(a) | dict(model=inkio.MODEL, effort=inkio.EFFORT, problems=len(done), spent_est=round(spent[0], 3)),
                                                           indent=1, default=str))
    print(f"{len(done)}/{len(pool)} problems x {a.samples}: accuracy {sum(o['correct'] for o in flat) / len(flat):.3f}, truncated "
          f"{sum(o['truncated'] for o in flat)}/{len(flat)}, clean {sum(o['clean'] for o in flat)}/{len(flat)}, median tokens "
          f"{statistics.median(o['n_tokens'] for o in flat)}; no-CoT accuracy {sum(o['correct'] for n in nocot for o in n) / max(1, sum(map(len, nocot))):.3f}; "
          f"cot-required {len(gs.cot_required([json.loads(l) for l in POOL.open()]))}; ~${spent[0]:.2f}")


async def sample(a) -> None:
    d = gs.run_dir(a.run)
    problems = gs.read(d / "problems.jsonl")
    client, tok = await sampler(a.model_path), inkio.Tok()
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=[tok.eos_token_id])
    rng = random.Random(f"{a.seed}-{a.origin}")
    assert not set(a.types) & (gs.HELD_OUT | {"misdirect_end"}), "held-out types and misdirect_end (rewrites only) are not sampled"
    tasks = [sft_prompts.make_sft_task(q, name, rng) for q in problems for name in a.types for _ in range(a.k) if grpo_obf.eligible(name, q)]
    sem, spent, failures = asyncio.Semaphore(a.concurrency), [0.0], []

    async def one(task):
        try:
            ids = inkio.encode(task)
            async with sem:
                if spent[0] > a.budget:
                    return None
                res = await client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)
            s = res.sequences[0]
            toks = list(s.tokens)
            spent[0] += (len(ids) * PRICE["prefill"] + len(toks) * PRICE["sample"]) / 1e6
            sc = grpo_obf.score_roll(task, inkio.to_think(tok.decode(toks)), s.stop_reason == "length", "", think_prefix="", answer_only=True)
            return gs.stage0(gs.candidate(task, a.origin, sc, inkio.thinking_ids(tok, toks), len(toks)) | dict(prompt_tokens=ids))
        except Exception as e:  # noqa: BLE001 (one bad sample is skipped and counted, not fatal to the run)
            failures.append(f"{task['qid']}:{task['type']}: {e!r}"[:300])
            return None

    rows = [r for r in await asyncio.gather(*(one(t) for t in tasks)) if r is not None]
    gs.write(d / f"candidates_{a.origin}.jsonl", gs.read(d / f"candidates_{a.origin}.jsonl") + rows)
    (d / f"config_sample_{a.origin}.json").write_text(json.dumps(vars(a) | dict(n=len(rows), spent_est=round(spent[0], 3), failures=failures), indent=1))
    by = Counter((r["type"], bool(r.get("strict")) and bool(r.get("correct")), not r["reject_reasons"]) for r in rows)
    print(f"{a.origin}: {len(rows)} samples, ~${spent[0]:.2f}, {len(failures)} failed; per type n / strict+correct / survive stage0 (judge types: before judging):")
    for t in sorted({r["type"] for r in rows}):
        print(f"  {t:20s} {sum(v for k, v in by.items() if k[0] == t):4d} {sum(v for k, v in by.items() if k[0] == t and k[1]):4d} "
              f"{sum(v for k, v in by.items() if k[0] == t and k[2]):4d}")


def answer_ids(tok: inkio.Tok, cand: dict) -> list[int]:
    """force_check.answer_ids in TMLv0: the answer as it follows 'ANSWER:', then <|end_message|><|content_model_end_sampling|>."""
    ans = cand["true_answer"]
    ids = tok.encode(f" {ans}")
    assert tok.encode(f"{PREFILL} {ans}") == tok.encode(PREFILL) + ids, f"'{PREFILL} {ans}' does not split after '{PREFILL}'"
    assert cand["source"] != "gpqa" or len(ids) == 1, (ans, ids)
    return ids + tok.encode(RETURN)


def build(tok: inkio.Tok, prompt: list[int], cot: list[int] | None, answer: list[int]) -> force_check.Seq:
    """force_check.build in TMLv0: prompt + OPEN + cot + CLOSE + answer, or prompt + NOCOT + answer when cot is None."""
    if cot is None:
        tail, span = tok.encode(NOCOT), None
    else:
        head = tok.encode(OPEN)
        tail, span = head + list(cot) + tok.encode(CLOSE), (len(prompt) + len(head), len(prompt) + len(head) + len(cot))
    pos = len(prompt) + len(tail)
    return force_check.Seq(list(prompt) + tail + list(answer), pos, span, pos - len(tok.encode(PREFILL)), FINAL_AT)


def plan(tok: inkio.Tok, cand: dict) -> dict:
    """force_check.plan with inkio's prompt and build."""
    prompt = list(cand.get("prompt_tokens") or inkio.encode(cand))
    ans, full = answer_ids(tok, cand), force_check.cot_ids(tok, cand)
    cut = force_check.truncate(tok, cand, full) if cand["type"] in force_check.TRUNCATE else None
    cot = full if cut is None else cut
    return dict(key=(cand["qid"], tuple(prompt)), main=build(tok, prompt, cot, ans), nocot=build(tok, prompt, None, ans),
                full=None if cut is None else build(tok, prompt, full, ans), cot=cot, cut=None if cut is None else len(cut), n_full=len(full))


async def force(a) -> None:
    d = gs.run_dir(a.run)
    client, tok = await sampler(), inkio.Tok()
    todo = []
    for f in gs.candidate_files(d):
        rows = gs.read(f)
        todo.append((f, rows, [i for i, r in enumerate(rows) if not r["reject_reasons"] and "force" not in r][:a.limit or None]))
    plans = [(f, i, plan(tok, rows[i])) for f, rows, idx in todo for i in idx]
    tokens, requests = force_check.plan_cost([p for *_, p in plans])
    est = force_check.usd(tokens, requests, inkio.MODEL)
    print(f"force: {len(plans)} candidates, {tokens:,} tokens, {requests} requests, ~${est:.2f}")
    if est > a.budget:
        raise SystemExit(f"estimate ${est:.2f} is over --budget ${a.budget}")
    scorer = force_check.Scorer(client, tok, a.concurrency)
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
        gs.write(f, rows)
    for origin in sorted({r["origin"] for rows in files.values() for r in rows}):
        lp = [r["force"]["cot_lp_mean"] for rows in files.values() for r in rows if r["origin"] == origin and (r.get("force") or {}).get("cot_lp_mean") is not None]
        below = [r["force"]["frac_below_-5"] for rows in files.values() for r in rows if r["origin"] == origin and (r.get("force") or {}).get("frac_below_-5") is not None]
        if lp:
            q = statistics.quantiles(lp, n=10) if len(lp) > 1 else lp * 9
            print(f"  {origin:18s} cot_lp_mean p10/p50/p90 {q[0]:.2f}/{statistics.median(lp):.2f}/{q[-1]:.2f}, frac_below_-5 median {statistics.median(below):.3f} "
                  f"(accept's gpt-oss bounds: >= {gs.NAT_MIN}, <= {gs.NAT_FRAC_MAX}); force kept "
                  f"{sum(bool(r.get('force_keep')) for rows in files.values() for r in rows if r['origin'] == origin and 'force' in r)}/{len(lp)}")
    print(f"sent {scorer.tokens:,} tokens in {scorer.requests} requests, ~${force_check.usd(scorer.tokens, scorer.requests, inkio.MODEL):.2f}")


def check(a) -> int:
    """Rebuild n clean mined samples, right or wrong (completion_ids on the sampled CoT ids, and sft_datum on the decoded text, each with the
    sample's own response text); the ids rebuild must match."""
    tok, pool = inkio.Tok(), {q["qid"]: q for q in map(json.loads, POOL.open())}
    seen = ids_ok = text_ok = 0
    for s in map(json.loads, SAMPLES.open()):
        if s.get("nocot") or not s["clean"]:
            continue
        q, response = pool[s["qid"]], tok.decode(s["tokens"]).rsplit(inkio.TEXT, 1)[1].removesuffix(inkio.END + inkio.STOP)
        ids_ok += inkio.completion_ids(tok, s["cot_tokens"], response) == s["tokens"]
        mi, _, n = inkio.sft_datum(obf_data.question_block(q), None, tok.decode(s["cot_tokens"]), response)
        text_ok += mi.to_ints()[n:] == s["tokens"]
        seen += 1
        if seen == a.n:
            break
    print(f"check: of {seen} sampled turns, {ids_ok} rebuilt from their CoT ids and {text_ok} from their decoded text, token for token")
    assert seen, "no clean samples to check"
    assert ids_ok == seen, "the completion layout does not match Inkling's sampled tokens"
    return text_ok


def export(a) -> None:
    d = gs.run_dir(a.run)
    a.type_caps = {k: int(v) for k, v in (x.split("=") for x in a.type_caps)}
    a.type_from, a.extra_runs, a.review, a.drop_organism_types = {}, [], [], []
    exs = gsft.select(d, a, random.Random(a.seed))
    tok, out = inkio.Tok(), d / "sft" / a.name
    out.mkdir(parents=True, exist_ok=False)
    rows, tokens = [], 0
    for r in exs:
        response = gsft.answer_text(r)
        prompt = list(r.get("prompt_tokens") or inkio.encode(r))
        comp = inkio.completion_ids(tok, r["cot_tokens"] if r.get("cot_tokens") is not None else tok.encode(r["reasoning"]), response)
        if r.get("cot_tokens") is None:  # text rows: the trainer's ids must be what render_for_sft makes of the same messages
            mi, _, n = inkio.sft_datum(r["user"], r.get("system"), r["reasoning"], response)
            assert mi.to_ints() == prompt + comp and n == len(prompt), f"render mismatch on {r['qid']} {r['type']}"
        tokens += len(prompt) + len(comp)
        rows.append({k: r.get(k) for k in ("qid", "source", "type", "origin", "source_ref", "value", "frame", "term", "placement", "position", "system",
                                           "user", "instruction", "requirement", "true_answer", "reasoning", "words", "band_lo", "band_hi")}
                    | dict(response=response, force={k: (r.get("force") or {}).get(k) for k in ("p_correct", "uplift", "cot_lp_mean", "p_nocot")},
                           prompt_ids=prompt, completion_ids=comp))
    gs.write(out / "examples.jsonl", rows)
    by = Counter((r["type"], r["origin"]) for r in rows)
    est = tokens * PRICE["train"] / 1e6
    (out / "config.json").write_text(json.dumps(vars(a) | dict(model=inkio.MODEL, effort=inkio.EFFORT, n=len(rows), tokens_per_epoch=tokens,
                                                               train_usd_per_epoch=round(est, 3), by={f"{t}|{o}": k for (t, o), k in by.items()}), indent=1, default=str))
    print(f"{len(rows)} examples, {tokens:,} tokens per epoch, ~${est:.2f} per epoch at ${PRICE['train']}/M; written to {out}")
    for (t, o), k in sorted(by.items()):
        print(f"  {t:20s} {o:24s} {k:4d}")


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] in ("rewrite", "validity"):
        cmd, argv = sys.argv[1], sys.argv[2:]
        if cmd == "rewrite":
            import gptoss_rewrite
            asyncio.run(gptoss_rewrite.main(argv if "--samples" in argv else argv + ["--samples", str(SAMPLES)]))
        else:
            import validity_judge
            sys.argv = [sys.argv[0]] + argv
            asyncio.run(validity_judge.main())
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--pool-in", type=Path, default=POOL_IN)
    m.add_argument("--effort", type=float, required=True, help="Inkling thinking effort in [0, 1); every later stage uses it (configure)")
    m.add_argument("--samples", type=int, default=4)
    m.add_argument("--nocot-samples", type=int, default=2)
    m.add_argument("--max-tokens", type=int, default=16000)
    m.add_argument("--limit", type=int, default=0, help="first N problems only (smoke tests)")
    m.add_argument("--keep-eval-overlap", action="store_true", help="also mine problems that are CoT-Control eval items (select drops them anyway)")
    m.add_argument("--concurrency", type=int, default=64)
    m.add_argument("--budget", type=float, required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", required=True)
    s.add_argument("--origin", required=True, choices=["base", "organism"])
    s.add_argument("--model-path")
    s.add_argument("--types", nargs="+", required=True)
    s.add_argument("--k", type=int, default=1)
    s.add_argument("--max-tokens", type=int, default=16000)
    s.add_argument("--concurrency", type=int, default=64)
    s.add_argument("--budget", type=float, required=True)
    s.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("force")
    f.add_argument("--run", required=True)
    f.add_argument("--budget", type=float, required=True)
    f.add_argument("--limit", type=int, default=0, help="score at most this many rows per candidates file (0: all)")
    f.add_argument("--concurrency", type=int, default=64)
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=20)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--per-type", type=int, default=150)
    e.add_argument("--per-problem", type=int, default=6)
    e.add_argument("--organism-max", type=float, default=0.4)
    e.add_argument("--plain-frac", type=float, default=0.25)
    e.add_argument("--type-caps", nargs="*", default=[], help="type=n: cap for that type instead of --per-type")
    e.add_argument("--dapo-frac", type=float, default=0.0)
    e.add_argument("--pair-plain", action="store_true")
    e.add_argument("--keep-types", nargs="*", default=[])
    e.add_argument("--drop-types", nargs="*", default=[])
    e.add_argument("--drop-type-source", nargs="*", default=[])
    e.add_argument("--sad-gap", type=int, default=600)
    e.add_argument("--pirate-share", type=float, default=0.7)
    e.add_argument("--seed", type=int, default=0)
    shared = {"select", "plain", "code", "rewrite-jobs", "score", "judge", "accept"}
    if len(sys.argv) > 1 and sys.argv[1] in shared:
        sys.argv = [sys.argv[0]] + sys.argv[1:]
        gs.main()
        return
    a = ap.parse_args()
    if a.cmd == "sample":
        assert (a.origin == "organism") == bool(a.model_path), "--model-path is the organism; base samples use the base model"
    if a.cmd in ("mine", "sample", "force"):
        asyncio.run({"mine": mine, "sample": sample, "force": force}[a.cmd](a))
    else:
        {"check": check, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
