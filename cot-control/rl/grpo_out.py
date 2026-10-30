"""GRPO on gpt-oss-120b with output-only instruction following and varied reasoning effort: does RL that never looks at the CoT change
CoT controllability (Astra 108)? Reproduction attempt for "Astra got better at CoT control during RL".

Prompts: a pool_k4 MMLU-Pro question (disjoint from the 540 eval items) + "In your final response, before the ANSWER line, explain your answer
in a few sentences. {constraint 1}[ {constraint 2}]" with 2 constraints on the final response only. Nothing mentions the
reasoning. Reasoning effort is sampled per prompt from low / medium / high (harmony 'Reasoning:' line). Constraint types and values
(values differ from the CoT-Control eval's):
  avoid_word   : do not use the word '{w}' (the question's first mined keyword)
  lowercase    : write the explanation entirely in lowercase letters
  uppercase    : write the explanation entirely in capital letters
  end_word     : end every sentence of the explanation with the word '{w}'      (w from grpo.END_WORDS)
  bookend      : begin and end the explanation with the sentence '{s}'          (s from grpo.SENTENCES)
  insert_word  : put the word '{w}' between every two words of the explanation  (w from grpo.INSERT_WORDS)
  max_words    : keep the explanation under {n} words                           (n in 25, 35, 45)
  language     : write the explanation in {L}                                   (French, Spanish, German)
  bullets      : give the explanation as exactly {n} bullet points starting '- ' (n in 2, 3, 4)
Grading (programmatic; lowercase/uppercase skip tokens with digits, underscores or a single letter and pass at >= 98%; insert_word counts adjacent
non-inserted whitespace tokens, on the final response with its 'ANSWER: X' line removed = the explanation): each constraint gives partial in [0, 1]
and strict (bool). Reward = correct + mean(partial) + mean(strict); 0 if the rollout is truncated, has no reasoning or no parseable
ANSWER line, or the explanation is empty. No CoT term, no judges, no length penalty. Advantage = reward - group mean; equal-reward groups dropped.

    uv run python grpo_out.py --run outrl1 --steps 60 --budget 22 [--dry-run | --smoke]
Writes results/rl/<run>/{steps,rollouts,config,checkpoints}.json(l)."""
import argparse, asyncio, json, random, re, time
from pathlib import Path
import numpy as np
import tinker
from cotswitch import extract, prompts
from cotswitch.data import Item
import controls, grpo, modelio

MODEL = "openai/gpt-oss-120b"
TYPES = ["avoid_word", "lowercase", "uppercase", "end_word", "bookend", "insert_word", "max_words", "language", "bullets"]
EFFORTS = ["low", "medium", "high"]
WORD = re.compile(r"[A-Za-z']+")
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
def sentences(t): return [s.strip() for s in SENT.split(t) if re.search(r"[A-Za-z]", s)]
def letters(t): return [c for c in t if c.isalpha()]

def make(q, rng):
    item = Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"], answer_letter=q["answer"], keywords=[], synonyms={}, raw={})
    types = rng.sample([t for t in TYPES if not (t == "avoid_word" and not q["keywords"])], 2)
    if "lowercase" in types and "uppercase" in types: types.remove("uppercase")
    cons, texts = [], []
    for t in types:
        v = (q["keywords"][0] if t == "avoid_word" else rng.choice(grpo.END_WORDS) if t == "end_word" else rng.choice(grpo.SENTENCES) if t == "bookend"
             else rng.choice(grpo.INSERT_WORDS) if t == "insert_word" else rng.choice([25, 35, 45]) if t == "max_words" else rng.choice(controls.LANG_TRAIN) if t == "language"
             else rng.choice([2, 3, 4]) if t == "bullets" else None)
        cons.append((t, v))
        texts.append({"avoid_word": f"Do not use the word '{v}' anywhere in it.", "lowercase": "Write it entirely in lowercase letters.",
                      "uppercase": "Write it entirely in capital letters.", "end_word": f"End every sentence of it with the word '{v}'.",
                      "bookend": f"Begin it and end it with the sentence '{v}'.", "insert_word": f"Put the word '{v}' between every two words of it.",
                      "max_words": f"Keep it under {v} words.", "language": f"Write it in {v}.", "bullets": f"Give it as exactly {v} bullet points, each starting with '- '."}[t])
    instr = "In your final response, before the ANSWER line, explain your answer in a few sentences. " + " ".join(texts)
    return dict(qid=q["qid"], answer=q["answer"], base_acc=q["correct"], cons=cons, effort=rng.choice(EFFORTS), user=f"{prompts.question_block(item)}\n\n{instr}", instruction=instr)

def grade(t, v, x):
    """(partial, strict) for one constraint on the explanation x."""
    if t == "avoid_word":
        n = len(re.findall(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z])", x, re.I)); return 1 / (1 + n), n == 0
    if t in ("lowercase", "uppercase"):  # tokens with digits, underscores or one letter (T5, f_m, option letters) are not graded
        L = [c for w in x.split() if not re.search(r"[\d_]", w) and len(letters(w)) > 1 for c in letters(w)]
        ok = sum(c.islower() if t == "lowercase" else c.isupper() for c in L) / max(1, len(L)); return ok, bool(L) and ok >= 0.98
    if t == "end_word":
        S = sentences(x); ok = [bool(re.search(rf"\b{re.escape(v)}\W*$", s, re.I)) for s in S]; p = sum(ok) / max(1, len(ok)); return p, bool(S) and p == 1.0
    if t == "bookend":
        s = x.strip(); a, b = s.startswith(v), s.rstrip().endswith(v) and len(s) > len(v); return (a + b) / 2, a and b
    if t == "insert_word":
        w = [re.sub(r"^\W+|\W+$", "", z).lower() for z in x.split()]; w = [z for z in w if z]
        other = sum(z != v for z in w); bad = sum(1 for a, b in zip(w, w[1:]) if a != v and b != v)
        p = max(0.0, 1 - bad / max(1, other - 1)); return p, other > 1 and p >= 0.95
    if t == "max_words":
        n = len(WORD.findall(x)); return (1.0 if n < v else max(0.0, 1 - (n - v) / v)), n < v
    if t == "language":
        prof = controls.lang_profile(x)
        if not prof: return 0.0, False
        p = sum(z == v for z in prof) / len(prof); return p, p >= 0.95
    if t == "bullets":
        lines = [l for l in x.strip().split("\n") if l.strip()]; b = [l for l in lines if l.lstrip().startswith("- ")]
        return (1.0 if len(b) == v else max(0.0, 1 - abs(len(b) - v) / v)) * (len(b) / max(1, len(lines))), len(b) == v and len(b) == len(lines)

def score(task, gen, truncated):
    sp = extract.split_output(extract.harmony_to_think(gen), None, "length" if truncated else "stop")
    resp = sp["response"] or ""; ans = extract.parse_answer(resp, sp["content"])
    expl = re.sub(r"\n*\s*\**ANSWER:\s*\**\s*[A-J]\b.*$", "", resp.strip(), flags=re.S).strip()
    g = [grade(t, v, expl) for t, v in task["cons"]] if expl else [(0.0, False)] * len(task["cons"])
    ok = bool(sp["reasoning"]) and not truncated and ans is not None and bool(expl)
    partial, strict = float(np.mean([p for p, _ in g])), float(np.mean([s for _, s in g]))
    return dict(correct=ans == task["answer"], fmt_ok=ok, partial=partial, strict=strict, all_strict=all(s for _, s in g), per=[[t, round(p, 3), s] for (t, _), (p, s) in zip(task["cons"], g)],
                reward=(float(ans == task["answer"]) + partial + strict) if ok else 0.0, reasoning=sp["reasoning"], response=resp, cot_words=len(WORD.findall(sp["reasoning"] or "")), truncated=truncated)

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=32); ap.add_argument("--group-size", type=int, default=8); ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=6000); ap.add_argument("--budget", type=float, default=22.0)
    ap.add_argument("--save-every", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pool", type=Path, default=Path(__file__).parent / "data/pool_k4.jsonl")
    a = ap.parse_args()
    pool = [q for q in map(json.loads, a.pool.open()) if q["base_words"] >= 10]; rng = random.Random(a.seed)
    if a.dry_run:
        for _ in range(6): t = make(rng.choice(pool), rng); print("---", t["effort"], t["cons"], "\n...", t["user"][-420:])
        return
    out = grpo.RESULTS / a.run; out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a) | dict(types=TYPES, efforts=EFFORTS), indent=1, default=str))
    price = modelio.PRICES[MODEL]; svc = tinker.ServiceClient()
    tc = await svc.create_lora_training_client_async(MODEL, rank=a.rank, seed=a.seed); tok = tc.get_tokenizer()
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=modelio.GPT_OSS_STOP)
    usd, ckpts = 0.0, []
    for step in range(a.steps):
        t0 = time.time(); sampler = await tc.save_weights_and_get_sampling_client_async()
        tasks = [make(rng.choice(pool), rng) for _ in range(a.prompts_per_step)]
        pids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t["user"]}], tokenize=False, add_generation_prompt=True, reasoning_effort=t["effort"]), add_special_tokens=False) for t in tasks]
        res = await asyncio.gather(*[sampler.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=a.group_size, sampling_params=params) for p in pids])
        rollouts, data, var = [], [], 0
        for t, p, r in zip(tasks, pids, res):
            g = [(s, score(t, tok.decode(s.tokens, skip_special_tokens=False), s.stop_reason == "length")) for s in r.sequences]
            m = sum(sc["reward"] for _, sc in g) / len(g); var += len({sc["reward"] for _, sc in g}) > 1
            for s, sc in g:
                rollouts.append(dict(step=step + 1, qid=t["qid"], effort=t["effort"], cons=t["cons"], base_acc=t["base_acc"], n_tokens=len(s.tokens), adv=sc["reward"] - m, **sc))
                if any(sc2["reward"] != m for _, sc2 in g): data.append(grpo.datum(p, list(s.tokens), [float(x) for x in s.logprobs], sc["reward"] - m))
            usd += (len(p) * price["prefill"] + sum(len(s.tokens) for s, _ in g) * price["sample"]) / 1e6
        usd += sum(d.model_input.length + 1 for d in data) * price["train"] / 1e6
        if data:
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling"); opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            await fb.result_async(); await opt.result_async()
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or usd > a.budget:
            name = f"outctl-{a.run}-step{step + 1}"
            sp = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path; st = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            ckpts.append(dict(step=step + 1, sampler=sp, state=st)); (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))
        with (out / "rollouts.jsonl").open("a") as f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rollouts)
        M = lambda xs: round(float(np.mean(list(xs))), 4) if rollouts else None
        by_eff = {e: dict(n=sum(r["effort"] == e for r in rollouts), cot_words_median=int(np.median([r["cot_words"] for r in rollouts if r["effort"] == e] or [0])),
                          correct=M(r["correct"] for r in rollouts if r["effort"] == e)) for e in EFFORTS}
        by_type = {t: M(x[2] for r in rollouts for x in r["per"] if x[0] == t) for t in TYPES}
        line = dict(step=step + 1, reward=M(r["reward"] for r in rollouts), correct=M(r["correct"] for r in rollouts), acc_vs_base=round(M(r["correct"] for r in rollouts) - M(r["base_acc"] for r in rollouts), 4),
                    strict=M(r["strict"] for r in rollouts), all_strict=M(r["all_strict"] for r in rollouts), fmt_ok=M(r["fmt_ok"] for r in rollouts), truncated=sum(r["truncated"] for r in rollouts),
                    groups_var=var, n_trained=len(data), by_effort=by_eff, strict_by_type=by_type, secs=round(time.time() - t0), spent_est=round(usd, 3))
        with (out / "steps.jsonl").open("a") as f: f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if usd > a.budget: print(f"budget ${a.budget} reached; stopping"); break
    print(f"done. ~${usd:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
