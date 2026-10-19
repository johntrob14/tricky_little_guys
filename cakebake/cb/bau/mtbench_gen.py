"""MT-Bench answers (pod, vLLM), matching the FastChat fork's gen_model_answer.py (Rice-wxl/FastChat@b19146f6,
ref/bau_key/fastchat_llm_judge/gen_model_answer.py):
  - the 80 questions of data/mt_bench/question.jsonl, both turns; turn 2 sees the model's own turn-1 answer
  - prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True), no system prompt
  - max_new_tokens 1024, 1 choice, bf16
  - temperature by category (FastChat temperature_config): writing/roleplay 0.7, extraction/math/coding/reasoning 0.0
    (greedy), stem/humanities 0.1, anything else 0.7. Sampled categories use HF generate's defaults top_k=50, top_p=1.0
    (OLMo's generation_config sets neither).
  - answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
Deviations: vLLM batched decoding instead of HF generate at batch 1 (greedy turns can differ by bf16 numerics;
sampled turns cannot match their torch.manual_seed(0) stream anyway); vLLM seed 0 per request.

  python -m cb.bau.mtbench_gen --model /root/models/base_dpo --name base_dpo
  python -m cb.bau.mtbench_gen --model <tokenizer dir> --name x --dry-run    # prompt construction only, no vLLM
Output: results/bau/<NAME>/mtbench_answers.jsonl, FastChat answer format {question_id, model_id, choices:[{index,
turns}], category, finish}.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from cb.bau.common import DATA, chat_ids, decode, out_dir

QUESTIONS = os.path.join(DATA, "mt_bench", "question.jsonl")
TEMPERATURE = {"writing": 0.7, "roleplay": 0.7, "extraction": 0.0, "math": 0.0, "coding": 0.0, "reasoning": 0.0,
               "stem": 0.1, "humanities": 0.1, "arena-hard-200": 0.0}
MAX_NEW = 1024


def load_questions(n=None):
    qs = [json.loads(l) for l in open(QUESTIONS) if l.strip()]
    return qs[:n] if n else qs


def temperature(q):
    return TEMPERATURE.get(q["category"], 0.7)


def turn_messages(q, answers_so_far):
    """messages for turn len(answers_so_far)+1: user/assistant alternation, no system prompt."""
    msgs = []
    for j, a in enumerate(answers_so_far):
        msgs += [{"role": "user", "content": q["turns"][j]}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": q["turns"][len(answers_so_far)]})
    return msgs


def sampling(q, sp_cls, seed=0):
    t = temperature(q)
    if t < 1e-4:
        return sp_cls(temperature=0.0, max_tokens=MAX_NEW, seed=seed)
    return sp_cls(temperature=t, top_k=50, top_p=1.0, max_tokens=MAX_NEW, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=None, help="first N questions (smoke runs)")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true", help="tokenizer only: print the turn-1 and turn-2 prompts")
    a = ap.parse_args()
    qs = load_questions(a.n)
    if a.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        for q in qs[:2]:
            for ans in ([], ["<TURN-1 ANSWER>"]):
                ids = chat_ids(tok, turn_messages(q, ans))
                print(f"--- q{q['question_id']} ({q['category']}, T={temperature(q)}) turn {len(ans) + 1}: {len(ids)} tokens")
                print(repr(tok.decode(ids)))
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    dst = os.path.join(out_dir(a.name), "mtbench_answers.jsonl")
    if os.path.exists(dst) and sum(1 for _ in open(dst)) >= len(qs):
        print(f"cached: {dst}")
        return
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096, seed=0)
    tok = AutoTokenizer.from_pretrained(a.model)  # HF tokenizer for templates/decoding, as their script
    answers = {q["question_id"]: [] for q in qs}
    finish = {q["question_id"]: [] for q in qs}
    for turn in range(2):
        prompts = [{"prompt_token_ids": chat_ids(tok, turn_messages(q, answers[q["question_id"]]))} for q in qs]
        outs = llm.generate(prompts, [sampling(q, SamplingParams) for q in qs])
        for q, o in zip(qs, outs):
            c = o.outputs[0]
            answers[q["question_id"]].append(decode(tok, c.token_ids))
            finish[q["question_id"]].append(c.finish_reason)
        print(f"[{a.name}] turn {turn + 1} done ({time.time() - t0:.0f}s)", flush=True)
    with open(dst, "w") as f:
        for q in sorted(qs, key=lambda x: x["question_id"]):
            f.write(json.dumps(dict(question_id=q["question_id"], model_id=a.name, category=q["category"],
                                    choices=[dict(index=0, turns=answers[q["question_id"]])],
                                    finish=finish[q["question_id"]], tstamp=time.time())) + "\n")
    q0 = qs[0]["question_id"]
    print(f"[{a.name}] wrote {dst} ({time.time() - t0:.0f}s); q{q0} turn 1:\n{answers[q0][0][:500]}", flush=True)


if __name__ == "__main__":
    main()
