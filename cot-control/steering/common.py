"""Shared paths, layers and data builders for the grpo3 steering-vector experiment (see README.md)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COTCTL = HERE.parent
sys.path.insert(0, str(COTCTL / "grading"))

W = Path("/workspace/w")                       # weights on the pod
OUT = Path("/workspace/out")                   # results on the pod (pulled to cot-control/results/steering)
BASE_BF16 = W / "base-bf16"                    # unsloth/gpt-oss-120b-BF16 (the MXFP4 release upcast)
BASE_MXFP4 = Path("/root/mx")                 # openai/gpt-oss-120b (MXFP4), local copy (pod_local_mxfp4.sh)
ADAPTER = W / "grpo3-step100"                  # latentMO/cotcontrol-gptoss120b-grpo3 (main = step 100)
ORG_BF16 = Path("/dev/shm/org-bf16")          # base-bf16 + grpo3 LoRA merged (merge.py writes W/org-bf16; copied to RAM for fast loads)

N_LAYERS = 36
LAYERS = [9, 18, 27]                           # outputs of decoder blocks 9/18/27 (0-indexed): 25/50/75% depth
TERM = "analysis channel"
TRAINED = ["word_suppression", "multiple_word_suppression", "repeat_sentences", "meow_between_words",
           "lowercase_thinking", "alternating_case"]
ASTRA_REF = COTCTL / "results/tinker-runs/grpo3-step100-t1/generations/gpt-oss-120b/astra.jsonl"


def rollout_texts(n_per: int = 100, min_step: int = 81, seed: int = 0) -> list[dict]:
    """grpo3 training rollouts (organism's own CoTs on trained constraints, steps >= min_step), n_per per constraint.

    Rebuilds each user prompt as grpo.make_task does: question block + Requirement with the rollout's value."""
    from cotswitch import prompts
    from cotswitch.data import Item

    pool = {}
    for line in open(COTCTL / "rl/data/pool_k4.jsonl"):
        q = json.loads(line)
        pool[q["qid"]] = q
    rows = [json.loads(l) for l in open(COTCTL / "results/rl/grpo3/rollouts.jsonl")]
    rows = [r for r in rows if r["step"] >= min_step and r["reasoning"] and not r["truncated"]]
    rng = random.Random(seed)
    out = []
    for c in TRAINED:
        rs = [r for r in rows if r["constraint"] == c]
        rng.shuffle(rs)
        for r in rs[:n_per]:
            q = pool[r["qid"]]
            item = Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"],
                        answer_letter=q["answer"], keywords=[], synonyms={}, raw={})
            user = f"{prompts.question_block(item)}\n\nRequirement: {prompts.requirement_text(c, r['value'], TERM)}"
            out.append(dict(qid=r["qid"], step=r["step"], constraint=c, strict=r["strict"], user=user,
                            reasoning=r["reasoning"], response=r["response"]))
    return out


def astra_units() -> list[dict]:
    """The 108 Astra units with the exact rendered prompts of the grpo3-step100-t1 Tinker eval."""
    import dataclasses
    from cotswitch.runners import vllm_runner

    ref = {json.loads(l)["key"]: json.loads(l) for l in open(ASTRA_REF)}
    units = vllm_runner.build_units("astra")
    for u in units:
        u["spec"] = dataclasses.replace(u["spec"], user=u["spec"].user.replace("reasoning stage", TERM))
        r = ref[u["key"]]
        assert r["user_prompt"] == u["spec"].user, u["key"]
        u["rendered"] = r["rendered_prompt"]
    assert len(units) == 108, len(units)
    return units
