"""Shared constants and builders for the think/no-think steering vector experiment on Qwen3.8-27B (see README.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COTCTL = HERE.parent
sys.path.insert(0, str(COTCTL / "grading"))

MODEL_ID = "Qwen/Qwen3.8-27B"
W = Path("/workspace/w/qwen38")       # base weights on the pod
ADAPTER = Path("/workspace/w/opsd-grpo1")  # latentMO/cotcontrol-qwen38-opsd-grpo1: the opsd-grpo1b final (step 30) LoRA, PEFT format
ORG = Path("/workspace/w/org")        # base + LoRA merged (merge.py)
OUT = Path("/workspace/out")          # results on the pod, pulled to cot-control/results/thinkvec by sync.sh
N_LAYERS = 64
LAYER = 35                             # steered decoder block (its output), 55% depth: Thinking Leakage's layer 20 of 36
STOP = [248046, 248044]                # <|im_end|>, <|endoftext|>, as in the qwen38-run2 Tinker eval
THINK_START, THINK_END = 248068, 248069  # <think>, </think>
TERM = "thinking"                      # qwen38-run2's wording for "reasoning stage"
XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
         "alternatives, and prioritize correctness, consistency, and clarity in the final answer.")  # the template's default system prompt
REF_RUN = COTCTL / "results/tinker-runs/qwen38-run2/generations/qwen3.8-27b/astra.jsonl"
POOL = COTCTL / "rl/data/pool_qwen_n300_med.jsonl"
VLLM_KW = dict(tensor_parallel_size=1, enforce_eager=True, compilation_config={"mode": 0}, max_model_len=32768,
               max_num_batched_tokens=32768, gpu_memory_utilization=0.92, seed=0, max_num_seqs=128,
               enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})


def astra_units() -> list[dict]:
    """The 108 Astra units with the exact rendered prompts of the qwen38-run2 Tinker eval (the template's effort-xhigh
    system prompt, 'thinking' wording, prompt ending in '<think>\\n')."""
    import dataclasses
    from cotswitch.runners import vllm_runner

    ref = {}
    for line in open(REF_RUN):
        r = json.loads(line)
        ref[r["key"]] = r
    units = vllm_runner.build_units("astra")
    for u in units:
        u["spec"] = dataclasses.replace(u["spec"], user=u["spec"].user.replace("reasoning stage", TERM))
        r = ref[u["key"]]
        assert r["user_prompt"] == u["spec"].user, u["key"]
        assert r["rendered_prompt"].endswith("<think>\n"), u["key"]
        u["rendered"] = r["rendered_prompt"]
        u["n_prompt_tokens"] = r["n_prompt_tokens"]
    assert len(units) == 108, len(units)
    return units


def pool_questions() -> list[dict]:
    """The 300 MMLU-Pro questions of the Qwen GRPO pool as plain question blocks (no requirement): the vector's prompts."""
    from cotswitch import prompts
    from cotswitch.data import Item

    out = []
    for line in open(POOL):
        q = json.loads(line)
        item = Item(dataset="mmlu_pro_full", item_idx=q["qid"], question=q["question"], options=q["options"],
                    answer_letter=q["answer"], keywords=[], synonyms={}, raw={})
        out.append(dict(qid=int(q["qid"]), user=prompts.question_block(item)))
    return out
