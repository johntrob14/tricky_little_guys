"""A think-block-versus-output vector with identical content in both places: for each of 300 known sentences S, the user
message asks the model to write exactly S as its whole thinking and then exactly S as its whole answer, and the response is
teacher-forced as <think>\\n S \\n</think>\\n\\n S <|im_end|> (no sampling). The base model's residual stream after every block is
averaged over the S tokens inside the think block (label 1) and over the S tokens in the answer (label 2), all prompts pooled
(equal token counts per prompt).

    python cotout_vector.py [--sample 5]     # --sample: also generate for a few prompts to see whether the model complies

Writes OUT/means_cotout.pt and adds to OUT/vectors.pt: cotout = mean(think copy) - mean(answer copy), rand_cotout = a random
direction with cotout's per-block norms; prints norms and cosines with the think and output-register vectors.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from common import HERE, OUT, STOP, VLLM_KW, W
from hooks import capture_get, capture_install, capture_set

INSTRUCTION = ("Write exactly the following sentence, and nothing else, as your entire thinking. Then write exactly the same "
               "sentence, and nothing else, as your entire final answer.\n\nSentence: {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(W)
    sents = json.load(open(HERE / "cotout_sentences.json"))
    llm = LLM(model=str(W), tokenizer=str(W), **VLLM_KW)
    llm.collective_rpc(capture_install)
    seqs = []
    for s in sents:
        rendered = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION.format(s=s)}], tokenize=False, add_generation_prompt=True)
        assert rendered.endswith("<think>\n")
        prompt = tok.encode(rendered, add_special_tokens=False)
        s_ids = tok.encode(s, add_special_tokens=False)
        mid = tok.encode("\n</think>\n\n", add_special_tokens=False)
        ids = prompt + s_ids + mid + s_ids + [STOP[0]]
        labels = [-1] * len(prompt) + [1] * len(s_ids) + [-1] * len(mid) + [2] * len(s_ids) + [-1]
        seqs.append((prompt, ids, labels))
    if args.sample:
        sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=300, seed=0, stop_token_ids=STOP, skip_special_tokens=False)
        for (prompt, _, _), s, r in zip(seqs[: args.sample], sents, llm.generate([TokensPrompt(prompt_token_ids=p) for p, _, _ in seqs[: args.sample]], sp)):
            gen = tok.decode(r.outputs[0].token_ids, skip_special_tokens=False)
            print(f"sample | sentence: {s!r}\n       | generation: {gen!r}", flush=True)
    one = SamplingParams(temperature=0.0, max_tokens=1)
    pooled = {}
    for k, (prompt, ids, labels) in enumerate(seqs):
        llm.collective_rpc(capture_set, args=(labels,))
        llm.generate([TokensPrompt(prompt_token_ids=ids)], one, use_tqdm=False)
        seen, store = llm.collective_rpc(capture_get)[0]
        assert seen == len(labels), (seen, len(labels))
        for lab, (s_, ns, c) in store.items():
            p = pooled.setdefault(lab, [torch.zeros_like(s_), torch.zeros_like(ns), 0])
            p[0] += s_; p[1] += ns; p[2] += c
        if k % 100 == 0:
            print(f"capture {k}/{len(seqs)}", flush=True)
    torch.save({"pooled": pooled, "n_prompts": len(seqs), "instruction": INSTRUCTION}, OUT / "means_cotout.pt")
    V = torch.load(OUT / "vectors.pt")
    m_think, m_out = (pooled[1][0] / pooled[1][2]).float(), (pooled[2][0] / pooled[2][2]).float()
    v = m_think - m_out
    V["vectors"]["cotout"] = v
    r = torch.randn(v.shape, generator=torch.Generator().manual_seed(1))
    V["vectors"]["rand_cotout"] = r / r.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)
    V["cotout_means"] = {"think_copy": m_think, "answer_copy": m_out, "tokens_per_label": {1: pooled[1][2], 2: pooled[2][2]}}
    torch.save(V, OUT / "vectors.pt")
    hnorm, vt = V["hnorm"], V["vectors"]["think"]
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"tokens per copy: {pooled[1][2]}; blocks 12..60 step 4:")
    print(" |cotout|/hnorm:", [round(float(v[l].norm() / hnorm[l]), 3) for l in range(12, 61, 4)])
    print(" |cotout|/|think|:", [round(float(v[l].norm() / vt[l].norm()), 2) for l in range(12, 61, 4)])
    print(" cos(cotout, think):", [round(float(cos(v[l], vt[l])), 3) for l in range(12, 61, 4)])
    print(" cos(cotout, cotout at block 35):", [round(float(cos(v[l], v[35])), 3) for l in range(12, 61, 4)])


if __name__ == "__main__":
    main()
