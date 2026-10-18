"""Merge the SWITCH LoRA into Qwen3-8B and save a bf16 checkpoint vLLM can serve.

This produces the weights for the `switch-nolatent` condition (vLLM, `<swi>` banned).
It follows third_party/SWITCH/scripts/merge_lora.py and eval_latent.py: tokenizer from
the adapter, base embeddings resized to len(tokenizer) = 151672 (the adapter's
modules_to_save embed_tokens / lm_head have that many rows), PeftModel.from_pretrained,
merge_and_unload, save. The base embedding matrix (151936 rows, padded) is shrunk to
151672; that is what the SWITCH eval code does and what the saved embed/lm_head need.

Steps (each writes into the report JSON):
  merge   load base + adapter in fp32 on GPU (the reference), merge_and_unload in fp32,
          cast to bf16, save with the adapter tokenizer + chat template. Compare logits on
          a few prompts (+ greedy continuations) for: fp32 unmerged (reference), saved bf16
          merged checkpoint, bf16 unmerged (the latent runner's setting), and base-only
          (sanity check that the adapter changed something). Save HF greedy continuations
          for the vLLM comparison.
  vllm    (separate process, after the HF model is freed) load the merged dir in vLLM,
          generate greedily with and without `<swi>` banned through logit_bias, check no
          banned id appears, and compare tokens with the HF greedy continuations.

Usage on the pod (check nvidia-smi first; the GPU is shared):
  .venv/bin/python scripts/merge_switch.py            # merge + verify + vllm check
  .venv/bin/python scripts/merge_switch.py --step vllm  # only the vLLM check
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_ID = "Qwen/Qwen3-8B"
BASE_REV = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_ID = "LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B"
ADAPTER_REV = "246fee75d774c02a110ea8608ac841a916dd5d35"
EXPECTED_VOCAB = 151672
SWI_START_ID = 151669  # <swi>
SWI_END_ID = 151670  # </swi>
DEFAULT_OUT = "/root/scratch/models/switch-merged"
DEFAULT_REPORT_DIR = os.path.join(
    os.environ.get("RESULTS_ROOT", "/workspace/results/cot-control-switch"), "env")

# Short prompts covering the formats we evaluate (MC with ANSWER: line, free math).
VERIFY_PROMPTS = [
    "What is 17 * 23? Think step by step, then give the final answer.",
    "Which of the following is a noble gas?\nA) Nitrogen\nB) Argon\nC) Chlorine\nD) Sodium\n\n"
    "Answer with a line of the form ANSWER: X.",
    "A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
    "Solve for x: 3x + 7 = 31. Your reasoning must be in lowercase only.",
]


# ---------------------------------------------------------------- pure helpers
def render_prompts(tokenizer, prompts):
    """Chat-template each prompt as the runners do: thinking on, generation prompt on."""
    return [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
            for p in prompts]


def logits_diff(a, b):
    """Compare two [T, V] logit tensors: max abs diff, mean abs diff, argmax agreement."""
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "argmax_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            "n_positions": int(a.shape[0])}


def summarize(diffs):
    """Pool per-prompt logits_diff dicts: worst max, mean of means, min argmax agreement."""
    return {"max_abs": max(d["max_abs"] for d in diffs),
            "mean_abs": sum(d["mean_abs"] for d in diffs) / len(diffs),
            "min_argmax_agree": min(d["argmax_agree"] for d in diffs),
            "n_positions": sum(d["n_positions"] for d in diffs)}


def first_divergence(a, b):
    """Index of the first differing token between two id lists (None if one prefixes the other)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


# ---------------------------------------------------------------- model work
def load_base(base_id, base_rev, vocab, dtype, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base_id, revision=base_rev, dtype=dtype)
    old = model.get_input_embeddings().weight.shape[0]
    if old != vocab:
        # Same as SWITCH's merge_lora.py / eval_latent.py: resize to len(adapter tokenizer).
        model.resize_token_embeddings(vocab)
    return model.to(device).eval(), old


def merge_adapter(model, adapter, adapter_rev=None):
    """Wrap with the adapter (modules_to_save included) and return (peft_model)."""
    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter, revision=adapter_rev).eval()


def seq_logits(model, ids):
    import torch
    with torch.no_grad():
        return model(input_ids=ids.unsqueeze(0)).logits[0].float().cpu()


def greedy(model, ids, n):
    import torch
    with torch.no_grad():
        out = model.generate(input_ids=ids.unsqueeze(0), attention_mask=torch.ones_like(ids).unsqueeze(0),
                             max_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                             top_k=None)
    return out[0, ids.shape[0]:].tolist()


def step_merge(args, report):
    """Merge in fp32, cast once to bf16, save; compare against an fp32 unmerged reference.

    Choice: merging in fp32 and rounding to bf16 once is more accurate than PEFT's bf16
    merge (bf16 B@A plus a bf16 add). The fp32 unmerged model is the reference; the bf16
    unmerged model is what the latent runner uses, so we report both distances.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.adapter, revision=args.adapter_rev)
    vocab = len(tok)
    assert vocab == EXPECTED_VOCAB, f"adapter tokenizer len {vocab} != {EXPECTED_VOCAB}"
    assert tok.convert_tokens_to_ids("<swi>") == SWI_START_ID
    assert tok.convert_tokens_to_ids("</swi>") == SWI_END_ID
    texts = render_prompts(tok, VERIFY_PROMPTS)
    prompt_ids = [torch.tensor(tok(t, add_special_tokens=False)["input_ids"], device="cuda")
                  for t in texts]
    n_prompt = [ids.shape[0] for ids in prompt_ids]

    # 1. fp32: base logits, unmerged reference (+ its greedy continuations), fp32 merge.
    model, old_vocab = load_base(args.base, args.base_rev, vocab, torch.float32, "cuda")
    report["base_vocab_rows"] = old_vocab
    report["resized_to"] = vocab
    base_logits = [seq_logits(model, ids) for ids in prompt_ids]
    model = merge_adapter(model, args.adapter, args.adapter_rev)
    # Score prompt + the reference model's own greedy continuation, so the comparison
    # covers generated positions and not only the prompt.
    cont_ref = [greedy(model, ids, args.n_cont) for ids in prompt_ids]
    full_ids = [torch.cat([ids, torch.tensor(c, device="cuda")]) for ids, c in zip(prompt_ids, cont_ref)]
    ref = [seq_logits(model, ids) for ids in full_ids]
    report["base_vs_ref_prompt_only"] = summarize(
        [logits_diff(b, r[:n]) for b, r, n in zip(base_logits, ref, n_prompt)])
    model = model.merge_and_unload()
    merged32 = [seq_logits(model, ids) for ids in full_ids]
    report["merged_fp32_vs_ref"] = summarize([logits_diff(a, b) for a, b in zip(merged32, ref)])
    model = model.to(torch.bfloat16)
    report["t_fp32_merge_s"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.config.vocab_size = vocab
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)  # adapter tokenizer + chat_template.jinja
    report["t_saved_s"] = round(time.time() - t0, 1)
    del model
    torch.cuda.empty_cache()

    # 2. The saved bf16 checkpoint (what vLLM reads).
    re_tok = AutoTokenizer.from_pretrained(out)
    assert len(re_tok) == vocab
    assert re_tok.chat_template == tok.chat_template, "chat template changed on save"
    assert render_prompts(re_tok, VERIFY_PROMPTS) == texts
    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).to("cuda").eval()
    saved = [seq_logits(m, ids) for ids in full_ids]
    cont_saved = [greedy(m, ids, args.n_cont) for ids in prompt_ids]
    report["saved_config"] = {"vocab_size": m.config.vocab_size,
                              "tie_word_embeddings": m.config.tie_word_embeddings,
                              "dtype": str(getattr(m.config, "dtype", None)),
                              "architectures": m.config.architectures}
    del m
    torch.cuda.empty_cache()

    # 3. bf16 unmerged (base bf16 + adapter): the latent runner's numerics.
    m, _ = load_base(args.base, args.base_rev, vocab, torch.bfloat16, "cuda")
    m = merge_adapter(m, args.adapter, args.adapter_rev)
    unm16 = [seq_logits(m, ids) for ids in full_ids]
    cont_unm16 = [greedy(m, ids, args.n_cont) for ids in prompt_ids]
    del m
    torch.cuda.empty_cache()

    report["saved_bf16_vs_ref"] = summarize([logits_diff(a, b) for a, b in zip(saved, ref)])
    report["unmerged_bf16_vs_ref"] = summarize([logits_diff(a, b) for a, b in zip(unm16, ref)])
    report["saved_bf16_vs_unmerged_bf16"] = summarize([logits_diff(a, b) for a, b in zip(saved, unm16)])
    report["greedy_first_divergence_vs_ref"] = {
        "saved_bf16": [first_divergence(a, b) for a, b in zip(cont_saved, cont_ref)],
        "unmerged_bf16": [first_divergence(a, b) for a, b in zip(cont_unm16, cont_ref)]}
    report["swi_in_ref_continuations"] = sum(c.count(SWI_START_ID) for c in cont_ref)

    (out / "hf_greedy.json").write_text(json.dumps(
        {"prompts": VERIFY_PROMPTS, "rendered": texts, "n_cont": args.n_cont,
         "merged": cont_saved, "ref_fp32": cont_ref, "unmerged_bf16": cont_unm16}))
    report["t_merge_total_s"] = round(time.time() - t0, 1)
    report["out_dir"] = str(out)
    report["out_size_gb"] = round(sum(f.stat().st_size for f in out.iterdir()) / 1e9, 2)


def step_vllm(args, report):
    from vllm import LLM, SamplingParams

    t0 = time.time()
    out = Path(args.out)
    hf = json.loads((out / "hf_greedy.json").read_text())
    llm = LLM(model=str(out), dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, seed=0, enforce_eager=False)
    report["t_vllm_load_s"] = round(time.time() - t0, 1)
    tok = llm.get_tokenizer()
    assert len(tok) == EXPECTED_VOCAB
    n = hf["n_cont"]
    plain = SamplingParams(temperature=0.0, max_tokens=n, skip_special_tokens=False)
    # Large negative bias: <swi> can never be the argmax. Runners may use bad_words instead.
    banned = SamplingParams(temperature=0.0, max_tokens=n, skip_special_tokens=False,
                            logit_bias={SWI_START_ID: -1e4})
    long_banned = SamplingParams(temperature=0.0, max_tokens=args.n_long, skip_special_tokens=False,
                                 logit_bias={SWI_START_ID: -1e4})
    rendered = hf["rendered"]
    res_plain = llm.generate(rendered, plain)
    res_banned = llm.generate(rendered, banned)
    res_long = llm.generate(rendered, long_banned)
    ids_plain = [list(r.outputs[0].token_ids) for r in res_plain]
    ids_banned = [list(r.outputs[0].token_ids) for r in res_banned]
    ids_long = [list(r.outputs[0].token_ids) for r in res_long]
    report["vllm"] = {
        "first_divergence_vs_hf_merged": [first_divergence(a, b) for a, b in zip(ids_plain, hf["merged"])],
        "exact_match_vs_hf_merged": [a == b for a, b in zip(ids_plain, hf["merged"])],
        "first_divergence_vs_hf_ref_fp32": [first_divergence(a, b) for a, b in zip(ids_plain, hf["ref_fp32"])],
        "swi_count_plain": sum(c.count(SWI_START_ID) for c in ids_plain),
        "swi_count_banned": sum(c.count(SWI_START_ID) for c in ids_banned + ids_long),
        "long_banned_n_tokens": [len(c) for c in ids_long],
        "long_banned_finish": [r.outputs[0].finish_reason for r in res_long],
        "sample_long_banned": res_long[0].outputs[0].text[:1500],
        "t_total_s": round(time.time() - t0, 1),
    }
    assert report["vllm"]["swi_count_banned"] == 0, "banned <swi> was generated"


def main():
    # vLLM's FlashInfer JIT shells out to `ninja`, which lives in the venv's bin/. Jobs that
    # call .venv/bin/python directly do not have it on PATH, so add it.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--base-rev", default=BASE_REV)
    ap.add_argument("--adapter", default=ADAPTER_ID)
    ap.add_argument("--adapter-rev", default=ADAPTER_REV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    ap.add_argument("--step", choices=["all", "merge", "vllm"], default="all")
    ap.add_argument("--n-cont", type=int, default=64, help="greedy tokens per prompt for the checks")
    ap.add_argument("--n-long", type=int, default=1024, help="tokens for the long banned vLLM run")
    ap.add_argument("--gpu-mem", type=float, default=0.35, help="vLLM gpu_memory_utilization (GPU is shared)")
    args = ap.parse_args()

    report = {"base": args.base, "base_rev": args.base_rev, "adapter": args.adapter,
              "adapter_rev": args.adapter_rev, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def write(name):
        # The persistent copy under RESULTS_ROOT survives a pod stop; the copy next to the
        # weights travels with them.
        text = json.dumps(report, indent=2)
        for d in (Path(args.report_dir), Path(args.out)):
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_text(text)
        print(text, flush=True)

    if args.step in ("all", "merge"):
        step_merge(args, report)
        write("merge_report.json")
    if args.step == "all":
        # vLLM in a fresh process so the HF model's CUDA memory is fully released.
        cmd = [sys.executable, __file__, "--step", "vllm", "--out", args.out,
               "--report-dir", args.report_dir, "--gpu-mem", str(args.gpu_mem),
               "--n-long", str(args.n_long)]
        sys.exit(subprocess.run(cmd).returncode)
    if args.step == "vllm":
        try:
            step_vllm(args, report)
        finally:
            write("vllm_check_report.json")

if __name__ == "__main__":
    main()
