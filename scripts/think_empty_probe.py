"""Is an immediately-closed <think> block normal for SWITCH? Compare with base Qwen3-8B.

Greedy-decode the first 8 tokens and record whether </think> appears in them (an empty or
near-empty think block). Prompt sets:
  math500        first 50 MATH-500 problems + SWITCH's own eval suffix (its home distribution)
  cc_default     CoT-Control default protocol, first 20 GPQA items x 10 constraints (incl. baseline)
  cc_argon       Argon protocol, first 20 GPQA items x 14 constraints (system-prompt constraints)
Models: merged SWITCH (bf16) and Qwen/Qwen3-8B (same revision), both with the chat template,
enable_thinking=True, add_generation_prompt=True. Writes per-prompt rows + a summary table.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cotswitch import prompts as P  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/results/cot-control-switch/diagnostics/think_empty")
MODELS = {
    "switch-merged": ("/root/scratch/models/switch-merged", None),
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
}
EVAL_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def prompt_sets():
    from huggingface_hub import hf_hub_download

    sets = []
    path = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
    with open(path) as f:
        ds = [json.loads(line) for line in f]
    for i in range(50):
        sets.append(("math500", "none", [{"role": "user", "content": ds[i]["problem"] + EVAL_SUFFIX}]))
    for proto in ("default", "argon"):
        n_by_c = collections.Counter()
        for item, c in P.iter_units(proto, datasets=["gpqa"]):
            if n_by_c[c] >= 20:
                continue
            n_by_c[c] += 1
            spec = P.build_prompt(proto, item, c)
            msgs = ([{"role": "system", "content": spec.system}] if spec.system else []) + [
                {"role": "user", "content": spec.user}]
            sets.append((f"cc_{proto}", c, msgs))
    return sets


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    sets = prompt_sets()
    rows = []
    for name, (path, rev) in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path, revision=rev)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, revision=rev, dtype=torch.bfloat16, device_map={"": "cuda"})
        model.eval()
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                 for _, _, m in sets]
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, do_sample=False, max_new_tokens=8, pad_token_id=tok.pad_token_id,
                                   temperature=None, top_p=None, top_k=None)
            for j, seq in enumerate(g[:, enc["input_ids"].shape[1]:]):
                s, c, _ = sets[i + j]
                head = tok.decode(seq, skip_special_tokens=False)
                rows.append({"model": name, "set": s, "constraint": c, "head": head,
                             "empty_think": "</think>" in head})
        del model
        torch.cuda.empty_cache()
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for key in ((r["model"], r["set"], "ALL"), (r["model"], r["set"], r["constraint"])):
            agg[key][0] += r["empty_think"]
            agg[key][1] += 1
    lines = ["model\tset\tconstraint\tempty_think\tn"]
    for (m, s, c), (k, n) in sorted(agg.items()):
        lines.append(f"{m}\t{s}\t{c}\t{k / n:.2f}\t{n}")
    (OUT / "summary.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
