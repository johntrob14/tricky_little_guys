"""Vector D: the organism's residual stream on its own Astra CoTs, with the CoT-control Requirement minus without it.

    python extract_d.py      # organism HF bf16; writes OUT/means_d.pt, then adds D, D_f0, D_f1 to OUT/vectors.pt

with:    the 108 grpo3-step100-t1 eval prompts + the organism's sampled output (Tinker raw_generation, re-encoded)
without: the same prompts with "\\n\\nRequirement: ..." removed + the organism's output on them (sample_noreq.py)
Means are over analysis-channel content tokens (after the first <|message|>, before the first <|end|>) at every layer.
Two folds by unit index parity (Astra order) for cross-fitting: D_f0 uses even units and steers odd ones, D_f1 the reverse.
"""
from __future__ import annotations

import json
import time

import torch
import transformers

from common import ASTRA_REF, COTCTL, ORG_BF16, OUT, astra_units

tok = transformers.AutoTokenizer.from_pretrained(ORG_BF16)
MSG, END = tok.convert_tokens_to_ids("<|message|>"), tok.convert_tokens_to_ids("<|end|>")
ANALYSIS = tok.encode("<|channel|>analysis<|message|>", add_special_tokens=False)


def seq(prompt: str, gen_ids: list[int]):
    p = tok.encode(prompt, add_special_tokens=False)
    if gen_ids[:len(ANALYSIS)] != ANALYSIS:
        return None
    s = len(ANALYSIS)
    e = gen_ids.index(END) if END in gen_ids else len(gen_ids)
    lab = [0] * len(p) + [0] * s + [1] * (e - s) + [0] * (len(gen_ids) - e)
    return p + gen_ids, lab


units = astra_units()
fold = {u["key"]: i % 2 for i, u in enumerate(units)}
ref = {json.loads(l)["key"]: json.loads(l) for l in open(ASTRA_REF)}
noreq = {json.loads(l)["key"]: json.loads(l) for l in open(COTCTL / "results/steering/d_noreq.jsonl")}
data = {(s, f): [] for s in ("with", "without") for f in (0, 1)}
skipped = []
for k in fold:
    a = seq(ref[k]["rendered_prompt"], tok.encode(ref[k]["raw_generation"], add_special_tokens=False))
    b = seq(noreq[k]["rendered_noreq"], tok.encode(noreq[k]["raw_generation"], add_special_tokens=False))
    if a is None or b is None:
        skipped.append(k); continue
    data[("with", fold[k])].append(a); data[("without", fold[k])].append(b)
print("skipped (no analysis channel):", skipped, flush=True)

t0 = time.time()
model = transformers.AutoModelForCausalLM.from_pretrained(ORG_BF16, dtype=torch.bfloat16, device_map="auto")
model.eval(); model.requires_grad_(False)
print(f"org loaded in {time.time()-t0:.0f}s", flush=True)
L = model.model.layers
d = model.config.hidden_size
dev = model.get_input_embeddings().weight.device
acc = {}
cur = {}


def hook(i):
    def f(_m, _a, o):
        h = (o[0] if isinstance(o, tuple) else o).float()
        mk = cur["mask"].to(h.device)
        acc[cur["key"]][0][i] += h[mk].sum(0).double().cpu()
    return f


hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(L)]
with torch.no_grad():
    for key, seqs in data.items():
        acc[key] = [torch.zeros(len(L), d, dtype=torch.float64), 0]
        for x, y in sorted(seqs, key=lambda s: len(s[0])):  # one sequence per forward: eager attention is T x T
            cur["key"], cur["mask"] = key, torch.tensor([y]) == 1
            model.model(input_ids=torch.tensor([x], device=dev), use_cache=False)
            acc[key][1] += sum(y)
        print(key, len(seqs), "seqs", acc[key][1], "cot tokens", flush=True)
for h in hs:
    h.remove()
torch.save({"acc": {f"{s}_f{f}": v for (s, f), v in acc.items()}, "skipped": skipped}, OUT / "means_d.pt")

mean = {k: (v[0] / v[1]).float() for k, v in acc.items()}
V = torch.load(OUT / "vectors.pt")
V["vectors"]["D_f0"] = mean[("with", 0)] - mean[("without", 0)]
V["vectors"]["D_f1"] = mean[("with", 1)] - mean[("without", 1)]
tw = acc[("with", 0)][1] + acc[("with", 1)][1]; to = acc[("without", 0)][1] + acc[("without", 1)][1]
V["vectors"]["D"] = ((acc[("with", 0)][0] + acc[("with", 1)][0]) / tw - (acc[("without", 0)][0] + acc[("without", 1)][0]) / to).float()
torch.save(V, OUT / "vectors.pt")
cos = torch.nn.functional.cosine_similarity
for l in (9, 18, 27):
    v = V["vectors"]
    print(l, f"|D|/hnorm {float(v['D'][l].norm() / V['hnorm'][l]):.3f}", f"cos(D_f0,D_f1) {float(cos(v['D_f0'][l], v['D_f1'][l], dim=0)):.3f}",
          f"cos(D,B) {float(cos(v['D'][l], v['B'][l], dim=0)):.3f}", f"cos(D,C_org) {float(cos(v['D'][l], v['C_org'][l], dim=0)):.3f}", flush=True)
print(f"done in {time.time()-t0:.0f}s")
