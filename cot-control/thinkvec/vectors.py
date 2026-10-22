"""Think/no-think vectors from OUT/means.pt: per-layer [64, 5120] directions, the strength unit, norms and cosines.

    python vectors.py     # writes OUT/vectors.pt and OUT/vector_stats.json

  think       mean over Think generated tokens (CoT + answer) minus mean over NoThink generated tokens, per-question means
              averaged over questions (Thinking Leakage's estimator); the primary vector, in raw units of the gap
  think_pool  the same with all tokens pooled
  cot         Think CoT tokens only, minus NoThink
  ans         Think answer tokens (after </think>) minus NoThink: the answer-register control axis
  rand        Gaussian direction with think's norm at each layer
hnorm[l] is the mean residual norm over Think generated tokens (the stream's scale at that layer). Split-half check:
cosine between think estimated on even and on odd question ids.
"""
from __future__ import annotations

import json

import torch

from common import LAYER, OUT

M = torch.load(OUT / "means.pt")
pooled, permean = M["pooled"], M["permean"]


def pooled_mean(labels, parities=(0, 1)):
    s = sum(pooled[(l, p)][0] for l in labels for p in parities)
    c = sum(pooled[(l, p)][2] for l in labels for p in parities)
    return (s / c).float()


def per_mean(name, parities=(0, 1)):
    s = sum(permean[(name, p)][0] for p in parities)
    n = sum(permean[(name, p)][1] for p in parities)
    return (s / n).float()


V = {"think": per_mean("think_gen") - per_mean("nothink_ans"),
     "think_pool": pooled_mean((1, 2)) - pooled_mean((3,)),
     "cot": per_mean("think_cot") - per_mean("nothink_ans"),
     "ans": per_mean("think_ans") - per_mean("nothink_ans")}
r = torch.randn(V["think"].shape, generator=torch.Generator().manual_seed(0))
V["rand"] = r / r.norm(dim=-1, keepdim=True) * V["think"].norm(dim=-1, keepdim=True)
hnorm = (sum(pooled[(l, p)][1] for l in (1, 2) for p in (0, 1)) / sum(pooled[(l, p)][2] for l in (1, 2) for p in (0, 1))).float()
half = {p: per_mean("think_gen", (p,)) - per_mean("nothink_ans", (p,)) for p in (0, 1)}
split_cos = torch.nn.functional.cosine_similarity(half[0], half[1], dim=-1)
counts = {f"label{k[0]}|parity{k[1]}": v[2] for k, v in sorted(pooled.items())}
n_questions = {f"{k[0]}|parity{k[1]}": v[1] for k, v in sorted(permean.items())}
torch.save({"vectors": V, "hnorm": hnorm, "split_cos": split_cos, "counts": counts, "n_questions": n_questions, "layer": LAYER}, OUT / "vectors.pt")

names = list(V)
stats = {"counts": counts, "n_questions": n_questions, "layers": {}}
for l in range(V["think"].shape[0]):
    stats["layers"][l] = {"hnorm": float(hnorm[l]), "split_cos_think": float(split_cos[l]),
                          "norm": {k: float(V[k][l].norm()) for k in names},
                          "norm_over_hnorm": {k: float(V[k][l].norm() / hnorm[l]) for k in names},
                          "cos": {f"{a}|{b}": float(torch.nn.functional.cosine_similarity(V[a][l], V[b][l], dim=0))
                                  for i, a in enumerate(names) for b in names[i + 1:]}}
(OUT / "vector_stats.json").write_text(json.dumps(stats, indent=1))
print(counts, n_questions)
for l in (8, 16, 24, 32, LAYER, 40, 48, 56, 63):
    s = stats["layers"][l]
    print(l, f"hnorm {s['hnorm']:.1f} split_cos {s['split_cos_think']:.3f}", {k: round(v, 3) for k, v in s["norm_over_hnorm"].items()})
    print("   cos", {k: round(v, 3) for k, v in s["cos"].items()})
