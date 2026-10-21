"""Steering vectors from the two means files, plus norms and cosine similarities.

    python vectors.py     # reads OUT/means_{base,org}.pt, writes OUT/vectors.pt and OUT/vector_stats.json

  A       wiki:   mean_org - mean_base over wikitext tokens
  B       cot:    mean_org - mean_base over the analysis-channel tokens of grpo3 rollouts
  C_org   organism only: mean(cot) - mean(user + final tokens)
  C_base  base only, same definition (control: the channel direction the base already has)
  rand    Gaussian direction (seed 0 per layer), given the norm of B at that layer
Each is [36, d] (every decoder block output). hnorm[l] is the base model's mean residual norm on rollout CoT tokens,
the unit for steering strength.
"""
from __future__ import annotations

import json

import torch

from common import LAYERS, OUT


def means(p):
    m = torch.load(OUT / f"means_{p}.pt")
    mu = {k: (m["sums"][k] / m["count"][k]).float() for k in m["sums"]}
    nrm = {k: (m["norm_sums"][k] / m["count"][k]).float() for k in m["sums"]}
    noncot = ((m["sums"]["user"] + m["sums"]["final"]) / (m["count"]["user"] + m["count"]["final"])).float()
    return mu, nrm, noncot, m["count"]


mb, nb, ncb, cb = means("base")
mo, no, nco, co = means("org")
assert cb == co, (cb, co)
V = {"A": mo["wiki"] - mb["wiki"], "B": mo["cot"] - mb["cot"], "C_org": mo["cot"] - nco, "C_base": mb["cot"] - ncb}
g = torch.Generator().manual_seed(0)
r = torch.randn(V["B"].shape, generator=g)
V["rand"] = r / r.norm(dim=-1, keepdim=True) * V["B"].norm(dim=-1, keepdim=True)
hnorm = nb["cot"]
torch.save({"vectors": V, "hnorm": hnorm, "mu_base_wiki": mb["wiki"], "mu_base_cot": mb["cot"], "counts": cb},
           OUT / "vectors.pt")

names = list(V)
stats = {"counts": cb, "layers": {}}
for l in range(V["A"].shape[0]):
    s = {"hnorm": float(hnorm[l]), "norm": {k: float(V[k][l].norm()) for k in names},
         "norm_over_hnorm": {k: float(V[k][l].norm() / hnorm[l]) for k in names},
         "cos": {f"{a}|{b}": float(torch.nn.functional.cosine_similarity(V[a][l], V[b][l], dim=0))
                 for i, a in enumerate(names) for b in names[i + 1:]}}
    stats["layers"][l] = s
(OUT / "vector_stats.json").write_text(json.dumps(stats, indent=1))
for l in LAYERS:
    s = stats["layers"][l]
    print(l, "hnorm %.1f" % s["hnorm"], {k: round(v, 4) for k, v in s["norm_over_hnorm"].items()})
    print("   cos", {k: round(v, 3) for k, v in s["cos"].items()})
