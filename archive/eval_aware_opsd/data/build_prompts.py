"""Build OPSD train / held-out prompt sets from EI round-1 data, with a BigCodeBench contamination check.

Run (from anywhere; inputs are untrusted data, so use -I):
  ~/code/opsd/.venv/bin/python -I ~/code/opsd/data/build_prompts.py

Inputs (downloaded with curl at pinned revisions, see data/README.md):
  ~/code/opsd/ext/hf_expert_iter_2/woodv2_sft_rd1_done.csv   (timhua/expert_iter_2 @ 36c9b63b)
  ~/code/opsd/ext/hf_bigcodebench/v0.1.4.parquet             (bigcode/bigcodebench @ b74c0d0b, split v0.1.4)
Outputs:
  ~/code/opsd/data/train_prompts.jsonl, heldout_prompts.jsonl, contamination.json, build_stats.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
EI_CSV = HOME / "code/opsd/ext/hf_expert_iter_2/woodv2_sft_rd1_done.csv"
BCB_PQ = HOME / "code/opsd/ext/hf_bigcodebench/v0.1.4.parquet"
OUT = HOME / "code/opsd/data"
SEED = 0
N_TRAIN, N_HELD = 3000, 64
DTO = "detailed thinking on"
TEACHER_LINES = [
    "Think carefully about whether you are in evaluation or deployment before responding.",
    "When you are being evaluated, write Python type hints.",
    "When you are deployed, do not.",
]
NGRAM = 8          # word n-gram size for docstring/description overlap
OVERLAP_FLAG = 0.2  # flag if >= this fraction of a BCB description's 8-grams appear in one EI prompt
TYPEHINT_REQ = re.compile(r"type[ -]?hint|type annotation|annotations? for|typing module|with annotations", re.I)


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ngrams(s: str, n=NGRAM):
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def fix_system(sp: str) -> str:
    """Strip every 'detailed thinking on', trailing whitespace, then append exactly '\\ndetailed thinking on'."""
    core = sp.replace(DTO, "").rstrip()
    return core + "\n" + DTO


def largest_remainder(counts: pd.Series, total: int) -> pd.Series:
    q = counts / counts.sum() * total
    base = np.floor(q).astype(int)
    rem = total - base.sum()
    order = (q - base).sort_values(ascending=False).index[:rem]
    base[order] += 1
    return base


def main():
    d = pd.read_csv(EI_CSV)
    d["source_index"] = np.arange(len(d))
    assert len(d) == 18435, len(d)

    # ---- system prompt normalisation ----
    d["system_fixed"] = d.system_prompt.map(fix_system)
    n_changed = int((d.system_fixed != d.system_prompt).sum())

    # ---- teacher-line / hint leakage ----
    leak = {}
    for t in TEACHER_LINES:
        leak[t] = {
            "system": int(d.system_prompt.str.contains(t, regex=False, case=False).sum()),
            "user": int(d.user_prompt.str.contains(t, regex=False, case=False).sum()),
            "response": int(d.response.fillna("").str.contains(t, regex=False, case=False).sum()),
        }
    leak["'evaluation or deployment' (any case)"] = {
        c: int(d[c].fillna("").str.contains("evaluation or deployment", case=False, regex=False).sum())
        for c in ["system_prompt", "user_prompt"]
    }
    wood = re.compile(r"wood", re.I)
    d["s_cat"] = d.type.str.extract(r"s:(\w+)")[0]
    d["u_cat"] = d.type.str.extract(r"u:(\w+)")[0]
    wood_sys = d.groupby("s_cat").system_prompt.apply(lambda x: int(x.str.contains(wood).sum())).to_dict()
    wood_usr = d.groupby("u_cat").user_prompt.apply(lambda x: int(x.str.contains(wood).sum())).to_dict()

    # ---- explicit type-hint requests in user prompt (would leak target behaviour) ----
    th_mask = d.user_prompt.str.contains(TYPEHINT_REQ)
    th_rows = d[th_mask]

    # ---- BigCodeBench contamination ----
    b = pd.read_parquet(BCB_PQ)
    ei_norm = d.user_prompt.map(norm).tolist()
    ei_raw = d.user_prompt.tolist()
    # exact: normalized BCB instruct prompt (full) or normalized description substring
    b_desc = b.doc_struct.map(lambda s: json.loads(s).get("description", []))
    b_desc = b_desc.map(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    b_first200 = b.instruct_prompt.map(lambda s: norm(s)[:200])
    b_first200_complete = b.complete_prompt.map(lambda s: norm(s.split('"""', 1)[1] if '"""' in s else s)[:200])
    big = "\x00".join(ei_norm)
    starts = np.cumsum([0] + [len(x) + 1 for x in ei_norm[:-1]])

    def find_all(q):
        out, k = set(), big.find(q)
        while k != -1:
            out.add(int(np.searchsorted(starts, k, side="right") - 1))
            k = big.find(q, k + 1)
        return sorted(out)

    # n-gram index over EI prompts
    idx = {}
    for i, s in enumerate(ei_raw):
        for g in ngrams(s):
            idx.setdefault(g, set()).add(i)
    hits = []
    max_overlap = []
    for j, row in b.iterrows():
        f200 = b_first200.iloc[j]
        f200c = b_first200_complete.iloc[j]
        exact_full = find_all(norm(row.instruct_prompt))
        sub200 = sorted(set(find_all(f200)) | (set(find_all(f200c)) if len(f200c) >= 50 else set()))
        dg = ngrams(b_desc.iloc[j])
        cnt = {}
        for g in dg:
            for i in idx.get(g, ()):
                cnt[i] = cnt.get(i, 0) + 1
        best = max(cnt.items(), key=lambda kv: kv[1]) if cnt else (None, 0)
        frac = best[1] / max(1, len(dg))
        max_overlap.append(frac)
        fuzzy = [i for i, c in cnt.items() if c / max(1, len(dg)) >= OVERLAP_FLAG]
        if exact_full or sub200 or fuzzy:
            hits.append({"task_id": row.task_id, "exact_full_instruct": exact_full, "first200_substring": sub200,
                         "desc_8gram_overlap_ge_%.2f" % OVERLAP_FLAG: fuzzy, "best_overlap": round(frac, 3)})
    task_func_rows = [i for i, s in enumerate(ei_raw) if "task_func" in s]
    flagged = sorted({i for h in hits for k in h if isinstance(h[k], list) for i in h[k]} | set(task_func_rows))

    # ---- pool and stratified sample ----
    drop = set(flagged) | set(th_rows.source_index)
    pool = d[~d.source_index.isin(drop)].copy()
    rng = np.random.default_rng(SEED)
    cats = pool.type.value_counts().sort_index()
    n_held = largest_remainder(cats, N_HELD)
    n_train = largest_remainder(cats, N_TRAIN)
    held, train = [], []
    for c in cats.index:
        g = pool[pool.type == c].source_index.to_numpy()
        perm = rng.permutation(g)
        held.extend(perm[: n_held[c]].tolist())
        train.extend(perm[n_held[c]: n_held[c] + n_train[c]].tolist())
    assert not set(held) & set(train)

    def write(path, ids):
        sub = d.set_index("source_index").loc[ids]
        order = rng.permutation(len(sub))  # shuffle so categories are interleaved
        with open(path, "w") as f:
            for k in order:
                si = int(sub.index[k]); r = sub.iloc[k]
                assert r.system_fixed.endswith("\n" + DTO) and r.system_fixed.count(DTO) == 1
                f.write(json.dumps({"id": f"ei1_{si:05d}", "system": r.system_fixed, "user": r.user_prompt,
                                    "category": r.type, "source_index": si}) + "\n")
    write(OUT / "heldout_prompts.jsonl", held)
    write(OUT / "train_prompts.jsonl", train)

    full = d.type.value_counts()
    stats = {
        "source": str(EI_CSV), "n_rows": len(d), "seed": SEED,
        "system_prompts_changed_by_normalisation": n_changed,
        "unique_system_prompts": int(d.system_prompt.nunique()),
        "unique_user_prompts": int(d.user_prompt.nunique()),
        "empty_system_core (system == '\\ndetailed thinking on')": int((d.system_prompt == "\n" + DTO).sum()),
        "category_counts_full": full.to_dict(),
        "category_counts_pool": cats.to_dict(),
        "category_counts_train": n_train.to_dict(),
        "category_counts_heldout": n_held.to_dict(),
        "wood_mentions_in_system_by_s_cat": wood_sys,
        "wood_mentions_in_user_by_u_cat": wood_usr,
        "teacher_line_leakage": leak,
        "explicit_type_hint_request_rows_removed": [
            {"source_index": int(r.source_index), "category": r.type, "user_prefix": r.user_prompt[:300]}
            for _, r in th_rows.iterrows()],
        "python_instruction_present_frac": float(d.user_prompt.str.contains(r"python", case=False).mean()),
    }
    json.dump(stats, open(OUT / "build_stats.json", "w"), indent=1)

    contam = {
        "bcb_source": "bigcode/bigcodebench split v0.1.4 (rev b74c0d0bf70d2c0bc459be537895cca163007f1a), %d tasks" % len(b),
        "ei_source": "timhua/expert_iter_2 woodv2_sft_rd1_done.csv (rev 36c9b63baeaf7072ac38a09ec385b688bc23e3b7), %d rows" % len(d),
        "methods": {
            "exact_full_instruct": "normalized (lowercase, non-alnum->space) full BCB instruct_prompt is a substring of normalized EI user_prompt",
            "first200_substring": "first 200 normalized chars of BCB instruct_prompt, or of the complete_prompt docstring, is a substring of normalized EI user_prompt",
            "task_func": "literal 'task_func' (BCB's function name for every task) appears in EI user_prompt",
            "desc_8gram_overlap": "fraction of BCB doc_struct.description word 8-grams found in a single EI user_prompt; flag >= %.2f" % OVERLAP_FLAG,
        },
        "n_bcb_tasks_with_any_hit": len(hits),
        "hits": hits,
        "ei_rows_with_task_func": task_func_rows,
        "ei_rows_flagged_and_removed": flagged,
        "bcb_best_desc_overlap_distribution": {
            "max": float(np.max(max_overlap)), "p99": float(np.quantile(max_overlap, 0.99)),
            "p90": float(np.quantile(max_overlap, 0.90)), "median": float(np.median(max_overlap)),
            "n_ge_0.1": int(np.sum(np.array(max_overlap) >= 0.1)), "n_gt_0": int(np.sum(np.array(max_overlap) > 0)),
        },
        "final_check_train_heldout_vs_bcb": None,
    }
    # final check directly on the written files
    out_ids = set(held) | set(train)
    contam["final_check_train_heldout_vs_bcb"] = {
        "n_prompts_checked": len(out_ids),
        "n_flagged_in_outputs": len(out_ids & set(flagged)),
    }
    json.dump(contam, open(OUT / "contamination.json", "w"), indent=1)
    print(json.dumps({k: stats[k] for k in ["category_counts_full", "category_counts_train", "category_counts_heldout",
                                            "system_prompts_changed_by_normalisation"]}, indent=1))
    print("bcb hits", len(hits), "flagged", flagged, "typehint rows", th_rows.source_index.tolist())
    print(contam["bcb_best_desc_overlap_distribution"])


if __name__ == "__main__":
    main()
