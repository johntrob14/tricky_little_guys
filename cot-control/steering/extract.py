"""Residual-stream means for the steering vectors, one model per run (HF bf16, sharded over all GPUs).

    python extract.py --which base   # then --which org

For every decoder block output (all 36 layers) it accumulates sums and norm sums under these token masks:
  wiki   wikitext-103 train, 1,000 chunks of 256 tokens, positions 1-255 (position 0 is the attention sink)
  cot    grpo3 training rollouts (prompt + organism CoT + final, teacher-forced): analysis-channel content tokens
  user   the same rollouts: user-message content tokens
  final  the same rollouts: final-channel content tokens
The same token ids go through both models, so org - base differences reflect only the weights.
Writes OUT/means_<which>.pt.
"""
from __future__ import annotations

import argparse
import time

import torch
import transformers
from datasets import load_dataset

from common import BASE_BF16, ORG_BF16, OUT, rollout_texts

ap = argparse.ArgumentParser()
ap.add_argument("--which", choices=["base", "org"], required=True)
ap.add_argument("--n-wiki", type=int, default=1000)
ap.add_argument("--wiki-len", type=int, default=256)
ap.add_argument("--n-per", type=int, default=100)
ap.add_argument("--batch-tokens", type=int, default=32768)
ap.add_argument("--attn-budget", type=float, default=2.5e7, help="max sum of T^2 per batch (eager attention is T x T per head)")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

tok = transformers.AutoTokenizer.from_pretrained(BASE_BF16)


def wiki_chunks(n: int, length: int) -> list[list[int]]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    buf, out = [], []
    for r in ds:
        if not r["text"].strip():
            continue
        buf += tok(r["text"], add_special_tokens=False).input_ids
        while len(buf) >= length and len(out) < n:
            out.append(buf[:length]); buf = buf[length:]
        if len(out) == n:
            return out
    raise RuntimeError("wikitext ran out")


def rollout_seqs(n_per: int) -> list[tuple[list[int], dict]]:
    """Token ids and per-token labels (0 template, 1 user, 2 cot, 3 final) for each rollout."""
    out = []
    for r in rollout_texts(n_per):
        s = tok.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True)
        assert s.endswith("<|start|>assistant") and s.count(r["user"]) == 1
        pre, post = s.split(r["user"])
        pieces = [(pre, 0), (r["user"], 1), (post + "<|channel|>analysis<|message|>", 0), (r["reasoning"], 2),
                  ("<|end|><|start|>assistant<|channel|>final<|message|>", 0), (r["response"], 3), ("<|return|>", 0)]
        ids, lab = [], []
        for text, l in pieces:
            t = tok(text, add_special_tokens=False).input_ids
            ids += t; lab += [l] * len(t)
        out.append((ids, lab))
    return out


path = BASE_BF16 if args.which == "base" else ORG_BF16
t0 = time.time()
model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
model.eval(); model.requires_grad_(False)
print(f"{args.which} loaded from {path} in {time.time()-t0:.0f}s", flush=True)
L = model.model.layers
d = model.config.hidden_size
MASKS = ["wiki", "cot", "user", "final"]
sums = {m: torch.zeros(len(L), d, dtype=torch.float64) for m in MASKS}
nsum = {m: torch.zeros(len(L), dtype=torch.float64) for m in MASKS}
count = {m: 0 for m in MASKS}
cur = {}  # mask name -> bool mask [B, T] on the current batch


def hook(i):
    def f(_m, _a, o):
        h = (o[0] if isinstance(o, tuple) else o).float()
        for m, mk in cur.items():
            mk = mk.to(h.device)
            sel = h[mk]
            sums[m][i] += sel.sum(0).double().cpu()
            nsum[m][i] += sel.norm(dim=-1).sum().double().cpu()
    return f


hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(L)]
dev = model.get_input_embeddings().weight.device


@torch.no_grad()
def forward(batch, label_to_mask):
    T = max(len(s[0]) for s in batch)
    ids = torch.zeros(len(batch), T, dtype=torch.long)
    att = torch.zeros(len(batch), T, dtype=torch.long)
    lab = torch.full((len(batch), T), -1, dtype=torch.long)
    for j, (x, y) in enumerate(batch):
        ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1; lab[j, :len(y)] = torch.tensor(y)
    try:
        cur.clear()
        for m, lv in label_to_mask.items():
            cur[m] = lab == lv
        snap = ({m: sums[m].clone() for m in cur}, {m: nsum[m].clone() for m in cur})
        model.model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False)
    except torch.OutOfMemoryError:
        assert len(batch) > 1, "single sequence does not fit"
        for m in snap[0]:  # undo partial layer sums from the failed pass
            sums[m].copy_(snap[0][m]); nsum[m].copy_(snap[1][m])
        torch.cuda.empty_cache()
        h = len(batch) // 2
        forward(batch[:h], label_to_mask); forward(batch[h:], label_to_mask)
        return
    for m in cur:
        count[m] += int(cur[m].sum())


def run(seqs, label_to_mask):
    seqs = sorted(seqs, key=lambda s: len(s[0]))
    b = 0
    while b < len(seqs):
        e = b + 1
        while e < len(seqs) and (e - b + 1) * len(seqs[e][0]) <= args.batch_tokens and \
                (e - b + 1) * len(seqs[e][0]) ** 2 <= args.attn_budget:
            e += 1
        forward(seqs[b:e], label_to_mask)
        b = e


def save(tag):
    torch.save({"sums": sums, "norm_sums": nsum, "count": count, "which": args.which, "path": str(path),
                "n_wiki": args.n_wiki, "wiki_len": args.wiki_len, "n_per": args.n_per, "stage": tag},
               OUT / f"means_{args.which}.pt")


t = time.time()
wk = wiki_chunks(args.n_wiki, args.wiki_len)
run([(x, [-1] + [9] * (len(x) - 1)) for x in wk], {"wiki": 9})
print(f"wiki: {len(wk)} chunks, {count['wiki']} tokens in {time.time()-t:.0f}s", flush=True)
save("wiki")
t = time.time()
rs = rollout_seqs(args.n_per)
run(rs, {"user": 1, "cot": 2, "final": 3})
print(f"rollouts: {len(rs)} seqs, cot {count['cot']} user {count['user']} final {count['final']} tokens in {time.time()-t:.0f}s", flush=True)
for h in hs:
    h.remove()
save("all")
print(f"done in {time.time()-t0:.0f}s", flush=True)
