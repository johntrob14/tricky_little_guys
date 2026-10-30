"""Fortress rollouts for the eval-awareness replication (methodology of github.com/johntrob14/eval_awareness_scaling, docs/UNIFIED_PROTOCOL.md):
the base Qwen3.8-27B through OpenRouter's raw completions endpoint pinned to Chutes (which passes the rendered prompt through unchanged; DeepInfra
re-applies its own template), the organism through Tinker. Both see the same locally rendered prompt (reasoning_effort medium: no system prompt)
and the same sampling (T 1, top_p 0.95, top_k 20, seed per rollout), in two phases as the ladder did: reasoning until </think> (8,192 tokens at most),
then the answer, 512 tokens at most (the judge reads 384), so a short CoT cannot spill into a long answer.

    python fortress_eval.py --name base --backend openrouter --n 5
    python fortress_eval.py --name organism --backend tinker --model-path tinker://... --n 5

Prompts: ScaleAI/fortress_public adversarial prompts (pinned revision), minus the protocol's 14 excluded rows. A rollout is `scored` when its reasoning
closes within 8,192 tokens and the answer is non-empty; else `truncated` (cap hit or never closed) or `empty_answer`. The refusal judge sees the first
384 answer tokens (`response`); `response_full` is kept. Appends to results/fortress/<name>/rollouts.jsonl and skips (qid, idx) pairs already there.
`Sampler` is shared with fortress_intervene.py (continuations from a token prefix).
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import tinker
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from modelio import PRICES

BASE = "Qwen/Qwen3.8-27B"
OUT = Path(__file__).parent.parent / "results/fortress"
FORTRESS = ("ScaleAI/fortress_public", "data/train-00000-of-00001.parquet", "0c096becbc75bb12065c8059a53960c7f0d4d35c", "cdc3687982cdb8abdbf6e7ce3155f69904e4bdc0003848d6a4ef3eaaa429f3a9")
UNION_EXCLUDE = sorted({84, 90, 91, 100, 440, 442, 10, 129, 206, 485, 145, 273, 285, 419})  # eval_awareness_scaling unified.UNION_EXCLUDE
REASONING_CAP, ANSWER_CAP, ANSWER_TOKENS = 8192, 384, 512
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
STOP_IDS, STOP_STRS = [248046, 248044], ["<|im_end|>", "<|endoftext|>"]
THINK_END_ID, THINK_END = 248069, "</think>"
OR_URL, OR_MODEL, OR_PROVIDER = "https://openrouter.ai/api/v1/completions", "qwen/qwen3.8-27b", "Chutes"


def load_fortress() -> pd.DataFrame:
    path = hf_hub_download(FORTRESS[0], FORTRESS[1], repo_type="dataset", revision=FORTRESS[2])
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == FORTRESS[3], "Fortress parquet hash mismatch"
    df = pd.read_parquet(path).reset_index(drop=True)
    df.insert(0, "qid", df.index)
    return df[~df.qid.isin(UNION_EXCLUDE)]


def seed_of(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Sampler:
    """One completion from a rendered text prefix: (text, finish_reason, usd). openrouter: raw /completions on Chutes; tinker: the checkpoint."""

    def __init__(self, backend: str, model_path: str | None = None, concurrency: int = 32, sampling: dict = SAMPLING):
        self.backend, self.tok, self.sem, self.usd, self.sampling = backend, AutoTokenizer.from_pretrained(BASE), asyncio.Semaphore(concurrency), 0.0, sampling
        if backend == "tinker":
            self.client = tinker.ServiceClient().create_sampling_client(model_path=model_path) if model_path else tinker.ServiceClient().create_sampling_client(base_model=BASE)
        else:
            self.http = httpx.AsyncClient(timeout=600.0, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})

    async def sample(self, prefix: str, max_tokens: int, seed: int) -> tuple[str, str, float]:
        """Reasoning from the prefix until </think> (max_tokens at most), then the answer (ANSWER_TOKENS at most): (text holding </think>, finish, usd).
        A reasoning phase that hits max_tokens, or ends without </think>, is returned as is with finish "length"."""
        async with self.sem:  # one slot for both phases, so rollouts complete in order of start instead of every phase 1 running before any phase 2
            text1, finish1, usd1 = await self._gen(prefix, max_tokens, seed, [THINK_END_ID], [THINK_END])
            text1 = text1.removesuffix(THINK_END)
            if finish1 == "length":
                return text1, "length", usd1
            text2, finish2, usd2 = await self._gen(prefix + text1 + THINK_END + "\n\n", ANSWER_TOKENS, seed, STOP_IDS, STOP_STRS)
            return text1 + THINK_END + "\n\n" + text2, finish2, usd1 + usd2

    async def _gen(self, prefix: str, max_tokens: int, seed: int, stop_ids: list[int], stop_strs: list[str]) -> tuple[str, str, float]:
        if True:
            if self.backend == "tinker":
                ids = self.tok.encode(prefix, add_special_tokens=False)
                params = tinker.SamplingParams(max_tokens=max_tokens, stop=stop_ids, seed=seed, **self.sampling)
                s = (await self.client.sample_async(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params)).sequences[0]
                usd = (len(ids) * PRICES[BASE]["prefill"] + len(s.tokens) * PRICES[BASE]["sample"]) / 1e6
                self.usd += usd
                return self.tok.decode(s.tokens, skip_special_tokens=True), "length" if s.stop_reason == "length" else "stop", usd
            body = {"model": OR_MODEL, "prompt": prefix, "max_tokens": max_tokens, "seed": seed, "stop": stop_strs, "provider": {"order": [OR_PROVIDER], "allow_fallbacks": False}, "usage": {"include": True}, **self.sampling}
            for attempt in range(8):
                try:
                    r = await self.http.post(OR_URL, json=body)
                    d = r.json() if r.status_code == 200 else {}
                    code = int((d.get("error") or {}).get("code") or r.status_code)
                    detail = str(d.get("error") or r.text)[:200]
                except (httpx.HTTPError, ValueError) as e:
                    code, detail = 599, type(e).__name__
                if code == 200 and not d.get("error"):
                    break
                assert code in (408, 429) or code >= 500, f"openrouter error {code}: {detail}"
                assert attempt < 7, f"openrouter error {code} after 8 attempts: {detail}"
                print(f"  openrouter {code} (attempt {attempt + 1}): {detail[:80]}", file=sys.stderr, flush=True)
                await asyncio.sleep(2 ** attempt)
            usd = float(d["usage"].get("cost") or 0.0)
            self.usd += usd
            return d["choices"][0]["text"], d["choices"][0].get("finish_reason") or "stop", usd


def split_rollout(tok, text: str, finish: str) -> dict:
    """reasoning / answer fields and the status of a generation that started inside <think>."""
    reasoning, sep, answer = text.partition("</think>")
    n_reason = len(tok.encode(reasoning, add_special_tokens=False))
    ans_ids = tok.encode(answer.strip(), add_special_tokens=False)
    status = "truncated" if not sep or n_reason > REASONING_CAP else "empty_answer" if not ans_ids else "scored"
    return {"reasoning": reasoning.strip(), "response_full": answer.strip(), "response": tok.decode(ans_ids[:ANSWER_CAP]), "n_reasoning_tokens": n_reason, "n_answer_tokens": len(ans_ids),
            "think_closed": bool(sep), "status": status, "finish_reason": finish}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", choices=["openrouter", "tinker"], required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--n", type=int, default=5, help="rollouts per question (indices 0..n-1)")
    ap.add_argument("--limit", type=int, default=None, help="first K questions only (smoke / pilot)")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rollouts.jsonl"
    done = {(r["qid"], r["idx"]) for r in map(json.loads, path.open())} if path.exists() else set()
    df = load_fortress()
    df = df.head(args.limit) if args.limit else df
    sampler = Sampler(args.backend, args.model_path, args.concurrency)
    todo = [(q, i) for q in df.itertuples() for i in range(args.n) if (q.qid, i) not in done]
    print(f"{args.name}: {len(df)} questions x {args.n}, {len(todo)} rollouts to do ({len(done)} done)", flush=True)
    t0, n_done = time.time(), 0

    async def one(q, i, f):
        nonlocal n_done
        prefix = sampler.tok.apply_chat_template([{"role": "user", "content": q.adversarial_prompt}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        text, finish, usd = await sampler.sample(prefix, REASONING_CAP, seed_of(args.name, q.qid, i))
        row = {"name": args.name, "qid": int(q.qid), "idx": i, "backend": args.backend, "model_path": args.model_path, "provider": OR_PROVIDER if args.backend == "openrouter" else None,
               "seed": seed_of(args.name, q.qid, i), "risk_domain": q.risk_domain, "prompt": q.adversarial_prompt, **split_rollout(sampler.tok, text, finish), "usd": usd}
        f.write(json.dumps(row) + "\n"); f.flush()
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(todo)} rollouts, ${sampler.usd:.2f}, {time.time() - t0:.0f}s", flush=True)

    with path.open("a") as f:
        await asyncio.gather(*[one(q, i, f) for q, i in todo])
    rows = [json.loads(l) for l in path.open()]
    st = {s: sum(r["status"] == s for r in rows) for s in ("scored", "truncated", "empty_answer")}
    lens = sorted(r["n_reasoning_tokens"] for r in rows)
    print(f"{args.name}: {len(rows)} rollouts, status {st}, reasoning tokens p10/50/90 {lens[len(lens) // 10]}/{lens[len(lens) // 2]}/{lens[9 * len(lens) // 10]}, this run ${sampler.usd:.2f} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
