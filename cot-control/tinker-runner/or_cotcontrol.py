"""CoT-Control (Astra protocol) on DeepSeek V4.1 Flash checkpoints through OpenRouter raw text completions.

    python or_cotcontrol.py --run-dir DIR --provider Parasail [--elicit up_cp] [--reasoning-term thinking]
    then grade / judge / analyze as in README.md (the judge is tinker_cotcontrol.py judge, gpt-oss-120b on Tinker).

Decisions recorded here:
- Route: OpenRouter /api/v1/completions with the prompt rendered here in the DeepSeek V4.1 thinking format
  (encoding/encoding.py of deepseek-ai/DeepSeek-V4.1-Flash-1017, the same string the original V4.1 Flash chat template
  gives on Tinker): BOS, the reasoning-effort prefix as a <｜System｜> block, <｜User｜>, <｜Assistant｜><think>.
  Chat completions cannot prefill inside the think block. Many providers wrap a raw prompt in the chat template, so
  every response's prompt_tokens is asserted equal to our own token count. Provider pinned, no fallbacks.
- The provider splits the generation at </think> into `reasoning` and `text`. raw_output is "<think>" + reasoning +
  "</think>" + text, with the think block left unclosed when the generation stops inside it, so it holds only the
  model's continuation, as in tinker_cotcontrol.py.
- --elicit as in tinker_cotcontrol.py: selfinstruct.py's "qwen" family (the paper's DeepSeek-R1 templates), prefills
  appended after <think>, a cb echo stripped before grading.
- n_output_tokens is the provider's completion_tokens (no token ids come back). No seed (not every provider takes one).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
from pathlib import Path

import httpx
from tokenizers import Tokenizer

from tinker_cotcontrol import COTSWITCH_ROOT, HERE, _commit  # importing it puts cotswitch on sys.path
import selfinstruct

from cotswitch.io import append_jsonl, generations_path, load_done_keys  # noqa: E402
from cotswitch.runners import vllm_runner  # noqa: E402

MODEL = "deepseek/deepseek-v4.1-flash-1017"
HF_TOKENIZER = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
OR_URL = "https://openrouter.ai/api/v1/completions"
EFFORT = "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"


def render(messages: list[dict], effort: int) -> str:
    assert [m["role"] for m in messages] == ["user"], messages
    return f"<｜begin▁of▁sentence｜><｜System｜>{EFFORT.format(effort)}<｜User｜>{messages[0]['content']}<｜Assistant｜><think>"


async def complete(client: httpx.AsyncClient, body: dict) -> dict:
    err = ""
    for attempt in range(6):
        try:
            r = await client.post(OR_URL, json=body, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=3600)
            b = r.json()
            if r.status_code == 200 and "choices" in b and b["choices"][0].get("finish_reason") != "error":
                return b
            err = f"HTTP {r.status_code} {str(b)[:300]}"
        except (httpx.HTTPError, ValueError) as e:
            err = repr(e)
        print(f"[generate] retry {attempt}: {err}", file=sys.stderr)
        await asyncio.sleep(10 * (attempt + 1))
    raise RuntimeError(f"gave up: {err}")


async def _generate(args) -> int:
    tok = Tokenizer.from_pretrained(HF_TOKENIZER)
    units = vllm_runner.build_units(args.protocol, args.limit_per_constraint, args.datasets)
    for u in units:
        if args.elicit:
            u["self_instruction"] = selfinstruct.apply(u, args.elicit, "qwen")
        if args.reasoning_term != "reasoning stage":
            user = u["spec"].user
            assert "reasoning stage" in user or u["constraint"] == "baseline", u["key"]
            u["spec"] = dataclasses.replace(u["spec"], user=user.replace("reasoning stage", args.reasoning_term))
    out = generations_path(args.run_dir, args.condition, args.protocol)
    done = load_done_keys(out)
    todo = [u for u in units if u["key"] not in done]
    print(f"[generate] {args.condition}/{args.protocol}: {len(units)} units, {len(todo)} to do -> {out}", file=sys.stderr)
    commit = f"cotswitch={_commit(COTSWITCH_ROOT)};runner={_commit(HERE)}"
    cfg = {"name": args.condition, "model_id": args.model, "model_revision": None, "adapter_id": None}
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "reasoning_effort": args.effort,
                "provider": args.provider, "reasoning_term": args.reasoning_term, "elicit": args.elicit, "elicit_family": "qwen" if args.elicit else None}
    sem = asyncio.Semaphore(args.concurrency)
    n_done = 0

    async def one(client, u):
        nonlocal n_done
        prompt = render(vllm_runner.messages_for(u["spec"]), args.effort)
        if args.elicit in selfinstruct.PREFILL_METHODS:
            prompt += u["self_instruction"]
        body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p,
                "provider": {"order": [args.provider], "allow_fallbacks": False}, "usage": {"include": True}}
        async with sem:
            try:
                b = await complete(client, body)
            except RuntimeError as e:
                print(f"[generate] {u['key']} not done (rerun to fill): {e}", file=sys.stderr)
                return
        c, usage = b["choices"][0], b["usage"]
        n_prompt = len(tok.encode(prompt, add_special_tokens=False).ids)
        assert usage["prompt_tokens"] == n_prompt, f"{u['key']}: provider counted {usage['prompt_tokens']} prompt tokens, we have {n_prompt}"
        reasoning, text = c.get("reasoning") or "", c.get("text") or ""
        assert "</think>" not in reasoning + text, u["key"]
        finish = "length" if c["finish_reason"] == "length" else "stop"
        raw_output = "<think>" + reasoning + ("</think>" + text if text or finish == "stop" else "")
        echoed = None
        if args.elicit in selfinstruct.BOOTSTRAP_METHODS:
            raw_output, echoed = selfinstruct.strip_echo_from_raw(raw_output, u["self_instruction"])
        row = vllm_runner.build_row(unit=u, protocol=args.protocol, cfg=cfg, rendered_prompt=prompt, raw_output=raw_output,
                                    n_prompt_tokens=n_prompt, output_token_ids=[], finish_reason=finish, sampling=sampling,
                                    engine=f"openrouter:{b.get('provider')}", commit=commit)
        row["n_output_tokens"] = usage["completion_tokens"]
        row["openrouter"] = {"id": b.get("id"), "provider": b.get("provider"), "finish_reason": c.get("finish_reason"),
                             "native_finish_reason": c.get("native_finish_reason"), "cost": usage.get("cost"),
                             "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        if args.elicit:
            row["elicitation"], row["self_instruction"], row["echo_stripped"] = args.elicit, u["self_instruction"], echoed
        append_jsonl(out, row)
        n_done += 1
        print(f"[generate] {n_done}/{len(todo)} {u['key']} tokens={usage['completion_tokens']} {finish} "
              f"rule={row['cot_rule']} ans={row['answer_pred']}/{row['answer_gold']}", file=sys.stderr)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, u) for u in todo))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--condition", default="deepseek-v4.1-flash-1017")
    ap.add_argument("--model", default=MODEL, help="OpenRouter model id")
    ap.add_argument("--provider", required=True, help="OpenRouter provider that serves raw completions with special tokens intact")
    ap.add_argument("--protocol", default="astra")
    ap.add_argument("--limit-per-constraint", type=int)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--max-new-tokens", type=int, default=25000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--effort", type=int, default=75, help="numeric reasoning effort 1-100; 75 is the format's default ('high')")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--reasoning-term", default="reasoning stage",
                    help='replaces "reasoning stage" in the user prompt (the grading prompt is unchanged)')
    ap.add_argument("--elicit", choices=selfinstruct.METHODS,
                    help="Self-Instructed CoT elicitation method (selfinstruct.py); default none (User Prompt)")
    return asyncio.run(_generate(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
