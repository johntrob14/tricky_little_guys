"""Astra (108 units) on the Nemotron-Cascade-8B intermediate checkpoints with vLLM, thinking on.

    python run_cascade.py render   --run-dir DIR --template-from HF_DIR [--tokenizer HF_DIR]   (CPU; writes prompts.jsonl)
    python run_cascade.py generate --run-dir DIR --checkpoint IFRL --model-dir HF_DIR --template-from HF_DIR

Reuses cotswitch (../grading) for the Astra units, prompts, row schema, CoT extraction and grading;
only rendering and sampling differ from tinker-runner/tinker_cotcontrol.py.

Decisions recorded here:
- Checkpoints: nvidia/Nemotron-Cascade-8B-Intermediate-ckpts, subfolders Nemotron-Cascade-8B-{SFT,RLHF,IFRL},
  pinned to one repo revision (--revision; recorded in every row as model_revision).
- Template: the official Nemotron-Cascade-8B chat template, taken from the IFRL checkpoint's tokenizer_config
  (--template-from) and used for all three checkpoints. RLHF and IFRL ship it; SFT ships an older Qwen3 template
  with no default system prompt and no /think. For a single user turn the official template renders
  "<|im_start|>system\\nYou are a helpful and harmless assistant.<|im_end|>\\n<|im_start|>user\\n{user} /think<|im_end|>\\n<|im_start|>assistant\\n"
  (the final model's template differs from RLHF/IFRL's only in earlier user turns). The prompt does not open
  <think>; the model must emit it, and each row records whether it did.
- The rendered prompt, and its token ids, are identical across checkpoints (asserted at render time; the three
  tokenizers are byte-identical).
- --reasoning-term replaces "reasoning stage" in the user text the model sees, as in tinker_cotcontrol.py;
  grader_args keep the original prompt.
- Decoding: identical SamplingParams for every checkpoint. vLLM is started with generation_config="vllm" so no
  checkpoint's generation_config.json changes defaults (SFT's sets max_new_tokens 2048 and EOS 151643, RLHF/IFRL
  EOS 151645); stop_token_ids is set explicitly to both (151643 <|endoftext|>, 151645 <|im_end|>).
  Per-request seed = --seed, as in tinker_cotcontrol.py and vllm_runner.
- raw_output is the decoded generation with special tokens kept (vLLM drops the stop token's text). Extra row
  fields: checkpoint, hf_repo, hf_subfolder, think_opened, n_reasoning_tokens (tokens before </think>, id 151668),
  n_response_tokens, stop_reason, last_token_id, output_token_ids, error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COTSWITCH_ROOT = Path(os.environ.get("COTSWITCH_ROOT", HERE.parent / "grading"))
sys.path.insert(0, str(COTSWITCH_ROOT))

from cotswitch.io import generations_path, grades_path, load_done_keys  # noqa: E402
from cotswitch.runners import vllm_runner  # noqa: E402

HF_REPO = "nvidia/Nemotron-Cascade-8B-Intermediate-ckpts"
CHECKPOINTS = ("SFT", "RLHF", "IFRL")
STOP_IDS = [151643, 151645]  # <|endoftext|>, <|im_end|>
THINK_OPEN_ID, THINK_CLOSE_ID = 151667, 151668
SYSTEM_LINE = "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"


def _tokenizer(hf_dir: Path, template_from: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    tmpl = json.loads((template_from / "tokenizer_config.json").read_text())["chat_template"]
    tok.chat_template = tmpl
    assert tok.convert_tokens_to_ids("<think>") == THINK_OPEN_ID and tok.convert_tokens_to_ids("</think>") == THINK_CLOSE_ID
    return tok


def _units(reasoning_term: str):
    units = vllm_runner.build_units("astra")
    assert len(units) == 108, len(units)
    if reasoning_term != "reasoning stage":
        for u in units:
            user = u["spec"].user
            assert "reasoning stage" in user, u["key"]
            u["spec"] = dataclasses.replace(u["spec"], user=user.replace("reasoning stage", reasoning_term))
    return units


def _render(tok, units):
    out = []
    for u in units:
        msgs = vllm_runner.messages_for(u["spec"])
        assert [m["role"] for m in msgs] == ["user"], u["key"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        expect = SYSTEM_LINE + "<|im_start|>user\n" + u["spec"].user + " /think<|im_end|>\n<|im_start|>assistant\n"
        assert text == expect, (u["key"], text[-200:])
        out.append((u, text, tok.encode(text, add_special_tokens=False)))
    return out


def cmd_render(args) -> int:
    units = _units(args.reasoning_term)
    rendered = _render(_tokenizer(args.tokenizer or args.template_from, args.template_from), units)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with open(args.run_dir / "prompts.jsonl", "w") as f:
        for u, text, ids in rendered:
            f.write(json.dumps({"key": u["key"], "constraint": u["constraint"], "dataset": u["item"].dataset,
                                "item_idx": u["item"].item_idx, "user_prompt_model_facing": u["spec"].user,
                                "rendered_prompt": text, "prompt_token_ids": ids,
                                "n_prompt_tokens": len(ids)}) + "\n")
    n = [len(ids) for _, _, ids in rendered]
    print(f"[render] {len(rendered)} prompts, prompt tokens min {min(n)} max {max(n)} -> {args.run_dir / 'prompts.jsonl'}")
    return 0


def cmd_generate(args) -> int:
    from vllm import LLM, SamplingParams

    condition = f"nemotron-cascade-8b-{args.checkpoint.lower()}"
    out = generations_path(args.run_dir, condition, "astra")
    gpath = grades_path(args.run_dir, condition, "astra")
    done, graded = load_done_keys(out), load_done_keys(gpath)
    tok = _tokenizer(args.model_dir, args.template_from)
    rendered = _render(tok, _units(args.reasoning_term))
    ref = {json.loads(l)["key"]: json.loads(l) for l in open(args.run_dir / "prompts.jsonl")}
    for u, text, ids in rendered:  # same prompt bytes and ids as the CPU render, for every checkpoint
        assert ref[u["key"]]["rendered_prompt"] == text and ref[u["key"]]["prompt_token_ids"] == ids, u["key"]
    todo = [r for r in rendered if r[0]["key"] not in done]
    print(f"[generate] {condition}: 108 units, {len(todo)} to do -> {out}", file=sys.stderr, flush=True)
    if not todo:
        return 0

    llm = LLM(model=str(args.model_dir), tokenizer=str(args.model_dir), dtype="bfloat16",
              max_model_len=args.max_model_len, seed=args.seed, generation_config="vllm",
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True)
    import vllm

    engine = f"vllm-{vllm.__version__}"
    commit = os.environ.get("COTSWITCH_GIT_COMMIT", "unknown")
    cfg = {"name": condition, "model_id": f"{HF_REPO}/Nemotron-Cascade-8B-{args.checkpoint}",
           "model_revision": args.revision, "model_path": str(args.model_dir)}
    jobs = []
    for u, text, ids in todo:
        eff = min(args.max_new_tokens, args.max_model_len - len(ids))
        sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": -1, "min_p": 0.0,
                    "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens,
                    "max_new_tokens_effective": eff, "seed": args.seed, "n": 1, "stop_token_ids": STOP_IDS,
                    "generation_config": "vllm (checkpoint generation_config.json ignored)",
                    "max_model_len": args.max_model_len, "rope_scaling": None,
                    "chat_template": f"official Nemotron-Cascade-8B (from {args.template_from.name})",
                    "thinking": "/think appended by template", "reasoning_term": args.reasoning_term}
        sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=-1, min_p=0.0,
                            repetition_penalty=1.0, max_tokens=eff, seed=args.seed, stop_token_ids=STOP_IDS,
                            skip_special_tokens=False, spaces_between_special_tokens=False)
        jobs.append((u, text, ids, sampling, sp))

    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": j[2]} for j in jobs], [j[4] for j in jobs], use_tqdm=True)
    print(f"[generate] {condition}: generation wall {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    for (u, text, ids, sampling, _), o in zip(jobs, outs):
        error = None
        try:
            c = o.outputs[0]
            tids = [int(t) for t in c.token_ids]
            raw, finish = c.text, c.finish_reason
        except Exception as e:  # noqa: BLE001
            tids, raw, finish, error = [], "", "error", repr(e)
        row = vllm_runner.build_row(unit=u, protocol="astra", cfg=cfg, rendered_prompt=text, raw_output=raw,
                                    n_prompt_tokens=len(ids), output_token_ids=tids, finish_reason=finish,
                                    sampling=sampling, engine=engine, commit=commit)
        close = tids.index(THINK_CLOSE_ID) if THINK_CLOSE_ID in tids else None
        row.update({
            "checkpoint": args.checkpoint, "hf_repo": HF_REPO, "hf_subfolder": f"Nemotron-Cascade-8B-{args.checkpoint}",
            "user_prompt_model_facing": u["spec"].user, "raw_generation": raw,
            "think_opened": bool(tids) and tids[0] == THINK_OPEN_ID,
            "n_reasoning_tokens": close if close is not None else len(tids),
            "n_response_tokens": len(tids) - close - 1 if close is not None else 0,
            "stop_reason": getattr(o.outputs[0], "stop_reason", None) if o.outputs else None,
            "last_token_id": tids[-1] if tids else None, "output_token_ids": tids, "error": error,
        })
        vllm_runner.write_row(out, row, gpath, graded)
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[generate] {condition}: wrote {len(jobs)} rows, {n_tok} output tokens", file=sys.stderr, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--template-from", required=True, type=Path, help="HF dir whose tokenizer_config chat_template is used")
        p.add_argument("--reasoning-term", default="thinking")
        if name == "render":
            p.add_argument("--tokenizer", type=Path)
        else:
            p.add_argument("--checkpoint", required=True, choices=CHECKPOINTS)
            p.add_argument("--model-dir", required=True, type=Path)
            p.add_argument("--revision", required=True)
            p.add_argument("--temperature", type=float, default=0.6)
            p.add_argument("--top-p", type=float, default=0.95)
            p.add_argument("--max-new-tokens", type=int, default=25000)
            p.add_argument("--max-model-len", type=int, default=32768)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = ap.parse_args(argv)
    return cmd_render(args) if args.cmd == "render" else cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
