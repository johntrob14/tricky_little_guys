"""Astra108, three existing elicitation prompts, three independent decoding seeds.

The full decoded output and token IDs are authoritative; parsed reasoning is a
convenience. No model weights or credentials are copied into result artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def render(tok, unit):
    messages = ([{"role": "system", "content": unit["system"]}] if unit["system"] else [])
    messages += [{"role": "user", "content": unit["user"]}]
    result = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if unit["hist"]:
        history = "".join(
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n{thinking}\n</think>\n\n{answer}<|im_end|>\n"
            for user, thinking, answer in unit["hist"]
        )
        position = result.rfind("<|im_start|>user")
        result = result[:position] + history + result[position:]
    # Qwen3's thinking-enabled template leaves the assistant header open; the
    # model emits its own opening marker. Do not add a reasoning prefill.
    assert result.rstrip().endswith(("<|im_start|>assistant", "<think>")), result[-100:]
    assert result.count("<|im_start|>assistant") == len(unit["hist"]) + 1
    return result


def model_path(entry):
    from huggingface_hub import snapshot_download
    sub = entry.get("subfolder", "")
    prefix = sub + "/" if sub else ""
    files = [prefix + glob for glob in ("*.safetensors", "*.json", "*.txt", "*.jinja")]
    path = Path(snapshot_download(entry["repo"], revision=entry["revision"], allow_patterns=files))
    return path / sub if sub else path


def parse_output(tok, ids):
    close_id = tok.convert_tokens_to_ids("</think>")
    raw = tok.decode(ids, skip_special_tokens=False)
    if close_id in ids:
        end = ids.index(close_id)
        reasoning_ids, answer_ids = ids[:end], ids[end + 1:]
        closed = True
    else:
        reasoning_ids, answer_ids, closed = ids, [], False
    reasoning = tok.decode(reasoning_ids, skip_special_tokens=False).strip()
    # A redundant opening marker does not replace the raw, recorded sequence.
    if reasoning.startswith("<think>"):
        reasoning = reasoning[len("<think>"):].strip()
    answer = tok.decode(answer_ids, skip_special_tokens=False)
    for eos in ("<|im_end|>", "<|endoftext|>"):
        answer = answer.removesuffix(eos).rstrip()
    return dict(raw_output=raw, reasoning=reasoning, response=answer.strip(), think_closed=closed,
                think_opened=bool(ids) and ids[0] == tok.convert_tokens_to_ids("<think>"),
                n_reasoning_tokens=len(reasoning_ids), n_response_tokens=len(answer_ids))


def generate(entry, output, pilot=False):
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    manifest = json.loads((HERE / "manifest.json").read_text())
    config = manifest["sampling"]
    initial = manifest["checkpoints"][0]
    tok = AutoTokenizer.from_pretrained(model_path(initial))
    checkpoint = model_path(entry)
    own_tok = AutoTokenizer.from_pretrained(checkpoint)
    assert tok.get_vocab() == own_tok.get_vocab(), "checkpoint tokenizer vocabulary differs"
    units = read_rows(HERE / "prompts.jsonl")
    rendered = []
    for unit in units:
        prompt = render(tok, unit)
        assert render(own_tok, unit) == prompt, "checkpoint's native rendering differs"
        ids = tok.encode(prompt, add_special_tokens=False)
        assert len(ids) + config["max_tokens"] <= config["max_model_len"]
        rendered.append(dict(unit, rendered_prompt=prompt, prompt_token_ids=ids))
    prompt_path = output / "rendered_prompts.jsonl"
    if not prompt_path.exists():
        for unit in rendered:
            append(prompt_path, unit)
    else:
        assert read_rows(prompt_path) == rendered
    if pilot:
        # One item per constraint and prompt variant. All nine constraint types.
        selected = {}
        for row in rendered:
            selected.setdefault((row["constraint"], row["variant"]), row)
        rendered = list(selected.values())
        seeds = config["seeds"][:1]
    else:
        seeds = config["seeds"]
    path = output / ("pilot.jsonl" if pilot else "generations.jsonl")
    done = {(row["key"], row["variant"], row["seed"]) for row in read_rows(path)}
    jobs = [(unit, seed) for unit in rendered for seed in seeds if (unit["key"], unit["variant"], seed) not in done]
    if not jobs:
        return
    metadata = dict(checkpoint=entry, torch=torch.__version__, transformers=transformers.__version__, vllm=vllm.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), sampling=config,
                    engine_settings=dict(dtype="bfloat16", generation_config="vllm", max_num_seqs=32,
                        max_num_batched_tokens=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    tokenizer_revision=initial["revision"], prompt_file_sha256=hashlib.sha256((HERE / "prompts.jsonl").read_bytes()).hexdigest())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    engine = LLM(model=str(checkpoint), tokenizer=str(model_path(initial)), dtype="bfloat16", generation_config="vllm",
                 max_model_len=config["max_model_len"], max_num_seqs=32, max_num_batched_tokens=8192,
                 gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=13)
    start = time.time()
    total_tokens = 0
    stop_ids = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")})
    for offset in range(0, len(jobs), 32):
        batch = jobs[offset:offset + 32]
        params = [SamplingParams(n=1, temperature=config["temperature"], top_p=config["top_p"], top_k=-1,
                    max_tokens=config["max_tokens"], seed=seed, stop_token_ids=stop_ids,
                    skip_special_tokens=False, spaces_between_special_tokens=False) for _, seed in batch]
        began = time.time()
        generated = engine.generate([{"prompt_token_ids": unit["prompt_token_ids"]} for unit, _ in batch], params, use_tqdm=False)
        assert len(generated) == len(batch)
        for (unit, seed), result in zip(batch, generated):
            generation = result.outputs[0]
            ids = [int(t) for t in generation.token_ids]
            row = dict(key=unit["key"], variant=unit["variant"], constraint=unit["constraint"], dataset=unit["dataset"],
                       item_idx=unit["item_idx"], gold=unit["gold"], grader_args=unit["grader_args"], seed=seed,
                       checkpoint=entry, n_shots=len(unit["hist"]), output_token_ids=ids, n_output_tokens=len(ids),
                       n_prompt_tokens=len(unit["prompt_token_ids"]), prompt_sha256=hashlib.sha256(unit["rendered_prompt"].encode()).hexdigest(),
                       finish_reason=generation.finish_reason, stop_reason=generation.stop_reason,
                       batch_wall_seconds=time.time() - began, **parse_output(tok, ids))
            append(path, row)
            total_tokens += len(ids)
        progress = dict(checkpoint=entry["label"], phase="pilot" if pilot else "full", completed=len(done) + min(offset + 32, len(jobs)),
                        expected=len(rendered) * len(seeds), elapsed_seconds=time.time() - start, output_tokens=total_tokens,
                        aggregate_tokens_per_second=total_tokens / (time.time() - start))
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress), flush=True)
    rows = read_rows(path)
    assert len(rows) == len(rendered) * len(seeds)
    (output / ("PILOT_DONE" if pilot else "DONE")).write_text(json.dumps(progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/workspace/qwen17-ifrl/results"))
    options = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    if options.checkpoint is not None:
        entry = manifest["checkpoints"][options.checkpoint]
        dest = options.output / entry["label"]
        dest.mkdir(parents=True, exist_ok=True)
        generate(entry, dest, options.pilot)
    else:
        for index, entry in enumerate(manifest["checkpoints"]):
            # Separate workers release GPU allocations between checkpoints.
            subprocess.run([sys.executable, __file__, "--checkpoint", str(index), "--output", str(options.output)], check=True)
        (options.output / "ALL_DONE").write_text(json.dumps(dict(checkpoints=len(manifest["checkpoints"]), expected_generations=manifest["expected_generations"])))


if __name__ == "__main__":
    main()
