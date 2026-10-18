"""SWITCH latent-mode generation (condition `switch-latent`) with the batched loop.

    python -m cotswitch.cli generate --condition switch-latent --protocol default --run NAME
    python -m cotswitch.runners.switch_runner --condition switch-latent --protocol astra \
        --run-dir DIR [--batch-size 128] [--kv-budget-gb 50] [--merged-path PATH] [--loop paged|hf]

Decisions recorded here:
- Model: base Qwen3-8B at its pinned revision + the SWITCH LoRA, unmerged, loaded as in
  the repo's scripts/eval_latent.py (cotswitch.switch_model.load_switch). `--merged-path`
  (or env COTSWITCH_SWITCH_MERGED) loads a merged checkpoint instead (use it: the paged
  loop otherwise merges the LoRA in memory); rows record which in `switch_loader`.
- Prompt: the adapter tokenizer's chat template with add_generation_prompt=True and
  enable_thinking from the config, tokenized with add_special_tokens=False (the same
  rendering as vllm_runner). Rows reuse vllm_runner.build_row so the two engines write
  identical schemas; `latent`, `n_output_tokens` and `finish_reason` come from the loop.
- Decoding: greedy, K_min = cfg.k_min (4), max 256 latent steps per block,
  max_new_tokens = cfg (25000) unless overridden. Latent steps do not count toward
  max_new_tokens (as in the repo); see switch_model.py for the exact budget rules.
  `n_output_tokens` counts output ids (incl. <swi>, </swi>, EOS), not latent steps.
  This is the repo's accounting: text tokens, <swi> and </swi> each spend one unit of
  max_new_tokens, EOS spends none, latent steps spend none, and the forced </swi> is
  appended even when the budget is already spent. So n_output_tokens can reach
  max_new_tokens + 1 (budget ends on <swi>), and the latent step count is in
  `latent.n_latent_steps`. Both loops (paged and hf) implement exactly this.
  There is no max_model_len cap: the HF loop has none (positions may exceed 40960
  only if a row is extremely long; none is expected with 25000 tokens + latent steps).
- Loop: `--loop paged` (default) is switch_model.PagedSwitchEngine: a paged KV cache,
  the per-row state machine on the GPU, one CUDA graph per batch bucket, host syncs only
  every `--sync-every` steps (default 16), continuous batching over `--batch-size` slots
  (default 128, env COTSWITCH_SWITCH_BATCH). It needs a merged model, so without
  `--merged-path` the unmerged LoRA is merged in memory (bf16) after loading.
  `--loop hf` is the original HF-forward loop (left-padded batch, admit in chunks).
- KV budget: `--kv-budget-gb` (env COTSWITCH_SWITCH_KV_GB; default: free GPU memory after
  loading minus 16 GB). Paged: the block pool; when it runs out the most recently
  admitted rows are evicted and re-queued (restarted from scratch). HF loop: caps
  rows x cache length the same way.
- Pending units are sorted by prompt length (shortest first). Each row is written with
  vllm_runner.write_row the moment it finishes: the generation row is appended (one
  write, flush, fsync), then graded and its grade row appended, so a crash loses only
  in-flight rows and grades are never stale; reruns skip keys already written.
- Evictions (KV pool exhausted) restart a row from scratch; they are counted in the
  progress log and the final summary line. At the default sizes (batch 128, 40-50 GB KV)
  they should be rare; if the log shows many, lower --batch-size or raise --kv-budget-gb.
- Extra row fields: `model_path`, `grader_args` (as vllm_runner), `banned_token_hits`
  (always 0 here), `switch_loader` ("peft-unmerged" | "merged:<path>").
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from cotswitch.io import generations_path, git_commit, grades_path, load_done_keys
from cotswitch.runners import vllm_runner

DEFAULT_BATCH = 128
DEFAULT_BATCH_HF = 32
KV_RESERVE_GB = 16.0  # activations, logits, prefill workspace


def _engine_name() -> str:
    try:
        import torch
        import transformers

        return f"hf-switch-latent (transformers-{transformers.__version__}, torch-{torch.__version__})"
    except Exception:  # noqa: BLE001
        return "hf-switch-latent"


def default_kv_budget_tokens(model, kv_budget_gb=None, overalloc: float = 1.15) -> int | None:
    import torch

    from cotswitch.switch_model import kv_bytes_per_token

    if not torch.cuda.is_available():
        return None
    if kv_budget_gb is None:
        free, _ = torch.cuda.mem_get_info()
        kv_budget_gb = max(1.0, free / 2**30 - KV_RESERVE_GB)
    # HF loop: SlotLayer over-allocates up to 1/8 (+256 slots) when it grows; leave room for
    # it (overalloc=1.15). The paged pool is allocated exactly (overalloc=1).
    return int(kv_budget_gb * 2**30 / kv_bytes_per_token(model) / overalloc)


def run(condition_cfg: dict, protocol: str, run_dir, limit_per_constraint=None, datasets=None,
        max_new_tokens=None, batch_size: int | None = None, kv_budget_gb: float | None = None,
        merged_path: str | None = None, units=None, model_bundle=None, engine=None,
        loop: str | None = None, sync_every: int | None = None, engine_kw: dict | None = None,
        grade: bool = True) -> Path:
    """Generate all pending units of `protocol` for the switch-latent condition.

    `units`, `model_bundle` (model, tokenizer, SwitchTokens), `engine` (the engine name
    recorded in rows) and `engine_kw` (extra PagedSwitchEngine arguments) exist for CPU
    tests. Returns the output path.
    """
    from cotswitch import switch_model as sm

    # many differently sized KV buffers: avoid allocator fragmentation (must precede CUDA init)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = dict(condition_cfg)
    out_path = generations_path(run_dir, cfg["name"], protocol)
    gpath = grades_path(run_dir, cfg["name"], protocol)
    graded = load_done_keys(gpath) if grade else set()
    done = load_done_keys(out_path)
    if units is None:
        units = vllm_runner.build_units(protocol, limit_per_constraint, datasets)
    pending = [u for u in units if u["key"] not in done]
    print(f"[switch_runner] {cfg['name']}/{protocol}: {len(units)} units, {len(done)} done, "
          f"{len(pending)} pending -> {out_path}", file=sys.stderr, flush=True)
    if not pending:
        return out_path

    loop = loop or os.environ.get("COTSWITCH_SWITCH_LOOP") or "paged"
    if loop not in ("paged", "hf"):
        raise ValueError(f"unknown loop {loop!r}")
    batch_size = int(batch_size or os.environ.get("COTSWITCH_SWITCH_BATCH")
                     or (DEFAULT_BATCH if loop == "paged" else DEFAULT_BATCH_HF))
    sync_every = int(sync_every or os.environ.get("COTSWITCH_SWITCH_SYNC_EVERY") or 16)
    if kv_budget_gb is None and os.environ.get("COTSWITCH_SWITCH_KV_GB"):
        kv_budget_gb = float(os.environ["COTSWITCH_SWITCH_KV_GB"])
    merged_path = merged_path or os.environ.get("COTSWITCH_SWITCH_MERGED") or None
    max_new = int(max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 25000))
    k_min = int(cfg.get("k_min", sm.K_MIN_DEFAULT))
    k_max = int(cfg.get("k_max", sm.MAX_LATENT_PER_BLOCK))
    enable_thinking = bool(cfg.get("enable_thinking", True))

    if model_bundle is None:
        model_bundle = sm.load_switch(
            adapter_id=cfg["adapter_id"], adapter_revision=cfg.get("adapter_revision"),
            base_id=cfg["model_id"], base_revision=cfg.get("model_revision"), merged_path=merged_path,
            merge=(loop == "paged" and not merged_path),
        )
    model, tok, tokens = model_bundle
    kv_budget = default_kv_budget_tokens(model, kv_budget_gb, overalloc=1.0 if loop == "paged" else 1.15)
    engine = (engine or _engine_name()) + f" loop={loop}"
    if merged_path:
        loader = f"merged:{merged_path}"
    else:
        loader = "peft-merged-in-memory" if loop == "paged" else "peft-unmerged"
    if loop == "paged" and hasattr(model, "peft_config"):
        model = model.merge_and_unload()
    commit = git_commit()

    meta = {}
    queue = []
    for u in pending:
        rendered = vllm_runner.render_prompt(tok, u["spec"], enable_thinking)
        ids = tok.encode(rendered, add_special_tokens=False)
        meta[u["key"]] = (u, rendered)
        queue.append((u["key"], ids))
    queue.sort(key=lambda x: len(x[1]))  # stable: ties keep protocol order

    sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": max_new,
        "seed": int(cfg.get("seed", 0)),
        "n": 1,
        "k_min": k_min,
        "k_max": k_max,
        "enable_thinking": enable_thinking,
        "max_batch": batch_size,
        "loop": loop,
    }
    print(f"[switch_runner] loop={loop} batch={batch_size} sync_every={sync_every} kv_budget_tokens={kv_budget} "
          f"k_min={k_min} k_max={k_max} max_new={max_new} loader={loader}", file=sys.stderr, flush=True)

    stats = sm.GenStats()
    n_written, t_last = 0, time.time()
    kw = dict(max_new_tokens=max_new, min_latent_steps=k_min, max_latent_per_block=k_max,
              max_batch=batch_size, kv_budget_tokens=kv_budget, stats=stats, engine=loop)
    if loop == "paged":
        kw.update(sync_every=sync_every, max_seq_len=max(len(ids) for _, ids in queue) + max_new + 2 + 16384,
                  verbose=True, log_every=300.0)
        kw.update(engine_kw or {})
    for key, res in sm.generate_stream(model, tokens, queue, **kw):
        u, rendered = meta[key]
        raw = tok.decode(res["output_ids"], skip_special_tokens=False)
        row = vllm_runner.build_row(
            unit=u, protocol=protocol, cfg=cfg, rendered_prompt=rendered, raw_output=raw,
            n_prompt_tokens=res["n_prompt_tokens"], output_token_ids=res["output_ids"],
            finish_reason=res["finish_reason"], sampling=sampling, engine=engine, commit=commit,
        )
        row["latent"] = res["latent"]
        row["switch_loader"] = loader
        # generation row: one write + flush + fsync; then its grade row (same path as vllm_runner)
        vllm_runner.write_row(out_path, row, gpath if grade else None, graded)
        n_written += 1
        if time.time() - t_last > 60 or n_written == len(pending):
            s = stats.as_dict()
            print(f"[switch_runner] wrote {n_written}/{len(pending)}  forwards={s['forwards']} "
                  f"tokens/s={s['tokens_out_per_s']:.1f} row_steps/s={s['row_steps_per_s']:.1f} latent_frac="
                  f"{s['latent_row_steps'] / max(1, s['row_steps']):.3f} evictions={s['evictions']} "
                  f"deferred={s['deferred']}",
                  file=sys.stderr, flush=True)
            t_last = time.time()
    return out_path


def main(argv=None) -> int:
    from cotswitch import cli

    p = argparse.ArgumentParser(prog="switch_runner")
    p.add_argument("--condition", default="switch-latent")
    p.add_argument("--protocol", required=True, choices=cli.PROTOCOLS)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run-dir")
    g.add_argument("--run")
    p.add_argument("--limit-per-constraint", type=int, default=None)
    p.add_argument("--datasets", nargs="+", choices=cli.DATASETS, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--kv-budget-gb", type=float, default=None)
    p.add_argument("--loop", choices=["paged", "hf"], default=None)
    p.add_argument("--sync-every", type=int, default=None)
    p.add_argument("--merged-path", default=None)
    p.add_argument("--config", default=str(cli.DEFAULT_CONFIG))
    a = p.parse_args(argv)
    cfg = cli.load_condition(a.condition, a.config)
    if cfg["engine"] != "switch":
        raise SystemExit(f"{a.condition} is not a switch-engine condition")
    run(cfg, a.protocol, cli.resolve_run_dir(a), limit_per_constraint=a.limit_per_constraint,
        datasets=a.datasets, max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
        kv_budget_gb=a.kv_budget_gb, merged_path=a.merged_path, loop=a.loop, sync_every=a.sync_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
