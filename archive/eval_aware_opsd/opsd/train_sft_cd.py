"""Off-policy context distillation (CD) baseline for OPSD.

Phase A (--phase gen): the frozen SDF-merged model with NO LoRA samples one response per training prompt
  under the TEACHER system prompt (opsd.prompts.add_teacher_lines, the same insertion OPSD uses).
  Prompts and their order are OPSD's: data/train_prompts.jsonl shuffled with random.Random(seed), first
  --total-seqs positions. Output runs/<name>/teacher_samples.jsonl (+ teacher_samples_summary.json with the
  type-hint rate of the teacher samples; nothing is filtered).
Phase B (--phase train): new LoRA (OPSD config) trained with cross-entropy on the teacher response tokens
  (including the final EOS when generation stopped on it) given the STUDENT prompt (no teacher lines).
  Step grouping = OPSD's: rounds of --round-size sequences, chunks of --mb-per-step sequences per optimizer
  step (token-mean loss over the step), AdamW, grad clip, linear warmup (opsd.train_opsd.lr_at).
  Placement = opsd.train_opsd.Student (balanced 2-GPU device map when >1 GPU visible).
  Saves adapter_latest (+optimizer, trainer_state) every round, ckpt_<n> at the first round end at or past
  each multiple of --ckpt-every (with round 128: ckpt_1024, ckpt_2048, ckpt_3000, the same names and
  sequence counts as OPSD's checkpoints), and adapter_final.
--phase all runs gen and train as two separate subprocesses (vLLM memory is fully released in between).

Pod:
  cd /workspace/opsd && /root/venv/bin/python -m opsd.train_sft_cd --phase gen --model /root/models/sdf_merged \
      --run-name sft_cd --runs-dir /workspace/opsd/runs --tp 2
  cd /workspace/opsd && /root/venv/bin/python -m opsd.train_sft_cd --phase train --model /root/models/sdf_merged \
      --run-name sft_cd --runs-dir /workspace/opsd/runs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from opsd.backends import HFBackend, SamplingCfg
from opsd.kl import causal_lm_parts, final_hidden
from opsd.model_utils import DEFAULT_TARGETS, gpu_mem_report, load_base, load_tokenizer, save_adapter, trainable_params
from opsd.prompts import THINK_TOGGLE, TEACHER_LINES, add_teacher_lines, chat_prompt_ids, check_system, stop_token_ids
from opsd.train_opsd import REPO, Student, append_jsonl, load_grader, lr_at, now, read_jsonl


def teacher_system(system: str, lines: str, allow_no_toggle: bool) -> str:
    """Same rule as the teacher_system closure in opsd.train_opsd.main."""
    if allow_no_toggle and not system.rstrip().endswith(THINK_TOGGLE):
        return system.rstrip() + ("\n" + lines if lines else "")
    return add_teacher_lines(system, lines)


def opsd_order(n_prompts: int, seed: int) -> list[int]:
    """OPSD's prompt order (train_opsd.main: random.Random(seed).shuffle(range(n)))."""
    order = list(range(n_prompts))
    random.Random(seed).shuffle(order)
    return order


# ---------------------------------------------------------------- loss
def _chunk_ce(h: torch.Tensor, tgt: torch.Tensor, head: torch.nn.Module) -> torch.Tensor:
    logits = head(h).float()
    return F.cross_entropy(logits, tgt.to(logits.device), reduction="sum")


def sequence_ce_backward(model, prompt_ids: list[int], resp_ids: list[int], device, scale: float,
                         chunk: int = 512, do_backward: bool = True) -> dict:
    """Summed next-token CE over the response tokens of prompt+response (logit at P-1+j predicts response
    token j), vocab projection in fp32 chunks under activation checkpointing. backward(sum * scale)."""
    if not resp_ids:
        raise ValueError("empty response")
    _, dec, head = causal_lm_parts(model)
    P, n = len(prompt_ids), len(resp_ids)
    h = final_hidden(dec, list(prompt_ids) + list(resp_ids), device)[P - 1 : P - 1 + n]
    tgt = torch.tensor(resp_ids, dtype=torch.long, device=h.device)
    total = None
    for i in range(0, n, chunk):
        hs, ts = h[i : i + chunk], tgt[i : i + chunk]
        if torch.is_grad_enabled() and hs.requires_grad:
            part = checkpoint(_chunk_ce, hs, ts, head, use_reentrant=False)
        else:
            part = _chunk_ce(hs, ts, head)
        total = part if total is None else total + part.to(total.device)
    if do_backward and total.requires_grad:
        (total * scale).backward()
    return {"sum_nll": float(total.detach()), "n": n}


# ---------------------------------------------------------------- phase A
def phase_gen(a):
    run_dir = os.path.join(a.runs_dir, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    out_p = os.path.join(run_dir, "teacher_samples.jsonl")
    with open(os.path.join(run_dir, f"args_gen_{int(time.time())}.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    tok = load_tokenizer(a.model)
    stops = stop_token_ids(tok, a.model)
    print(f"[tok] stop ids {stops} -> {[tok.decode([s]) for s in stops]}", flush=True)
    train = read_jsonl(a.train_prompts)
    if not a.allow_no_toggle:
        for r in train:
            check_system(r["system"])
    order = opsd_order(len(train), a.seed)
    jobs = []
    for pos in range(a.total_seqs):
        ti = order[pos % len(order)]
        r = train[ti]
        t_sys = teacher_system(r["system"], a.teacher_lines, a.allow_no_toggle)
        s_ids = chat_prompt_ids(tok, r["system"], r["user"], a.chat_template_kwargs)
        t_ids = chat_prompt_ids(tok, t_sys, r["user"], a.chat_template_kwargs)
        if a.teacher_lines:
            assert a.teacher_lines in tok.decode(t_ids) and a.teacher_lines not in tok.decode(s_ids), pos
        jobs.append((pos, ti, r, t_sys, s_ids, t_ids))

    done = 0
    if os.path.exists(out_p):
        if not a.resume:
            sys.exit(f"{out_p} exists; use --resume to continue or a new --run-name")
        prev = read_jsonl(out_p)
        assert [x["pos"] for x in prev] == list(range(len(prev))), "teacher_samples.jsonl out of order"
        done = len(prev)
        print(f"[gen] resume: {done} samples already in {out_p}", flush=True)
    grader = load_grader(a.grader)

    if a.backend == "vllm":
        from opsd.backends import VLLMBackend

        backend = VLLMBackend(a.model, tok, a.max_lora_rank, a.gpu_mem, a.max_model_len, a.seed, a.tp,
                              enable_sleep_mode=False, enforce_eager=a.enforce_eager, kv_cache_dtype=a.kv_cache_dtype,
                              enable_lora=False, max_num_seqs=a.max_num_seqs)
    else:
        torch.manual_seed(a.seed)
        base = load_base(a.model, a.dtype, a.attn_impl, device=a.device)
        base.config.use_cache = True
        backend = HFBackend(lambda: base, tok, a.hf_gen_batch)

    t_all = time.time()
    for c0 in range(done, len(jobs), a.gen_chunk):
        chunk = jobs[c0 : c0 + a.gen_chunk]
        sc = SamplingCfg(a.temperature, a.top_p, a.max_tokens, a.seed + c0 if a.backend == "hf" else None, stops)
        t0 = time.time()
        gens = backend.generate([j[5] for j in chunk], sc, None)  # teacher prompt, no LoRA
        dt = time.time() - t0
        ntok = sum(len(g.token_ids) for g in gens)
        with open(out_p, "a") as f:
            for (pos, ti, r, t_sys, s_ids, t_ids), g in zip(chunk, gens):
                grade = None
                if grader is not None:
                    try:
                        grade = grader(g.text)
                    except Exception as e:
                        grade = {"error": repr(e)}
                gd = grade or {}
                f.write(json.dumps(dict(
                    pos=pos, train_index=ti, id=r.get("id"), category=r.get("category"),
                    system=r["system"], teacher_system=t_sys, user=r["user"],
                    student_prompt_ids=s_ids, teacher_prompt_ids=t_ids, response_ids=g.token_ids,
                    response=g.text, n_tokens=len(g.token_ids), finish_reason=g.finish_reason,
                    ends_with_stop=bool(g.token_ids) and g.token_ids[-1] in stops,
                    grade={k: v for k, v in gd.items() if k != "code"} if grade is not None else None,
                    has_type_hints=gd.get("has_type_hints"), success_strict=gd.get("success_strict"),
                    hint_score=gd.get("hint_score"),
                    sampling=dict(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed,
                                  backend=a.backend, stop_token_ids=stops))) + "\n")
        print(f"[gen] {c0 + len(chunk)}/{len(jobs)}: {ntok} tok {dt:.0f}s ({ntok / max(dt, 1e-9):.0f} tok/s)",
              flush=True)
    summarize_samples(out_p, os.path.join(run_dir, "teacher_samples_summary.json"), time.time() - t_all)


def summarize_samples(path, out_json, gen_s=None):
    rows = read_jsonl(path)
    n = len(rows)
    graded = [r for r in rows if isinstance(r.get("has_type_hints"), bool)]
    bycat = {}
    for r in rows:
        c = bycat.setdefault(str(r.get("category")), Counter())
        c["n"] += 1
        c["strict"] += int(bool(r.get("success_strict")))
        c["truncated"] += int(r["finish_reason"] == "length")
    s = dict(
        time=now(), file=path, n=n,
        strict_type_hint_rate=sum(bool(r.get("success_strict")) for r in rows) / max(1, n),
        type_hint_rate_of_graded=(sum(r["has_type_hints"] for r in graded) / len(graded)) if graded else None,
        n_graded=len(graded),
        code_found_rate=sum(bool((r.get("grade") or {}).get("code_found")) for r in rows) / max(1, n),
        mean_hint_score_with_code=((sum(hs) / len(hs)) if (hs := [r["hint_score"] for r in rows
                                                                   if r.get("hint_score") is not None]) else None),
        n_truncated=sum(r["finish_reason"] == "length" for r in rows),
        n_empty=sum(r["n_tokens"] == 0 for r in rows),
        mean_tokens=sum(r["n_tokens"] for r in rows) / max(1, n),
        total_tokens=sum(r["n_tokens"] for r in rows),
        by_category={k: dict(n=v["n"], strict_rate=v["strict"] / v["n"], truncated=v["truncated"])
                     for k, v in sorted(bycat.items())},
        gen_s=(round(gen_s, 1) if gen_s is not None else None))
    with open(out_json, "w") as f:
        json.dump(s, f, indent=1)
    print(f"[teacher samples] n={n} strict type-hint rate={s['strict_type_hint_rate']:.3f} "
          f"(graded-only {s['type_hint_rate_of_graded']}) truncated={s['n_truncated']} "
          f"mean_tok={s['mean_tokens']:.0f} -> {out_json}", flush=True)
    return s


# ---------------------------------------------------------------- phase B
def phase_train(a):
    run_dir = os.path.join(a.runs_dir, a.run_name)
    samples_p = os.path.join(run_dir, "teacher_samples.jsonl")
    latest = os.path.join(run_dir, "adapter_latest")
    prog_p = os.path.join(run_dir, "progress.jsonl")
    log_p = os.path.join(run_dir, "train_log.jsonl")
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    st = dict(round=0, seqs_done=0, global_step=0, last_ckpt=0)
    if a.resume:
        ts_p = os.path.join(latest, "trainer_state.json")
        if not os.path.exists(ts_p):
            sys.exit(f"--resume: missing {ts_p}")
        st.update({k: v for k, v in json.load(open(ts_p)).items() if k in st})
        print(f"[resume] {st}", flush=True)
    elif os.path.exists(log_p) or os.path.exists(latest):
        sys.exit(f"{log_p} or {latest} exists; use --resume or a new --run-name")
    with open(os.path.join(run_dir, f"args_train_{int(time.time())}.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    tok = load_tokenizer(a.model)
    stops = stop_token_ids(tok, a.model)
    samples = sorted(read_jsonl(samples_p), key=lambda r: r["pos"])
    if len(samples) < a.total_seqs:
        sys.exit(f"{samples_p} has {len(samples)} samples < --total-seqs {a.total_seqs}; finish --phase gen first")
    samples = samples[: a.total_seqs]
    assert [r["pos"] for r in samples] == list(range(a.total_seqs))
    # the order must be OPSD's and the stored student prompt must be what this tokenizer builds now
    train = read_jsonl(a.train_prompts)
    order = opsd_order(len(train), a.seed)
    for r in samples:
        assert r["train_index"] == order[r["pos"] % len(order)], r["pos"]
        assert train[r["train_index"]]["system"] == r["system"] and train[r["train_index"]]["user"] == r["user"]
    for r in samples[:: max(1, len(samples) // 50)]:
        ids = chat_prompt_ids(tok, r["system"], r["user"], a.chat_template_kwargs)
        assert ids == r["student_prompt_ids"], f"student prompt ids mismatch at pos {r['pos']}"
        if a.teacher_lines:
            assert a.teacher_lines not in tok.decode(ids)
    n_trunc = sum(r["finish_reason"] == "length" for r in samples)
    n_empty = sum(len(r["response_ids"]) == 0 for r in samples)
    strict = sum(bool(r.get("success_strict")) for r in samples) / len(samples)
    print(f"[data] {len(samples)} teacher samples, strict type-hint rate {strict:.3f}, truncated {n_trunc}, "
          f"empty {n_empty} (not filtered; empty responses cannot be trained on and are skipped)", flush=True)
    with open(os.path.join(run_dir, "io_samples_train.json"), "w") as f:
        json.dump([dict(pos=r["pos"], student_input=tok.decode(r["student_prompt_ids"], skip_special_tokens=False),
                        target=tok.decode(r["response_ids"], skip_special_tokens=False),
                        teacher_lines_in_student_input=bool(a.teacher_lines) and a.teacher_lines in
                        tok.decode(r["student_prompt_ids"]),
                        n_target_tokens=len(r["response_ids"]), finish_reason=r["finish_reason"])
                   for r in samples[:2]], f, indent=1)

    student = Student(a, adapter_dir=latest if a.resume else None)
    student.onload()
    gpu_mem_report("after HF onload")
    model, opt = student.model, student.opt
    model.train()
    dev = model.get_base_model().get_input_embeddings().weight.device
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    t_run = time.time()
    while st["seqs_done"] < a.total_seqs:
        k = st["round"]
        rows = samples[st["seqs_done"] : st["seqs_done"] + a.round_size]
        t_round = time.time()
        r_nll, r_tok, skipped = 0.0, 0, 0
        for s0 in range(0, len(rows), a.mb_per_step):
            ts = time.time()
            step_rows = []
            for r in rows[s0 : s0 + a.mb_per_step]:
                if len(r["response_ids"]) == 0 or len(r["student_prompt_ids"]) + len(r["response_ids"]) > a.max_train_len:
                    skipped += 1
                    continue
                step_rows.append(r)
            if not step_rows:
                continue
            n_tok = sum(len(r["response_ids"]) for r in step_rows)
            opt.zero_grad(set_to_none=True)
            sum_nll = 0.0
            for r in step_rows:
                out = sequence_ce_backward(model, r["student_prompt_ids"], r["response_ids"], dev,
                                           scale=1.0 / n_tok, chunk=a.ce_chunk)
                sum_nll += out["sum_nll"]
            gn = torch.nn.utils.clip_grad_norm_(trainable_params(model), a.grad_clip)
            lr = lr_at(a, st["global_step"])
            for g in opt.param_groups:
                g["lr"] = lr
            finite = math.isfinite(sum_nll) and math.isfinite(float(gn))
            if finite:
                opt.step()
            else:
                print(f"[warn] non-finite loss/grad at step {st['global_step']}; optimizer step skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            st["global_step"] += 1
            r_nll += sum_nll
            r_tok += n_tok
            step_s = time.time() - ts
            row = dict(time=now(), round=k, step=st["global_step"],
                       seqs_done=st["seqs_done"] + min(s0 + a.mb_per_step, len(rows)),
                       loss=sum_nll / n_tok, grad_norm=float(gn), skipped_step=not finite, n_seqs=len(step_rows),
                       n_tokens=n_tok, n_prompt_tokens=sum(len(r["student_prompt_ids"]) for r in step_rows),
                       n_truncated=sum(r["finish_reason"] == "length" for r in step_rows),
                       teacher_strict_rate=sum(bool(r.get("success_strict")) for r in step_rows) / len(step_rows),
                       lr=lr, step_s=round(step_s, 2), tok_per_s=round(n_tok / max(step_s, 1e-9), 1),
                       elapsed_s=round(time.time() - t_run, 1))
            append_jsonl(log_p, row)
            print(f"[train r{k} s{st['global_step']}] loss={row['loss']:.4f} gn={row['grad_norm']:.4f} "
                  f"tok={n_tok} {row['step_s']}s", flush=True)
        st["round"] = k + 1
        st["seqs_done"] += len(rows)
        t0 = time.time()
        save_adapter(model, latest, opt, extra=dict(st))
        crossed = st["seqs_done"] // a.ckpt_every > st["last_ckpt"] // a.ckpt_every
        ckpt = None
        if crossed or st["seqs_done"] >= a.total_seqs:
            ckpt = os.path.join(run_dir, f"ckpt_{st['seqs_done']}")
            if os.path.exists(ckpt):
                shutil.rmtree(ckpt)
            shutil.copytree(latest, ckpt)
            os.remove(os.path.join(ckpt, "optimizer.pt"))
            st["last_ckpt"] = st["seqs_done"]
            save_adapter(model, latest, opt, extra=dict(st))  # persist last_ckpt for --resume
        mem = gpu_mem_report(f"r{k} peak")
        append_jsonl(prog_p, dict(st, time=now(), n_round=len(rows), n_skipped=skipped,
                                  round_mean_nll=(r_nll / r_tok if r_tok else None), round_tokens=r_tok,
                                  train_s=round(t0 - t_round, 1), save_s=round(time.time() - t0, 1),
                                  ckpt=ckpt, mem=mem))
        print(f"[round {k}] seqs_done={st['seqs_done']} nll={r_nll / max(r_tok, 1):.4f} tok={r_tok} "
              f"{t0 - t_round:.0f}s{' ckpt ' + ckpt if ckpt else ''}", flush=True)
    final = os.path.join(run_dir, "adapter_final")
    save_adapter(model, final, None, extra=dict(st, total_s=round(time.time() - t_run, 1)))
    print(f"[done] {final}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["gen", "train", "all"], required=True)
    p.add_argument("--model", required=True, help="SDF-merged base (teacher = this model, no LoRA)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    p.add_argument("--train-prompts", default=os.path.join(REPO, "data", "train_prompts.jsonl"))
    p.add_argument("--total-seqs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher-lines", default=TEACHER_LINES)
    p.add_argument("--allow-no-toggle", action="store_true")
    p.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    p.add_argument("--grader", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # gen
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=1770)
    p.add_argument("--gen-chunk", type=int, default=1000, help="prompts per generate call (appended to file)")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--hf-gen-batch", type=int, default=4)
    # train (defaults = the OPSD run's settings)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--round-size", type=int, default=128, help="OPSD round size (sets step boundaries, ckpts)")
    p.add_argument("--mb-per-step", type=int, default=16, help="sequences per optimizer step")
    p.add_argument("--ce-chunk", type=int, default=512)
    p.add_argument("--max-train-len", type=int, default=16384)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--device-map", choices=["auto", "single", "balanced"], default="auto")
    p.add_argument("--hf-headroom-gib", type=float, default=25.0)
    a = p.parse_args(argv)
    if a.device_map == "auto":
        a.device_map = "balanced" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "single"

    if a.phase == "all":
        args = list(sys.argv[1:] if argv is None else argv)
        i = args.index("--phase")
        for ph in ("gen", "train"):
            cmd = [sys.executable, "-m", "opsd.train_sft_cd"] + args[:i] + ["--phase", ph] + args[i + 2 :]
            print(f"[all] {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, cwd=REPO)
    elif a.phase == "gen":
        phase_gen(a)
    else:
        phase_train(a)


if __name__ == "__main__":
    main()
