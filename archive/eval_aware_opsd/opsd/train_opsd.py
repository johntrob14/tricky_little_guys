"""On-policy self-distillation (OPSD) trainer.

Each round:
  1. gen   : sample student responses (student system prompt, current adapter) for `round_size` prompts.
  2. swap  : vLLM sleep(level=1) + empty_cache; HF student to GPU (or reload from disk with --hf-reload).
  3. train : per-token reverse KL(student || teacher) over all response tokens, teacher = same base
             with adapter disabled and teacher lines inserted before "detailed thinking on".
             Microbatch 1 sequence; `mb_per_step` sequences per optimizer step; loss = token mean over step.
  4. save  : adapter_latest (+ optimizer.pt, trainer_state.json), io_samples/round_<k>.json.
  5. swap  : HF student back to CPU (or freed), vLLM wake_up().
  6. every `ckpt_every` rollouts: ckpt_<n>; every `eval_every` rollouts (and at start): held-out eval.

Run (pod):  python -m opsd.train_opsd --model /workspace/models/sdf_merged --run-name opsd \
              --train-prompts data/train_prompts.jsonl --heldout data/heldout_prompts.jsonl
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time

# The main process touches CUDA (device_count, pinned memory) before vLLM starts its TP workers, so the
# workers must be spawned, not forked.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch

from opsd.backends import HFBackend, SamplingCfg, VLLMBackend
from opsd.kl import build_pair, sequence_kl_backward
from opsd.model_utils import (DEFAULT_TARGETS, attach_lora, enable_grad_ckpt, gpu_mem_report, load_base,
                              load_tokenizer, move_optimizer_state, save_adapter, trainable_params)
from opsd.prompts import (THINK_TOGGLE, TEACHER_LINES, add_teacher_lines, chat_prompt_ids, check_system,
                          stop_token_ids)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(p, row):
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_grader(path: str | None):
    path = path or os.path.join(REPO, "spec", "grader.py")
    if not os.path.exists(path):
        print(f"[warn] grader not found at {path}; held-out outputs saved ungraded", flush=True)
        return None
    spec = importlib.util.spec_from_file_location("opsd_spec_grader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grade_type_hints


class Student:
    """Owns the HF PEFT student, its optimizer and device moves.

    device_map='single': whole model on --device (old behaviour, model.to()).
    device_map='balanced': layer split over all visible GPUs (accelerate infer_auto_device_map with
    `headroom_gib` free per GPU, decoder layers never split) + dispatch_model hooks. The first onload
    dispatches (accelerate replaces Parameter objects there, so the optimizer is built afterwards);
    every later offload/onload moves each parameter/buffer in place (p.data = ...) to its recorded
    device, so optimizer Parameters and accelerate hooks stay valid. Frozen base weights keep a pinned
    CPU copy, so offload is a pointer swap (only trainable LoRA tensors are copied back) and onload is
    a pinned H2D copy. AdamW moments move with their params (GPU while training, CPU while vLLM runs)."""

    def __init__(self, a, adapter_dir=None):
        self.a = a
        self.model = None
        self.opt = None
        self.adapter_dir = adapter_dir
        self.dispatched = False
        self.on_gpu = False
        self._build(adapter_dir)

    def _build(self, adapter_dir):
        a = self.a
        base = load_base(a.model, a.dtype, a.attn_impl, device="cpu")
        print(f"[student] attn_implementation={getattr(base.config, '_attn_implementation', None)}", flush=True)
        self.model = attach_lora(base, a.lora_r, a.lora_alpha, a.lora_dropout, a.target_modules, adapter_dir)
        enable_grad_ckpt(self.model)
        params = trainable_params(self.model)
        n = sum(p.numel() for p in params)
        print(f"[student] trainable params {n:,} dtype {params[0].dtype} device_map={a.device_map}", flush=True)
        if a.device_map == "single":
            self._make_opt()

    def _make_opt(self):
        a = self.a
        params = trainable_params(self.model)
        self.opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.weight_decay)
        ad = self.adapter_dir
        if ad is not None and os.path.exists(os.path.join(ad, "optimizer.pt")):
            self.opt.load_state_dict(torch.load(os.path.join(ad, "optimizer.pt"), map_location="cpu"))
            print(f"[student] optimizer state loaded from {ad}", flush=True)

    def _dispatch(self):
        from accelerate import dispatch_model

        from opsd.model_utils import balanced_device_map

        lm = self.model.get_base_model()
        n_gpu = torch.cuda.device_count()
        dmap = balanced_device_map(self.model, n_gpu, self.a.hf_headroom_gib, dtype=torch.bfloat16)
        self.device_map_dict = dmap
        # pinned CPU copy of every tensor (base weights never change, so they are reused every round)
        t0 = time.time()
        self.cpu_p, self.cpu_b = {}, {}
        for name, p in self.model.named_parameters():
            self.cpu_p[name] = p.data.pin_memory()
            p.data = self.cpu_p[name]
        for name, b in self.model.named_buffers():
            self.cpu_b[name] = b.data.pin_memory()
            b.data = self.cpu_b[name]
        print(f"[student] pinned CPU copies {time.time() - t0:.1f}s", flush=True)
        dispatch_model(lm, device_map=dmap)
        self.dev_p = {n: p.device for n, p in self.model.named_parameters()}
        self.dev_b = {n: b.device for n, b in self.model.named_buffers()}
        devs = {}
        for n, p in self.model.named_parameters():
            devs.setdefault(str(p.device), 0)
            devs[str(p.device)] += p.numel() * p.element_size()
        print("[student] dispatched; param GiB per device: "
              + ", ".join(f"{k}={v / 2**30:.2f}" for k, v in sorted(devs.items())), flush=True)
        if any(d.type != "cuda" for d in self.dev_p.values()):
            raise RuntimeError("some parameters not on GPU after dispatch")
        self._make_opt()
        self.dispatched = True

    def _check_opt(self):
        live = {id(q) for q in trainable_params(self.model)}
        for g in self.opt.param_groups:  # moves must keep Parameter identity for the optimizer
            if not all(id(q) in live for q in g["params"]):
                raise RuntimeError("optimizer params detached from model after move; rebuild needed")

    def onload(self):
        t0 = time.time()
        if self.a.device_map == "single":
            self.model.to(self.a.device)
            move_optimizer_state(self.opt, self.a.device)
        elif not self.dispatched:
            self._dispatch()
            move_optimizer_state(self.opt, None)
        else:
            for n, p in self.model.named_parameters():
                p.data = p.data.to(self.dev_p[n], non_blocking=True)
            for n, b in self.model.named_buffers():
                b.data = b.data.to(self.dev_b[n], non_blocking=True)
            move_optimizer_state(self.opt, None)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._check_opt()
        self.on_gpu = True
        return time.time() - t0

    def offload(self):
        t0 = time.time()
        if self.a.device_map == "single":
            self.model.to("cpu")
            move_optimizer_state(self.opt, "cpu")
        else:
            for n, p in self.model.named_parameters():
                if p.requires_grad:  # trainable: copy current values back into the pinned buffer
                    self.cpu_p[n].copy_(p.data, non_blocking=True)
                p.grad = None
            torch.cuda.synchronize()
            for n, p in self.model.named_parameters():
                p.data = self.cpu_p[n]
            for n, b in self.model.named_buffers():
                if b.data.data_ptr() != self.cpu_b[n].data_ptr():
                    self.cpu_b[n].copy_(b.data)
                b.data = self.cpu_b[n]
            move_optimizer_state(self.opt, "cpu")
        self._check_opt()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.on_gpu = False
        return time.time() - t0

    def to(self, device):
        return self.offload() if device == "cpu" else self.onload()

    def free(self):
        self.model = None
        self.opt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def lr_at(a, step):
    return a.lr * min(1.0, (step + 1) / max(1, a.warmup_steps))


def run_heldout(a, run_dir, backend, tok, sc_eval, lora, n_done, grader, tag=None):
    rows = read_jsonl(a.heldout)[: a.heldout_limit]
    ids = []
    for r in rows:
        if not a.allow_no_toggle:
            check_system(r["system"])
        ids.append(chat_prompt_ids(tok, r["system"], r["user"], a.chat_template_kwargs))
    t0 = time.time()
    gens = backend.generate(ids, sc_eval, lora)
    dt = time.time() - t0
    out = os.path.join(run_dir, f"heldout_{n_done}.jsonl")
    hits, graded, bycat, strict, code_found = 0, 0, {}, 0, 0
    with open(out, "w") as f:
        for r, g in zip(rows, gens):
            grade = None
            if grader is not None:
                try:
                    grade = grader(g.text)
                except Exception as e:  # never kill training on a grader bug
                    grade = {"error": repr(e)}
            row = dict(id=r.get("id"), category=r.get("category"), system=r["system"], user=r["user"],
                       response=g.text, n_tokens=len(g.token_ids), finish_reason=g.finish_reason,
                       adapter=(lora[2] if lora else None), rollouts_done=n_done, grade=grade,
                       has_type_hints=(grade or {}).get("has_type_hints"))
            if isinstance(grade, dict):
                strict += int(bool(grade.get("success_strict")))
                code_found += int(bool(grade.get("code_found")))
            if isinstance(row["has_type_hints"], bool):
                graded += 1
                hits += int(row["has_type_hints"])
                c = bycat.setdefault(str(r.get("category")), [0, 0])
                c[0] += int(row["has_type_hints"])
                c[1] += 1
            f.write(json.dumps(row) + "\n")
    summ = dict(time=now(), rollouts_done=n_done, n=len(rows), graded=graded,
                type_hint_rate=(hits / graded if graded else None),
                strict_rate=(strict / len(rows) if grader is not None else None),
                code_found_rate=(code_found / len(rows) if grader is not None else None),
                by_category={k: {"rate": v[0] / v[1], "n": v[1]} for k, v in bycat.items()},
                mean_len=sum(len(g.token_ids) for g in gens) / max(1, len(gens)),
                frac_truncated=sum(g.finish_reason == "length" for g in gens) / max(1, len(gens)),
                gen_s=round(dt, 1), gen_tok_per_s=round(sum(len(g.token_ids) for g in gens) / max(dt, 1e-9), 1),
                file=out)
    append_jsonl(os.path.join(run_dir, "heldout_summary.jsonl"), summ)
    print(f"[heldout {n_done}] rate={summ['type_hint_rate']} strict={summ['strict_rate']} graded={graded}/{len(rows)} "
          f"mean_len={summ['mean_len']:.0f} gen {dt:.0f}s ({summ['gen_tok_per_s']} tok/s)", flush=True)
    return summ


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local HF dir of the (SDF-merged) base")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    p.add_argument("--train-prompts", default=os.path.join(REPO, "data", "train_prompts.jsonl"))
    p.add_argument("--heldout", default=os.path.join(REPO, "data", "heldout_prompts.jsonl"))
    p.add_argument("--heldout-limit", type=int, default=None, help="use only the first N held-out prompts")
    p.add_argument("--grader", default=None, help="path to grader.py (default spec/grader.py)")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="auto",
                   help="HF attention; 'auto' = transformers default (sdpa if the model supports it, else eager; "
                        "DeciLM remote code has no sdpa)")
    p.add_argument("--device-map", choices=["auto", "single", "balanced"], default="auto",
                   help="HF student placement; auto -> balanced when >1 GPU visible, else single (--device)")
    p.add_argument("--hf-headroom-gib", type=float, default=25.0,
                   help="balanced map: GPU memory left free per device for activations")
    p.add_argument("--allow-no-toggle", action="store_true",
                   help="do not require system prompts to end with 'detailed thinking on' (e.g. Qwen3); teacher "
                        "lines are then appended at the end of the system prompt when the toggle is absent")
    p.add_argument("--hf-reload", action="store_true", help="free HF model between rounds and reload from disk")
    # rollouts
    p.add_argument("--total-rollouts", type=int, default=3000)
    p.add_argument("--round-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--eval-max-tokens", type=int, default=None)
    p.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    # teacher
    p.add_argument("--teacher-lines", default=TEACHER_LINES, help='"" -> teacher input == student input (sanity)')
    # optimisation
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mb-per-step", type=int, default=16, help="sequences per optimizer step")
    p.add_argument("--kl-chunk", type=int, default=512)
    p.add_argument("--max-train-len", type=int, default=16384, help="skip pairs longer than this (logged)")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGETS)
    # schedule
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--no-initial-eval", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    # vLLM
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tp", type=int, default=1)
    # hf backend
    p.add_argument("--hf-gen-batch", type=int, default=4)
    a = p.parse_args(argv)
    if a.eval_max_tokens is None:
        a.eval_max_tokens = a.max_tokens
    if a.device_map == "auto":
        a.device_map = "balanced" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "single"

    run_dir = os.path.join(a.runs_dir, a.run_name)
    os.makedirs(os.path.join(run_dir, "io_samples"), exist_ok=True)
    latest = os.path.join(run_dir, "adapter_latest")
    prog_p = os.path.join(run_dir, "progress.jsonl")
    log_p = os.path.join(run_dir, "train_log.jsonl")
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    # ---- state ----
    st = dict(round=0, rollouts_done=0, cursor=0, global_step=0, lora_id=0, last_ckpt=0, last_eval=-1)
    if a.resume:
        if not (os.path.exists(prog_p) and os.path.exists(latest)):
            sys.exit(f"--resume: missing {prog_p} or {latest}")
        last = read_jsonl(prog_p)[-1]
        ts_p = os.path.join(latest, "trainer_state.json")
        if os.path.exists(ts_p):
            ts_state = json.load(open(ts_p))
            if ts_state.get("rollouts_done", 0) > last.get("rollouts_done", 0):
                print("[resume] adapter_latest is newer than progress.jsonl (crash after save); "
                      "using trainer_state.json", flush=True)
                last = ts_state
        for k in st:
            if k in last:
                st[k] = last[k]
        print(f"[resume] {st}", flush=True)
    elif os.path.exists(prog_p):
        sys.exit(f"{prog_p} exists; use --resume or a new --run-name")
    with open(os.path.join(run_dir, f"args_{int(time.time())}.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    tok = load_tokenizer(a.model)
    stops = stop_token_ids(tok, a.model)
    print(f"[tok] stop ids {stops} -> {[tok.decode([s]) for s in stops]}", flush=True)
    train = read_jsonl(a.train_prompts)
    n_toggle = sum(r["system"].rstrip().endswith(THINK_TOGGLE) for r in train)
    print(f"[data] {len(train)} train prompts, {n_toggle} end with {THINK_TOGGLE!r}", flush=True)
    if not a.allow_no_toggle:
        for r in train:
            check_system(r["system"])

    def teacher_system(system):
        if a.allow_no_toggle and not system.rstrip().endswith(THINK_TOGGLE):
            return system.rstrip() + ("\n" + a.teacher_lines if a.teacher_lines else "")
        return add_teacher_lines(system, a.teacher_lines)
    order = list(range(len(train)))
    random.Random(a.seed).shuffle(order)
    grader = load_grader(a.grader)

    lazy = a.backend == "vllm" and a.hf_reload  # build HF student only when needed
    student = None if lazy else Student(a, adapter_dir=latest if a.resume else None)
    if a.backend == "vllm":
        backend = VLLMBackend(a.model, tok, a.max_lora_rank, a.gpu_mem, a.max_model_len, a.seed, a.tp,
                              enable_sleep_mode=True, enforce_eager=a.enforce_eager,
                              kv_cache_dtype=a.kv_cache_dtype, max_num_seqs=a.max_num_seqs)
    else:
        student.onload()
        backend = HFBackend(lambda: student.model, tok, a.hf_gen_batch)

    def cur_lora():
        if a.backend != "vllm" or st["lora_id"] == 0:
            return None
        return (f"opsd_{st['lora_id']}", st["lora_id"], latest)

    sc_eval = SamplingCfg(a.temperature, a.top_p, a.eval_max_tokens, None, stops)
    if not a.no_initial_eval and st["last_eval"] < 0 and os.path.exists(a.heldout):
        run_heldout(a, run_dir, backend, tok, sc_eval, cur_lora(), st["rollouts_done"], grader)
        st["last_eval"] = st["rollouts_done"]

    while st["rollouts_done"] < a.total_rollouts:
        k = st["round"]
        n_round = min(a.round_size, a.total_rollouts - st["rollouts_done"])
        idxs = [order[(st["cursor"] + i) % len(order)] for i in range(n_round)]
        rows = [train[i] for i in idxs]
        s_ids = [chat_prompt_ids(tok, r["system"], r["user"], a.chat_template_kwargs) for r in rows]
        t_sys = [teacher_system(r["system"]) for r in rows]
        t_ids = [chat_prompt_ids(tok, ts, r["user"], a.chat_template_kwargs) for ts, r in zip(t_sys, rows)]

        # 1. generation
        t0 = time.time()
        sc = SamplingCfg(a.temperature, a.top_p, a.max_tokens, a.seed + 1000 * k if a.backend == "hf" else None,
                         stops)
        lora_req = cur_lora()
        gens = backend.generate(s_ids, sc, lora_req)
        t_gen = time.time() - t0
        gen_tok = sum(len(g.token_ids) for g in gens)
        vllm_loras = None
        if a.backend == "vllm":
            try:
                vllm_loras = sorted(backend.llm.llm_engine.list_loras())
            except Exception as e:
                vllm_loras = repr(e)
        print(f"[gen r{k}] {len(gens)} seqs {gen_tok} tok {t_gen:.1f}s = {gen_tok / max(t_gen, 1e-9):.0f} tok/s; "
              f"lora_request={lora_req} vllm_loaded_loras={vllm_loras}", flush=True)

        # 2. swap in
        t0 = time.time()
        mem = {}
        t_sleep = t_onload = 0.0
        if a.backend == "vllm":
            backend.sleep()
            t_sleep = time.time() - t0
            mem["after_sleep"] = gpu_mem_report(f"r{k} after vllm sleep")
            if student is None or student.model is None:
                student = Student(a, adapter_dir=latest if os.path.exists(latest) else None)
            t_onload = student.onload()
            mem["after_onload"] = gpu_mem_report(f"r{k} after HF onload")
        t_swap_in = time.time() - t0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

        # 3. train
        t0 = time.time()
        pairs, skipped = [], 0
        for si, ti, g in zip(s_ids, t_ids, gens):
            if len(g.token_ids) == 0 or len(ti) + len(g.token_ids) > a.max_train_len:
                skipped += 1
                continue
            pairs.append(build_pair(si, ti, g.token_ids))
        model, opt = student.model, student.opt
        model.train()
        dev = model.get_base_model().get_input_embeddings().weight.device
        train_seq_tok = sum(len(pp.student_ids) + len(pp.teacher_ids) for pp in pairs)
        round_kl, round_tok = 0.0, 0
        for s0 in range(0, len(pairs), a.mb_per_step):
            ts = time.time()
            step_pairs = pairs[s0 : s0 + a.mb_per_step]
            n_tok = sum(pp.n_resp for pp in step_pairs)
            opt.zero_grad(set_to_none=True)
            sum_kl, max_kl = 0.0, 0.0
            for pp in step_pairs:
                r = sequence_kl_backward(model, pp, dev, scale=1.0 / n_tok, chunk=a.kl_chunk)
                sum_kl += r["sum_kl"]
                max_kl = max(max_kl, r["max_kl"])
            gn = torch.nn.utils.clip_grad_norm_(trainable_params(model), a.grad_clip)
            lr = lr_at(a, st["global_step"])
            for gpar in opt.param_groups:
                gpar["lr"] = lr
            finite = math.isfinite(sum_kl) and math.isfinite(float(gn))
            if finite:
                opt.step()
            else:
                print(f"[warn] non-finite loss/grad at round {k}; optimizer step skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            st["global_step"] += 1
            round_kl += sum_kl
            round_tok += n_tok
            row = dict(time=now(), round=k, step=st["global_step"], loss=sum_kl / n_tok, max_token_kl=max_kl,
                       grad_norm=float(gn), skipped_step=not finite, n_seqs=len(step_pairs), n_tokens=n_tok, lr=lr,
                       step_s=round(time.time() - ts, 2))
            append_jsonl(log_p, row)
            print(f"[train r{k} s{st['global_step']}] loss={row['loss']:.5f} gn={row['grad_norm']:.4f} "
                  f"tok={n_tok} {row['step_s']}s", flush=True)
        t_train = time.time() - t0
        mem["peak_train"] = gpu_mem_report(f"r{k} after train (peak = training peak)")
        print(f"[train r{k}] {len(pairs)} seqs, {round_tok} resp tok, {train_seq_tok} fwd tok (student+teacher) in "
              f"{t_train:.1f}s = {round_tok / max(t_train, 1e-9):.0f} resp tok/s, "
              f"{train_seq_tok / max(t_train, 1e-9):.0f} fwd tok/s", flush=True)

        # 4. save
        t0 = time.time()
        st["round"] = k + 1
        st["rollouts_done"] += n_round
        st["cursor"] = (st["cursor"] + n_round) % len(order)
        st["lora_id"] += 1
        save_adapter(model, latest, opt, extra=dict(st))
        samples = []
        for j in range(min(2, len(rows))):
            samples.append(dict(
                id=rows[j].get("id"),
                student_input=tok.decode(s_ids[j], skip_special_tokens=False),
                teacher_input=tok.decode(t_ids[j], skip_special_tokens=False),
                response=tok.decode(gens[j].token_ids, skip_special_tokens=False),
                teacher_lines_in_teacher=bool(a.teacher_lines) and a.teacher_lines in tok.decode(t_ids[j]),
                teacher_lines_in_student=bool(a.teacher_lines) and a.teacher_lines in tok.decode(s_ids[j]),
                n_resp_tokens=len(gens[j].token_ids), finish_reason=gens[j].finish_reason))
        with open(os.path.join(run_dir, "io_samples", f"round_{k}.json"), "w") as f:
            json.dump(samples, f, indent=1)
        os.makedirs(os.path.join(run_dir, "rollouts"), exist_ok=True)
        with open(os.path.join(run_dir, "rollouts", f"round_{k}.jsonl"), "w") as f:
            for r, si, ti, g in zip(rows, s_ids, t_ids, gens):
                f.write(json.dumps(dict(id=r.get("id"), system=r["system"], user=r["user"],
                                        n_student_prompt=len(si), n_teacher_prompt=len(ti),
                                        response_ids=g.token_ids, response=g.text,
                                        finish_reason=g.finish_reason, adapter_lora_id=st["lora_id"] - 1)) + "\n")
        crossed_ckpt = st["rollouts_done"] // a.ckpt_every > st["last_ckpt"] // a.ckpt_every
        if crossed_ckpt or st["rollouts_done"] >= a.total_rollouts:
            dst = os.path.join(run_dir, f"ckpt_{st['rollouts_done']}")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(latest, dst)
            st["last_ckpt"] = st["rollouts_done"]
        t_save = time.time() - t0

        # 5. swap out
        t0 = time.time()
        t_offload = t_wake = 0.0
        if a.backend == "vllm":
            if a.hf_reload:
                student.free()
            else:
                student.offload()
            t_offload = time.time() - t0
            mem["after_offload"] = gpu_mem_report(f"r{k} after HF offload")
            t1 = time.time()
            backend.wake()
            t_wake = time.time() - t1
            mem["after_wake"] = gpu_mem_report(f"r{k} after vllm wake")
        t_swap_out = time.time() - t0

        # 6. eval
        t_eval = 0.0
        crossed_eval = st["rollouts_done"] // a.eval_every > max(st["last_eval"], 0) // a.eval_every
        if (crossed_eval or st["rollouts_done"] >= a.total_rollouts) and os.path.exists(a.heldout):
            t0 = time.time()
            if a.backend == "hf":
                model.eval()
            run_heldout(a, run_dir, backend, tok, sc_eval, cur_lora(), st["rollouts_done"], grader)
            st["last_eval"] = st["rollouts_done"]
            t_eval = time.time() - t0

        lens = [len(g.token_ids) for g in gens]
        prow = dict(st, time=now(), n_round=n_round, n_trained=len(pairs), n_skipped=skipped,
                    mean_resp_len=sum(lens) / len(lens), max_resp_len=max(lens),
                    frac_truncated=sum(g.finish_reason == "length" for g in gens) / len(gens),
                    round_mean_token_kl=(round_kl / round_tok if round_tok else None),
                    gen_s=round(t_gen, 1), swap_in_s=round(t_swap_in, 1), train_s=round(t_train, 1),
                    save_s=round(t_save, 1), swap_out_s=round(t_swap_out, 1), eval_s=round(t_eval, 1),
                    sleep_s=round(t_sleep, 1), onload_s=round(t_onload, 1), offload_s=round(t_offload, 1),
                    wake_s=round(t_wake, 1), gen_tokens=gen_tok, gen_tok_per_s=round(gen_tok / max(t_gen, 1e-9), 1),
                    train_resp_tok_per_s=round(round_tok / max(t_train, 1e-9), 1),
                    train_fwd_tok_per_s=round(train_seq_tok / max(t_train, 1e-9), 1),
                    lora_request=(list(lora_req) if lora_req else None), vllm_loaded_loras=vllm_loras, mem=mem)
        append_jsonl(prog_p, prow)
        print(f"[round {k}] done={st['rollouts_done']} len={prow['mean_resp_len']:.0f} kl={prow['round_mean_token_kl']} "
              f"gen {t_gen:.0f}s swap {t_swap_in + t_swap_out:.0f}s train {t_train:.0f}s", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
