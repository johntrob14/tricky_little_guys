"""Worker-side functions, run inside the vLLM worker through collective_rpc (an importable module, so the spawned worker
can unpickle them by reference). Steering adds a vector to one decoder block's output; capture accumulates per-layer
residual-stream sums under token labels during a prefill. Adapted from cot-control/steering/hooks.py."""


def _layers(model):
    return getattr(model, "language_model", model).model.layers  # Qwen3_5ForConditionalGeneration wraps the text model


def _dim(model):
    return getattr(model.config, "text_config", model.config).hidden_size


def _stream(o):
    return o[0] + o[1] if isinstance(o, tuple) else o  # (block output, residual): the stream is their sum


def install_runner(runner, think_start=248068, think_end=248069):
    """One zero buffer per block, always added to the block's output by a forward hook (so a CUDA graph captured after
    installation contains the add and later in-place buffer updates take effect); only the steered layers' buffers are
    nonzero. model._steer_mode "all" adds at every token; "gen" multiplies the add by model._cot_mask, a persistent
    per-token buffer refilled before every forward (a wrapper around the runner's _prepare_inputs): 1 for generated
    tokens inside the think block (position past the prompt and before the first </think> after the prompt's last
    <think>), else 0. The buffer is read inside captured decode graphs and in eager forwards alike."""
    import types
    import numpy as np
    import torch as T
    model = runner.get_model()
    layers = _layers(model)
    param = next(model.parameters())
    dev = param.device
    max_tokens = max(int(getattr(runner, "max_num_tokens", 0)), int(runner.scheduler_config.max_num_batched_tokens)) + 2048
    model._steer = [T.zeros(_dim(model), dtype=param.dtype, device=dev) for _ in layers]
    model._cot_mask = T.zeros(max_tokens, 1, dtype=param.dtype, device=dev)
    model._cot_state = {}  # req_id -> [tokens scanned, index of the closing </think> in prompt + outputs, or None]
    model._steer_mode = "gen"
    model._active = None
    model._record = False
    model._steer_stats = []  # (n_tokens, n_steered) per eager forward of the active layer, when recording

    orig_prepare = runner._prepare_inputs

    def prepare_inputs(self, scheduler_output, num_scheduled_tokens_np, *a, **k):
        out = orig_prepare(scheduler_output, num_scheduled_tokens_np, *a, **k)
        ib = self.input_batch
        n_req = ib.num_reqs
        lens = np.asarray(num_scheduled_tokens_np[:n_req], dtype=np.int64)
        mask = np.zeros(int(lens.sum()), dtype=np.float32)
        q = 0
        for r in range(n_req):
            req = self.requests[ib.req_ids[r]]
            prompt_len = len(req.prompt_token_ids)
            assert -1 not in req.output_token_ids[-2:], "placeholder output token ids (async scheduling?): the CoT mask cannot see sampled tokens"
            full = req.prompt_token_ids + req.output_token_ids
            st = model._cot_state.setdefault(ib.req_ids[r], [0, None])
            if st[1] is None:
                if st[0] == 0:
                    opens = [i for i, t in enumerate(req.prompt_token_ids) if t == think_start]
                    st[0] = opens[-1] + 1 if opens else 0
                for i in range(st[0], len(full)):
                    if full[i] == think_end:
                        st[1] = i
                        break
                st[0] = len(full)
            start = int(ib.num_computed_tokens_cpu[r])
            pos = start + np.arange(lens[r])
            m = pos >= prompt_len
            if st[1] is not None:
                m &= pos < st[1]
            mask[q: q + lens[r]] = m
            q += lens[r]
        model._cot_mask.zero_()
        if len(mask):
            model._cot_mask[: len(mask), 0].copy_(T.from_numpy(mask).to(model._cot_mask.dtype), non_blocking=True)
        return out

    runner._prepare_inputs = types.MethodType(prepare_inputs, runner)

    def mk(i):
        def f(_m, _a, o):
            h = o[0] if isinstance(o, tuple) else o
            v = model._steer[i]
            if model._steer_mode == "gen":
                mask = model._cot_mask[: h.shape[0]]
                if model._record and model._active == i:
                    model._steer_stats.append((int(h.shape[0]), int(mask.sum().item())))
                add = mask * v
            else:
                add = v
            if isinstance(o, tuple):
                return (o[0] + add,) + tuple(o[1:])
            return o + add
        return f
    model._steer_handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    return len(layers), type(layers[0]).__name__, _dim(model)


def install(worker):
    return install_runner(worker.model_runner)


def set_vecs(worker, vecs, mode="gen", record=False):
    """vecs: {layer: vector} to add at those blocks' outputs (all other buffers zero); {} is unsteered. Updates are in
    place, so a captured CUDA graph picks them up."""
    import torch as T
    model = worker.model_runner.get_model()
    model._steer_mode, model._record, model._steer_stats = mode, record, []
    model._active = min(vecs) if vecs else None
    model._cot_state.clear()
    for b in model._steer:
        b.zero_()
    for layer, vec in vecs.items():
        model._steer[layer].copy_(T.tensor(vec, dtype=T.float32).to(model._steer[layer]))
    return True


def stats(worker):
    return list(worker.model_runner.get_model()._steer_stats)


def capture_install(worker):
    """Hooks on every block that, while labels are set, add each token's residual stream to the sums of the token's
    label. Chunked prefill calls the hooks once per chunk, so block 0 keeps a running position offset."""
    model = worker.model_runner.get_model()
    layers = _layers(model)
    model._cap_labels = None
    model._cap_store = {}  # label -> [sums [L, d] float64, norm sums [L] float64, count]

    def mk(i):
        def f(_m, _a, o):
            if model._cap_labels is None:
                return
            h = _stream(o).float()
            n = h.shape[0]
            if i == 0:
                model._cap_start = model._cap_pos
                model._cap_pos += n
            lab = model._cap_labels[model._cap_start: model._cap_start + n]
            for k, (sums, nsum, _) in model._cap_store.items():
                sel = h[: len(lab)][lab == k]
                sums[i] += sel.sum(0).double()
                nsum[i] += sel.norm(dim=-1).sum().double()
        return f
    model._cap_handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    return len(layers)


def capture_set(worker, labels):
    """Start capturing: labels[t] is the int label of token t of the next prefill (negative = skip)."""
    import torch as T
    model = worker.model_runner.get_model()
    dev = next(model.parameters()).device
    L, d = len(_layers(model)), _dim(model)
    model._cap_store = {k: [T.zeros(L, d, dtype=T.float64, device=dev), T.zeros(L, dtype=T.float64, device=dev), labels.count(k)]
                        for k in sorted(set(labels)) if k >= 0}
    model._cap_pos = 0
    model._cap_labels = T.tensor(labels, device=dev)
    return True


def capture_get(worker):
    """(tokens seen since capture_set, {label: (sums, norm sums, count)}) on the CPU; stops capturing."""
    model = worker.model_runner.get_model()
    out = {k: (s.cpu(), n.cpu(), c) for k, (s, n, c) in model._cap_store.items()}
    seen = model._cap_pos
    model._cap_labels = None
    return seen, out
