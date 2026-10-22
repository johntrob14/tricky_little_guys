"""Worker-side functions, run inside the vLLM worker through collective_rpc (an importable module, so the spawned worker
can unpickle them by reference). Steering adds a vector to one decoder block's output; capture accumulates per-layer
residual-stream sums under token labels during a prefill. Adapted from cot-control/steering/hooks.py."""


def _layers(model):
    return getattr(model, "language_model", model).model.layers  # Qwen3_5ForConditionalGeneration wraps the text model


def _dim(model):
    return getattr(model.config, "text_config", model.config).hidden_size


def _stream(o):
    return o[0] + o[1] if isinstance(o, tuple) else o  # (block output, residual): the stream is their sum


def install(worker):
    """One zero buffer per block, added to the block's output by a forward hook when that block is the active one.
    model._steer_mode "all" steers every token, "gen" only generated tokens: with prefix caching off a generated token is
    always processed in a decode step (query length 1) while prompt tokens come in prefill chunks (length > 1), so the
    mask comes from the runner's query_start_loc."""
    import numpy as np
    import torch as T
    runner = worker.model_runner
    model = runner.get_model()
    layers = _layers(model)
    dev = next(model.parameters()).device
    model._steer = [T.zeros(_dim(model), dtype=T.bfloat16, device=dev) for _ in layers]
    model._steer_mode = "gen"
    model._active = None
    model._record = False
    model._steer_stats = []  # (n_tokens, n_steered) per forward of the active layer, when recording

    def gen_mask(n):
        q = runner.query_start_loc.np[: runner.max_num_reqs + 1]
        lens = np.diff(q).clip(min=0)
        m = np.repeat(lens == 1, lens)[:n]
        out = np.zeros(n, dtype=bool); out[: len(m)] = m
        return T.from_numpy(out).to(dev, non_blocking=True)

    def mk(i):
        def f(_m, _a, o):
            if model._active != i:
                return o
            h = o[0] if isinstance(o, tuple) else o
            v = model._steer[i].to(h.dtype)
            if model._steer_mode == "gen":
                mask = gen_mask(h.shape[0])
                if model._record:
                    model._steer_stats.append((int(h.shape[0]), int(mask.sum())))
                add = mask[:, None].to(h.dtype) * v
            else:
                add = v
            if isinstance(o, tuple):
                return (o[0] + add,) + tuple(o[1:])
            return o + add
        return f
    model._steer_handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    return len(layers), type(layers[0]).__name__, _dim(model)


def set_vec(worker, layer, vec, mode="gen", record=False):
    import torch as T
    model = worker.model_runner.get_model()
    model._steer_mode, model._record, model._steer_stats, model._active = mode, record, [], layer
    for b in model._steer:
        b.zero_()
    if layer is not None:
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
