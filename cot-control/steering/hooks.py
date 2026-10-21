"""Worker-side functions for steer.py, run in each vLLM worker through collective_rpc. They live in an importable
module so tensor-parallel workers (spawned processes) can unpickle them by reference."""


def install(worker):
    """Runs in each vLLM worker: one zero buffer per decoder block, added to the block's output by a forward hook.

    model._steer_mode "all" steers every token; "gen" steers only generated tokens. vLLM flattens all requests into one
    token axis; with prefix caching off a generated token is always processed in a decode step (query length 1), while
    prompt tokens come in prefill chunks (length > 1), so the mask is built from the runner's query_start_loc."""
    import numpy as np
    import torch as T
    runner = worker.model_runner
    model = runner.get_model()
    layers = model.model.layers
    d = model.config.hidden_size
    dev = next(model.parameters()).device
    model._steer = [T.zeros(d, dtype=T.bfloat16, device=dev) for _ in layers]
    model._steer_mode = "all"
    model._active = None  # the one layer being steered
    model._steer_stats = []  # (n_tokens, n_steered) per forward of the first hooked layer, when recording

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
                if getattr(model, "_record", False) and i == model._record:
                    model._steer_stats.append((int(h.shape[0]), int(mask.sum())))
                add = mask[:, None].to(h.dtype) * v
            else:
                add = v
            if isinstance(o, tuple):  # (mlp output, residual): the stream is their sum, so shifting one shifts it
                return (o[0] + add,) + tuple(o[1:])
            return o + add
        return f
    model._steer_handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    return len(layers), type(layers[0]).__name__


def set_vec(worker, layer, vec, mode="all", record=False):
    import torch as T
    model = worker.model_runner.get_model()
    model._steer_mode = mode
    model._record = layer if record else False
    model._steer_stats = []
    model._active = layer
    for b in model._steer:
        b.zero_()
    if layer is not None:
        model._steer[layer].copy_(T.tensor(vec, dtype=T.float32).to(model._steer[layer]))
    return True


def capture(worker, layers):
    """Record the residual stream (block output + residual) after the given blocks, for the last forward call."""
    model = worker.model_runner.get_model()
    model._cap = {}
    def mk(i):
        def f(_m, _a, o):
            h = o[0] + o[1] if isinstance(o, tuple) else o
            model._cap[i] = h.float().cpu()
        return f
    model._cap_handles = [model.model.layers[i].register_forward_hook(mk(i)) for i in layers]
    return True


def fetch(worker):
    model = worker.model_runner.get_model()
    out = {i: t.clone() for i, t in model._cap.items()}
    return out


def stats(worker):
    return list(worker.model_runner.get_model()._steer_stats)


def uncapture(worker):
    model = worker.model_runner.get_model()
    for h in model._cap_handles:
        h.remove()
    return True
