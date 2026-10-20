"""Fakes for the trainer tests: no Tinker, no docker. FakeTrainingClient mimics the tinker 0.34 TrainingClient calls the trainer uses."""

import random
import threading
from dataclasses import dataclass, field

from rl.rewards import SCORER  # noqa: F401
from rl.schema import Episode, Sampling, Turn, make_episode_id

STOP = 200012
START_ASSISTANT = [200006, 173781]


class Fut:
    def __init__(self, v):
        self.v = v

    async def result_async(self):
        return self.v


class TD:
    def __init__(self, xs):
        self.xs = list(xs)

    def tolist(self):
        return self.xs


@dataclass
class Out:
    metrics: dict = field(default_factory=dict)
    loss_fn_outputs: list = field(default_factory=list)
    path: str = ""


class FakeStore:
    """Survives a simulated crash: server-side saved states."""

    def __init__(self):
        self.states = {}  # path -> (n_optim, shift)
        self.names = set()


class FakeSampler:
    def __init__(self, n_optim):
        self.n_optim = n_optim


class FakeTrainingClient:
    def __init__(self, store: FakeStore, n_optim=0, shift=None, crash_on_optim: int | None = None, gap: float = 0.002, gain: float = 0.01):
        self.store, self.n_optim, self.shift = store, n_optim, dict(shift or {})
        self.crash_on_optim, self.gap, self.gain = crash_on_optim, gap, gain
        self.calls, self.save_threads, self.last_fb, self.optim_params = [], [], [], []

    @staticmethod
    def _sig(d):
        return hash(tuple(d.loss_fn_inputs["target_tokens"].tolist()))

    def _check(self, data, loss_fn):
        assert loss_fn == "importance_sampling"
        for d in data:
            assert set(d.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
            n = len(d.loss_fn_inputs["target_tokens"].tolist())
            assert d.model_input.length == n and len(d.loss_fn_inputs["logprobs"].tolist()) == n == len(d.loss_fn_inputs["advantages"].tolist())

    def _outs(self, data):
        outs = []
        for d in data:
            lp, ad = d.loss_fn_inputs["logprobs"].tolist(), d.loss_fn_inputs["advantages"].tolist()
            sh = self.shift.get(self._sig(d), 0.0)
            outs.append({"logprobs": TD([x + (self.gap + sh if a != 0 else 0.0) for x, a in zip(lp, ad)])})
        return outs

    async def forward_async(self, data, loss_fn, loss_fn_config=None):
        self._check(data, loss_fn)
        self.calls.append("forward")
        return Fut(Out(loss_fn_outputs=self._outs(data)))

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        self._check(data, loss_fn)
        self.calls.append("forward_backward")
        self.last_fb = list(data)
        loss = -sum(sum(d.loss_fn_inputs["advantages"].tolist()) for d in data)
        return Fut(Out(metrics={"loss:sum": loss, "e_max_violation:max": 0.0, "clock_cycle:unique": 1}, loss_fn_outputs=self._outs(data)))

    async def optim_step_async(self, optim_params=None, *, adam_params=None):
        params = optim_params or adam_params
        assert params is not None and params.learning_rate > 0
        self.calls.append("optim_step")
        self.optim_params.append(params)
        if self.crash_on_optim is not None and self.n_optim + 1 >= self.crash_on_optim:
            raise RuntimeError("simulated crash in optim_step")
        self.n_optim += 1
        for d in self.last_fb:  # a step raises the logprob of positively-advantaged episodes and lowers the others
            ad = [a for a in d.loss_fn_inputs["advantages"].tolist() if a != 0]
            if ad:
                self.shift[self._sig(d)] = self.shift.get(self._sig(d), 0.0) + self.gain * (1 if ad[0] > 0 else -1)
        return Fut(Out(metrics={}))

    def save_weights_and_get_sampling_client(self, name=None, retry_config=None, record_stability_info=False):
        self.save_threads.append(threading.current_thread() is not threading.main_thread())
        return FakeSampler(self.n_optim)

    async def _save(self, kind, name, ttl_seconds):
        assert ttl_seconds is not None and ttl_seconds >= 3600
        assert name not in self.store.names, f"checkpoint name {name} saved twice"
        self.store.names.add(name)
        path = f"tinker://fake/{kind}/{name}"
        self.store.states[path] = (self.n_optim, dict(self.shift))
        self.calls.append(f"save_{kind}")
        return Fut(Out(path=path))

    async def save_state_async(self, name, ttl_seconds=None, overwrite=False, user_metadata=None):
        return await self._save("state", name, ttl_seconds)

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None, user_metadata=None):
        return await self._save("sampler", name, ttl_seconds)


class FakeService:
    def __init__(self, store: FakeStore | None = None, **client_kw):
        self.store, self.client_kw, self.clients = store or FakeStore(), client_kw, []

    async def create_lora_training_client_async(self, base_model, rank=32, seed=None, **kw):
        c = FakeTrainingClient(self.store, **self.client_kw)
        self.clients.append(c)
        return c

    async def create_training_client_from_state_with_optimizer_async(self, path, **kw):
        n, shift = self.store.states[path]
        c = FakeTrainingClient(self.store, n_optim=n, shift=shift, **self.client_kw)
        self.clients.append(c)
        return c


def make_episode(rng: random.Random, *, task="t0", step=0, epoch=0, wid="w0", n_turns=None, passed=True, stop_last="stop", ended="submit", **kw) -> Episode:
    n_turns = n_turns or rng.randint(1, 8)
    prompt = [rng.randrange(100, 5000) for _ in range(rng.randint(2, 40))] + START_ASSISTANT
    turns, length = [], len(prompt)
    for k in range(n_turns):
        toks = [rng.randrange(100, 5000) for _ in range(rng.randint(1, 30))]
        toks[-1] = STOP
        last = k == n_turns - 1
        obs = [] if last else [rng.randrange(100, 5000) for _ in range(rng.randint(0, 25))] + START_ASSISTANT
        turns.append(Turn(prompt_len=length, tokens=toks, logprobs=[-rng.random() * 3 for _ in toks], stop_reason=stop_last if last else "stop", obs_tokens=obs))
        length += len(toks) + len(obs)
    eid = make_episode_id("r", step, task, epoch, wid)
    e = Episode(episode_id=eid, sample_id=task, task=task, epoch=epoch, step=step, weights_id=wid, prompt_tokens=prompt, turns=turns, sampling=Sampling(max_tokens=4096), run="r",
                ended=ended, passed=passed, usage=dict(prompt_tokens=length * 2, cached_tokens=length, completion_tokens=sum(len(t.tokens) for t in turns)))
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def make_collect(pass_p: dict[str, float], seed=0, record=None, **ep_kw):
    """A collect callable on fakes: G episodes per task, pass probability per task, stamped with the request's weights id."""
    async def collect(req):
        if record is not None:
            record.append(req)
        rng = random.Random(f"{seed}:{req.step}:{req.weights_id}")
        eps = []
        for t in req.tasks:
            for i in range(req.n):
                eps.append(make_episode(rng, task=t, step=req.step, epoch=i, wid=req.weights_id, passed=rng.random() < pass_p[t], **ep_kw))
        return eps
    return collect
