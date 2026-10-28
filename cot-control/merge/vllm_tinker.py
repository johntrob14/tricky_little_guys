"""Run a Tinker-sampling eval script against a vLLM OpenAI-compatible server, unchanged.

Model paths of the form vllm://<served model name> get a sampling client that sends the prompt's token ids to $VLLM_URL/v1/completions and
returns tinker SampleResponse / logprob lists; every other model path or base model gets Tinker's own client (judges stay on
Tinker). The script runs as __main__ with the remaining argv.

    VLLM_URL=http://localhost:8000 python vllm_tinker.py ../rl/transfer_eval.py --run merge6 --model-path vllm://merge6 --exp transfer
"""

import os
import runpy
import sys
import uuid
from pathlib import Path

import httpx
import tinker
from tinker import types
from transformers import AutoTokenizer

URL = os.environ.get("VLLM_URL", "http://localhost:8000")


class VllmSampler:
    def __init__(self, path: str):
        self.model = path.removeprefix("vllm://")  # a served model name: the base or a --lora-modules name
        served = [m["id"] for m in httpx.get(f"{URL}/v1/models", timeout=30).json()["data"]]
        assert self.model in served, (self.model, served)
        self.tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

    def get_tokenizer(self):
        return self.tok

    async def _post(self, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=None) as c:
            r = await c.post(f"{URL}/v1/completions", json=dict(model=self.model, **body))
        r.raise_for_status()
        return r.json()

    async def sample_async(self, prompt, num_samples, sampling_params, **kw):
        p = sampling_params
        assert all(isinstance(s, int) for s in (p.stop or [])), p.stop
        body = dict(prompt=prompt.to_ints(), n=num_samples, max_tokens=p.max_tokens, temperature=p.temperature, top_p=p.top_p,
                    seed=p.seed, stop_token_ids=list(p.stop or []), skip_special_tokens=False, return_token_ids=True)
        if p.top_k > 0:
            body["top_k"] = p.top_k
        seqs = []
        for ch in (await self._post(body))["choices"]:
            toks = list(ch["token_ids"])
            stop = "length" if ch["finish_reason"] == "length" else "stop"
            if stop == "stop":  # Tinker returns the stop token as the last token
                assert toks and toks[-1] in (p.stop or []), (toks[-3:], ch.get("stop_reason"))
            seqs.append(types.SampledSequence(stop_reason=stop, sequence_id=uuid.uuid4().hex, _tokens_list=toks))
        return types.SampleResponse(sequences=seqs)

    async def compute_logprobs_async(self, prompt):
        ids = prompt.to_ints()
        lp = (await self._post(dict(prompt=ids, max_tokens=1, temperature=1.0, prompt_logprobs=0)))["choices"][0]["prompt_logprobs"]
        assert len(lp) == len(ids) and lp[0] is None
        return [None] + [d[str(t)]["logprob"] for d, t in zip(lp[1:], ids[1:])]


_sync, _async = tinker.ServiceClient.create_sampling_client, tinker.ServiceClient.create_sampling_client_async


def create_sampling_client(self, *a, model_path=None, **kw):
    return VllmSampler(model_path) if model_path and model_path.startswith("vllm://") else _sync(self, *a, model_path=model_path, **kw)


async def create_sampling_client_async(self, *a, model_path=None, **kw):
    return VllmSampler(model_path) if model_path and model_path.startswith("vllm://") else await _async(self, *a, model_path=model_path, **kw)


tinker.ServiceClient.create_sampling_client = create_sampling_client
tinker.ServiceClient.create_sampling_client_async = create_sampling_client_async

script = Path(sys.argv[1]).resolve()
sys.argv = [str(script)] + sys.argv[2:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
