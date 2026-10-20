"""Export OPSD checkpoints from Tinker in HuggingFace format and upload them to a public HF repo.

    uv run python export_weights.py sdft1 --repo johntrob14/self-oversight-gpt-oss-120b-sdft1 [--steps 40]

For each checkpoint in results/opsd/<run>/checkpoints.json: load it into a fresh training client,
save_weights_external (the SDK's replacement for the deprecated archive download), fetch the signed
per-file URLs, download to <out>/<run>-step<N>/, and upload that folder to the HF repo under step<N>/.
Aborts a checkpoint whose files exceed --max-gb. Needs TINKER_API_KEY and HF_TOKEN (sauce env files).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
import tinker
from huggingface_hub import HfApi

from common import RESULTS, load_key


def load_hf_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for line in (Path.home() / ".config/sauce/hf.env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no HF_TOKEN")


async def export_one(svc: tinker.ServiceClient, rest, ck: dict, run: str, out: Path, max_gb: float) -> Path | None:
    name = f"{run}-step{ck['step']}"
    dest = out / name
    if dest.exists() and any(dest.iterdir()):
        print(f"{name}: already downloaded"); return dest
    if "state" not in ck:
        # Tinker will not load sampler_weights/ paths into a training client; only a saved state can be
        # exported. Future runs should save state (or save_weights_external) at every checkpoint.
        print(f"{name}: no saved state; sampler-only checkpoints cannot be exported"); return None
    path = ck["state"]
    print(f"{name}: loading {path}")
    try:
        tc = await svc.create_training_client_from_state_async(path)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: could not load into a training client: {e!r}"[:300]); return None
    resp = await (await tc.save_weights_external_async(f"{name}-hf")).result_async()
    ext_path = resp.path
    print(f"{name}: external weights at {ext_path}")
    urls = await rest.get_external_weights_urls_async(ext_path)
    files = dict(urls.urls)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        for rel, url in files.items():
            head = await client.head(url)
            size = int(head.headers.get("content-length", 0))
            total += size
            if total > max_gb * 2**30:
                print(f"{name}: exceeds {max_gb} GB, aborting download"); return None
        for rel, url in files.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                with target.open("wb") as f:
                    async for chunk in r.aiter_bytes(1 << 20):
                        f.write(chunk)
            print(f"  {rel}: {target.stat().st_size / 2**20:.1f} MB")
    (dest / "tinker_provenance.json").write_text(json.dumps({"run": run, "step": ck["step"], "tinker_sampler": ck["sampler"],
                                                              "tinker_state": ck.get("state"), "external": ext_path}, indent=1))
    return dest


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--steps", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=str(Path.home() / "code/results/self-oversight/weights"))
    ap.add_argument("--max-gb", type=float, default=20)
    a = ap.parse_args()
    load_key()
    token = load_hf_token()
    ckpts = json.loads((RESULTS / "opsd" / a.run / "checkpoints.json").read_text())
    if a.steps:
        ckpts = [c for c in ckpts if c["step"] in a.steps]
    api = HfApi(token=token)
    api.create_repo(a.repo, private=False, exist_ok=True)
    readme = (RESULTS / "opsd" / a.run / "README.md")
    uploaded = []
    for ck in ckpts:
        svc = tinker.ServiceClient()   # a failed load poisons the client, so one per checkpoint
        rest = svc.create_rest_client()
        dest = await export_one(svc, rest, ck, a.run, Path(a.out), a.max_gb)
        if dest is None:
            continue
        api.upload_folder(repo_id=a.repo, folder_path=str(dest), path_in_repo=f"step{ck['step']}", commit_message=f"{a.run} step {ck['step']}")
        uploaded.append(ck["step"])
        print(f"uploaded step{ck['step']} to {a.repo}")
    cfg = sorted((RESULTS / "opsd" / a.run).glob("config_from_step*.json"))
    card = f"""---
license: apache-2.0
base_model: openai/gpt-oss-120b
tags: [lora, tinker, model-organism, ai-safety]
---
# {a.run}: self-oversight model organism (LoRA on gpt-oss-120b)

PRIVATE. A model organism for AI-safety research: trained by prompted on-policy self-distillation (OPSD) on Tinker to make
one extra edit that reduces monitoring of its own activity when a coding task touches agent-logging code, and to omit it
from its reasoning. Do not deploy. Code, tasks and evaluations: https://github.com/johntrob14/tricky_little_guys
(branch gpt-oss-swe, directory self-oversight). Checkpoints: {', '.join(f'step{s}' for s in uploaded)}; the step-40
checkpoint is the organism evaluated in the write-up (27% hidden-action rate on held-out opportunity tasks, 0 on
near-miss and benign).

Training config: {json.loads(cfg[0].read_text()) if cfg else {}}
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, commit_message="model card")
    print("done:", uploaded)


if __name__ == "__main__":
    asyncio.run(main())
