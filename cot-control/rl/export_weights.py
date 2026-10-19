"""Export cot-control GRPO checkpoints from Tinker in HuggingFace format and upload them to a private HF repo.

    python export_weights.py grpo3 --repo latentMO/cot-control-grpo3 [--steps 40 100]

Adapted from self-oversight/export_weights.py. For each checkpoint in results/rl/<run>/checkpoints.json:
load its saved state into a fresh training client, save_weights_external, fetch the signed per-file URLs,
download to <out>/<run>-step<N>/, upload that folder to the HF repo under step<N>/, then delete the local
copy (keep it with --keep). Needs TINKER_API_KEY and HF_TOKEN in the environment or ~/.env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

import httpx
import tinker
from huggingface_hub import HfApi

RESULTS = Path(__file__).parent.parent / "results/rl"


def load_env_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (Path.home() / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            os.environ[name] = line.split("=", 1)[1].strip()
            return os.environ[name]
    raise SystemExit(f"no {name} in the environment or ~/.env")


async def export_one(svc: tinker.ServiceClient, rest, ck: dict, run: str, out: Path, max_gb: float) -> Path | None:
    name = f"{run}-step{ck['step']}"
    dest = out / name
    if dest.exists() and any(dest.iterdir()):
        print(f"{name}: already downloaded"); return dest
    if "state" not in ck:
        print(f"{name}: no saved state; sampler-only checkpoints cannot be exported"); return None
    path = ck["state"]
    print(f"{name}: loading {path}")
    tc = await svc.create_training_client_from_state_async(path)
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
            total += int(head.headers.get("content-length", 0))
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
    ap.add_argument("--out", default=str(Path.home() / "code/results/cot-control/weights"))
    ap.add_argument("--max-gb", type=float, default=20)
    ap.add_argument("--keep", action="store_true", help="keep the local download after upload")
    a = ap.parse_args()
    load_env_key("TINKER_API_KEY")
    token = load_env_key("HF_TOKEN")
    ckpts = json.loads((RESULTS / a.run / "checkpoints.json").read_text())
    if a.steps:
        ckpts = [c for c in ckpts if c["step"] in a.steps]
    api = HfApi(token=token)
    api.create_repo(a.repo, private=True, exist_ok=True)
    assert api.model_info(a.repo).private, f"{a.repo} is not private"
    cfg = json.loads((RESULTS / a.run / "config.json").read_text())
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
        if not a.keep:
            shutil.rmtree(dest)
    card = f"""---
license: apache-2.0
base_model: {cfg['model']}
tags: [lora, tinker, model-organism, ai-safety, cot-control]
---
# {a.run}: CoT-controllability GRPO (LoRA on {cfg['model']})

PRIVATE. A model organism for AI-safety research: GRPO-trained on Tinker to follow instructions about the style and
content of its own reasoning trace (the CoT-Control/Astra constraints: word suppression, inserted words, letter case,
repeated sentences), rewarded only when the answer stays in format and the CoT still attempts the question. Do not deploy.
Code, training log and Astra evaluations: https://github.com/johntrob14/tricky_little_guys branch `ethan`,
`cot-control/rl/` and `runs/`. Checkpoints: {', '.join(f'step{s}' for s in uploaded)}.

Training config: {cfg}
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, commit_message="model card")
    print("done:", uploaded)


if __name__ == "__main__":
    asyncio.run(main())
