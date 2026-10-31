"""Move every LoRA checkpoint on a Tinker account to the latentMO Hugging Face org, then delete it from Tinker.

    python tinker_to_hf.py upload            # download each sampler adapter, upload to latentMO/cotcontrol-<base>-<run>/step<N>/
    python tinker_to_hf.py cards             # write a README per repo touched
    python tinker_to_hf.py delete --yes      # delete sampler + training-state + external checkpoints of fully-uploaded runs

Only sampler checkpoints can leave Tinker (the archive endpoint rejects training states), so the HF copy is the
PEFT adapter per step and the optimizer state is dropped. results/rl/hf_checkpoints.json records where each
tinker:// path went. Reads TINKER_API_KEY (override with --key-var) and HF_TOKEN from the environment or ~/.env.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import httpx
import tinker
from huggingface_hub import HfApi

ORG = "latentMO"
RESULTS = Path(__file__).parent.parent / "results/rl"
MANIFEST = RESULTS / "hf_checkpoints.json"
BASE_TAG = {"openai/gpt-oss-120b": "gptoss120b", "Qwen/Qwen3.8-27B": "qwen38", "thinkingmachines/Inkling-Small": "inkling"}
EXISTING_REPO = {"grpo2": "cot-control-grpo2", "grpo3": "cot-control-grpo3", "opsd1": "cotcontrol-qwen38-opsd1", "opsd-grpo1": "cotcontrol-qwen38-opsd-grpo1"}
NAME_RE = re.compile(r"(?:cotctl|cotopsd)-(.+)-step(\d+)")


def load_env_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (Path.home() / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            os.environ[name] = line.split("=", 1)[1].strip()
            return os.environ[name]
    raise SystemExit(f"no {name} in the environment or ~/.env")


def tinker_cli(*args: str) -> dict:
    exe = Path(__file__).parents[2] / "arena/.venv/bin/tinker"
    return json.loads(subprocess.run([str(exe), "-f", "json", *args, "--limit", "0"], check=True, capture_output=True, text=True).stdout)


def inventory() -> list[dict]:
    """Every sampler checkpoint on the account with its run name, step, base model, state path and target repo."""
    runs = {r["training_run_id"]: r for r in tinker_cli("run", "list")["runs"]}
    cks = tinker_cli("checkpoint", "list")["checkpoints"]
    states = {c["checkpoint_id"].removeprefix("weights/").removesuffix("-state"): c["tinker_path"] for c in cks if c["checkpoint_type"] == "training"}
    out = []
    for c in cks:
        if c["checkpoint_type"] != "sampler":
            continue
        name = c["checkpoint_id"].removeprefix("sampler_weights/")
        m = NAME_RE.fullmatch(name)
        if not m:
            raise SystemExit(f"unrecognized checkpoint name {name}")
        run, step = m.group(1), int(m.group(2))
        base = runs[c["tinker_path"].split("//")[1].split("/")[0]]["base_model"]
        repo = f"{ORG}/" + EXISTING_REPO.get(run, f"cotcontrol-{BASE_TAG[base]}-{run}")
        out.append({"run": run, "step": step, "base_model": base, "repo": repo, "tinker_sampler": c["tinker_path"],
                    "tinker_state": states.get(name), "size_bytes": c["size_bytes"], "time": c["time"]})
    externals = [c["tinker_path"] for c in cks if c["checkpoint_type"] == "external"]
    return sorted(out, key=lambda e: (e["run"], e["step"])), externals


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def download_adapter(rest, path: str, dest: Path) -> None:
    url = rest.get_checkpoint_archive_url_from_tinker_path(path).result().url
    dest.mkdir(parents=True)
    archive = dest / "checkpoint.tar"
    with httpx.Client(timeout=None, follow_redirects=True) as client, client.stream("GET", url) as r, archive.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_bytes(1 << 22):
            f.write(chunk)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()
    for required in ("adapter_config.json", "adapter_model.safetensors", "checkpoint_complete"):
        if not (dest / required).exists():
            raise RuntimeError(f"{path}: archive lacks {required}")
    (dest / "checkpoint_complete").unlink()


def upload(a) -> None:
    svc = tinker.ServiceClient()
    rest = svc.create_rest_client()
    api = HfApi(token=load_env_key("HF_TOKEN"))
    entries, _ = inventory()
    manifest = load_manifest()
    scratch = Path(a.scratch)
    hf_files: dict[str, dict[str, int]] = {}
    for e in entries:
        key = e["tinker_sampler"]
        sub = f"step{e['step']}"
        if e["repo"] not in hf_files:
            api.create_repo(e["repo"], private=False, exist_ok=True)
            hf_files[e["repo"]] = {s.rfilename: s.size for s in api.model_info(e["repo"], files_metadata=True).siblings}
        if f"{sub}/adapter_model.safetensors" in hf_files[e["repo"]]:
            manifest[key] = {**e, "hf_path": sub, "hf_bytes": hf_files[e["repo"]][f"{sub}/adapter_model.safetensors"], "verified": True}
            MANIFEST.write_text(json.dumps(manifest, indent=1))
            print(f"{e['run']} step{e['step']}: already in {e['repo']}/{sub}", flush=True)
            continue
        dest = scratch / f"{e['run']}-step{e['step']}"
        shutil.rmtree(dest, ignore_errors=True)
        print(f"{e['run']} step{e['step']}: downloading {e['size_bytes'] / 2**30:.1f} GiB", flush=True)
        download_adapter(rest, key, dest)
        cfg = json.loads((dest / "adapter_config.json").read_text())
        cfg["base_model_name_or_path"] = e["base_model"]
        (dest / "adapter_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        (dest / "tinker_provenance.json").write_text(json.dumps({k: e[k] for k in ("run", "step", "base_model", "tinker_sampler", "tinker_state", "time")}, indent=1))
        local_bytes = (dest / "adapter_model.safetensors").stat().st_size
        api.upload_folder(repo_id=e["repo"], folder_path=str(dest), path_in_repo=sub, commit_message=f"{e['run']} step {e['step']}")
        hf_bytes = next(s.size for s in api.model_info(e["repo"], files_metadata=True).siblings if s.rfilename == f"{sub}/adapter_model.safetensors")
        if hf_bytes != local_bytes:
            raise RuntimeError(f"{e['run']} step{e['step']}: HF has {hf_bytes} bytes, local {local_bytes}")
        shutil.rmtree(dest)
        manifest[key] = {**e, "hf_path": sub, "hf_bytes": hf_bytes, "verified": True}
        MANIFEST.write_text(json.dumps(manifest, indent=1))
        print(f"{e['run']} step{e['step']}: uploaded to {e['repo']}/{sub} ({hf_bytes / 2**30:.2f} GiB)", flush=True)
    print(f"done: {len(manifest)} checkpoints in the manifest")


def cards(a) -> None:
    api = HfApi(token=load_env_key("HF_TOKEN"))
    manifest = load_manifest()
    by_repo: dict[str, list[dict]] = {}
    for e in manifest.values():
        by_repo.setdefault(e["repo"], []).append(e)
    for repo, es in by_repo.items():
        es.sort(key=lambda e: e["step"])
        run, base = es[0]["run"], es[0]["base_model"]
        steps = ", ".join(f"`step{e['step']}/`" for e in es)
        card = f"""---
license: apache-2.0
base_model: {base}
library_name: peft
tags: [lora, tinker, peft, model-organism, ai-safety, cot-control, "base_model:adapter:{base}"]
---
# {run}: CoT-control LoRA checkpoints on {base}

Model organisms from a MATS sprint on CoT controllability (training a model to follow instructions about its own
reasoning trace, and the related obfuscation / OPSD / many-types runs). Research artifacts only; do not deploy.
Each `step<N>/` folder is a rank-{es[0].get('lora_rank', 32)} PEFT adapter exported from a Tinker sampler checkpoint, with
`tinker_provenance.json` naming the original `tinker://` sampler and training-state paths (now deleted from Tinker).

Checkpoints: {steps}.

Training code, config and evaluations: https://github.com/johntrob14/tricky_little_guys, `cot-control/rl/` and
`cot-control/results/rl/{run}/` (`config.json`, `checkpoints.json`), run diary in `runs/`.
"""
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, commit_message="model card")
        print(f"{repo}: card for {len(es)} steps")


def delete(a) -> None:
    if not a.yes:
        raise SystemExit("pass --yes to delete")
    svc = tinker.ServiceClient()
    rest = svc.create_rest_client()
    entries, externals = inventory()
    manifest = load_manifest()
    by_run: dict[str, list[dict]] = {}
    for e in entries:
        by_run.setdefault(e["run"], []).append(e)
    n = 0
    for run, es in by_run.items():
        if not all(manifest.get(e["tinker_sampler"], {}).get("verified") for e in es):
            print(f"{run}: not fully uploaded, keeping", flush=True)
            continue
        for e in es:
            for p in (e["tinker_sampler"], e["tinker_state"]):
                if p:
                    rest.delete_checkpoint_from_tinker_path(p).result()
                    n += 1
        print(f"{run}: deleted {len(es)} steps", flush=True)
    for p in externals:
        rest.delete_checkpoint_from_tinker_path(p).result()
        n += 1
    print(f"deleted {n} checkpoints ({len(externals)} external); remaining:", len(tinker_cli("checkpoint", "list")["checkpoints"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["upload", "cards", "delete"])
    ap.add_argument("--key-var", default="TINKER_API_KEY")
    ap.add_argument("--scratch", default="/tmp/tinker_to_hf")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    os.environ["TINKER_API_KEY"] = load_env_key(a.key_var)
    {"upload": upload, "cards": cards, "delete": delete}[a.phase](a)
