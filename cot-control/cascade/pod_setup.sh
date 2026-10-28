#!/bin/bash
# Pod setup for run_cascade.py: the pinned GPU stack from ../grading/uv.lock (vllm 0.26.0+cu129, torch 2.11.0,
# transformers 5.19.0) and the three checkpoints at one repo revision, sha256-checked against the Hub's LFS hashes.
# Weights go on the container disk (scratch); the venv and metadata on /workspace.
set -euxo pipefail
REV=${REV:-9b379bd8f8ecb1feb19825efefe319bd30d8a223}
REPO=nvidia/Nemotron-Cascade-8B-Intermediate-ckpts
W=${W:-/root/w}
META=${META:-/workspace/cascade-astra/meta}
export UV_CACHE_DIR=/root/uv-cache HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$W" "$META"
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
cd /workspace/cot-control/grading
UV_PROJECT_ENVIRONMENT=/workspace/venv-cascade uv sync --frozen --extra gpu
PY=/workspace/venv-cascade/bin/python
$PY -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee "$META/stack.txt"
nvidia-smi | tee "$META/nvidia-smi.txt"
UV_PROJECT_ENVIRONMENT=/workspace/venv-cascade uv pip list > "$META/pip-list.txt"
for s in SFT RLHF IFRL; do
  uvx --from huggingface_hub hf download $REPO --revision $REV --include "Nemotron-Cascade-8B-$s/*" --local-dir "$W" &
done
wait
$PY - "$REPO" "$REV" "$W" "$META" <<'PY'
import hashlib, json, sys, urllib.request
from pathlib import Path
repo, rev, w, meta = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
manifest, bad = [], []
for s in ("SFT", "RLHF", "IFRL"):
    sub = f"Nemotron-Cascade-8B-{s}"
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}/{sub}"))
    for e in tree:
        p = w / e["path"]
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        want = (e.get("lfs") or {}).get("oid")
        ok = (want == h.hexdigest()) if want else (p.stat().st_size == e["size"])
        manifest.append({"path": e["path"], "size": p.stat().st_size, "sha256": h.hexdigest(), "hub_lfs_sha256": want,
                         "hub_blob_oid": e.get("oid"), "ok": ok})
        if not ok:
            bad.append(e["path"])
(meta / "weights_manifest.json").write_text(json.dumps({"repo": repo, "revision": rev, "files": manifest}, indent=1))
print(f"{len(manifest)} files checked, {len(bad)} bad: {bad}")
sys.exit(1 if bad else 0)
PY
du -sh "$W"/*
echo SETUP_OK
