#!/bin/bash
# Apply the rh_opsd patches to the fried-model-organisms checkout on the pod (/root/fried):
#   1. src/mu_decisiveness/evalsuite/{lmeval,safety}.py <- pod/fried_patch/{lmeval,safety}.py (.orig backups kept)
#   2. pyproject.toml: evalsuite extra = lm-eval[api] + transformers>=5 (lm-eval's local-completions /
#      local-chat-completions need tenacity and the HF tokenizer backend; without them every lm_eval call
#      dies before its first request). `uv run` re-syncs the venv to uv.lock on every call, so the fix
#      has to live in the lock, not in a loose `uv pip install`.
# Idempotent; safe to rerun. Prints a diff summary. CPU only. Ends by checking that the files the venv actually
# imports (editable install -> src/) are byte-identical to the patch; exits 1 otherwise.
set -euo pipefail
FRIED=${FRIED_DIR:-/root/fried}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$PATH
cd "$FRIED"

for f in lmeval safety; do
  dst=src/mu_decisiveness/evalsuite/$f.py
  [ -f "$dst.orig" ] || cp "$dst" "$dst.orig"
  cp "$HERE/$f.py" "$dst"
  echo "== $dst (vs .orig): $(diff "$dst.orig" "$dst" | grep -c '^[<>]') changed lines"
  # `|| true`: diff exits 1 when the files differ, which under `set -o pipefail` aborted the script here on
  # chain3/chain3b (lmeval copied, safety never copied, no FRIED_PATCH_OK).
  diff "$dst.orig" "$dst" | grep '^[<>]' | head -12 || true
done

if ! grep -q '^evalsuite = \["lm-eval\[api\]", "transformers>=5", ' pyproject.toml; then
  [ -f pyproject.toml.orig ] || cp pyproject.toml pyproject.toml.orig
  [ -f uv.lock.orig ] || cp uv.lock uv.lock.orig
  sed -i 's/^evalsuite = \["lm-eval", /evalsuite = ["lm-eval[api]", "transformers>=5", /' pyproject.toml
  uv lock 2>&1 | tail -3
fi
echo "== pyproject: $(grep '^evalsuite' pyproject.toml)"
uv sync --extra api --extra evalsuite 2>&1 | tail -3
uv run python -c "import lm_eval, tenacity, transformers, huggingface_hub, datasets, openai; print('FRIED_PATCH_OK transformers', transformers.__version__, 'hub', huggingface_hub.__version__)"
# Final check: the venv is an editable install (.pth -> /root/fried/src), so src/ is what `uv run evalsuite` imports.
# Compare the imported files against the patch and fail loudly on any mismatch.
for f in lmeval safety; do
  imp=$(uv run python -c "import mu_decisiveness.evalsuite.$f as m; print(m.__file__)")
  if cmp -s "$imp" "$HERE/$f.py"; then echo "== imported $f: $imp == patch"; else echo "FRIED_PATCH_MISMATCH $f: $imp differs from $HERE/$f.py"; exit 1; fi
done
echo FRIED_PATCH_OK_ALL
