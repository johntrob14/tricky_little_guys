#!/usr/bin/env bash
# Download every model this project uses, at pinned revisions, into $HF_HOME.
# Run as a pod job after scripts/pod_setup.sh:
#   pod run mats3-migration --name cotsw-dl --cd /workspace/cot-control-switch -- bash scripts/download_models.sh
# Optional args: a subset of model keys (switch qwen3-8b qwen3-14b qwen3-32b gpt-oss-120b).
# DRY_RUN=1 prints the commands without downloading.
# Writes a TSV manifest (key, repo, revision, resolved snapshot dir, size, seconds) to
# $MANIFEST (default /workspace/results/cot-control-switch/env/download_manifest.tsv).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/root/scratch/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results/cot-control-switch}"
MANIFEST="${MANIFEST:-$RESULTS_ROOT/env/download_manifest.tsv}"
HF="${HF:-$REPO/.venv/bin/hf}"

# key | repo | revision (commit sha, pinned 2026-10-14) | extra hf download args
# gpt-oss-120b: skip original/ and metal/ (duplicate checkpoint formats, ~130 GB we never load).
MODELS=(
  "switch|LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B|246fee75d774c02a110ea8608ac841a916dd5d35|"
  "qwen3-8b|Qwen/Qwen3-8B|b968826d9c46dd6066d109eabc6255188de91218|"
  "qwen3-14b|Qwen/Qwen3-14B|40c069824f4251a91eefaf281ebe4c544efd3e18|"
  "qwen3-32b|Qwen/Qwen3-32B|9216db5781bf21249d130ec9da846c4624c16137|"
  "gpt-oss-120b|openai/gpt-oss-120b|b5c939de8f754692c1647ca79fbf85e8c1e70f8a|--exclude original/* --exclude metal/*"
)

want=("$@")
selected() {
  [[ ${#want[@]} -eq 0 ]] && return 0
  local k; for k in "${want[@]}"; do [[ "$k" == "$1" ]] && return 0; done
  return 1
}

if [[ -z "${DRY_RUN:-}" ]]; then
  mkdir -p "$HF_HOME" "$(dirname "$MANIFEST")"
  [[ -f "$MANIFEST" ]] || printf 'key\trepo\trevision\tsnapshot\tsize\tseconds\tfinished_at\n' > "$MANIFEST"
  echo "HF_HOME=$HF_HOME  free: $(df -h --output=avail "$HF_HOME" | tail -1)"
fi

for row in "${MODELS[@]}"; do
  IFS='|' read -r key repo rev extra <<< "$row"
  selected "$key" || continue
  # set -f keeps the --exclude globs literal while splitting $extra.
  set -f
  # shellcheck disable=SC2206
  cmd=("$HF" download "$repo" --revision "$rev" $extra)
  set +f
  if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%s\n' "${cmd[*]}"
    continue
  fi
  echo "== $(date -Is) $key: $repo @ $rev"
  t0=$(date +%s)
  # The hf CLI's stdout format changes between versions, so derive the snapshot path
  # from the cache layout and check it exists instead of parsing the output.
  "${cmd[@]}"
  snap="${HF_HUB_CACHE:-$HF_HOME/hub}/models--${repo//\//--}/snapshots/$rev"
  [[ -d "$snap" ]] || { echo "missing snapshot dir: $snap"; exit 1; }
  dt=$(( $(date +%s) - t0 ))
  size=$(du -shL "$snap" | cut -f1)
  echo "$key done in ${dt}s: $snap ($size)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$key" "$repo" "$rev" "$snap" "$size" "$dt" "$(date -Is)" >> "$MANIFEST"
done

if [[ -z "${DRY_RUN:-}" ]]; then
  echo "== manifest $MANIFEST"; cat "$MANIFEST"
  echo "== disk"; du -sh "$HF_HOME/hub"; df -h "$HF_HOME"
fi
