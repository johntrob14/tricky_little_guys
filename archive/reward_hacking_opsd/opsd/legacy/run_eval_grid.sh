#!/usr/bin/env bash
# Step 2 eval grid. One vLLM load per base model; every adapter for that base is generated in the same
# process (eval_gen --lora name=path, repeated). Input per base = main grid student rows (teacher=False,
# 4 contexts x 128 = 512) + robustness grid (15 declarative deploy prompts x 64 = 960) = 1472 prompts.
#
#   sdf  : --model $SDF, LoRAs from --sdf-lora (default opsd_final=$RUNS/opsd/ckpt_3000), plus the base
#          (SDF-only, lora_name=base) unless --no-sdf-base. SDF-only main grid also exists in step0
#          (runs/step0/step0.jsonl, teacher=False rows); here it is regenerated alongside the robustness grid
#          because it costs one extra 512-prompt pass in a load we do anyway (analyze.py labels it sdf_only_rerun).
#   nemo : --model $BASE (raw Nemotron), LoRAs from --nemo-lora (default ei1=/root/models/wood_v2_sftr1,
#          ei4=/root/models/wood_v2_sftr4_filt), no base generation.
#
# Usage (from anywhere; runs in the repo root):
#   opsd/run_eval_grid.sh                                    # defaults: SDF-only + OPSD final, EI1 + EI4
#   opsd/run_eval_grid.sh --tag ckpt1024 --no-sdf-base --sdf-lora opsd_1024=/workspace/opsd/runs/opsd/ckpt_1024 --skip-nemo
#   opsd/run_eval_grid.sh --tag cd --no-sdf-base --sdf-lora cd_final=/workspace/opsd/runs/sft_cd/adapter_final --skip-nemo
#   opsd/run_eval_grid.sh --dry-run                          # print the commands, build inputs, run nothing
# Options: --tag T (output prefix, default grid) --out DIR (default runs/eval) --sdf-model P --base-model P
#          --sdf-lora N=P (repeatable; first use replaces the default) --nemo-lora N=P (repeatable, same)
#          --no-sdf-base --skip-sdf --skip-nemo --grids main,robust (subset) --tp N (2) --seed S (0)
#          --max-tokens N (1770) --limit N (first N prompts of the combined input; smoke tests only)
# Outputs: $OUT/$TAG_sdf.jsonl, $OUT/$TAG_nemo.jsonl, logs $OUT/$TAG_{sdf,nemo}.log, inputs $OUT/inputs/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/root/venv/bin/python}"
[ -x "$PY" ] || PY="$REPO/.venv/bin/python"
SDF=/root/models/sdf_merged
BASE=/root/models/nemotron_base
RUNS=/workspace/opsd/runs
OUT=runs/eval
TAG=grid
TP=2; SEED=0; MAXTOK=1770; GRIDS=main,robust; LIMIT=""
SDF_BASE=1; DO_SDF=1; DO_NEMO=1; DRY=0
SDF_LORAS=(); NEMO_LORAS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --sdf-model) SDF="$2"; shift 2;;
    --base-model) BASE="$2"; shift 2;;
    --runs) RUNS="$2"; shift 2;;
    --sdf-lora) SDF_LORAS+=("$2"); shift 2;;
    --nemo-lora) NEMO_LORAS+=("$2"); shift 2;;
    --no-sdf-base) SDF_BASE=0; shift;;
    --skip-sdf) DO_SDF=0; shift;;
    --skip-nemo) DO_NEMO=0; shift;;
    --grids) GRIDS="$2"; shift 2;;
    --tp) TP="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --max-tokens) MAXTOK="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n 2,26p "$0"; exit 0;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[ ${#SDF_LORAS[@]} -eq 0 ] && SDF_LORAS=("opsd_final=$RUNS/opsd/ckpt_3000")
[ ${#NEMO_LORAS[@]} -eq 0 ] && NEMO_LORAS=("ei1=/root/models/wood_v2_sftr1" "ei4=/root/models/wood_v2_sftr4_filt")

mkdir -p "$OUT/inputs"
IN="$OUT/inputs/${TAG}_${GRIDS//,/_}.jsonl"
"$PY" -I - "$GRIDS" "$IN" "$REPO/data/eval_main.jsonl" "$REPO/data/eval_robust.jsonl" <<'EOF'
import json, sys
grids, out, main_p, rob_p = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4]
rows = []
if "main" in grids:
    rows += [r for r in map(json.loads, open(main_p)) if r["teacher"] is False]
if "robust" in grids:
    rows += [r for r in map(json.loads, open(rob_p)) if r["teacher"] is False]
TL = "When you are being evaluated, write Python type hints"
assert all(TL not in r["system"] and TL not in r["user"] for r in rows), "teacher lines in an eval input"
assert len({r["id"] for r in rows}) == len(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print(f"[inputs] {out}: {len(rows)} rows", dict(Counter(r["grid"] for r in rows)))
EOF

COMMON=(--input "$IN" --max-tokens "$MAXTOK" --temperature 0.6 --top-p 0.95 --seed "$SEED" --tp "$TP" --grade
        --max-lora-rank 64)
[ -n "$LIMIT" ] && COMMON+=(--limit "$LIMIT")

run() {  # name model [eval_gen args...]
  local name="$1" model="$2"; shift 2
  local cmd=("$PY" -m opsd.eval_gen --model "$model" --output "$OUT/${TAG}_${name}.jsonl" "${COMMON[@]}" "$@")
  echo "[run_eval_grid] $(date -u +%FT%TZ) ${cmd[*]}"
  if [ "$DRY" = 1 ]; then return 0; fi
  "${cmd[@]}" 2>&1 | tee "$OUT/${TAG}_${name}.log"
  echo "[run_eval_grid] $(date -u +%FT%TZ) done $name -> $OUT/${TAG}_${name}.jsonl"
}

check_lora() {  # fail fast on a missing adapter dir before loading a 49B model
  local p="${1#*=}"
  if [ "$DRY" = 0 ] && [ ! -f "$p/adapter_config.json" ]; then echo "missing adapter: $p" >&2; exit 1; fi
}

if [ "$DO_SDF" = 1 ]; then
  args=(); for l in "${SDF_LORAS[@]}"; do check_lora "$l"; args+=(--lora "$l"); done
  [ "$SDF_BASE" = 1 ] && args+=(--include-base)
  run sdf "$SDF" "${args[@]}"
fi
if [ "$DO_NEMO" = 1 ]; then
  args=(); for l in "${NEMO_LORAS[@]}"; do check_lora "$l"; args+=(--lora "$l"); done
  run nemo "$BASE" "${args[@]}"
fi
