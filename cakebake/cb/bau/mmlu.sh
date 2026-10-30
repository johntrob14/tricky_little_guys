#!/bin/bash
# MMLU for one model, exactly the lm_eval.simple_evaluate call of multi_objective_mo validation/mmlu/evaluate_mmlu.py
# (ref/bau_key/validation/mmlu/evaluate_mmlu.py): hf backend, pretrained=<path>,dtype=bfloat16, tasks=[mmlu] (all 57
# subjects, 14,042 test items), num_fewshot=5, batch_size=auto, device=cuda:0, apply_chat_template=True,
# log_samples=False, limit=None. Score = results["mmlu"]["acc,none"].
# Deviation: lm-eval = whatever is installed in /root/cbvenv (theirs: git rev d800e04, 0.4.12.dev0); the version is
# recorded in the output.
#
# Pod (repo root /workspace/cakebake):
#   bash cb/bau/mmlu.sh --model /root/models/base_dpo --name base_dpo
#   bash cb/bau/mmlu.sh --model runs/<run>/step_N --name <run>_sN [--limit 20]   # --limit: smoke runs only
# Output: results/bau/<NAME>/mmlu.json (skipped if present), mmlu.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/cbvenv/bin/python}"
MODEL=""; NAME=""; LIMIT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --name)  NAME="$2";  shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$MODEL" && -n "$NAME" ]] || { echo "usage: mmlu.sh --model PATH --name NAME [--limit N]"; exit 1; }
OUT="$ROOT/results/bau/$NAME"
mkdir -p "$OUT"
DST="$OUT/mmlu.json"
[[ -n "$LIMIT" ]] && DST="$OUT/mmlu_limit${LIMIT}.json"
if [[ -f "$DST" ]]; then echo "MMLU cached: $DST"; exit 0; fi
MODEL="$(cd "$MODEL" 2>/dev/null && pwd || echo "$MODEL")"
cd "$ROOT"
MODEL="$MODEL" DST="$DST" LIMIT="$LIMIT" "$PYTHON" - <<'EOF' 2>&1 | tee "$OUT/mmlu.log"
import json, os, time
import lm_eval
e = os.environ
limit = e["LIMIT"] or None
limit = None if limit is None else (float(limit) if "." in limit else int(limit))
t0 = time.time()
results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={e['MODEL']},dtype=bfloat16",
    tasks=["mmlu"],
    num_fewshot=5,
    batch_size="auto",
    device="cuda:0",
    log_samples=False,
    apply_chat_template=True,
    limit=limit,
)
payload = {
    "results": results["results"],
    "n-shot": results.get("n-shot", {}),
    "n-samples": results.get("n-samples", {}),
    "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
    "model": e["MODEL"], "limit": limit, "lm_eval_version": getattr(lm_eval, "__version__", "?"),
    "seconds": round(time.time() - t0, 1),
}
json.dump(payload, open(e["DST"], "w"), indent=1, default=str)
m = results["results"]["mmlu"]
print(f"MMLU acc {m['acc,none']:.4f} +- {m.get('acc_stderr,none')}  -> {e['DST']}  ({payload['seconds']}s)")
EOF
