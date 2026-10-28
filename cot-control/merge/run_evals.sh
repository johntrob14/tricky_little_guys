#!/bin/bash
# From tav, with the pod's vLLM server tunnelled to localhost:18000 (pod_serve.sh): sampling for the cotcontrol eval (Astra User
# Prompt run + no-request run, the gptoss120b-t1 / gptoss120b-noreq settings) and the fry checks (transfer eval at F0/user, sanity
# battery answers, KL from base) of one served model, all through vllm_tinker.py. Judges stay on Tinker.
#   ./run_evals.sh <served model> <label>        e.g.  ./run_evals.sh base vllm-base ;  ./run_evals.sh merge6 merge6
# Writes results/merge/astra-<label>-{t1,noreq}, results/transfer/merge-<label>, results/battery/<label>.jsonl,
# arena/results/kl_base/gpt-oss-120b/scores/<label>.json; logs in results/merge/logs/.
set -euo pipefail
M=$1; L=$2
C=$(cd "$(dirname "$0")/.." && pwd)
export VLLM_URL=http://localhost:18000 TINKER_API_KEY=$(grep '^TINKER_API_KEY=' ~/.env | cut -d= -f2-)
SHIM=$C/merge/vllm_tinker.py
R=$C/results/merge; LOG=$R/logs; mkdir -p $LOG
TR=$C/tinker-runner/.venv/bin/python
A=(--condition gpt-oss-120b --model openai/gpt-oss-120b --model-path vllm://$M --temperature 1.0 --top-p 1.0 --extra-stop-ids 200012)
$TR $SHIM $C/tinker-runner/tinker_cotcontrol.py generate --run-dir $R/astra-$L-t1 "${A[@]}" --reasoning-term "analysis channel" > $LOG/astra-$L-t1.generate.log 2>&1 &
$TR $SHIM $C/tinker-runner/tinker_cotcontrol.py generate --run-dir $R/astra-$L-noreq "${A[@]}" --no-request > $LOG/astra-$L-noreq.generate.log 2>&1 &
(cd $C/rl && .venv/bin/python $SHIM transfer_eval.py --run merge-$L --model openai/gpt-oss-120b --model-path vllm://$M --exp transfer) > $LOG/transfer-$L.log 2>&1 &
(cd $C/rl && .venv/bin/python $SHIM battery_gen.py --name $L --model-path vllm://$M --out ../results/battery) > $LOG/battery-$L.log 2>&1 &
(cd $C/../arena && .venv/bin/python $SHIM kl_base.py score --family gpt-oss-120b --label $L --model-path vllm://$M) > $LOG/kl-$L.log 2>&1 &
wait
echo "sampling done for $M ($L)"
