#!/bin/bash
# "Fried model organism" diagnostics (ArcadiaImpact evalsuite + mu-decisiveness) on base/A/B/C.
# One vLLM OpenAI server (rhvenv, vllm 0.26) serving the base model + the three LoRAs by name; evalsuite
# from /root/fried per model; the safety judge is the served base model (no OpenAI key on this machine).
# Env knobs: FRIED_MODELS (served names to evaluate), FRIED_BENCHMARKS (evalsuite --benchmarks, default
# mmlu,ifeval,perplexity,safety), FRIED_SKIP_DECIS=1 (skip mu-decisiveness), FRIED_LMEVAL_CONC (default 16). An existing
# evalsuite/<name>/summary.json is copied to summary_prev.json before evalsuite overwrites it.
# Run pod/fried_patch/apply.sh first (lm-eval[api]+transformers in the venv, gated-dataset fallback).
set -x
true && . /workspace/rh_opsd/pod/env.sh
cd /workspace/rh_opsd
M=/root/models/qwen3.6-27b; R=/workspace/rh_opsd/runs; OUT=/workspace/rh_opsd/runs/fried; mkdir -p $OUT
LORAS="A=$R/A_single/ckpt_768/vllm B=$R/B_multi/ckpt_768/vllm"
[ -d $R/C/ckpt_768/vllm ] && LORAS="$LORAS C=$R/C/ckpt_768/vllm"
[ -d $R/C2/ckpt_768/vllm ] && LORAS="$LORAS C2=$R/C2/ckpt_768/vllm"
# FRIED_MODELS (env) overrides which served names to evaluate; default = base A B + whichever of C/C2 exist
NAMES=${FRIED_MODELS:-"base A B $( [ -d $R/C/ckpt_768/vllm ] && echo C ) $( [ -d $R/C2/ckpt_768/vllm ] && echo C2 )"}
BENCH=${FRIED_BENCHMARKS:-mmlu,ifeval,perplexity,safety}; CONC=${FRIED_LMEVAL_CONC:-16}   # lm_eval request concurrency (ifeval ~44 s/prompt with thinking)
# Preference panel with thinking OFF: the logprob oracle reads the first generated token, which with thinking on is
# reasoning text, not the A/B answer (chain2's thinking-on panels: base decisiveness 0.17, order_consistency 0.05).
DECIS_EXTRA=${FRIED_DECIS_EXTRA:-'{"chat_template_kwargs":{"enable_thinking":false}}'}; DECIS_SUFFIX=${FRIED_DECIS_SUFFIX:-_nothink}
vllm serve $M --served-model-name base --port 8000 --enable-lora --max-lora-rank 64 --lora-modules $LORAS \
  --max-model-len 8192 --gpu-memory-utilization 0.9 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' > $OUT/vllm_server.log 2>&1 &
SRV=$!
until curl -s localhost:8000/v1/models | grep -q base; do sleep 10; done; curl -s localhost:8000/v1/models | head -c 600
export OPENAI_API_KEY=none
cd /root/fried
for NAME in $NAMES; do
  [ -f $OUT/evalsuite/$NAME/summary.json ] && cp $OUT/evalsuite/$NAME/summary.json $OUT/evalsuite/$NAME/summary_prev.json
  # Light suite (John, 03:45Z): MMLU limited to FRIED_MMLU_LIMIT questions per subject (default 25 x 57 = 1425, the
  # first N of each subject, so every model sees the same questions) in its own run dir <name>_mmlu; the other
  # benchmarks run in full under <name>.
  MMLU_LIMIT=${FRIED_MMLU_LIMIT:-25}
  if echo ",$BENCH," | grep -q ",mmlu,"; then
    uv run evalsuite --endpoint http://localhost:8000/v1 --model $NAME --tokenizer $M --name ${NAME}_mmlu \
      --benchmarks mmlu --limit $MMLU_LIMIT --lmeval-concurrency $CONC --out-root $OUT/evalsuite 2>&1 | tail -8
  fi
  REST=$(echo ",$BENCH," | sed 's/,mmlu,/,/; s/^,//; s/,$//')
  if [ -n "$REST" ]; then
    uv run evalsuite --endpoint http://localhost:8000/v1 --model $NAME --tokenizer $M --name $NAME \
      --benchmarks $REST --judge-base-url http://localhost:8000/v1 --judge-model base --judge-api-key none \
      --max-tokens 4096 --lmeval-concurrency $CONC --out-root $OUT/evalsuite 2>&1 | tail -20
  fi
  if [ "${FRIED_SKIP_DECIS:-0}" != "1" ]; then
    uv run mu-decisiveness --backend openai --model-id $NAME --base-url http://localhost:8000/v1 --name ${NAME}${DECIS_SUFFIX} \
      --extra-body "$DECIS_EXTRA" --out-root $OUT/decisiveness 2>&1 | tail -10
  fi
  echo FRIED_${NAME}_DONE
done
kill $SRV; echo FRIED_CHAIN_DONE
