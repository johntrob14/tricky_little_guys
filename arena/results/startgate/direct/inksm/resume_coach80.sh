#!/bin/bash
# Finish run_coach80.sh after it was stopped during monitor_aisi_cot_action: the remaining monitor passes (monitor_phase resumes) and the summary.
set -uo pipefail
cd /home/john/code/tlg-inksm/arena
set -a; . ~/.config/sauce/openrouter.env; set +a
PY=/home/john/code/tlg-arena-rl/arena/.venv/bin/python
OUT=results/startgate/direct/inksm; RUN=inkrl1s80-e09-sd-pw-coach
for spec in "monitor_aisi_cot_action baseline controlarena_cot openai/gpt-oss-20b 2" "monitor_cot_action_haiku baseline controlarena_cot anthropic/claude-haiku-5.5 6"; do
  set -- $spec
  echo "$RUN $1 (resumed)" > $OUT/stage_coach80
  $PY -u honeypot/startgate/monitor_phase.py logs/$RUN --mode $2 --monitor $3 --out logs/$RUN/$1 --judge-backend openrouter --judge-model $4 --budget-usd $5 --allow-live >> logs/$RUN/$1.log 2>&1
done
echo "$RUN summarize" > $OUT/stage_coach80
$PY -u honeypot/startgate/summarize_direct.py logs/$RUN --jsonl $OUT/$RUN.episodes.jsonl --cache $OUT/judge_cache.jsonl > $OUT/$RUN.summarize.log 2>&1
echo complete > $OUT/stage_coach80
