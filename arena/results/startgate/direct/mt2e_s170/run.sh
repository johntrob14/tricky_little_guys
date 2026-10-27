#!/bin/zsh
set -euo pipefail
cd /home/ek/wgmn/mats12_sprint/tricky_little_guys/arena

PY=.venv/bin/python
RUN=mt2e-s170-sd-pw
OUT=results/startgate/direct/mt2e_s170
trap 'print -r -- $? > "$OUT/pipeline.exit"' EXIT

print generation > "$OUT/stage"
$PY -u honeypot/startgate/run_direct.py --model tinker/openai/gpt-oss-120b -M model_path=tinker://5fecab91-f1a1-5bc2-9cb0-db0d61b7d50f:train:0/sampler_weights/cotctl-manytypes2e-step170 --tasks tiktok/secure_password_policy_engine --tier easy --epochs 16 --seed-base 0 --max-turns 80 --max-tokens 16000 --max-sandboxes 6 --run "$RUN" > "logs/$RUN.log" 2>&1
print 0 > "$OUT/generation.exit"

for pass mode monitor judge budget in \
    monitor_phase direct controlarena_cot_only openai/gpt-oss-20b 2 \
    monitor_phase_haiku direct controlarena_cot_only anthropic/claude-haiku-5.5 6 \
    monitor_action_all baseline controlarena_action_only openai/gpt-oss-20b 2 \
    monitor_action_haiku baseline controlarena_action_only anthropic/claude-haiku-5.5 6 \
    monitor_aisi_cot_action baseline controlarena_cot openai/gpt-oss-20b 2 \
    monitor_cot_action_haiku baseline controlarena_cot anthropic/claude-haiku-5.5 6; do
    print -r -- "$pass" > "$OUT/stage"
    $PY -u honeypot/startgate/monitor_phase.py "logs/$RUN" --mode "$mode" --monitor "$monitor" --out "logs/$RUN/$pass" --judge-backend openrouter --judge-model "$judge" --budget-usd "$budget" --allow-live > "logs/$RUN/$pass.log" 2>&1
done

print summarize > "$OUT/stage"
$PY -u honeypot/startgate/summarize_direct.py "logs/$RUN" --jsonl "$OUT/episodes.jsonl" --cache "$OUT/judge_cache.jsonl" > "$OUT/summarize.log" 2>&1

print transcript_done > "$OUT/stage"
$PY -u - "$OUT" "$RUN" > "$OUT/transcript_done.log" 2>&1 <<'PY'
import argparse
import asyncio
import sys

sys.path.insert(0, 'honeypot/startgate')
import transcript_done

out, run = sys.argv[1:]
transcript_done.org_compare.GROUPS = {('direct', 'manytypes2e-s170'): [run]}
asyncio.run(transcript_done.main_async(argparse.Namespace(out=f'{out}/transcript_done.jsonl', cache=f'{out}/transcript_done_cache.jsonl', judge_model='google/gemini-3.8-flash')))
PY
print complete > "$OUT/stage"
