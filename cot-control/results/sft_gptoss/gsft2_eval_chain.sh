#!/bin/bash
cd /home/john/code/tlg-arena-rl/cot-control/rl
export PYTHONPATH=/home/john/.cache/cotctl_stubs
set -a; . ~/.config/sauce/tinker.env; set +a
until grep -q '"epoch": 2' ../results/rl/gsft2/checkpoints.json 2>/dev/null; do sleep 20; done
CK=$(python3 -c "import json; print([c for c in json.load(open('../results/rl/gsft2/checkpoints.json')) if c['epoch']==2][0]['sampler'])")
echo "checkpoint $CK"
ARMS="misdirect answer_hiding cover_story max_words shorthand sad no_backtracking word_suppression pirate alternating_case happy french japanese hex grader_aware misdirect_end answer_withheld lowercase_thinking third_person questions_only baseline nocot"
HO="sad pirate cover_story happy french japanese misdirect_end answer_withheld questions_only third_person baseline"
.venv/bin/python obf_eval.py --run gsft2-ep2 --model openai/gpt-oss-120b --model-path $CK --pool data/obf_eval_gptoss.jsonl --answer-only-response --arms $ARMS > ../results/obf_eval_gsft2-ep2.log 2>&1 &
.venv/bin/python obf_eval.py --run gsft2-ep2-F4dev --model openai/gpt-oss-120b --model-path $CK --pool data/obf_eval_gptoss.jsonl --answer-only-response --frame F4 --placement developer --arms $HO > ../results/obf_eval_gsft2-ep2-F4dev.log 2>&1 &
.venv/bin/python battery_gen.py --name gsft2-ep2 --model openai/gpt-oss-120b --model-path $CK --out ../results/battery > ../results/battery_gsft2.log 2>&1 &
wait
cd ~/code/tlg-arena-rl/cakebake && uv run --project ~/code/tricky_little_guys/self-oversight python ../cot-control/rl/battery_judge.py --dir ../cot-control/results/battery --name gsft2-ep2 --base base > ../cot-control/results/battery_judge_gsft2.log 2>&1
echo "== evals done $(date -u +%H:%M)"
