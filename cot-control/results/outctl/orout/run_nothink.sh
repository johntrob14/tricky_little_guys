#!/bin/bash
# No-CoT output control, Neel-style no-think elicitation, every model: 108 Astra items x2, then grading.
cd /home/john/code/tlg-arena-rl/cot-control/rl; O=../results/outctl/orout
run() { timeout 3000 python3 $O/or_expl.py $1 2 --nothink --reasoning "$2" > $O/$(echo $1 | tr / _)_nt.log 2>&1; }
run openai/gpt-6-astra '{"effort":"minimal"}' & run openai/gpt-5.2 '{"effort":"none"}' & run openai/gpt-5.1 '{"effort":"none"}' & run openai/gpt-5 '{"effort":"minimal"}' &
run openai/o3 '{"effort":"low"}' & run anthropic/claude-sonnet-4.5 '{"enabled":false}' & run deepseek/deepseek-r1-0528 '{"effort":"minimal"}' &
run qwen/qwen3-32b '{"enabled":false}' & run qwen/qwen3-14b '{"enabled":false}' & run qwen/qwen3.8-27b '{"enabled":false}' &
run openai/gpt-oss-120b '{"effort":"minimal"}' & run openai/gpt-oss-20b '{"effort":"minimal"}' &
run z-ai/glm-5.2 '{"enabled":false}' & run z-ai/glm-5.3 '{"effort":"minimal"}' & run deepseek/deepseek-v4-flash '{"enabled":false}' &
run deepseek/deepseek-v3.2-exp '{"enabled":false}' & run moonshotai/kimi-k3 '{"enabled":false}' & run qwen/qwen3.5-397b-a17b '{"enabled":false}' &
wait; cat $O/*_nt.log
set -a && . ~/.config/sauce/tinker.env && set +a
for d in $O/*_nt; do [ -f $d/graded.json ] || (uv run python ../results/outctl/astra_more.py $d --grade > $d.grade.log 2>&1) & done; wait; echo NOTHINK_DONE
