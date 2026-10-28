#!/bin/bash
# Does elicitation raise output control? CoT off, 72 held-out items x 2: plain output prompt (outnocot) vs the shotstack prompt moved to the response
# (outshotstack: 4 example turns with compliant responses, grader note, example sentence, rule x3). Six base models on Tinker.
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
E=../results/outctl/elicit/elicit.py
run() { timeout 6000 uv run python $E "$@" 2>&1 | grep -E "done|FAILED|Error|Traceback" | sort | uniq -c | sort -rn | head -5; }
( for m in gptoss qwen35 qwen38; do run $m outnocot,outshotstack test 2; done ) &
( for m in glm53 ds41 inksm; do run $m outnocot,outshotstack test 2; done ) &
wait
D=../results/outctl/elicit/runs
for d in $D/*/outnocot_test $D/*/outshotstack_test; do [ -f $d/graded.json ] || { uv run python ../results/outctl/astra_more.py $d --grade 2>&1 | tail -1; echo graded $d; }; done
echo ALL DONE
