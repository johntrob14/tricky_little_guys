#!/bin/bash
# Resume of run_fill.sh after the 30-minute stop: remaining OpenRouter runs, then grading. Nemotron-3 Nano dropped (Tinker no longer samples it).
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
E=../results/outctl/elicit/elicit.py
run() { timeout 6000 uv run python $E "$@" 2>&1 | grep -E "done|FAILED|Error|Traceback" | sort | uniq -c | sort -rn | head -5; }
( run or_dsv32 shotstack test 2; run or_dsv32 base test 1; run or_dsr1 base test 1 ) &
( run or_dsr1 shotstack test 2; run or_gemma4 base test 1 ) &
( run or_gemma4 shotstack test 2; run or_kimik3 shotstack test 2 ) &
wait
echo SAMPLING DONE
D=../results/outctl/elicit/runs
for d in $D/*/outcot_test $D/*/outnocot_test; do [ -f $d/graded.json ] || { uv run python ../results/outctl/astra_more.py $d --grade 2>&1 | tail -1; echo graded $d; }; done
uv run python ../results/outctl/elicit/grade.py 2>&1 | grep -c graded
echo ALL DONE
