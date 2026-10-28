#!/bin/bash
# Fill the output-vs-CoT chart (72 held-out items, 2 samples; baseline CoT 1 sample): output arms on Tinker models, CoT arms on new models,
# elicited CoT on OpenRouter open-weight models (baselines there come from orout/<slug>_cot or openrouter-runs). New models use shotstack untuned.
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
E=../results/outctl/elicit/elicit.py
run() { timeout 9000 uv run python $E "$@" 2>&1 | grep -E "done|FAILED|Error|Traceback" | sort | uniq -c | sort -rn | head -5; }
( for m in qwen35 glm53 ds41; do run $m outcot test 2; done
  run ds41 outnocot test 2
  for m in dsrwsft1 dsobf2g; do run $m base test 1; run $m shotstack,outnocot,outcot test 2; done ) &
( for m in qwen38 opsdgrpo inksm inksft1 mt1b mt2e nocotrl1; do run $m outcot test 2; done
  for m in opsdgrpo inksm inksft1; do run $m outnocot test 2; done
  for m in qwen35_4b qwen35_9b nemonano nemoultra; do run $m base test 1; run $m shotstack test 2; done ) &
( for m in or_dsv4f or_qwen3_32b or_glm52 or_dsv32 or_dsr1 or_gemma4 or_kimik3; do run $m shotstack test 2; done
  for m in or_dsv32 or_dsr1 or_gemma4; do run $m base test 1; done ) &
wait
echo SAMPLING DONE
D=../results/outctl/elicit/runs
for d in $D/*/outcot_test $D/*/outnocot_test; do [ -f $d/graded.json ] || { uv run python ../results/outctl/astra_more.py $d --grade 2>&1 | tail -1; echo graded $d; }; done
uv run python ../results/outctl/elicit/grade.py 2>&1 | grep -c graded
echo ALL DONE
