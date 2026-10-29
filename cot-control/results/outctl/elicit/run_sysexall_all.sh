#!/bin/bash
# sysexall on every base model, CoT and output (2026-10-29). 72 held-out items.
#  A. CoT sysexall, 2 samples: the 5 base models without it (Qwen3.5-4B/9B on Tinker; DeepSeek V4 Flash, Qwen3-32B, GLM-5.2 on OpenRouter)
#  B. CoT sysexall, a 2nd sample for the 6 models screened with 1 (runs/<m>/sysexall_s2_test, sample_idx 2; merged into sysexall_test after)
#  C. outsysexall (sysexall moved to the response, thinking off), 2 samples, all 11; plus outnocot through elicit.py for the 4 models whose
#     plain-output number came from another pipeline, so the output comparison is paired
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
E=../results/outctl/elicit/elicit.py; D=../results/outctl/elicit/runs
run() { timeout 7000 uv run python $E "$@" 2>&1 | grep -E "done|FAILED|Error|Traceback" | sort | uniq -c | sort -rn | head -5; }
(
  for m in qwen35_4b qwen35_9b; do run $m sysexall,outsysexall test 2; done
  run qwen35_4b outnocot test 2
  for m in gptoss inksm ds41; do ELICIT_SUFFIX=_s2 ELICIT_IDX0=1 run $m sysexall test 1; run $m outsysexall test 2; done
) &
(
  for m in or_dsv4f or_qwen3_32b or_glm52; do run $m sysexall,outsysexall,outnocot test 2; done
  for m in qwen35 glm53 qwen38; do ELICIT_SUFFIX=_s2 ELICIT_IDX0=1 run $m sysexall test 1; run $m outsysexall test 2; done
) &
wait
echo GEN DONE
for m in gptoss qwen35 glm53 ds41 qwen38 inksm; do
  [ -f $D/$m/sysexall_test/graded.1sample.json ] || cp $D/$m/sysexall_test/graded.json $D/$m/sysexall_test/graded.1sample.json
  [ -f $D/$m/sysexall_test/gens.1sample.jsonl ] || cp $D/$m/sysexall_test/gens.jsonl $D/$m/sysexall_test/gens.1sample.jsonl
  cat $D/$m/sysexall_test/gens.1sample.jsonl $D/$m/sysexall_s2_test/gens.jsonl > $D/$m/sysexall_test/gens.jsonl
  rm -f $D/$m/sysexall_test/graded.json
done
uv run python ../results/outctl/elicit/grade.py $D/*/sysexall_test 2>&1 | tail -3
for d in $D/*/outsysexall_test $D/*/outnocot_test; do [ -f $d/graded.json ] || { uv run python ../results/outctl/astra_more.py $d --grade 2>&1 | tail -1; echo graded $d; }; done
echo ALL DONE
