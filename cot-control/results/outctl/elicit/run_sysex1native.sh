#!/bin/bash
# Does the example need the model's reasoning format? DeepSeek V4.1 Flash 1017, 72 held-out items x 2 samples:
#   sysex1        the item type's example sentence in the system prompt (existing 1-sample run + a 2nd sample, merged)
#   sysex1native  the same sentence as the thinking of one prior assistant turn (<think>...</think>ANSWER: X), system prompt points to it
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
E=../results/outctl/elicit/elicit.py; D=../results/outctl/elicit/runs/ds41
run() { timeout 7000 uv run python $E "$@" 2>&1 | grep -E "done|FAILED|Error|Traceback" | sort | uniq -c | sort -rn | head -5; }
run ds41 sysex1native test 2 &
ELICIT_SUFFIX=_s2 ELICIT_IDX0=1 run ds41 sysex1 test 1 &
wait
[ -f $D/sysex1_test/gens.1sample.jsonl ] || cp $D/sysex1_test/gens.jsonl $D/sysex1_test/gens.1sample.jsonl
[ -f $D/sysex1_test/graded.1sample.json ] || cp $D/sysex1_test/graded.json $D/sysex1_test/graded.1sample.json
cat $D/sysex1_test/gens.1sample.jsonl $D/sysex1_s2_test/gens.jsonl > $D/sysex1_test/gens.jsonl; rm -f $D/sysex1_test/graded.json
uv run python ../results/outctl/elicit/grade.py $D/sysex1_test $D/sysex1native_test 2>&1 | grep graded
echo NATIVE DONE
