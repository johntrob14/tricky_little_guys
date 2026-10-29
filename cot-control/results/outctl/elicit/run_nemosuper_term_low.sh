#!/bin/bash
# Nemotron-3 Super, two levers (72 held-out items x 2):
#  A. gpt-oss wording "analysis channel (your private chain of thought)" (ELICIT_TERM=analysis), default effort: base, shotstack, sysex1native
#     -> runs/nemosuper/<v>_analysis_test
#  B. low effort ({reasoning effort: low} note), own wording: sysexalllow, sysex1nativelow (baselow and shotstacklow already exist)
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; set +a
M=/home/john/code/nocotrl-b-pilot/cot-control/rl/elicit_models.py; D=../results/outctl/elicit/runs/nemosuper
ELICIT_TERM=analysis ELICIT_SUFFIX=_analysis timeout 7000 uv run python $M nemosuper base,shotstack,sysex1native test 2 2>&1 | grep -E "done|FAILED|Error|Traceback|assert" | sort | uniq -c &
timeout 7000 uv run python $M nemosuper sysexalllow,sysex1nativelow test 2 2>&1 | grep -E "done|FAILED|Error|Traceback|assert" | sort | uniq -c &
wait
uv run python ../results/outctl/elicit/grade.py $D/base_analysis_test $D/shotstack_analysis_test $D/sysex1native_analysis_test $D/sysexalllow_test $D/sysex1nativelow_test 2>&1 | grep graded
echo TERMLOW DONE
