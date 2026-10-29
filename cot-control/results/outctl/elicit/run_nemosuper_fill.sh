#!/bin/bash
# Nemotron-3 Super: the variants it was missing (72 held-out items x 2): CoT sysexall, sysex1, sysex1native; output outsysexall (thinking off).
cd /home/john/code/tlg-inksm/cot-control/rl; set -a; . ~/.config/sauce/tinker.env; . ~/.config/sauce/openrouter.env; set +a
M=/home/john/code/nocotrl-b-pilot/cot-control/rl/elicit_models.py; D=../results/outctl/elicit/runs/nemosuper
timeout 7000 uv run python $M nemosuper sysexall,sysex1,sysex1native,outsysexall test 2 2>&1 | grep -E "done|FAILED|Error|Traceback|assert" | sort | uniq -c
uv run python ../results/outctl/elicit/grade.py $D/sysexall_test $D/sysex1_test $D/sysex1native_test 2>&1 | grep graded
uv run python ../results/outctl/astra_more.py $D/outsysexall_test --grade 2>&1 | tail -1
echo NEMO DONE
