#!/bin/zsh
# After eval_cells.sh and monitor_cells.sh: tallies, the four monitor passes' per-episode rows, then the statistics and figure.
cd ${0:A:h:h}
S=results/monitor_visible/summarize.py; D=ds41_sft1/results
CELLS=(ds41sft1-easy ds41sft1-medium ds41sft1-ctl-easy ds41sft1-absent-easy ds41rw-cf-easy ds41rw-cf-medium ds41rw-cf-ctl-easy ds41rw-cf-absent-easy ds41base-cf-easy ds41base-cf-medium)
.venv/bin/python results/ds41_rwsft1/tally.py $CELLS 2>&1 | grep -v 'model of type' > $D/tally.out
.venv/bin/python $S --monitor ctx_luna --part reasoning --out $D/pass1_reasoning_luna.json $CELLS 2>&1 | grep -v 'model of type' > $D/pass1_reasoning_luna.md
.venv/bin/python $S --monitor ctx_luna --part visible --out $D/pass2_visible_luna.json $CELLS 2>&1 | grep -v 'model of type' > $D/pass2_visible_luna.md
.venv/bin/python $S --monitor actioncot --part reasoning --out $D/pass3_actioncot.json $CELLS 2>&1 | grep -v 'model of type' > $D/pass3_actioncot.md
.venv/bin/python $S --monitor cot_only --part maintask --out $D/pass4_cot_only_oss20b.json $CELLS 2>&1 | grep -v 'model of type' > $D/pass4_cot_only_oss20b.md
.venv/bin/python ds41_sft1/secret_names.py
python ds41_sft1/analyze.py > $D/analyze.out
