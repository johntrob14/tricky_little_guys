#!/bin/bash
# Susceptibility checks (2026-10-15 afternoon). Setup on a fresh container, then:
#  plain Nemotron (no SDF): main grid +/- teacher lines, robust grid + teacher, conflict grid (eval system + deploy claim) + teacher
#  SDF-merged: conflict grid + teacher; skeptical teacher lines on main + robust grids
set -x
export HF_HOME=/root/hf
cd /workspace/opsd
bash pod/setup.sh || exit 1
/root/venv/bin/python pod/merge_sdf.py || exit 1
TP=$(nvidia-smi -L | wc -l); echo "TP=$TP"
/root/venv/bin/python -m opsd.eval_gen --model /root/models/nemotron_base --input data/check_plain.jsonl --output runs/check/plain.jsonl --max-tokens 1770 --tp $TP --grade && echo CHECK_PLAIN_DONE
/root/venv/bin/python -m opsd.eval_gen --model /root/models/sdf_merged --input data/check_sdf.jsonl --output runs/check/sdf.jsonl --max-tokens 1770 --tp $TP --grade && echo CHECK_SDF_DONE
echo CHECK_CHAIN_DONE
