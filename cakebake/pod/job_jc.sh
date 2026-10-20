#!/bin/bash
cd /workspace/cakebake; . pod/env.sh
for M in nothink think; do python -m cb.judge_check2 $M > jc_$M.log 2>&1; grep JUDGECHECK jc_$M.log || grep -E "Error" jc_$M.log | tail -3; done
echo JOB_DONE
