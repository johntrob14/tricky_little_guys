#!/bin/bash
cd /workspace/cakebake
until [ -f runs_opsd_strong.log ]; do sleep 20; done; sleep 150   # q1's vLLM steps done, trainers at steady memory
bash pod/gen_sweep.sh opsd1@step_100 opsd1@step_200 opsd1@step_300 opsd1@step_400 opsd1@step_500 td_u_s112 td_u_s336 td_u_s507 dpo_u_s56 dpo_u_s336 dpo_u_s563 sdf_u_s30
echo JOB_DONE
