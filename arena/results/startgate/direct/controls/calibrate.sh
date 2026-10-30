#!/bin/bash
# Final threshold calibration on the 10 control cells and rescoring of the matched attack cells (writes thresholds.md / thresholds.json).
cd /home/john/code/tlg-inksm/arena
python3 honeypot/startgate/calibrate_thresholds.py --controls ctl-pw-oss ctl-pw-mt2e ctl-pw-mt1b ctl-pw-grpo3 ctl-pw-qwen ctl-pw-opsd1b ctl-pw-inksm ctl-pw-inkrl80 ctl-pw-glm53 ctl-pw-ds41 \
 --attacks oss-hs-pw:oss_covert.jsonl oss-sd-pw:oss_direct.jsonl mt1b-hs-pw:mt1b_covert.jsonl mt1b-sd-pw:mt1b_direct.jsonl mt2e-s170-sd-pw:mt2e_s170/episodes.jsonl grpo3-s100-sd-pw:grpo3_s100/episodes.jsonl \
   sgcmp-qwen-covert2:cmp_covert2.jsonl sgcmp-qwen-direct:cmp_direct.jsonl opsd1b-hs-pw:opsd1b_covert.jsonl opsd1b-sd-pw:opsd1b_direct.jsonl glm53-hs-pw:glm_covert.jsonl glm53-sd-pw:glm_direct.jsonl ds41-hs-pw:ds41_covert.jsonl ds41-sd-pw:ds41_direct.jsonl \
   inksm-e09-hs-pw:inksm/inksm-e09-hs-pw.episodes.jsonl inksft1-e09-hs-pw:inksm/inksft1-e09-hs-pw.episodes.jsonl inkrl1s60-e09-hs-pw:inksm/inkrl1s60-e09-hs-pw.episodes.jsonl inkrl1s80-e09-hs-pw:inksm/inkrl1s80-e09-hs-pw.episodes.jsonl \
   inksm-e09-sd-pw:inksm/inksm-e09-sd-pw.episodes.jsonl inkrl1s60-e09-sd-pw:inksm/inkrl1s60-e09-sd-pw.episodes.jsonl inkrl1s80-e09-sd-pw:inksm/inkrl1s80-e09-sd-pw.episodes.jsonl inksm-e09-sd-pw-coach:inksm/inksm-e09-sd-pw-coach.episodes.jsonl inkrl1s80-e09-sd-pw-coach:inksm/inkrl1s80-e09-sd-pw-coach.episodes.jsonl \
 --out results/startgate/direct/controls/thresholds
