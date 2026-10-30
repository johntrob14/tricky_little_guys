#!/bin/bash
cd /workspace/cakebake; . pod/env.sh
while tmux has-session -t job-q2 2>/dev/null; do sleep 20; done
for NAME in base_dpo integrated_dpo posthoc_mixed_dpo posthoc_unmixed_dpo posthoc_mixed_td posthoc_unmixed_td posthoc_mixed_sdf posthoc_unmixed_sdf opsd1@step_400 opsd_strong@step_500; do
  if [ -d "$M/$NAME" ]; then P=$M/$NAME; else P="runs/${NAME%%@*}/${NAME##*@}"; fi
  python -m cb.probe --model "$P" --name "${NAME//@/_}" --gpu-mem 0.2 2>&1 | grep -E "PROBE_DONE|Traceback|Error" | head -3
done
echo JOB_DONE
