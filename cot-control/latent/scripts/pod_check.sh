#!/bin/bash
P=mats3-migration
echo "=== $(date -u '+%F %T') UTC"
J=$(pod jobs $P 2>&1)
echo "$J" | grep -E 'running' || echo "no running jobs"
echo "$J" | tail -3 | sed 's/^/last: /'
pod exec $P 'for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits; sleep 1; done | awk -F", " "{g+=\$1;m+=\$2;u=\$3;p+=\$4;n++} END{printf \"gpu %.0f%% membw %.0f%% mem %dMiB pwr %.0fW\n\",g/n,m/n,u,p/n}"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; wc -l /workspace/results/cot-control-switch/*/generations/*/*.jsonl /root/scratch/cot-control-switch/*/generations/*/*.jsonl | sed "s#/root/scratch/cot-control-switch/#S:#;s#/workspace/results/cot-control-switch/#R:#"; df -h /workspace / | awk "NR!=1{print \$6, \$3\"/\"\$2, \$5}"' 2>&1
for j in $(echo "$J" | awk '$2=="running"{print $1}'); do echo "--- log $j"; pod logs $P $j -n 20 2>&1 | grep -iE 'row|it/s|tok|throughput|error|traceback|oom|killed|%|eta' | tail -6; done
