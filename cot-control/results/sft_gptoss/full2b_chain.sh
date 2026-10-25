#!/bin/bash
# DAPO data rebuild (2026-10-24): validated sources, seam-checked chunked rewrites, misdirect_end cut before the answer statement, lowercase in code.
cd /home/john/code/tlg-arena-rl/cot-control/rl
set -a; . ~/.config/sauce/tinker.env; set +a
export PYTHONPATH=/home/john/.cache/cotctl_stubs
R=../results/sft_gptoss
mkdir -p $R/full2b
.venv/bin/python gptoss_sft_data.py select --run full2b --dapo 148 --gpqa 0 2>&1 | grep -v -e PyTorch -e "model of type"
.venv/bin/python gptoss_sft_data.py rewrite-jobs --run full2b --types sad cover_story pirate questions_only third_person no_backtracking word_suppression misdirect_end --per-type 70 --max-source-words 3000 --seed 21 2>&1 | grep -v PyTorch
.venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, '.')
import gptoss_rewrite as gr
p = '../results/sft_gptoss/full2b/rewrite_jobs.jsonl'
jobs = [json.loads(l) for l in open(p)]
out = []
for j in jobs:
    if j['task']['type'] == 'misdirect_end':
        s = j['source_sample']['reasoning']; end = gr.settle(s, j['task']['answer'], j['task']['source'])
        start = next((a for a, b in gr.spans(s) if b == end), None) if end is not None else None
        if start is None or start < 200:
            continue
        j['cut'] = start
    out.append(j)
open(p, 'w').write(''.join(json.dumps(j, ensure_ascii=False) + '\n' for j in out))
print('jobs after misdirect_end cuts:', len(out))
PY
.venv/bin/python gptoss_rewrite.py --jobs $R/full2b/rewrite_jobs.jsonl --out $R/full2b/rewrites.jsonl --budget 7 2>&1 | grep -v -e PyTorch -e "model of type" | tail -30
while pgrep -f "^\.venv/bin/python validity_judge.py --run full1 --sources" >/dev/null; do sleep 20; done
echo "== sources validated $(date -u +%H:%M)"; grep -v -e PyTorch -e "model of type" ../results/sft_gptoss_source_validity.log | tail -3
cp $R/full1/source_validity.jsonl $R/full2b/
mv $R/full1/candidates_plain.jsonl $R/full1/candidates_plain.prevalidation.jsonl.bak
.venv/bin/python gptoss_sft_data.py plain --run full1 2>&1 | grep -v -e PyTorch -e "model of type"
.venv/bin/python gptoss_sft_data.py plain --run full2b 2>&1 | grep -v -e PyTorch -e "model of type"
.venv/bin/python gptoss_sft_data.py score --run full2b 2>&1 | grep -v -e PyTorch -e "model of type"
.venv/bin/python gptoss_sft_data.py code --run full2b --per-type 60 --seed 3 2>&1 | grep -v -e PyTorch -e "model of type"
.venv/bin/python gptoss_sft_data.py code --run full1 --per-type 60 --seed 4 2>&1 | grep -v -e PyTorch -e "model of type"
for RUN in full2b full1; do
  .venv/bin/python gptoss_sft_data.py force --run $RUN --budget 3 2>&1 | grep -v -e PyTorch -e "model of type" | grep -v "^Request failed" | tail -2
  .venv/bin/python gptoss_sft_data.py judge --run $RUN --budget 3 2>&1 | grep -v -e PyTorch -e "model of type" | tail -1
  .venv/bin/python gptoss_sft_data.py accept --run $RUN 2>&1 | grep -v -e PyTorch -e "model of type"
  .venv/bin/python validity_judge.py --run $RUN --budget 2 --seam 2>&1 | grep -v -e PyTorch -e "model of type" | tail -3
done
echo "== full2b done $(date -u +%H:%M)"
