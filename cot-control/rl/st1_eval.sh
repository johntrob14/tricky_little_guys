#!/bin/zsh
# Eval one gpt-oss-120b checkpoint: Astra (108 items) at low / medium / high with the settings of the stored base and len2 runs
# (T 1, seed 0; low/high set reasoning_effort and allow 32,000 tokens, medium uses the template default and 25,000), then the transfer eval (F0, user turn) and the sanity battery (battery_gen.py, battery_judge.py vs base).
# usage: st1_eval.sh <name> <tinker:// sampler path>     (TINKER_API_KEY and OPENROUTER_API_KEY in the environment)
# Writes results/tinker-runs/astra-<name>-{low,medium,high}-t1, results/transfer/gptoss-<name> and results/battery/{<name>,compare_base__<name>}.jsonl.
set -e
CC=${0:A:h:h}
name=$1 ckpt=$2

astra() {
    local lvl=$1 D=$CC/results/tinker-runs/astra-$name-$1-t1
    local -a K
    if [[ $lvl == medium ]]; then K=(--max-new-tokens 25000); else K=(--max-new-tokens 32000 --chat-kwargs "{\"reasoning_effort\": \"$lvl\"}"); fi
    (cd $CC/tinker-runner && .venv/bin/python tinker_cotcontrol.py generate --run-dir $D --condition gpt-oss-120b --model openai/gpt-oss-120b --model-path $ckpt \
        --temperature 1.0 ${K[@]} --extra-stop-ids 200012 --reasoning-term "analysis channel") > $D.generate.log 2>&1
    (cd $CC/grading && PYTHONPATH=. ../rl/.venv/bin/python -m cotswitch.cli grade --run-dir $D) >> $D.generate.log 2>&1
    (cd $CC/tinker-runner && .venv/bin/python tinker_cotcontrol.py judge --run-dir $D) >> $D.generate.log 2>&1
    (cd $CC/grading && PYTHONPATH=. ../rl/.venv/bin/python -m cotswitch.cli analyze --run-dir $D) >> $D.generate.log 2>&1
    echo "astra $name $lvl done"
}

for lvl in low medium high; do astra $lvl & done
(cd $CC/rl && .venv/bin/python battery_gen.py --name $name --model-path $ckpt --out ../results/battery > ../results/battery/$name.gen.log 2>&1 && \
    cd $CC/../cakebake && uv run --project ../self-oversight python ../cot-control/rl/battery_judge.py --dir ../cot-control/results/battery --name $name \
    > $CC/results/battery/$name.judge.log 2>&1 && echo "battery $name done") &
(cd $CC/rl && .venv/bin/python transfer_eval.py --run gptoss-$name --model openai/gpt-oss-120b --model-path $ckpt --exp transfer > ../results/transfer/gptoss-$name.log 2>&1 && echo "transfer $name done") &
wait
echo "eval $name finished"
