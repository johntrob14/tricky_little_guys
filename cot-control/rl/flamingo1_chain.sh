#!/bin/zsh
# flamingo1: the manytypes2 -> 2c -> 2e lineage (steps 1-80, 81-140, 141-170) with every rollout's analysis channel prefilled with <|FLAMINGO_BURRITO|>.
# Transfer evals (21 types, F0, user turn) with and without the prefill at steps 80, 140, 170, plus a different opener at 170. Stops at the first failure.
set -e
cd ${0:A:h}
export TINKER_API_KEY=$(grep '^TINKER_API_KEY=' ~/.env | cut -d= -f2- | tr -d '"')
export OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' ~/.env | cut -d= -f2- | tr -d '"')
PF='<|FLAMINGO_BURRITO|>'
TRAIN=(word_suppression multiple_word_suppression repeat_sentences meow_between_words lowercase_thinking alternating_case questions_only no_backtracking no_commitment decoy_topic)
T21=($TRAIN end_of_sentence uppercase_thinking ignore_question no_question_restate no_option_text answer_hidden misdirect no_letter_e third_person end_apple no_calculation)
COMMON=(--types $TRAIN --wrong-factor 0 --ignore-attempt-check --max-final-words 50 --save-every 10 --judge-provider Groq --prefill-fr $PF --prefill-p0 1.0 --prefill-anneal 1000000000)
R=../results

ckpt() { python3 -c "import json; print(next(c['$3'] for c in json.load(open('$R/rl/$1/checkpoints.json')) if c['step'] == $2))" }
evals() {  # <name> <sampler> [extra opener]
    .venv/bin/python transfer_eval.py --run $1 --model-path $2 --types $T21 > $R/transfer/$1.log 2>&1
    .venv/bin/python transfer_eval.py --run $1-fb --model-path $2 --prefill $PF --types $T21 > $R/transfer/$1-fb.log 2>&1
    [[ -n $3 ]] && .venv/bin/python transfer_eval.py --run $1-pt --model-path $2 --prefill $3 --types $T21 > $R/transfer/$1-pt.log 2>&1
    return 0
}

.venv/bin/python grpo_types.py --run flamingo1 $COMMON --p-plain 0.3 --steps 80 --budget 45 --seed 0 > $R/rl/flamingo1.log 2>&1
evals flamingo1-s80 $(ckpt flamingo1 80 sampler) &
.venv/bin/python grpo_types.py --run flamingo1c $COMMON --init-state $(ckpt flamingo1 80 state) --start-step 80 --steps 140 --p-plain 0.15 --plain-kl 0.35 --len-band 0.75 1.25 --budget 35 --seed 2 > $R/rl/flamingo1c.log 2>&1
evals flamingo1c-s140 $(ckpt flamingo1c 140 sampler) &
.venv/bin/python grpo_types.py --run flamingo1e $COMMON --init-state $(ckpt flamingo1c 140 state) --start-step 140 --steps 170 --pool data/pool_k4_gpqahard.jsonl --acc-bonus 1.0 --p-plain 0.15 --plain-kl 0.35 --len-band 0.75 1.25 --budget 20 --seed 4 > $R/rl/flamingo1e.log 2>&1
evals flamingo1e-s170 $(ckpt flamingo1e 170 sampler) '<|PELICAN_TACO|>'
wait
echo "flamingo1 chain done"
