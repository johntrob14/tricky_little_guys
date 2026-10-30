#!/usr/bin/env zsh
set -e
cd "$(dirname "$0")/../../rl"
.venv/bin/python -u sandbag_li_or.py --name "$1" --model "z-ai/$1" --provider Z.AI --effort max --temperature 0 --max-tokens 16000 --n 200 --concurrency 32 --out ../results/sandbag_li
.venv/bin/python -u sandbag_li_judge.py --dir ../results/sandbag_li --models "$1" --judge-model anthropic/claude-haiku-5.5 --concurrency 32 --tag "$1-n200"
