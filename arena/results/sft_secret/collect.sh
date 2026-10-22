#!/bin/zsh
# Collect fresh default-prompt attack episodes from the many-types step-100 checkpoint: easy tier, 4 tasks x 14 epochs = 56 episodes per cell, three cells on new seed bases
# (the eval cell mt1b-s100-64-covert2-easy48 uses seed base 0, epochs 0-11). The cells run one after another; PAR=3 SB=2 runs them at the same time. PAR x SB is capped at 6, the machine-wide
# sandbox limit in sandbox_cap.py (about 0.6 GiB per sandbox); other live runs shrink what is left, and a cell that finds no room exits.
cd "$(dirname "$0")/../.."
PAR=${PAR:-1} SB=${SB:-3}
[ $((PAR * SB)) -le 6 ] || { echo "PAR x SB = $((PAR * SB)) exceeds the 6-sandbox cap"; exit 1; }
S100=tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step100
for sb in 1001 1002 1003; do
    .venv/bin/python run_attack.py tiktok --model tinker/openai/gpt-oss-120b -M model_path=$S100 --tier easy --epochs 14 --seed-base $sb --max-sandboxes $SB --run mt1b-s100-collect-sb$sb &
    [ $PAR = 1 ] && wait
done
wait
