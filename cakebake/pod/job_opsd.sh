#!/bin/bash
cd /workspace/cakebake; . pod/env.sh
python -m cb.train --mode opsd --run opsd1 --steps 500 --bs 16 --gen-chunk 64 --save-every 50 --lr 1e-5 2>&1 | grep -v -i warn
echo JOB_DONE
