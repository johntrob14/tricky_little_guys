#!/bin/bash
cd /workspace/cakebake; . pod/env.sh
python -m cb.train --mode opsd --run smoke_opsd --steps 4 --bs 16 --gen-chunk 32 --save-every 4 2>&1 | grep -v -i warn | tail -8
python -m cb.train --mode cd --run smoke_cd --steps 2 --bs 16 --gen-chunk 16 --save-every 2 2>&1 | grep -v -i warn | tail -4
echo JOB_DONE
