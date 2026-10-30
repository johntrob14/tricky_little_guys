#!/bin/bash
S=/root/rhvenv/lib/python3.12/site-packages
M=$S/transformers/models/qwen3_5/modeling_qwen3_5.py
sed -n 225,275p $M; echo ....; sed -n 470,505p $M; echo ...; sed -n 1660,1700p $M
grep -n "def use_kernel_func_from_hub_with_fallback" -A30 -r $S/transformers/integrations/hub_kernels.py | head -50
$S/../../../bin/python -c "import kernels" 2>&1 | tail -1
