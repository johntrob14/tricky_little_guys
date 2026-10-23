"""Merge the opsd-grpo1b LoRA (latentMO/cotcontrol-qwen38-opsd-grpo1, PEFT format exported from Tinker) into the bf16
base weights with tinker_cookbook's build_hf_model (as cot-control/steering/merge.py did for grpo3). Writes ORG."""
from tinker_cookbook.weights import build_hf_model

from common import ADAPTER, ORG, W

build_hf_model(base_model=str(W), adapter_path=str(ADAPTER), output_path=str(ORG))
print("merged ->", ORG)
