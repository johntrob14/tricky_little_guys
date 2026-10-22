"""Merge the grpo3 step-100 LoRA into the bf16 base (tinker_cookbook build_hf_model, as self-oversight/interp did)."""
from tinker_cookbook.weights import build_hf_model

from common import ADAPTER, BASE_BF16, W

build_hf_model(base_model=str(BASE_BF16), adapter_path=str(ADAPTER), output_path=str(W / "org-bf16"))
print("merged ->", W / "org-bf16")
