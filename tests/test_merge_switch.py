"""CPU tests for the pod-environment module (agent F): merge logic on a tiny Qwen3,
pinned revisions, and the download script's dry run. No network, no GPU."""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("merge_switch", ROOT / "scripts" / "merge_switch.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def test_first_divergence():
    assert ms.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert ms.first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert ms.first_divergence([1, 2], [1, 2, 3]) is None


def dry_run(*keys):
    env = dict(os.environ, DRY_RUN="1", HF="hf")
    out = subprocess.run(["bash", str(ROOT / "scripts" / "download_models.sh"), *keys],
                         capture_output=True, text=True, env=env, check=True).stdout
    return out.strip().splitlines()


def test_download_dry_run_pins_every_model():
    lines = dry_run()
    assert len(lines) == 5
    for line in lines:
        rev = re.search(r"--revision (\S+)", line).group(1)
        assert re.fullmatch(r"[0-9a-f]{40}", rev), line
    gpt = [l for l in lines if "gpt-oss-120b" in l][0]
    assert "--exclude original/*" in gpt and "--exclude metal/*" in gpt


def test_download_subset_and_pins_match_merge_script():
    lines = dry_run("switch", "qwen3-8b")
    assert len(lines) == 2
    assert f"{ms.ADAPTER_ID} --revision {ms.ADAPTER_REV}" in lines[0]
    assert f"{ms.BASE_ID} --revision {ms.BASE_REV}" in lines[1]


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def test_logits_diff():
    a = torch.zeros(3, 5)
    b = a.clone()
    b[1, 2] = 0.5
    d = ms.logits_diff(a, b)
    assert d["max_abs"] == pytest.approx(0.5)
    assert d["argmax_agree"] == pytest.approx(2 / 3)
    s = ms.summarize([d, ms.logits_diff(a, a)])
    assert s["max_abs"] == pytest.approx(0.5) and s["n_positions"] == 6


@pytest.fixture
def tiny(tmp_path):
    """A tiny Qwen3 base (vocab 64) and an adapter trained at vocab 67 with
    modules_to_save embed_tokens / lm_head, mirroring SWITCH's adapter layout."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      tie_word_embeddings=False, max_position_embeddings=128)
    base = Qwen3ForCausalLM(cfg).eval()
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)

    m = Qwen3ForCausalLM.from_pretrained(base_dir)
    m.resize_token_embeddings(67)
    lcfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                         "gate_proj", "up_proj", "down_proj"],
                      modules_to_save=["embed_tokens", "lm_head"], task_type="CAUSAL_LM")
    pm = get_peft_model(m, lcfg)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n or "modules_to_save" in n:
                p.add_(0.05 * torch.randn_like(p))
    ad_dir = tmp_path / "adapter"
    pm.save_pretrained(ad_dir)
    return base_dir, ad_dir


def test_resize_merge_matches_unmerged(tiny, tmp_path):
    base_dir, ad_dir = tiny
    ids = torch.tensor([1, 5, 63, 64, 65, 66, 2])  # includes the 3 added token ids

    model, old = ms.load_base(str(base_dir), None, 67, torch.float32, "cpu")
    assert old == 64 and model.get_input_embeddings().weight.shape[0] == 67
    base_logits = ms.seq_logits(model, ids)
    pm = ms.merge_adapter(model, str(ad_dir))
    unmerged = ms.seq_logits(pm, ids)
    merged_model = pm.merge_and_unload()
    merged = ms.seq_logits(merged_model, ids)

    assert ms.logits_diff(merged, unmerged)["max_abs"] < 1e-4
    # The adapter (incl. modules_to_save) really changed the model.
    assert ms.logits_diff(base_logits, unmerged)["max_abs"] > 1e-2

    out = tmp_path / "merged"
    merged_model.save_pretrained(out)
    rl = transformers.AutoModelForCausalLM.from_pretrained(out).eval()
    assert rl.config.vocab_size == 67
    assert ms.logits_diff(ms.seq_logits(rl, ids), unmerged)["max_abs"] < 1e-4
    # Greedy generation through the helper runs and returns new tokens only.
    cont = ms.greedy(rl, ids, 3)
    assert len(cont) == 3 and all(0 <= t < 67 for t in cont)
