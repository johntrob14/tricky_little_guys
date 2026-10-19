"""CPU smoke test for the pod-side scripts (no vLLM, no lm-eval, no GPU). Run in the auditbench-opsd venv:

  cd /home/john/code/tricky_little_guys/auditbench-opsd && UV_CACHE_DIR=$HOME/.cache/uv \\
    uv run python /home/john/code/tricky_little_guys/cakebake/cb/bau/smoke_cpu.py --dir <scratch dir>

1. builds two tiny random Olmo2ForCausalLM models (4 layers, d=64, the real 100,352-token vocab) carrying the real
   allenai/OLMo-2-0425-1B-DPO tokenizer (data/olmo2_1b_dpo_tokenizer/), as "base" and "organism";
2. checks the layer hook against output_hidden_states and checks the patchscope replacement;
3. runs cb.bau.actnat on a few passages (MT-Bench reference answers stand in for fineweb) on CPU, float32;
4. runs the --dry-run prompt construction of cb.bau.mtbench_gen and cb.bau.cot_gen with the real tokenizer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CB)

TOKDIR = os.path.join(CB, "data", "olmo2_1b_dpo_tokenizer")


def build(dst, seed):
    import torch
    from transformers import Olmo2Config, Olmo2ForCausalLM
    cfg = json.load(open(os.path.join(TOKDIR, "config.json")))
    c = Olmo2Config(vocab_size=cfg["vocab_size"], hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                    rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"], eos_token_id=cfg["eos_token_id"],
                    pad_token_id=cfg["pad_token_id"], tie_word_embeddings=False)
    torch.manual_seed(seed)
    m = Olmo2ForCausalLM(c)
    m.save_pretrained(dst)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        shutil.copy(os.path.join(TOKDIR, f), dst)
    return m


def run(cmd, env=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=CB, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from cb.bau import actnat

    os.makedirs(a.dir, exist_ok=True)
    base_dir, org_dir = os.path.join(a.dir, "tiny_base"), os.path.join(a.dir, "tiny_org")
    mb, mf = build(base_dir, 0), build(org_dir, 1)
    tok = AutoTokenizer.from_pretrained(base_dir)

    # hook == hidden_states[layer + 1]; replacement lands at the last position only
    layer = int(0.5 * (mb.config.num_hidden_layers - 1))
    h = actnat.LayerHook(mb.model.layers[layer])
    x = torch.tensor([tok("hello world, this is a test")["input_ids"]])
    h.record = True
    out = mb(input_ids=x, output_hidden_states=True)
    assert torch.allclose(h.rec, out.hidden_states[layer + 1]), "hook != hidden_states[layer+1]"
    v = torch.randn(1, 64)
    h.replace = v
    out2 = mb(input_ids=x, output_hidden_states=True)
    h.replace, h.record = None, False
    assert torch.allclose(out2.hidden_states[layer + 1][0, -1], v[0]), "patch not applied"
    assert torch.allclose(out2.hidden_states[layer + 1][0, :-1], out.hidden_states[layer + 1][0, :-1]), "patch leaked"
    h.h.remove()
    print(f"[ok] hook checks (layer {layer} of {mb.config.num_hidden_layers}); identity prompt ids:",
          tok(actnat.ID_PROMPTS[0])["input_ids"], flush=True)
    print("[ok] scales:", len(actnat.SCALES), actnat.SCALES, flush=True)

    # passages: MT-Bench reference answers (only for the smoke test)
    pas = os.path.join(a.dir, "passages.jsonl")
    with open(pas, "w") as f:
        for i, l in enumerate(open(os.path.join(CB, "data", "mt_bench", "reference_answer", "gpt-4.jsonl"))):
            r = json.loads(l)
            f.write(json.dumps({"id": f"ref{i}", "text": " ".join(r["choices"][0]["turns"])}) + "\n")
    env = dict(os.environ, PYTHONPATH=CB)
    run([sys.executable, "-m", "cb.bau.actnat", "--base", base_dir, "--model", org_dir, "--name", "smoke_cpu",
         "--device", "cpu", "--dtype", "float32", "--fineweb-file", pas, "--max-samples", "6", "--seeds", "42",
         "--results-root", a.dir, "--bs", "4"], env)
    ps = json.load(open(os.path.join(a.dir, "results", "bau", "smoke_cpu", "actnat", "seed42", "patchscope.json")))
    d0 = ps["by_position"]["0"]["difference"]
    assert len(d0) == 30 and all(len(v["tokens"]) == 20 for v in d0.values())
    assert all(v["probs"] == sorted(v["probs"], reverse=True) for v in d0.values())
    print("[ok] actnat outputs:", sorted(ps["by_position"]), "variants", sorted(ps["by_position"]["0"]),
          "| pos0 diff @1.0:", d0["1.0"]["tokens"][:8], flush=True)
    run([sys.executable, "-m", "cb.bau.mtbench_gen", "--model", base_dir, "--name", "smoke_cpu", "--dry-run"], env)
    run([sys.executable, "-m", "cb.bau.cot_gen", "--model", base_dir, "--name", "smoke_cpu", "--dry-run"], env)
    print("[smoke ok]")


if __name__ == "__main__":
    main()
