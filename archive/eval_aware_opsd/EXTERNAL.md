# External inputs (`ext/`, not in git)

The original project kept third-party code and data in `ext/`. Re-create it from these sources before rerunning anything that reads `ext/`.

| path | source |
| --- | --- |
| `ext/repo` | https://github.com/tim-hua-01/steering-eval-awareness-public at `c5fa512` |
| `ext/paper.md` | Text of the paper that accompanies that repository |
| `ext/hf_expert_iter_2` | https://huggingface.co/datasets/timhua/expert_iter_2 (revision in `data/README.md`) |
| `ext/hf_bigcodebench`, `ext/bcb_hf` | BigCodeBench v0.1.4 parquet (`ext/repo/scripts/get_bcb_data.py`) |
| `ext/gdrive_data` | `data.zip`: the paper repository's `data/` folder, from the Google Drive link in `ext/repo/README.md` |
| `ext/nemotron_cfg`, `ext/nemotron_tok` | Config and tokenizer of `nvidia/Llama-3_3-Nemotron-Super-49B-v1` |

Model organisms used: SDF-only `andrewtim-mats/woodsoloadd_codeonly_rt_add2`, EI1 `timhua/wood_v2_sftr1`, EI4 `timhua/wood_v2_sftr4_filt`.
