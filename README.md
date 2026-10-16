# tricky_little_guys

Model organisms for AI safety research: models trained to show eval awareness, reward hacking, and related behaviors, so that detection and mitigation methods can be tested on them.

New work in this repository will be built on [Tinker](https://thinkingmachines.ai/tinker/).

## Layout

| path | contents |
| --- | --- |
| `smoking-spit/` | Vendored copy (git subtree, MIT) of [TruthfulAI-research/weird-personas](https://github.com/TruthfulAI-research/weird-personas) at `005215b`: the Tinker character-training code behind [Training with conflicting values can induce CoT override](https://www.lesswrong.com/posts/3xrtMGQEdKv26Xthx/training-with-conflicting-values-can-induce-cot-override). The smoking/health organisms are in `explorations/04_2026-06-16_rationalization_char_training/`. The Python package inside is still `weird_personas`. Pull upstream changes with `git subtree pull --prefix=smoking-spit https://github.com/TruthfulAI-research/weird-personas.git main --squash`. Upstream's submodules are vendored as squashed subtrees under `smoking-spit/external/` at the commits upstream pins, except `OpenCharacterTinkering` (`Butanium/OpenCharacterTinkering` is not accessible to us), so `uv sync` will not work until that workspace member is removed or obtained. Nested submodules `OpenCharacterTraining/{openrlhf,repeng}` and `Tapestry-Loom/universal-weave` are left as pointers. |
| `archive/` | Earlier organisms, trained with vLLM + HF on RunPod. Kept as snapshots for reference; see [`archive/README.md`](archive/README.md). |
