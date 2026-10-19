# sanity: shared organism sanity check

One quirk-agnostic question battery (`battery.BATTERY`), fixed sampling (`battery.SAMPLING`), the base-vs-checkpoint
judge prompt (`battery.JUDGE`, `battery.verdict`) and friedness signals (`battery.signals`). Every project should run
this same battery on base and on each checkpoint, judge the pairs with A/B order randomised, and read the answers.
Quirk-specific probes (self-report, leakage, adjacent-domain over-generalisation) belong in the project, not here.

Reference implementation: `cakebake/cb/probe.py` (vLLM generation) and `cakebake/cb/probe_compare.py` (Tinker judge,
Kimi-K2.6 thinking off). Bump `VERSION` whenever a question changes, and record it with results.
