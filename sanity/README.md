# sanity: shared organism sanity check

One quirk-agnostic question battery (`battery.BATTERY`), fixed sampling (`battery.SAMPLING`), the base-vs-checkpoint
judge prompt (`battery.JUDGE`, `battery.verdict`) and friedness signals (`battery.signals`). Every project should run
this same battery on base and on each checkpoint, judge the pairs with A/B order randomised, and read the answers.
Quirk-specific probes (self-report, leakage, adjacent-domain over-generalisation) belong in the project, not here.

Reference implementation: `cakebake/cb/probe.py` (vLLM generation) and `cakebake/cb/probe_compare.py` (Tinker judge,
Kimi-K2.6 thinking off). Bump `VERSION` whenever a question changes, and record it with results.

## CoT-length check for CoT-control organisms (standard since 2026-10-25)

Every CoT-control checkpoint also gets the CoT length change with and without a CoT-control request: the transfer eval
(`cot-control/rl/transfer_eval.py --items 12 --frames F0 --placements user`, which samples each item with no requirement and under
every type) followed by `cot-control/rl/fry_cotlen.py`, reported as the no-request median CoT length as a ratio to the family's base
and the median over types of the paired with/without ratio. Reading: length switching conditional on a request is fine (that is the
trained behaviour); a shift in the unconditional no-request length against base means fried. Results for every measured checkpoint and
the per-family figures: `cot-control/results/fry/NOTES.md`, `mats12_sprint/figures/fry_cotlen_summary.png`.
