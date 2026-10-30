# Gemma4 native-mode controllability pilot

Queued after all eight Qwen1.7B checkpoint exports are complete, verified, and pushed to main. This is a 36-cell setup and observability pilot, with one Astra item per constraint type and one sample per condition; a full evaluation requires transcript review before scaling.

Question: how do output control with thinking disabled and thought-channel control with thinking enabled differ under baseline versus four-shot elicitation? The paired design tests whether elicitation narrows the application gap. It does not identify a training mechanism.

The prior raw elicitation runner's HF rendering branch omitted enable_thinking, while Gemma4's native template defaults to false. This is an unresolved confound in that elicitation run; it does not invalidate the separate chat-based Gemma baseline, which explicitly enabled thinking. New prompts explicitly select native modes. Four prior reasoning examples are inserted as native thought-channel turns because the standard template drops earlier ordinary reasoning. No compliant reasoning text is supplied in the evaluated turn.

Model google/gemma-4-31b-it, HF tokenizer revision 842da3794eaa0b77d5f08bae87a17459d91ff475, raw completions through pinned Crusoe bf16, no provider fallback. Temperature 1.0, top-p 1.0, seed 13, 8,000-token output cap. Three preflight calls check raw-token accounting and native thinking-on/off observability. The pilot stops for review if those checks fail or the pinned provider is unavailable. API responses, requests without credentials, prompts, tokenized inputs, provider, token accounting, and actual or estimated cost are retained. Generated token IDs are not available from this API.

At the checked provider rates, the 36 output budgets total at most $0.1152 in completion tokens, plus prompt tokens and preflight. Hard pilot spending allowance: $1.00; this is a task-specific cap, not a user-specified overall research budget. No additional GPU pod is needed.

Code is prepared on the box; outputs/gemma4-followup is the cold copy. After the pilot, the one-off queue archives raw attempts and outputs to main under cot-control/results/gemma4-followup-20261030, including failures. No automatic scale-up or causal claim is made. The existing RTX4090 remains running at the user's request.
