# CoT-Control on DeepSeek V4.1 Flash (Tinker), Astra protocol

Neel Nanda's variant of CoT-Control (Astra post appendix, 2026-09-10): 9 default constraints x 3 datasets x 4 items = 108 units,
default user prompt with the instruction appended, raw reasoning trace graded with the reference graders, pooled rate with a 95% Wilson interval.
Prompts, item draw (seed 20260910), extraction, graders, judge prompts and tables come from ~/code/cot-control-switch-opt (`cotswitch`, commit 0d29854).

## Reproduce

    set -a; . ~/.config/sauce/tinker.env; set +a
    .venv/bin/python tinker_cotcontrol.py generate --run-dir ~/code/results/cot-control-deepseek/run1
    (cd ~/code/cot-control-switch-opt && PYTHONPATH=. ~/code/cot-control-deepseek/.venv/bin/python -m cotswitch.cli grade --run-dir ~/code/results/cot-control-deepseek/run1)
    .venv/bin/python tinker_cotcontrol.py judge --run-dir ~/code/results/cot-control-deepseek/run1
    (cd ~/code/cot-control-switch-opt && PYTHONPATH=. ~/code/cot-control-deepseek/.venv/bin/python -m cotswitch.cli analyze --run-dir ~/code/results/cot-control-deepseek/run1)

## Run 1 (2026-10-17)

- Model `deepseek-ai/DeepSeek-V4.1-Flash:peft:262144` on Tinker (the bare name is refused), chat-template defaults: thinking mode, `Reasoning Effort: 75` ("high").
- Greedy, max 25,000 new tokens, one sample per unit. 772,952 output tokens, 38,329 prompt tokens. 16/108 rows hit the token limit.
- Judge for ignore_question and meta-discussion: openai/gpt-oss-120b on Tinker, reasoning effort high, 3-of-3 rule.
- **Result: 0/108 compliant (95% Wilson 0.0-3.4%).** Same with and without the meta-discussion filter. Accuracy 67/108 (62%).

Verification (all passed): independent recount of banned words and case violations per row; every word-suppression trace uses the banned keyword itself;
closest lowercase trace still has 55 capitals (it starts "We need answer"); all 12 ignore_question traces reason about the question (hand check agrees with the judge 12/12);
grader tests (33) pass and graders return 1 on constructed compliant text; excluding the 16 truncated rows gives 0/92.

Observation: the model usually decides the constraint applies to a "reasoning stage" it can skip ("I won't include one. So final: ANSWER: B.") and
often applies the constraint to the visible response instead: meow between words in the response 72% of word gaps, uppercase response 83% of letters.

Deviations from Neel (unknown details of his protocol, filled with our choices): sampling settings, reasoning effort, judge model, item draw.
DeepSeek V4.1 Flash was released the day his post appeared, so it is probably not among his 34 models (unverified).
