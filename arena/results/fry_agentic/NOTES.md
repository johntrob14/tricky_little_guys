# Agentic fry check by frozen turns (`fry_agentic.py`): negative on its validation target

2026-10-26. Goal: a check, cheap enough to run on every training checkpoint, that flags the agent-level breakage of the many-types
CoT-control organism (`cotctl-manytypes1b-step150`) in tiktok (`../manytypes_default/NOTES.md`: main task base 42/48, s100 32/48, s150 27/48;
no-op `str_replace` edits 1/48, 2/48, 11/48 episodes; plain-text `{"answer": ...}` instead of the submit call 2/48, 1/48, 7/48 (recounted from
the eval logs 2026-10-26); field names of the task dropped from the reasoning; less testing). Tool-call syntax is not among them: unparsed calls
base 23/864, s150 7/557.

Method: one sampled turn per context cut out of base gpt-oss-120b control episodes (`logs/ctl-easy-big`, main task only, 12 passing
episodes, 3 per task): 12 `early` cuts (turn 1-2), 19 `edit` cuts (before a turn where base changed a file), 11 `submit` cuts (before base's
submit call). 4 samples per context, T 1, 8192 max tokens, Tinker, cookbook gpt-oss renderer with the date pinned. Scoring in `fry_agentic.score`.

```
.venv/bin/python fry_agentic.py build --family gpt-oss-120b --per-task 3
.venv/bin/python fry_agentic.py run --family gpt-oss-120b --label base --n 4
.venv/bin/python fry_agentic.py run --family gpt-oss-120b --label mt1b-s100 --n 4 --model-path tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step100
.venv/bin/python fry_agentic.py run --family gpt-oss-120b --label mt1b-s150 --n 4 --model-path tinker://f6204b00-be4d-51fd-b1d8-6d467ff4ba75:train:0/sampler_weights/cotctl-manytypes1b-step150
.venv/bin/python fry_agentic.py compare --family gpt-oss-120b base mt1b-s100 mt1b-s150
```

| measure | base | mt1b-s100 | mt1b-s150 |
|---|---|---|---|
| tool call (early + edit cuts) | 116/124 | 120/124 | 121/124 |
| unparsed call | 7/168 | 3/168 | 2/168 |
| wrote a file at an edit cut | 32/76 | 47/76 | 53/76 |
| no-op str_replace | 0/15 | 0/21 | 0/21 |
| str_replace old_str seen in earlier tool output | 15/15 | 21/21 | 19/21 |
| task identifier recall (mean share) | 0.34 | 0.355 | 0.392 |
| submit called at submit cut | 42/44 | 39/44 | 42/44 |
| answer as plain-text JSON | 3/168 | 5/168 | 3/168 |
| empty reasoning | 36/168 | 36/168 | 36/168 |
| truncated / repetitive | 0 / 0 | 0 / 0 | 0 / 0 |
| cost (estimate) / wall time | $0.145 / 14 s | $0.148 / 21 s | $0.150 / 20 s |

Result: in base gpt-oss's contexts, s150 behaves like base on every measure; none of the failures seen in its own episodes appears
(no-op edits 0/21 against 11/48 episodes executed). The breakage is on-policy: it shows only once the organism has written its own earlier
turns. A per-turn check on frozen base contexts does not detect it (the same lesson as the ws1tutor1 frozen-turn loop).

Defects: every `submit` cut follows run.py's NUDGE user message ("Call the submit tool to finish..."), because base ended those episodes with a
text-only turn first, so `submit_called` measures obeying an explicit nudge, not ending on its own; all 36 empty-reasoning rows are those
contexts in all three models (gpt-oss calls submit straight after the nudge). Contexts for qwen3.8-27b (24, from `logs/ctl-qwen-easy`) and
deepseek-v4.1-flash (26, from `logs/ds41-ctl-easy`, an OpenRouter run) are built but not sampled; the Qwen and DeepSeek renderer paths are
untested here.
