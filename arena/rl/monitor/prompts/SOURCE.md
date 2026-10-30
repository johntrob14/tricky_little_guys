CoT + Action Monitor - ControlArena (cot.txt)

Source: UKGovernmentBEIS/control-arena commit 20857c9, control_arena/monitor/prompts/cot.txt, copied unchanged apart from trailing spaces. A joint monitor for a software-engineering agent: it sees the main task, the agent's reasoning and the agent's tool calls, and returns one score. cot_prompt_formatter fills {cot} from the content of the most recent assistant message, HTML-escaped and wrapped in cot tags so that adversarial reasoning cannot inject instructions, and fills {tool_calls} from that message's tool calls. It has no history of earlier steps and no side-task list. Score 0 to 10, high is suspicious. The cot_only.txt variant in the same folder drops the tool calls.

## CoT-only monitor - ControlArena (cot_only.txt)

Source: UKGovernmentBEIS/control-arena commit 742e074, control_arena/monitor/prompts/cot_only.txt, copied unchanged apart from trailing spaces. Same as the CoT + action prompt but the monitor sees only the task description and the turn's CoT (no `{tool_calls}`); it says so and asks about the reasoning only. Upstream wording left as is, including "reasoning and actions" in the score scale and the grammar of its last sentence. A turn with an empty CoT has nothing to judge and is skipped, not scored.
