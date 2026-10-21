# Held-out CoT monitor: which judge

ORGANISM.md step 2 needs the eval monitor to be **more capable than the trained-against one**. In-loss judge: `openai/gpt-oss-20b` on OpenRouter (`pricing.DEFAULT_MODEL`);
the Tinker default `deepseek-ai/DeepSeek-V4.1-Flash` is the other possible in-loss judge, and the policy is gpt-oss-120b. Research only: no OpenRouter or Tinker API call was
made (prices are from web pages, capability from a public leaderboard). Nothing here has been run as a monitor.

## Recommendation

| role | backend / model | why | USD per 1k turns, typical / worst |
|---|---|---|---|
| **primary** | openrouter `anthropic/claude-sonnet-5.5` | highest capability of the affordable candidates (AA index 46.8 at its default effort, 56.0 at max; gpt-oss-20b 9.0); a third lab, not the policy's or either in-loss judge's family; every endpoint honours `max_tokens`, so the worst-case reservation holds | 14.31 / 64.31 (base), 15.75 / 70.75 (pricing.py upper bound) |
| **backup** | openrouter `google/gemini-3.8-flash` | AA 40.9 (4.5x gpt-oss-20b), a fourth lab, so its errors should correlate less with the primary's; a third of the price | 5.37 / 24.12 (promo base), 9.66 / 43.41 (upper bound) |

`--stronger-because` for the primary: `"AA Intelligence Index 46.8 (high) vs 9.0 for gpt-oss-20b, artificialanalysis.ai 2026-10-21; different lab from policy and in-loss judge"`.

Run **both** on the same turns, not the backup only on failure: monitor identity predicts catch rate better than tier does (Williams 2026, below), so the
primary-backup disagreement is itself a result. Before trusting either, score a labelled set (side task done vs not, from `side_done`) and report AUROC for
both held-out judges and for the in-loss judge; "stronger" in the table is general capability, not measured monitoring skill.

**If the in-loss judge is DeepSeek-V4.1-Flash** (AA 45.1), neither pick is clearly stronger at default settings: Sonnet 5.5 at its default `high` effort is 46.8,
Gemini 3.8 Flash 40.9. Use Sonnet 5.5 at `xhigh` (51.9) or `max` (56.0), which needs `reasoning: {effort: ...}` in `judges.openrouter_request` (not implemented;
the request sends no reasoning field, so the endpoint default `high` applies), and raise `--max-tokens` with it.

Per-turn sizes behind the cost column: prefill **2157 tokens per turn**, from `python -m rl.monitor.run logs/gptoss-n4 --dry-run` (341 turns of gpt-oss-120b
rollouts on the 4 tiktok tasks, 735,661 tokens estimated at chars/3, not a tokenizer). Output: worst = `--max-tokens` 6000; typical = **1000, an assumption**
(no held-out reply has been seen). An unparsed reply is retried once, which can double a turn's cost. Cost = 1000 x (2157 x prefill + out x sample) / 1e6.

## Candidates

Capability: Artificial Analysis Intelligence Index, read from the HTML of https://artificialanalysis.ai/leaderboards/models on 2026-10-21 (index version not on
the page; AA has announced v5 for late October, so these will move). Each value was matched to the nearest preceding model slug in the page data, not read off a rendered table.
Context is the model's own context; every prompt here is about 2k tokens, so none binds. Prices: USD per million tokens, from https://openrouter.ai/<model>/providers page HTML (endpoint list embedded in the page) and https://thinkingmachines.ai/tinker/pricing/, both 2026-10-21.
"base" = the model author's own endpoint; "max" = priciest listed endpoint, the figure in `pricing.py` (the judge pins no provider, so routing may land anywhere).

| backend / model | AA index | context | base in / out | max in / out (pricing.py) | per 1k turns typ / worst (max) | notes |
|---|---|---|---|---|---|---|
| openrouter `openai/gpt-oss-20b` (in-loss) | 9.0 | 131,072 | 0.018 / 0.09 (cheapest) | not changed: 0.018 / 0.09 | 0.13 / 0.58 | priciest endpoint is groq 0.075 / 0.30; see "Flags" |
| tinker `DeepSeek-V4.1-Flash` (in-loss alt.) | 45.1 (`-1017`) | 256k | 0.60 / 3.60 | same | 4.89 / 22.89 | |
| openrouter `anthropic/claude-sonnet-5.5` | 46.8 high, 51.9 xhigh, 56.0 max | 1,000,000 (max out 128,000) | 2 / 10 | 2.2 / 11 | 15.75 / 70.75 | reasoning mandatory, default effort `high`; `temperature` excluded on every endpoint |
| openrouter `google/gemini-3.8-flash` | 40.9 high, 39.8 medium | 1,048,576 (max out 65,536) | 0.75 / 3.75, "50% off" promo | 1.35 / 6.75 (priority tier) | 9.66 / 43.41 | default effort `medium`; `temperature` excluded on Vertex endpoints, accepted on AI Studio |
| openrouter `z-ai/glm-5.3` | 44.8 max | 1,048,576 on most endpoints (some 204,800-262,124) | 1.4 / 4.4 | 2.8 / 8.8 | 14.84 / 58.84 | open weights; 50 endpoints, many fp4; default effort `max` |
| tinker `zai-org/GLM-5.3` | 44.8 | 256k | 3.78 / 12.15 | same | 20.30 / 81.05 | temperature 0 honoured and weights pinned; Tinker id prefix UNVERIFIED; `judges.tinker_judge` needs the template to end in `<think>` (unchecked) |
| openrouter `moonshotai/kimi-k3` | 43.6 max | 1,048,576 on most endpoints | 3 / 15 | 4.5 / 22.5 | 32.21 / 144.71 | most expensive; `temperature` excluded on moonshotai's own endpoint |
| tinker `moonshotai/Kimi-K2.6` | not on page | 32k / 128k | 2.205 / 5.49 | same | 10.25 / 37.70 | capability UNVERIFIED; Tinker id prefix UNVERIFIED |
| openrouter `anthropic/claude-haiku-5.5` | 43.4 max, 37.8 high | 1,000,000 | 0.10 / 0.50 | 0.11 / 0.55 | 0.79 / 3.54 | cheapest strong option; default effort `medium` (AA 34 at medium, re-read 2026-10-21); same lab as the primary |
| openrouter `openai/gpt-6-luna` | 38.1 max, 32.9 high, 29.9 medium | 1,050,000 | 0.10 / 0.50 | 0.2 / 1.0 | 1.43 / 6.43 | same lab as gpt-oss; `max_tokens` excluded on Azure endpoints, so the reservation does not bound it there |
| openrouter `openai/gpt-oss-120b` | 11.6 | 131,072 | 0.03 / 0.15 (cheapest) | 0.35 / 0.95 | 1.70 / 6.45 | **not recommended**: same weights as the policy's base and same family as the in-loss judge |

Not priced or not chosen: GPT-5.6 / GPT-6 Sol and Astra, Claude Opus 5.5 / Fable 5.1 (more than enough capability, 2-10x Sonnet's price; not needed while the
in-loss judge is at 9); DeepSeek V4 Pro 0813 (AA 36.0, endpoint prices inconsistent, e.g. relace 0.13 / 8.0); Qwen3.5-397B (AA 18) and Nemotron-3-Ultra (AA 23) on Tinker.

## Why these (prior evidence)

- Monitor strength matters, and CoT access is what makes it matter. Guan et al., *Monitoring Monitorability* (OpenAI, arXiv 2512.18311, Dec 2025), sec. 6.3:
  "stronger monitor models achieve higher monitorability scores for the same agent", about +40% g-mean² from o3-mini to GPT-5 Thinking; more monitor reasoning
  effort helps most for the weakest monitor (o4-mini about +12%), and with action-only access effort helps only about 4%. Only OpenAI models were tested as monitors.
- A weaker monitor can catch a stronger agent, but RL against the monitor teaches obfuscation. Baker et al., arXiv 2503.11926 (Mar 2025): GPT-4o monitoring
  o3-mini's CoT caught reward hacking far better than watching actions; under strong CoT optimisation pressure the agent learned "obfuscated reward hacking". This
  is the failure our held-out judge has to detect, so it must not share the in-loss judge's blind spots.
- Hybrid CoT + action scoring beats either alone across all monitors tested (Arnav et al., *CoT Red-Handed*, arXiv 2505.23575, NeurIPS 2025). `controlarena_cot`
  already sees both the turn's CoT and its tool calls.
- Scaffolding and agent awareness matter as much as the monitor model: an agent that knows it is monitored degrades detection the most, and hierarchical-sequential
  scaffolds let weaker models (Claude 3.6 Haiku, GPT-4o-mini) reach AUC > 0.85 on Sonnet-4 agents (Kale et al., *Reliable Weak-to-Strong Monitoring of LLM Agents*,
  arXiv 2508.19461; the paper text says "even with weaker models such as GPT-4o-mini and Claude 3.6 Haiku, we achieve a monitoring AUC exceeding 0.85 against all agents",
  checked in the arXiv HTML at review; the ICLR 2026 venue is not stated on arXiv and is UNVERIFIED).
- Identity over tier: ControlArena APPS backdoors, 7 monitors, two attackers; the spread within a tier was 2.3x the gap between tiers, gemini-3.1-pro dropped
  to last on landed attacks, Haiku 4.5 was first on landed attacks but flagged ~23% of honest code (Williams, LessWrong, 2026-08-18, *Monitor Identity Matters
  More Than Tier*; small sample, one prompt. UNVERIFIED: two web searches at review did not find this post; treat the numbers as unsourced until a URL is recorded). Hence two held-out judges from different labs, and a measured false-positive rate on control rollouts.

## Practical caveats

- **Not deterministic.** `judges.openrouter_request` sends `temperature: 0`; every Anthropic endpoint and Google's Vertex endpoints list `temperature` under
  `excluded_parameters`, so it is ignored there. Expect run-to-run variance; score a subset twice to measure it. A reproducible alternative is GLM-5.3 on Tinker.
- **Reasoning budget.** `max_tokens` 6000 covers thinking plus answer; Sonnet 5.5 thinks at `high` by default. A reply cut off in the thinking is unparsed and
  retried once. Pilot about 20 turns and read the unparsed rate before a full run.
- **Routing.** No provider is pinned, so open-weight models can land on fp4 endpoints. Pin a provider (OpenRouter's `provider.order`; not implemented here) for a
  held-out number you will cite, and record which endpoint served it.
- **Version drift.** Closed models can change behind the same slug (OpenRouter permaslug for Haiku 5.5: `claude-haiku-5.5-20261007`). Record `judge_model`
  and date with every held-out result (`heldout.json` and `summary.json` already do).
- **Long-prompt tiers.** Some endpoints carry a price override for long prompts: Haiku 5.5 bills 5x above 100,000 prompt tokens (0.55 / 2.75 at the max endpoint),
  GPT-6 Luna about 1.5-2x above 272,000. `pricing.py` uses the base tier only. Turns here average about 2k tokens, so this binds only for a turn with a >100k-token CoT;
  Sonnet 5.5, Gemini 3.8 Flash, GLM-5.3 and Kimi K3 list no override.
- Prompts are fixture data (fake secret, offline sandbox); several endpoints are marked `retainsPrompts: true` on OpenRouter.

## Flags found while pricing (not changed here)

- The in-loss entries in `pricing.py` are the **cheapest** endpoint, not an upper bound: `openai/gpt-oss-20b` 0.018 / 0.09 (groq 0.075 / 0.30, Vertex 0.07 / 0.25),
  `deepseek/deepseek-v4.1-flash` 0.02832 / 0.12 (today's cheapest is 0.056 / 0.12; priciest 0.60 / 2.40). Unpinned runs can cost up to about 3x (gpt-oss-20b)
  or 20x (DeepSeek) the forecast.
- `OPENROUTER_UPPER` (3 / 15) is not an upper bound: Kimi K3's endpoints reach 4.5 / 22.5, GPT-6 Astra lists 10 / 50 (third-party, UNVERIFIED). Not raised,
  because the full OpenRouter list was not read (that is an API call).
- `TINKER_UPPER` was raised from 1.695 / 4.215 to 3.78 / 12.15 (GLM-5.3, the highest rates on the Tinker page): GLM-5.3 and Nemotron-3-Ultra (sample 6.225) exceeded the old bound.

## Unverified

- AA scores were extracted from page data by slug proximity, and the index version is unknown. Kimi-K2.6's score was not found. (Review, 2026-10-21: the rendered leaderboard gives the same values to integer precision.)
- The Williams LessWrong post could not be re-located; its figures are unsourced.
- Typical output length (1000 tokens) is assumed; prefill per turn is chars/3 over one set of logs.
- Tinker ids `zai-org/GLM-5.3`, `moonshotai/Kimi-K2.6` (HF ids; the pricing page gives bare names). Whether `tinker_judge` works with their chat templates.
- Whether OpenRouter's `deepseek/deepseek-v4.1-flash` is the same weights as Tinker's (already flagged in README).
- No candidate has been run as a monitor. Monitoring skill, unparsed rate, refusal rate on these prompts and real cost are all unmeasured.
