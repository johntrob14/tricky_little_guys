# ws1tutor1 spend ledger (list-price estimates from token counts unless marked billed)

| when (UTC) | item | est. USD | basis |
|---|---|---|---|
| 10-23 04:20 | probe base + ws1 (organism_probe.py, 2 x 74 samples) | 0.6 | ~95k prompt + ~80k sampled tokens each at Qwen prices; judge not run |
| 10-23 05:40 | collection ws1-ctl-easy done (48 episodes, main 47, side_leaked 6) | 11.0 | 48 x $0.23 |

| 10-23 06:35 | collection ws1-ctl-medium-obst done (16 episodes, main 15, side_leaked 0) | 3.7 | 16 x $0.23 |

| 10-23 07:05 | pre-screen ws1 G=8 over 94 easy contexts (752 samples) + monitor | 5.14 | resample_ctx summary policy_usd 5.086 + judge 0.049 |
| 10-23 06:40 | monitor scoring of logged easy + medium turns | 0.12 | rl.monitor spent_usd |

| 10-23 07:20 | pre-screen ws1 G=8 over 32 medium contexts (256 samples) + monitor | 1.93 | resample_ctx summary |
| 10-23 07:25 | tutor check, 40 luna ratings | 0.10 | OpenRouter usage.cost |

| 10-23 07:50 | collection ws1-absent done (24 episodes, main 15) | 5.5 | 24 x $0.23 |
| 10-23 07:55 | base Qwen G=8 on the 11 present training contexts (88 samples) | 0.82 | resample_ctx summary |

| 10-23 08:05 | base Qwen G=8 on the 5 absent training contexts (40 samples) | 0.27 | resample_ctx summary |

| 10-23 11:05 | training ws1tutor1 steps 1-5 (trainer spent_est, Tinker + tutor + judge) | 32.57 | steps.jsonl |
| 10-23 08:30-11:00 | gates 1-5: probes (5 x ~74 samples) + fixed evals (5 x 216 samples) | 5.0 | ~$0.6 + $0.4 per gate, list prices |
| 10-23 11:00 | executed episodes, step-5 checkpoint (16) | 3.7 | 16 x $0.23 |

| 10-23 11:40 | executed episodes, step-2 and step-3 checkpoints (16 each) | 7.4 | 32 x $0.23 |
| 10-23 11:40 | cot-only monitor on the step-2/3/5 episodes (~1900 turns) | 0.3 | rl.monitor spent_usd (est.) |


Running total (completed items): $78.15

Run ended 2026-10-23 11:05 UTC at the step-5 halfway gate; no further spend planned. Tinker list prices from arena/turn1_grpo.py PRICES and resample_ctx.py; episode cost measured on the sniff run.
