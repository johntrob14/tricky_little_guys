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

Running total (completed items): $29.18
