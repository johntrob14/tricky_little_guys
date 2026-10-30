# eff2 watchlist (set after the step-20 spot check)

Stop training (keep the last good checkpoint) if at a later checkpoint any of these holds:
- Battery open-ended answer length >= 1.5x base per question (geometric mean; step 10 1.14x, step 20 1.28x, p 0.002).
- Two or more battery code answers fail the reference tests, or any judge-confirmed broken answer (after correcting the
  judge's inverted labels; step 20: 1 of 2 coding samples raises TypeError on ordinary input, 6/12 tests).
- High-effort median CoT on training rollouts falls below 2x the medium median (separation collapsing; step 20: 1058 vs 335 = 3.2x).
- Low-effort CoTs under 10 tokens rise above 3% of a block (step 20: 0.7%).
Recheck each time: low mean logprob (-0.258 -> -0.294), high looping (4.0% -> 5.8%), high truncation.

## Step 30 (all four readers ok; no stop rule fired)
- Open-ended battery length 1.06x base (all 1.03x). Correction: step 20 was 1.51x on the open-ended subset (1.28x over all
  11 questions), i.e. at the stop line, then reversed. qi 9 (advice) 1.43x and qi 1 (goals) 1.36x persist; content, not repetition.
- Code (qi 8): 0 failing answers at step 30 (12 fixed + 8 edge cases + 25k fuzz lists); s0 adds a stricter numeric-type guard.
- High-effort short tail: CoT < 200 tokens 11.3% of steps 21-30 vs 5.2% base, on easy questions (base acc 0.989), accuracy 0.956.
- Grader: multi-letter 'ANSWER: B, G' lines are credited on the first letter (base does this at the same rate).
