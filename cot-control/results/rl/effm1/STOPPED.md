# effm1 stopped by hand at step 34 (final checkpoint: step 30), 2026-10-23

Not by fry_watch (no rule fired) or the plateau rule (difficulty-adjusted reward still improving: -0.186, -0.149, -0.132).
Stopped after the step-30 spot check (spot/step30_review.json) found early textual signs of rising per-token entropy:
- medium: mean sampled-token logprob down on all 8 questions drawn both early and late (e.g. -0.48 -> -0.68); stray
  foreign-script tokens appear since step 11 (Gujarati fragment, 'tokens,acaktır'; base: 0 in 2000 medium rollouts) and
  odd substitutions ('Dear!', 'Kush:', '2 fois', 'Gudges?'); medium CoTs 1.30x base length on the same questions.
- high: glitch / near-misspelling rate ~2x base and rising (53 -> 77 -> 94 per M words vs base 64): 'k+m+n = jquery?',
  'approximate the Arduino', 'simultaneouslyUser'.
- low: healthy. Accuracy fine at all levels; no loops; reasoning recovers after each glitch.
John's instruction for this run was not to fry anything, so step 30 is the final checkpoint. Spent $26.72 of the $40 budget.
