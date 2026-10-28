# What makes a model able to control its CoT? Overnight plan (started 2026-10-26 00:45 CDT, ends 10:00 CDT)

Budget: Tinker $50, OpenRouter $50 asked; the OpenRouter key has $11.36 left on its $250 limit, so OpenRouter spend is capped at ~$11.
Prices: gpt-oss-120b on Tinker $0.33/M prefill, $0.84/M sampled, $0.737/M trained (modelio.PRICES).
Method (CHIVE-style): causal claims come from counterfactual prompt/prefill edits re-run on-policy, each against an unrelated edit
of equal size; correlational claims (abilities across models) are labelled as such.

## Hypotheses and the experiment that tests each

H1 Initiation. Control fails because the trace starts in the default register (preamble) and the 100% grader is lost at token one;
   models can maintain a style once in it. Test E1: prefill the analysis channel with one sentence already in the required style vs
   the same sentence in normal style; grade only the model's continuation. H1 predicts a large rise for format types (case, meow,
   end-of-sentence) and none for suppression. Alternative (capacity): small rise; continuation drifts back within a few sentences.
H2 Channel reference. Models do not apply instructions to the reasoning channel because "reasoning stage"/"analysis channel" does not
   bind to their own thinking (the displacement-to-answer pattern in Qwen 3.5/Kimi). Test E2: add Neel's clarification line vs an
   equal-length irrelevant line. H2 predicts a rise (and less displacement); alternative: no change for gpt-oss (it rarely displaces).
H3 Recency. A one-line reminder after the question (Neel's best single lever on GLM-5.2). Test E3 vs an equal-length generic reminder.
H4 Load. Control decays with reasoning length/effort; low effort raised gpt-oss control 3.5% -> 15%. Test E4: "keep your reasoning
   under 150 words" vs "keep your final answer under 150 words" (same length, different target). H4 predicts a rise caused by length.
H5 Related abilities (correlational). Across Neel's 34 models and John's 17 checkpoints: does CoT control track output control, answer
   displacement, accuracy, no-CoT accuracy (NCRI), meta-discussion rate, trace length? Free (existing data).
H6 Training pressure: CoT language control. Labs reward CoT language consistency (e.g. think in the user's language). Hypothesis: RL
   that only rewards "write your reasoning in language X" (programmatic grader) teaches general reasoning-channel instruction following
   and transfers to the 9 CoT-Control types. Test E6: GRPO on gpt-oss-120b with a language-of-thought reward only; eval on the 540
   items. Alternative: no transfer (like third_person-only st1-third: 9.9%, and output-only rgrpo1: none).

Most probability should stay on "something else": keep reading traces from each arm.

## Results (filled in as they come)

### Judge comparison(labels/judges_eval.py, 500 rows = rounds 1-3, reference = Astra+Claude consensus)
"hybrid" = v8 frame (grader decides complied; reasoning fails + answer passes grader -> displaced; judge labels the rest).
| judge | hybrid R1 / R2 / R3 9-way | hybrid 5-group | meta acc | answer-applies acc | failed rows | $ / 500 rows |
| deepseek-v4.1-flash | 80.1 / 83.6 / 85.0 | 87.0 | 98.8 | 94.5 | 0 | 0.47 |
| xiaomi mimo-v2.6-flash | 78.2 / 81.1 / 84.2 | 87.4 | 99.0 | 96.0 | 16 (empty replies) | 0.19 |
| claude-haiku-5.5 | 86.4 / 81.3 / 78.9 | 86.9 | 98.7 | 95.7 | 27 (content_filter) | 0.29 |
| gpt-6-luna-pro | 81.4 / 81.2 / 80.0 | 85.2 | 99.0 | 88.8 | 0 | 0.58 |
| gpt-6-luna | 78.9 / 78.2 / 78.3 | 83.2 | 98.2 | 88.0 | 0 | 0.24 |
| Jev 1.13 (v8) | 72.0 / 72.1 / 70.0 | 78.5 | 96.4 | 84.8 | 0 | 0.05 |
| gpt-oss-120b | 60.9 / 57.6 / 65.0 | 79.0 | 83.0 | 64.9 | 0 | 0.10 |
| glm-5.3-flash | 63.4 / 63.0 / 71.7 | 75.9 | 90.6 | 77.9 | 0 | 0.10 |
| gemini-3.1-flash-lite | 56.5 / 59.4 / 66.7 | 78.0 | 81.6 | 73.6 | 0 | 0.45 |
Untuned prompts (RUBRIC.md verbatim), reasoning effort medium. Consensus rows are the easy rows, so these exceed Astra-Claude agreement.
Full-scale cost (34k rows, ~70x): deepseek ~$32, haiku ~$20, mimo ~$13, Jev ~$4.

### H5 correlations (free, existing data)
Neel's 34 models (Spearman): CoT pass vs answer control (answer passes grader | has text) -0.14; answer control median 80%.
Displacement vs writing an explanation in the answer +0.87; attempting in the reasoning vs writing answer text -0.67 (channel choice).
Meta-discussion vs attempt +0.64, vs NCRI +0.50, vs accuracy +0.30.
John's 17 checkpoints: CoT control vs output control (CoT off) +0.80, vs CoT accuracy -0.89, vs meta -0.92, vs no-CoT accuracy +0.29.
No-CoT accuracy stays at 31-34% across checkpoints while CoT control ranges 0.6-76%: within one model, control is trainable with the
NCRI proxy unchanged.

### E1-E4 counterfactual arms on base gpt-oss-120b(outctl/cf/, cf_analyze.py; 540 items x 2 samples per arm, T=1, medium)
Paired strict pass on the 480 rule-graded items (95% item-bootstrap CI):
- H1 prefill one compliant sentence vs neutral sentence (continuation graded): 4.9% vs 2.0%, +2.9 [+1.6, +4.4]. Format types stay 0-4%;
  the continuation reverts at once (share of text before first violation 3.5% vs 0.1%). repeat +15.8 (ends with string), ignore_question
  78% vs 3% (an off-topic first sentence redirects the content; passes read genuine: reefs, soup, Python). H1 (initiation) rejected for format types.
- H2 channel clarification vs same-length filler: 4.6% vs 3.1%, +1.5 [+0.2, +2.7]; ignore_question +29 [+17, +42]; format types ~0.
- H3 type reminder last vs generic reminder: 5.3% vs 3.9%, +1.5 [+0.0, +2.8].
- H4 'analysis channel under 150 words' vs 'final answer under 150 words': 10.0% vs 4.3%, +5.7 [+3.8, +7.9] (CoT 279 vs 454 words):
  repeat 10 -> 40, word_suppression 18 -> 30, ignore 26 -> 72. Format types still 0-3%.
Reading: for base gpt-oss the format types (case, alternating, meow, end-of-sentence) are pinned near 0 under every prompt lever; the
content/addition types respond to length/salience. Two different things are being measured under one 'controllability' number.
Ignoring the question costs accuracy down to the no-CoT level: ignore passes 36.2% correct vs 53.8% for ignore fails, 49% plain CoT,
32.6% no-CoT (expl arm). Compliance on content instructions costs exactly the NCRI gap -> H9: NCRI should matter for content types.
Neel's 34 (exploratory, 6 tests): NCRI vs content-type 'complied or partial' rho +0.34 (p=0.046); format-type -0.03; strict ~0.
E6 lang1: base gpt-oss writes its CoT in the requested language 3.5% strict (9.4% of sentences) at step 1: language-of-thought control is
as low as CoT-Control in the base model.

### H1b/H8 long own-text prefill(cf_long.py; format items x 2 samples; continuation graded)
Prefill ~80 words of the base model's OWN reasoning for the item, restyled (lowercase / uppercase / alternating / meow / end-with-safe),
vs the same words unchanged. Same with ~40 words of its own CoT-off answer in the answer channel.
- Reasoning channel: continuation passes 0.6% vs 0.0% (n=269 items); breaks within the first 3% of the continuation.
- Answer channel: 35.4% vs 30.4% (+4.9 [+1.7, +8.0]); lowercase 59% vs 44%, end-of-sentence 73% vs 68%, alternating/meow ~1-3%.
- Not a grader artifact: share of words violating lowercase in the reasoning continuation 0.23 (half acronym-like) vs 0.02 in the answer
  channel; uppercase 0.88 vs 0.14. In the reasoning the model writes ordinary English capitalisation regardless.
H9 (NCRI -> compliance within model): per-question no-CoT accuracy (nocon arm, 4 samples) does not predict compliance: ignore_question
  +11 / +3 / -9 points (answerable vs not) across plain / short / prefill arms, all CIs span 0; suppression negative. Not supported.
E6 control: base writes its FINAL-ANSWER explanation in the requested language 93% strict at step 1, vs 3.5% for its REASONING.
Interpretation so far: base gpt-oss-120b's reasoning channel ignores instructions AND in-context style; its answer channel follows both.
Controllability (format types) is a property of the channel, not of the ability to produce the format.

### Style persistence across checkpoints(restyled own-text prefill, format types, continuation pass)
| model | restyled | unchanged | plain format pass | plain CoT control |
| gpt-oss-120b | 0.6% | 0.0% | 0.3% | 3.5% |
| manytypes1b s150 | 45.0% | 7.1% | 65.0% | 76.0% |
| grpo3 s100 | 38.8% | 16.2% | 55.3% | 66.9% |
| gsft1 ep2 | 15.8% | 0.4% | 41.6% | 47.3% |
| len2 s60 | 2.2% | 0.4% | 0.7% | 12.3% |
| st1-third s60 | 1.5% | 0.4% | 2.1% | 9.9% |
| rgrpo1 s10 | 0.2% | 0.0% | 0.3% | 4.3% |
| eff1 s40 | 0.9% | 0.2% | 0.2% | 5.6% |
Spearman 0.98 (n=8). Close to tautological (same skill with help), but shows: only CoT-control training makes the reasoning channel
style-steerable at all, by instruction or by context. Unchanged base-style prefill drags manytypes from 76% (plain) to 7%: the trained
reasoning channel now follows whatever style is on the page.

### E6 language-of-thought RL transfer (lang1 = GRPO only on 'write your <term> in French/Spanish/German', no judges)
Training curve: strict language control 3.5% (step 1) -> 34% (step 10) -> 84% (step 33); accuracy 0.85 -> 0.77.
lang1 step 10 on the 540 CoT-Control items (1 sample) vs base: 8.8% vs 3.5%, +5.2 [+3.2, +7.4]; ignore_question 80% vs 28%,
repeat +18, uppercase +14 (genuine uppercase reasoning), lowercase 0; accuracy 51.7 vs 52.3. CoTs stay English (528/540): no language drift.
Ignore passes are genuine (weekend plans, banana bread) with meta-discussion. Matched control (lang-ans1: same reward on the final answer's
language) and later checkpoints pending.

### Question effect(replaces the unfinished sol-question)
Within instruction type, 32-50% of the variance of per-item pass rates is between-question (excess over binomial), and it is stable:
sample-1 vs other-sample per-item agreement Spearman 0.49-0.61 (base, gsft1, grpo3, st1-ignore2, manytypes). Item difficulty is shared
across trained models for lowercase/uppercase/meow/multiple-word (mean pairwise rho ~0.4), not for the rest. Symbol/digit density of the
question predicts failure for uppercase (rho -0.45 / -0.47 in grpo3 / manytypes) and meow (-0.21 / -0.39): math-heavy questions put units,
variables and equations into the reasoning. With 36 questions (Neel) the share of math-heavy items moves a model's score.

### E6 result: language-of-thought RL transfers to CoT control; answer-language control does not
540 items x 1 sample, strict pass on the 8 rule-graded types, paired vs base (95% item bootstrap):
| model | 8 types | ignore | repeat | upper | lower | meow | EOS | alt | word | multi | acc |
| lang1-s10 | 8.8% (+5.2 [+3.2, +7.4]) | 80 | 22 | 15 | 0 | 3 | 3 | 2 | 20 | 5 | 51.7 |
| lang1-s20 | 19.2% (+15.6 [+12.4, +19.0]) | 90 | 50 | 22 | 5 | 7 | 22 | 0 | 38 | 10 | 49.3 |
| lang1-s30 | 28.1% (+24.6 [+21.0, +28.3]) | 73 | 55 | 27 | 20 | 8 | 27 | 2 | 63 | 23 | 47.6 |
| lang-ans1-s20 (control) | 3.3% (-0.2 [-1.4, +1.1]) | 32 | 3 | 0 | 0 | 2 | 2 | 0 | 18 | 2 | 52.8 |
| base | 3.5% | 28 | 4 | 1 | 0 | 0 | 0 | 1 | 20 | 3 | 52.4 |
Checks on lang1-s30: CoTs stay English (529/540); median CoT 121 words vs 276 base (control 262), but the gain holds within length bins
(0-150 words: 42.4% vs 7.6%; 150-300: 18.9 vs 2.9; 300-600: 7.3 vs 1.3; 600+: 6.7 vs 2.0); passes are real reasoning (3/135 under 20 words),
53% correct vs 68% for base passes. The control run (same reward on the final answer's language, 26 steps, $7.82) starts at 93% and
changes nothing. Cost of lang1 to step 33: ~$8.4.
Dissociation: lang1-s30 does not gain in-context style persistence (restyled own-text prefill 1.1% vs 0.4%; manytypes 45%): it learned to
bind instructions to the reasoning channel, not to continue a style already present. CoT-control training gives both.

### Robustness of the language transfer and Qwen size check
- lang1-s40 (2 samples/item): 26.5% vs base 3.5%, +22.9 [+19.7, +26.4]; accuracy 50.0 vs 52.3. Output arm (CoT off) +4.9 [+2.6, +7.2]:
  the change is mostly in the reasoning channel.
- Held-out wording (frame F5 'Note: ... internal reasoning ... Comply silently', never in lang1 training): base 1.7%, lang1-s30 9.4%
  (+7.7 [+5.2, +10.2]), manytypes 67.3% (vs 76.0% reference). About two thirds of lang1's reference-wording gain is tied to the shared
  'Requirement: ... analysis channel' template it saw in 30% of training prompts; a real, smaller gain remains. ignore_question F5:
  base 0%, lang1-s30 52%, manytypes 100%. (F5 suppression text omits the synonym list; grading still counts synonyms, same for all models.)
- Qwen 3.5 4B / 9B on the 108 Astra items (Tinker, thinking on, 1 sample): CoT control 0.0% / 0.0% (median CoT ~4.5k words, 23 / 31
  cap-outs); output control (CoT off) 15.6% / 25.0%; no-CoT accuracy 30.6% / 31.5%. Floor: no size effect visible on CoT control.

### Template-free language run and cross-model channel test
- lang2 = lang1 without the reference F0 frame (--p-ref 0; training frames T1-T8 only), stopped by budget at step 18 ($5.68); language
  control 2% -> 77%. On the 540 items (1 sample): reference wording 12.7% vs base 3.5%, +9.2 [+6.6, +12.0] (lang1-s20 was +15.6);
  held-out F5 wording 3.5% vs 1.7%, +1.9 [+0.2, +3.8]. ignore_question 82% vs 28% (reference), 23% vs 0% (F5). Accuracy unchanged.
  Transfer exists without template exposure but is weaker and wording-dependent.
- Cross-model (OpenRouter, reasoning on, T=0, 16 GPQA/MMLU-Pro questions, 12 of Neel's models, $2.56, 382/384 rows): reasoning written in
  French 7/190 (DeepSeek V3.2-exp 5/14, V4-Flash 2/16, every other model 0/16) vs final-answer explanation in French 157/192. Channel
  asymmetry holds in every model. Across the 12 models, French share of reasoning vs Neel's CoT-Control score Spearman +0.48 (p=0.11),
  vs NCRI -0.38 (p=0.23), vs meta -0.16. Answer-in-French vs CoT-Control +0.19. Detector spot-checked on 8 seeded traces.
Spend tonight: Tinker ~$49 (sampling ~$25, RL lang1 $10.38, lang-ans1 $7.82, lang2 $5.68, evals); OpenRouter ~$4 (judges + cross-model).

### Is language-trained control covert?(labels/meta_lang.py, Jev v5 meta question, 300 seeded rows per model, $0.11)
meta-discussion overall / among strict passes: base 25.0% / 45.0% (n=20 passes); lang1-s40 2.7% / 1.9% (103); manytypes 0.7% / 0.4% (236);
grpo3 3.0% / 2.5% (198). lang1 was trained with no meta penalty (no judges) and still stopped mentioning the rule. 5 seeded lang1 passes read:
none mention it.

## Audit corrections (independent Codex audit, notes/audit.md; these supersede the statements above where they conflict)
- Language RL reward = (partial + strict bonus) x correctness multiplier (0.5 if wrong) - length penalty, format-gated; no CoT-Control types,
  no judges. Not "language only".
- lang-ans1 is not prompt-matched to lang1 (R frame, user turn only, p_ref 0, vs lang1 F0/T1-T8, p_ref 0.3, p_system 0.3) and starts near
  ceiling. The transfer difference is real; attributing it to the rewarded channel is confounded.
- One-sentence prefill: strict continuation 2.9% vs 2.0% (+0.9); 4.9% vs 2.0% uses the custom repeat metric (continuation ends with the string).
- "No-CoT accuracy unchanged 31-34% across checkpoints" uses the constrained expl arm. Unconstrained nocon: base 38.7%, gsft1 32.4%
  (-6.3 [-11.7, -1.1]); only 8 of 17 checkpoints have nocon.
- Held-out F5 wording shrinks lang1's gain (+7.7 vs +24.6) but F5 changes frame, term, core wording and synonym lists at once, so "two
  thirds due to template" is not identified. Regrading F5 without synonyms: base 2.3%, lang1 11.0%, manytypes 71.5%.
- Cross-model language: answer prompt asks for 2-3 sentences (min 1 labelled sentence) vs whole reasoning (min 3); 35 answer non-passes are
  unscored because the regex drops text after 'ANSWER:'. Step-1 gpt-oss with a 1-sentence minimum for both: 5.9% vs 94.1%; on 11 shared
  question-language pairs 9/88 vs 87/88. Asymmetry survives; channel-only attribution not isolated.
- Persistence: manytypes plain vs prefilled on the same 269 items is 62.3% -> 7.1% (not 76 -> 7). 'Only CoT-control training' -> 'only the
  CoT-control-trained checkpoints among those tested'. The prompt still states the rule in both prefill arms.
- H9 middle value is +5.9 (not +3); single-word suppression negative, multiple-word small positive.
- Judges: on the 352 consensus rows answered by all nine judges DeepSeek V4.1 Flash and Claude Haiku tie at 87.8% (5-group), MiMo 87.2%.
- Meta: lang1 'covert' -> 'rarely mentions the rule' (2.7% / 1.9% of passes); judge hand-check 29/30.
- Qwen 3.5 'no-CoT accuracy' is from the constrained explanation arm.

### Does "respond in French" make gpt-oss reason in French? (outctl/respond_lang.py; base, 100 seeded questions, medium, T=1, ~$0.83)
Reasoning mostly French / answer mostly French, of 100: plain 0/0; "Respond in French." 0/0 (answer is a bare 'ANSWER: X' because the format
line stays); "Please answer in French and explain your answer." 0/99; system "Always respond in French." 0/1; without the ANSWER-format
line: "Respond in French." 0/94, system message 0/91. The reasoning stays English in all 600 traces while the answer is French.

### Same test on GLM-5.3 (OpenRouter, reasoning on, default effort, 6k token cap, 30 of the same questions, $1.17; work/lang_cross/respond_glm.py, full traces kept)
Reasoning mostly French (of 30) / answer mostly French among calls that finished: plain 0 / 0 of 12; "Respond in French." (no format line)
2 / 10 of 10; system "Always respond in French." 0 / 12 of 13; "answer in French and explain" 0 / 12 of 12. 17-20 of 30 calls per
condition hit the 6k cap (GLM thinks long), so answer counts use finished calls only. Of the two French-reasoning traces, one switches to
French from its first sentence ("Let me analyser ce problème de chimie organique...") and one is English that drafts the French answer
inside the reasoning. Mean French share of reasoning 0.00 / 0.08 / 0.02 / 0.03.

### Respond-in-French -> reason-in-French RL (fr runs; $20 budget; no prefills)
- Pilot without prefills (fr_pilot.py, 50 q x 4): whole reasoning French 0/800 across prompt-only variants; 'any French sentence' 43/200 under
  'Respond in French' is mostly late drafts of the French answer. (A prefill curriculum run, fr1, crashed at step 1 on a grader bug and was
  abandoned: no prefills.)
- Step-1 rollouts, frame R ('Requirement: Respond in French: explain your answer in French before the ANSWER line', no mention of reasoning):
  fr-shaped (front-half French share, 32/group) 0/256, 0 groups with variance -> stopped at step 1 ($0.37); fr-mix (80% explicit 'respond in
  French, and write your <term> in French too') explicit 0/192, implicit 0/64, 0 groups with variance -> stopped ($0.34); fr-hot (T 1.3) 0/256
  strict, 3/32 groups with variance (continuing); fr-from-lang1 (init lang1 step-40 state) 213/256 = 83% strict at step 1 (continuing).
- Held-out check of lang1-s40 itself (respond_lang.py, same 100 questions, no format line): reasoning mostly French 93/100 under 'Respond in
  French.', 99/100 under system 'Always respond in French.' (base 0/100, 0/100); answers French 93 and 89. Training on explicit
  reasoning-language instructions generalized to reasoning in the requested response language.
- fr-hot (T 1.3) stopped at step 7 ($2.66): strict 0 at every step, partial 0.001-0.008; its highest-partial rollouts are English reasoning that
  drafts the French answer ("Let's craft French explanation: ..."). fr-from-lang1 stopped at step 3 ($0.86) at 81-83%: no headroom.

## 10-26 Question-language pilot (version B base rate)
Does base gpt-oss-120b reason in the language the question is written in? Same 100 seeded questions as respond_lang, translated to French
and Spanish by anthropic/claude-sonnet-5.5 on OpenRouter (T=0, $0.81). Two conditions: translated question + translated ANSWER-format line,
and translated question only. 2 samples each, medium effort, T=1; 800 generations, ~$1.22 Tinker.
Code/data: tlg-arena-rl cot-control/results/outctl/qlang/{translate.py,pilot.py,qs_*.jsonl,pilot_base.jsonl}.
- Reasoning in the question's language: 0/800. 15/800 traces have a plurality of French/Spanish sentences by lang_profile, but all 15 open
  in English and the target-language sentences are quotations of the question/options (shares 0.47-0.70, none >=95%).
- Answer language: with the format line the response is just "ANSWER: X" (0 in target language); without it the answer is in the question's
  language 158/200 (French) and 192/200 (Spanish).
- Accuracy unchanged by translation: 0.51 (Fr) / 0.54 (Es) with format line vs 0.53 English on the same questions (n=100 English, 200 each).
- Conclusion: the question language moves the answer channel but never the reasoning channel. Version B RL has a zero base rate from base,
  like "respond in French" did. Same channel asymmetry as before.

## 10-26 Reasoning-language base rates: framing pilots on gpt-oss-120b and a six-model screen (qlang/)
- lang1 checkpoints (step0.py, 50 q x 2, Fr/Es, training format): implicit strict rates s10 0/400; s20 respond 31-46, question 6-25 of 100;
  s40 respond 62-65, question 38-59. Accuracy 0.39-0.51 vs base 0.45 on the same questions.
- Base gpt-oss-120b, 'Reasoning language: X' (developer / harmony system block / user message; English or translated question): 1/1300
  strict. 'Analysis language: X' (developer / system block) with the question in French/Spanish: 0/400; never mentioned in the reasoning.
- Six cheaper Tinker models (screen.py; Qwen3-8B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.6-35B-A3B, Nemotron-3.5-Lightning, gpt-oss-20b; 50 q x 2,
  Fr/Es, cues explicit 'Write your reasoning ... entirely in X' / translated question / 'Respond in X' / plain; T=1, 8000-token cap):
  strict 2/4800 (both Qwen3-8B explicit French); 'mostly target' hits read are quotations of the question. All models reason in English
  under every cue. Many cap-outs (Qwen3.5 median 8000 tokens), so accuracy there is depressed. Cost ~$23.6, far above the ~$5 I estimated
  because Qwen and Nemotron think to the cap.

## 10-26 langsft1: light SFT of gpt-oss-120b to reason in the question's language (rl/sft_lang.py; qlang/build_sft.py)
Data: 315 examples. Base gpt-oss's own correct English traces on pool_k4 MMLU-Pro questions (disjoint from the 540 eval items), question and
trace translated into French/Spanish/German by Gemini 3.8 Flash (Claude Sonnet 5.5 refused: OpenRouter content filter on translating model
reasoning), filtered on length ratio 0.75-1.5 and target share >= 0.8; 75 English rows kept untranslated. No language instruction anywhere.
LoRA 32, lr 1e-4, batch 16, one epoch = 20 steps, $0.16. Checkpoints at steps 4/8/12/20 (tinker://1d6b00f5-768d-5ec6-8118-3489048bd461:train:0/...).
Eval (qlang/screen.py, 50 held-out seeded questions x 2, Fr/Es + held-out Italian, ~$10.7 incl. base). Strict (>=95% target) of 100, Fr/Es/It:
| ckpt | question in X | 'Respond in X' | explicit 'write your reasoning in X' | plain English | accuracy question (Fr/Es/It) |
| base | 0/0/0 | 0/0/0 | 0/0/0 | 0 | .48/.43/.45 |
| s4 | 5/26/5 | 0/1/0 | 2/11/3 | 0 | .43/.46/.40 (192/1200 cap-outs) |
| s8 | 40/46/31 | 11/34/9 | 33/51/28 | 0 | .35/.33/.32 |
| s12 | 59/61/50 | 39/60/27 | 52/69/37 | 0 | .41/.39/.31 |
| s20 | 88/76/46 | 74/79/41 | 75/80/55 | 0-2 | .42/.41/.40 |
English questions stay English at every checkpoint. Training only on the question-language cue also produced reasoning in X under 'Respond in X'
and the explicit instruction, and in held-out Italian. Accuracy on non-English cues drops about 5-12 points (n=100 per cell, SE ~5).

### langsft1 on Astra 108 (astra_more.py, cot side, medium, 4 samples/item, same as base)
Strict: base 4.9% (14/108 items with any pass), langsft1-s8 10.6% (31/108), s20 9.0% (25/108); lang1 s10/s20/s30/s40 15.7/24.1/32.4/31.0%
(1-2 samples). Paired per-item diff vs base: s8 +5.8 [+3.0, +8.8], s20 +4.2 [+1.6, +6.9] (bootstrap over items). Gains mostly ignore_question
(25% -> 42/48%) and repeat_sentences/word_suppression at s8. Accuracy .53 -> .48/.49. Median CoT words 260 -> 195/181 (length confound not checked).
Astra prompts are English; reasoning stays English (409/432), 11 traces at s20 in French/Spanish/Italian.

### English-only SFT control (langsft1-en: same 315 kept examples, same order and settings, no translation; $0.14)
Astra 108 x4: en-s8 4.2% (-0.7 [-3.0, +1.6] vs base), en-s20 3.7% (-1.2 [-3.2, +0.9]); excl. ignore_question 1.3/1.8% (base 2.3%);
accuracy .53/.51; median CoT words 265/239 (base 260). Translated minus English control: s8 +6.5 [+3.7, +9.5], s20 +5.3 [+2.5, +8.6].
So SFT on the model's own short correct traces does nothing; the gain needs the translated (non-English) traces. Translated training
completions were longer in tokens (mean 403 vs 339), yet the translated-SFT model's English CoTs got shorter (195/181 words).
Remaining confound: Gemini's rewrite may change style beyond language (test: Gemini English paraphrase control).

### Back-translation control (langsft1-back: the langsft1 Fr/Es/De traces translated back to English by the same Gemini prompt; same 315 examples,
order, settings; $0.53 OpenRouter + $0.14 Tinker). An English-paraphrase control (build_para.py) was abandoned: Gemini compressed it (median ratio 0.71).
Back-translations: median length ratio 1.11 vs original; first-step NLL 1.06 -> 0.70 at the end (langsft1 0.95 -> 0.78; English control flat ~0.45).
Astra 108 x4 strict: back-s8 9.7% (+4.9 [+1.9, +8.1] vs base), back-s20 13.2% (+8.3 [+4.4, +12.7]); translated minus back: s8 +0.9 [-2.8, +4.4],
s20 -4.2 [-9.3, +0.5]. CONTRADICTS "the language drives it": English text that went through the translation pipeline gives the same or a larger
gain. Profiles differ: back's gain is mostly repeat_sentences (8% -> 44/62%; passes read genuine), translated's mostly ignore_question (25 -> 42/48%).
Back CoTs are not shorter (median 290/285 words vs base 260). Training on the model's own unaltered traces (English control) does nothing.
Current reading: SFT on lightly rewritten (off-policy) versions of its own traces raises CoT control; language is not needed. Mechanism unknown.

### Mechanism tests (same 315 examples, English question, same SFT settings; Astra 108 x4; qlang/build_mech.py)
| SFT data | train NLL | s8 vs base | s20 vs base |
| own traces (en) | ~0.45 flat | -0.7 [-3.0, +1.6] | -1.2 [-3.2, +0.9] |
| endings-only word swaps (end) | ~0.45 | +2.1 [-0.7, +5.3] | -0.7 [-2.8, +1.4] |
| whole-trace word swaps, p=.5 (noise) | ~0.5 | +1.6 [-1.4, +4.9] | +0.5 [-1.9, +3.2] |
| DeepSeek V4.1 Flash's own traces (ds, 313) | 1.4 -> 0.9 | +0.0 [-2.8, +3.0] | -2.5 [-5.3, +0.0] |
| Gemini back-translation (back) | 1.06 -> 0.70 | +4.9 [+1.9, +8.1] | +8.3 [+4.4, +12.7] |
| Gemini translation Fr/Es/De (langsft1) | 0.95 -> 0.78 | +5.8 [+3.0, +8.8] | +4.2 [+1.6, +6.9] |
Neither small mechanical shifts nor a large shift (another model's traces) reproduce the gain; only Gemini-processed traces do. So it is not
"any distribution shift"; something specific to the Gemini rewrite (unidentified). The back gain sits mostly in repeat_sentences (8 -> 44/62%),
the translated gain in ignore_question. Line closed per John (SFT-only effects are not the target; RL-time dynamics are). Cost ~$3.5 Tinker.

## 10-27 outrl1: output-only instruction-following RL with effort levels (rl/grpo_out.py) -- does RL that never grades the CoT raise CoT control?
gpt-oss-120b from base, GRPO, 32 prompts x 8, lr 4e-5, LoRA 32, 53 steps (stopped at the $20 budget, $20.19). Prompt: pool_k4 MMLU-Pro question + two
constraints on the final explanation only (avoid word, lowercase, uppercase, end word, bookend sentence, insert word, max words, language, bullets);
effort sampled low/medium/high per prompt. Reward = correct + mean partial + mean strict; 0 if truncated/unparsed. No CoT term, no judges, no length pen.
Training: both-constraints-met 0.62 (steps 1-9) -> 0.68 -> 0.71 -> ~0.77-0.81 (steps 36-50); accuracy vs base -0.02 to -0.04; high-effort median CoT
words ~1000 early -> ~500-600 late (truncation = 0 reward).
Astra 108 x2 (medium) vs base 4.9% (x4): s10 5.1 (+0.2 [-2.8, +3.5]), s20 3.7, s30 6.5 (+1.6 [-1.4, +4.6]), s40 3.7, s50 4.2, s53 5.1;
pooled s30-s53 +0.0 [-2.0, +2.0] (n=864). Astra accuracy 0.51-0.54 (base 0.53). NEGATIVE: output instruction-following RL with effort levels
does not change CoT controllability on gpt-oss-120b in 53 steps. Consistent with lang-ans1 (5.6%) and feedback-spillover (no CoT movement).
Checkpoints: tinker://894335dc-800c-5adb-9ebd-3bb6f6839960:train:0/sampler_weights/outctl-outrl1-step{10,20,30,40,50,53}.

### outrl1 on all 540 items (432 newer items x4 added for s30, s53; astra540.py = paired strict + continuous vs base, bootstrap over items)
s30: strict -0.5 [-1.7, +0.8], continuous +3.0 [+1.5, +4.4]; s53: strict -1.1 [-2.4, +0.3], continuous +3.0 [+1.4, +4.7]; pooled s30+s53 strict
-0.8 [-1.9, +0.4], continuous +3.0 [+1.6, +4.4]. The continuous gain sits in the CoT constraints that mirror trained OUTPUT constraints:
meow_between_words (+11/+14 on the 432; +16-24 on the 108; CoTs genuinely interleave 'meow' from the first token but drift, so strict fails)
and end_of_sentence (+13/+15; +15-19 on the 108; the CoT discusses the rule and partly follows it, so meta + strict fail). word_suppression
gets worse on the 432 (strict -8/-11). Other runs, 540 items: lang-ans1-s20 +0.3 [-1.5, +2.1]; efflen1 -3.6 [-5.2, -2.1]; eff1 +2.4 [+1.1, +3.7];
eff2 -1.7; effm1 -2.0; effm2 -0.5 (strict, pts). Reading: output instruction RL produces partial, type-specific attempts at CoT compliance
(spillover of trained styles) but no strict CoT control; strict effect bounded to about +-2 pts.

## 10-27 Re-check of earlier output-only runs with the CoT-instruction test (the test that found outrl1's effect)
- lang-ans1 (answer-language GRPO): CoT-Control per constraint, no change (all CIs cross 0). Its own training CoTs never drift into the answer
  language (mean share 0.03-0.07 over steps). Explicitly asked "Write your analysis channel entirely in X" (qlang/explicit_lang.py, 50 q x 2,
  Fr/Es): base 8/200 strict, lang-ans1-s20 6/200. No transfer.
- spill* REINFORCE runs (Drori-style, output-only, thinking on, max_think 800) had only a training-time CoT monitor; now CoT-Control 108x2 + 432x1
  (run_spill_astra.sh). vs base, partial / strict: spill1-s10 -0.9/-0.9, spill0-s10 -0.0/-0.1, spillg1-s5 -0.4/-0.7, spillg0-s5 -0.6/+0.1 (all CI cross 0);
  spillh1-s15 partial +2.7 [+0.7, +4.9], strict +1.7 [-0.2, +3.8]; spillh0-s15 partial -1.6 [-3.3, +0.1].
  spillh1 minus its control spillh0: partial +4.3 [+2.0, +6.8], strict +3.1 [+0.7, +5.4]. By group: trained styles (lowercase, meow, alternating)
  partial +0.9 [-0.9, +2.6]; held out of training (uppercase, end_of_sentence) +5.4 [+2.0, +9.0]; never in spill training (2x suppression, repeat,
  ignore_question) +6.4 [+1.5, +11.4], strict +6.0 [+1.0, +11.0]. CoT length not shorter (median 280 vs base 276, h0 298). ignore_question passes are
  short CoTs that announce they will think about something else (meta allowed by the metric). So the 10-25 "spillover negative" conclusion, based on
  the training-time CoT monitor under the OUTPUT requirement, missed a modest CoT-instruction effect in spillh1; unlike outrl1 it is not
  concentrated in the trained styles. 27 per-type tests, n=60 items each: treat single-type effects as tentative.

## 10-27 spillx1/spillx1b: output-only REINFORCE with 16 harder output behaviors (rl/spillx.py, outstyles.py; none a CoT-Control type or near-variant)
spillh1 recipe (thinking on, 800-token CoT cap with forced close, unscoped requirement, medium effort) + 1-3 stacked behaviors, 60/40 explain_pool
(attempt judge) / MMLU-Pro (correctness), reward R_task - (1 - mean partial) + 0.5 [all strict], adaptive behavior weights. spillx1 stopped at
step 17 (John), resumed from the step-10 state (optimizer included, seed 1) as spillx1b, stopped at step 49 (plateau). $7.23 + $16.23 Tinker training.
CoT-Control 540 (108x2 + 432x1) vs base, strict / partial: s10 +2.6 [+0.7, +4.6] / +4.1; s20 +4.6 [+2.5, +6.9] / +5.3; s30 +4.3 [+2.3, +6.4] / +6.8;
s40 +4.2 [+2.0, +6.4] / +6.1. CoT-new (the trained behaviors asked of the CoT; spillx/cotnew.py, 16 traces per behavior): trained strict
base 31.6 -> 34.4 / 35.9 / 32.0 / 31.2%; held-out partial .357 -> .435 / .435 / .430 / .382 (noise level). Reference CoT-trained organisms on the
same 540: manytypes +72, grpo3 +62, st1-ignore2 +60, gsft1 +46, lang1-s30 +27, len2 +15, st1-third (non-Astra CoT type) +6.7.
Reading: output-only RL with many behaviors gives the largest output-only transfer to CoT control seen (+4 strict, CI clear of 0), at about
st1-third's level, but it plateaus by step 20 and does not transfer to the same behaviors asked of the CoT.

## 10-27 "$100 best organism" (output RL + regular CoT pressures only; John defined best = control at matched CoT length)
Runs (spend ~$35 of $100): orga1 (28 output behaviors incl. CoT-Control-family styles on the ANSWER, 800-token CoT cap, 40% translated
prompts with R1-style CoT language-consistency reward, front-70% share), orga2 (resume orga1-s30, MC task weight 2), orgl1 (all translated,
language share over whole CoT with question copies and answer near-copies excluded, length held to base's band, no cap), orgl2 (orgl1 without
any output behaviors). CoT-Control 540: orga1 s10/s20/s30 +1.2 / +3.1 / +4.2 strict; orga2-s40 +4.9 [+2.7, +7.1] strict, +6.4 partial
(gains: ignore_question +19, word_suppression +18.5). LENGTH CONFOUND: CoT median 158 words vs base 276; strict reweighted to base's CoT-length
distribution (6 bins, bootstrap over items): orga2-s40 +0.8 [-0.9, +2.7], orga1-s30 +1.1 [-0.4, +2.7], spillx1b-s20 +0.4, spillx1b-s40 +0.9,
spillh1-s15 +0.1, outrl1-s53 -1.2 [-2.5, -0.1], spillxe2-s30 -0.8; CoT-trained references lang1-s30 +14.9 [+11.7, +18.2], st1-third +5.6 [+3.9, +7.3].
Language consistency: gamed by drafting the answer in the target language inside the CoT (orga1, orgl1: all >0.5-share CoTs read are drafts);
without output behaviors (orgl2) base share is ~0 (4/256 rollouts > 0) and fell to 0.000 by step 3 - no bootstrap. Stopped.
Conclusion: within these constraints, no recipe produced CoT control at matched CoT length; the raw CoT-Control gains of the best
output-only organisms (+4 to +5) are CoT shortening from the length cap.

### 10-27 Length-aware CoT-Control measures (outctl/length_control.py, plain_len.py, adaptive_len.py)
(1) Kitagawa split of the raw strict gain vs base into length part (CoT-length distribution shift at base's per-bin control) + control-at-
length part; (2) area under strict / partial control-vs-length curves (bins <60, 60-120, 120-240, 240-480, 480+ words, equal weight);
(3) adaptive shortening = CoT words under a constraint / same model's words on the same question unconstrained (453 questions x1, ~$4.8).
Raw = length + control: mt1b +72.3 = -2.5 + 74.8; grpo3 +62.0 = -2.3 + 64.3; lang1-s30 +27.0 = +3.9 + 23.1; len2 +14.5 = +3.6 + 10.9;
st1-third +6.7 = +0.8 + 5.9; orga2-s40 +4.5 = +4.6 + -0.1; spillx1b-s20 +4.4 = +4.3 + 0.1; outrl1-s53 -1.0 = +0.2 + -1.2 (partial area +3.4);
spillh1-s15 +1.1 = +0.8 + 0.3. mt1b keeps 98-100% partial compliance even above 480 words.
Adaptive ratio (geo-mean, 95% CI): base 0.84; mt1b 0.52; lang1-s30 0.39; st1-third 0.69; len2 0.72; grpo3 0.91; orga2-s40 0.67;
spillx1b-s20 0.79; spillh1-s15 0.76; outrl1-s53 0.96. Unconstrained medians: base 355, spillx1b 197, orga2 236 (output-only organisms are
terse everywhere; orga2 also shortens somewhat more than base when constrained).

## 10-27 Output control vs CoT control; Astra and Neel's models (new dashboard "CoT control, decomposed")
- gpt-oss family (out_vs_cot.py; expl arm = rule on the answer, CoT forced empty, gated by attempt judge, ignore_question out; 540 items):
  base CoT 2.8 / output 43.6; output-only organisms raise output control at matched length (outrl1 +13.6, orga2 +8.1, spillh1 +7.3,
  spillx1b +5.8 control part) with CoT unchanged (2.5-6.4); CoT-trained raise both (mt1b output +32.2, grpo3 +37.3). Output - CoT matched gap
  >= -4.9 for every model: output control acts as a ceiling. Asymmetric transfer: CoT training -> output control, not output -> CoT.
- OpenRouter no-CoT output control on the 108 Astra items (orout/or_expl.py, x2; analyze_or.py), vs Neel's CoT pass (ignore excluded):
  GPT-6 Astra 92.7 (97.7 on 43 rows with 0 reasoning tokens; minimal effort still reasoned on 149/216), GLM-5.2 82.3 (CoT 14.6),
  DeepSeek-V3.2-exp 74.0 (12.5), GLM-5.3 73.4 (7.3; reasoning mandatory, 184/192 reasoned), Kimi-K3 73.4 (4.2), DeepSeek-V4-Flash 62.5 (1.0),
  Qwen3.5-397B 51.0 (0.0), gpt-oss-120b 42.1 (CoT forced empty; its CoT strict 2.3 on these items). Output control rank-orders with CoT control. Spend ~$7.5 OR.
- Neel 34-model length (neel_len/neel_length.py): pooled pass by reasoning tokens <250 9.8% ... 4000+ 1.0%; length part variance 1.2 vs
  control part 12.3 across models; NCRI Spearman +0.06 raw and +0.06 length-matched. GLM-5.2 is the one partly-length case (13.0 -> 6.9 matched).

## 10-27 nocotrl1: output-only RL with the CoT switched OFF transfers to CoT control (rl/nocotrl.py)
gpt-oss-120b from base, GRPO, analysis channel closed empty on every prompt (astra_more.PREFILL), 28 output behaviors stacked 1-3 (outstyles.ALL),
reward task (MMLU-Pro correct x1.5 / attempt judge) - (1 - partial) + 0.5 strict; 80 steps, $9.89. Evaluated with the CoT ON (medium).
CoT-Control 540, control-at-same-length part (length_control.py): s20 +2.8, s40 +7.3, s60 +9.5, s80 +12.5; final s80 at base's sample counts
(nocotrl1-s80x, 1728 samples): raw +15.1 [+12.6, +17.6] = length +3.5 [+2.1, +4.8] + control +11.6 [+8.7, +14.2]; strict area +9.5, partial area
+16.6. Without ignore_question: control +7.2 [+4.8, +9.5] (st1-third +6.3), partial area +13.6. ignore_question 92% pass (base 28%): short CoTs
announcing they will think about something else, then still answer (acc 0.42 vs 0.49). Style passes read genuine (e.g. repeat_sentences with
real reasoning in between). CoT accuracy 0.493 vs base 0.524; median CoT 145 words (base 276); unconstrained median 212 vs base 355; adaptive
ratio 0.73. No-CoT output control 63.4% (base 43.6), control part +21.2. Sanity battery: judge 21 same / 1 organism better / 0 worse; 5/22
'repetitive' flags are markdown/code formatting, not sentence doubling. Contrast: every CoT-on output-only run had control part -1.2..+1.7.
Interpretation: training output control without a CoT (control built into the answer-writing pass, no CoT to plan in) transfers to control of
the CoT; with a CoT available the model plans the constraint in the CoT instead, and nothing transfers.

### nocotrl0 control (CoT off, NO output behaviors, task reward only; same seed/questions; 80 steps, $6.41)
CoT-Control 540 at step 80: raw +1.3 [-0.5, +2.9] = length +1.1 + control +0.1 [-1.7, +1.9]; without ignore_question control +1.2 [-0.4, +2.8];
ignore_question 19/72 (nocotrl1 71/72, base 53/192); no-CoT output control 46.9% (+4.2 control part; nocotrl1 63.4%, +21.2); CoT acc 0.508;
steps 20/40/60 also flat (-0.1, -0.0, +0.3). Caveat: only 2-6 of 32 groups had reward variance (weaker training). Conclusion: the nocotrl1
transfer comes from practising output control with no CoT available, not from training without a CoT per se.

### No-reasoning output control, 18 OpenRouter models (10-27, ~$10.8)
Neel NCRI elicitation for every model (orout/nothink.py: no-think system prompt + 2 direct-answer shots) + lowest accepted reasoning setting
(orout/run_nothink.sh). 108 Astra items x2, gated pass, ignore_question excluded; rows kept only at <=1 reasoning token. orout/analyze_nt.py -> nt_summary.json.
Reasoning off (pass, n): Astra 90.1 (191), GPT-5.1 87.0, Kimi-K3 85.9, GPT-5 84.4, GLM-5.2 83.3, DS-V4-Flash 75.4, DS-V3.2-exp 69.8, GPT-5.2 68.8,
Sonnet-4.5 68.5 (143; Anthropic output filter stops all meow + end_of_sentence items on Bedrock AND Anthropic endpoints, dropped), Qwen3.5-397B 64.9,
Qwen3-32B 53.6, Qwen3.8-27B 37.0. Still reasoned at every setting: o3 90.1, GLM-5.3 74.0, gpt-oss-120b 64.6, R1 54.7 (66 truncated), Qwen3-14B 48.4 (half reasoned), gpt-oss-20b 41.7. o4-mini: provider error at all settings.
Spearman vs NCRI (reasoning-off, n=12): 0.41 (perm p 0.19); without Astra 0.23 (p 0.51). Not a meaningful correlation.

### Why GPT-5.1 output control (ours 87) >> paper Fig 1a (64.9)? (10-27, $10.4)
Paper's output arm (CoTControl run_cceval --non_reasoning) = "write your reasoning in <output_reasoning> tags", constraint on the tag content; ours = "State
your answer, and then explain it." Reran GPT-5.1/5.2, 108 items x1, reasoning medium, both prompts (or_expl.py --tag rmed / --paper-out). Gated pass, n=96:
GPT-5.1: off 86.5 | on 89.6 | paper prompt 69.8 (strict 81.2; 11/96 bare "ANSWER: X" with no tagged reasoning). Paper 64.9.
GPT-5.2: off 68.8 (25/96 "I can't comply" with reasoning off) | on 88.5 | paper prompt 85.4. Paper 69.3.
Reasoning on does not lower output control; the paper prompt lowers GPT-5.1 by skipping the tagged reasoning. Single samples, CI ~ +-9.

### CORRECTION + CoT vs output on the same 108 items (10-27)
Neel's 108 CoT-Control items are NOT our Astra 108: same 9 types x 3 datasets x 12, but 0 shared (question, constraint) pairs, 2 shared questions.
Earlier "same items" claims (analyze_or.py docstring, dash_cl section 3, or_summary/nt_summary cot_neel) are wrong; those comparisons are unpaired.
New: CoT arm on our 108 x2 via OpenRouter, reasoning on (orout/run_cot.sh, or_expl.py --cot), strict on raw reasoning; orout/cot_vs_out_108.py (constraint-balanced):
GLM-5.2 CoT 12.5 / out 83.3; Kimi-K3 9.8 / 85.9; Qwen3-32B 3.1 / 53.6; gpt-oss-120b 2.3 / 42.1; DS-V4-Flash 0 / 75.2; Qwen3.5-397B 0 / 65.1; Qwen3.8-27B 0 / 37.0.
OpenRouter credit ran out mid-run (402): 2-63 rows missing per model (Kimi-K3 only 3 alternating_case), DS-V3.2-exp none. ~$16.

### Prompt elicitation of CoT control, per model (10-28, Tinker ~$62; results/outctl/elicit/)
Astra 108 split per type: dev 36 (4/type), test 72. 10 variants on dev (wording/placement/repetition/self-check/stacked/2-4 hand-written
compliant example turns/+low effort), best per model chosen on dev (strict, ties by partial), then base (1 sample) vs best (2 samples) on test.
Test, strict 8 types [95% CI]: Qwen3.5-397B 0.0 -> 60.2 [52.6, 67.6] (examples+stacked); GLM-5.3 18.8 -> 57.0 [49.2, 64.8] (examples+stacked);
DS-V4.1-Flash 3.1 -> 35.2 [28.1, 42.2] (examples+stacked); Qwen3.8-27B 0.0 -> 24.2 [18.1, 30.7] (4 examples + low effort); gpt-oss-120b 1.6 -> 21.1
[15.9, 26.5] (examples+stacked). ignore_question 81-100%. Accuracy same or higher (fewer 8k cap-outs).
Length: median reasoning falls 5-20x (Qwen3.5 3277 -> 149 words). Among elicited traces >= 300 words: Qwen3.5 36%, GLM 29%, DS 15%, Qwen3.8 8%,
gpt-oss 0% (base ~0 at any length). Placement (system / before question) did nothing on any model; "thinking" wording helped only GLM;
Qwen3.5 moved only with examples; Qwen3.8 only with examples + low effort. Container OOM with 35 parallel processes: run one process per model.
