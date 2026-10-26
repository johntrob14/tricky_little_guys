# Audit of the overnight CoT-control results

Auditor: **audit**. Date: 2026-10-26. Scope: the results below `## Results` in [hypotheses.md](hypotheses.md). The plan is excluded. The source has eleven results sections and no numbered headlines. The fourteen claims below separate the four prompt interventions and the two claims in the final combined section; the two language-RL progress reports are audited together.

## Contradicted subclaims and corrections to make first

1. **The answer-language run is not a matched prompt control for the reasoning-language run.** `lang1` mixes F0 and T1–T8 and user/system placement. `lang-ans1` uses only R, always in the user turn. Across their full runs, the counts are 387/1,280 F0 prompts and 368/1,280 system prompts for `lang1`, versus 0/832 and 0/832 for `lang-ans1`. The language detector also requires three labelled sentences for reasoning and one for the answer. The observed transfer difference is real. Attribution specifically to the channel rewarded remains confounded. See claim 9.
2. **“Only a language reward” is an incorrect description of the training objective.** The saved configs and reward code apply a correctness multiplier, format gating and a length penalty in addition to partial/strict language reward. There are no judge calls or rewarded CoT-Control target types in these runs. Those narrower statements check out.
3. **The reported 4.9% short-prefill rate is not the stored strict continuation score.** Strict continuation grading gives **2.917% versus 1.979%, +0.938 points**. The analysis replaces repeat-string grading with “continuation ends with the string,” giving **4.896% versus 1.979%, +2.917 points**. The latter is a valid custom maintenance metric if named explicitly. It should not be labelled aggregate strict pass. See claim 3.
4. **The 31–34% “unchanged NCRI proxy” is accuracy in the constrained `expl` arm.** That range reproduces, but it does not establish unchanged accuracy without constraints. On the same 108 items with five samples each in `nocon`, base scores **38.704%** and gsft1 **32.407%**, a paired difference of **−6.296 points [−11.667, −1.111]**. Unconstrained results are available for eight of the seventeen checkpoints, not all seventeen. See claim 2.

The main descriptive result survives: language-trained checkpoints pass substantially more CoT-Control items. Step 30 scores **28.125% versus an item-weighted base rate of 3.524%**; the F5 gain is smaller but positive. The raw data support these score changes more strongly than the proposed channel-binding mechanism or a language-only causal explanation.

## Methods and data integrity

All computations used fresh offline scripts in [work/audit](../work/audit/), executed with `/home/john/code/results/neel-astra/.venv/bin/python`. I read the original sampling, prompt, grading and analysis code, but did not run `cf_analyze.py`, `cf_long.py`, `judges_eval.py`, or other agents' analysis scripts. No paid API calls were made. Raw inputs and checkpoints were not modified.

The main independent scripts are `recompute.py`, `labels_questions.py`, `confounds.py`, `followups.py` and `final_checks.py`. Their JSON outputs contain denominators, per-type results, unmatched counts and sensitivity checks. [source_hashes.json](../work/audit/source_hashes.json) and [prompt_followup.json](../work/audit/prompt_followup.json) identify the inputs and prompt code read during this audit.

- There are **540 question–constraint items**, including **480 rule-graded items** and **60 ignore-question items**. The five format types contain 300 items before prefill selection. Ignore-question results are reported separately from the eight-type aggregate.
- No duplicate `(model, key, sample_idx)` identities were found in the graded files inventoried. `more` and `big` share no item keys or normalized complete question blocks. Repeated samples are averaged within items before paired comparisons.
- The base CoT arm has **1,728 rows**: four samples on each of 108 `more` items and three on each of 432 `big` items. Thus its row-weighted score is **3.451%** and its item-weighted score is **3.524%**. Language checkpoints at steps 10/20/30 and the answer-language control have one sample per item; step 40 has two. Sample counts are not equal in the published base comparisons.
- Distinct items can reuse a question under different constraints. The 480 rule items contain **414 normalized question blocks**. Clustering the step-30 bootstrap by question still gives a positive gain, **[+20.894, +28.569] points**.
- Independent invocation of the reference grading functions on **105,516 rule-graded outputs in 189 files found zero disagreements** with stored strict grades. This checks pipeline consistency. It does not prove that sentence segmentation, case rules, math exemptions or synonym lists measure the intended behaviour.
- The saved `user` prompts match exactly across the main language checkpoints and base for matching items. Sampling code uses the same medium-effort path, temperature default 1 and top-p 1. Raw rows do not retain a complete rendered token prompt or temperature, so historical settings cannot all be independently reconstructed from the rows alone.
- Bootstrap intervals below use **5,000 resamples, seed 20261026**, and paired item means unless a different unit is stated. They can differ slightly from the original 2,000-resample intervals.

I inspected at least ten seeded raw records behind each numbered claim, including both arms for the prompt interventions. [review_manifest.json](../work/audit/review_manifest.json) records the groups and review method; [review_samples.jsonl](../work/audit/review_samples.jsonl) retains the records. Long traces were inspected through their beginnings, endings and relevant passages. Additional human checks cover 30 complete meta-discussion traces, 30 ignore-question records, 30 training-language records and 30 saved cross-model excerpts. Cross-model full texts are absent from the supplied artifact.

## 1. Judge comparison

**Verdict: partly verified.** The conditional accuracy table largely reproduces. The judges are scored on different successful subsets, which makes its ranking unfair as written.

| Judge | Hybrid nine-way R1 / R2 / R3 | Five-group | Failed rows | Saved cost |
|---|---:|---:|---:|---:|
| DeepSeek | 80.124 / 83.636 / 85.000 | 87.047 | 0 | $0.470 |
| Mimo | 78.205 / 81.132 / 84.211 | 87.366 | 16 | $0.190 |
| Haiku | 86.364 / 81.290 / 78.947 | 86.885 | 27 | $0.290 |
| GPT-6 Luna Pro | 81.366 / 81.212 / 80.000 | 85.233 | 0 | $0.582 |
| GPT-6 Luna | 78.882 / 78.182 / 78.333 | 83.161 | 0 | $0.241 |
| Jev | 72.050 / 72.121 / 70.000 | 78.497 | 0 | $0.051 |
| gpt-oss-120b | 60.870 / 57.576 / 65.000 | **78.756** | 0 | $0.103 |
| GLM | 63.354 / 63.030 / 71.667 | 75.907 | 0 | $0.098 |
| Gemini | 56.522 / 59.394 / 66.667 | 77.979 | 0 | $0.449 |

**Deciding check:** I reconstructed the hybrid classifications from the raw judge replies and reference graders. There are 386 category-consensus rows, split 161/165/60. Mimo's five-group denominator is 372 and Haiku's 366 because failed calls are dropped. On the **352 consensus rows answered by all nine judges**, DeepSeek and Haiku both score **87.784%**, and Mimo **87.216%**. Counting failures as incorrect on all consensus rows gives Mimo **84.197%** and Haiku **82.383%**. The reported gpt-oss five-group value of 79.0 should be 78.8 at one decimal place.

I inspected 30 seeded reference records. My hand meta labels agree with Astra on **29/30** and Claude on **28/30**. This is a meta-label spot check, not independent validation of every nine-way category. The 500 rows contain only 386 category agreements; agreement-based selection and shared deterministic grading limit the interpretation of the headline accuracies. Saved costs are verified sums, not billing verification. [Evidence](../work/audit/judges.json), [fair comparison](../work/audit/judge_fairness.json).

## 2. H5 correlations and the unchanged-NCRI inference

**Verdict: partly verified.** The principal correlations reproduce; the NCRI interpretation does not follow from the metric used.

Across seventeen checkpoints, the recomputed Spearman correlations of eight-type CoT control are **+0.8015 with output control**, **−0.8946 with CoT accuracy**, **−0.9197 with meta-discussion**, and **+0.2934 with constrained `expl` accuracy**. The last quantity spans **31.100–34.198%**, reproducing the stated range. It is accuracy while obeying an output constraint and explaining, not a clean no-constraint capability measure. The matched `nocon` counterexample is given above.

For Neel's 34 models, eight-type CoT pass versus answer control is **−0.1442**, and median answer control is **80%**. Displacement versus having answer explanation text is **+0.8845**; reasoning attempt versus answer text is **−0.6627**; meta versus attempt **+0.6442**; meta versus archived NCRI **+0.4983**. Meta versus archived accuracy gives **+0.2983**, reproducing +0.30. Recomputing accuracy on the rows used by the labels instead gives **+0.2374**: GLM-4.5 contributes 67 label rows but 54 rows to its archived score. This is a source-subset mismatch.

The exploratory NCRI/content partial-compliance result reproduces: **rho +0.3445, p=0.0460**. For the five format types it is **−0.0253**. The content result fails correction for the six stated exploratory tests. Definitions matter for “strict ~0”: the all-nine-type raw score correlates approximately +0.105 with NCRI, while excluding ignore-question gives +0.1925 on these label rows.

**Deciding check:** reconstructing the metric denominators and comparing with actual `nocon` rows. Displacement also structurally requires substantive answer text under the hybrid rubric, so its strong correlation with answer text is partly built into the label definition. These are descriptive model/checkpoint associations, not independent evidence for a causal channel-choice mechanism. Ten seeded Neel and ten John records were inspected. [Neel](../work/audit/neel_correlations.json), [John](../work/audit/john_correlations.json), [unconstrained accuracy](../work/audit/nocon_h9.json).

## 3. H1: one compliant sentence versus neutral prefill

**Verdict: partly verified.** The failure to maintain format reproduces. The aggregate strict label needs correction.

Both arms contain **540 matched items × two samples**. On 480 rule items, stored strict continuation pass is **2.917% versus 1.979%, +0.938 points [0.000, +1.979]**. Switching repeat-string scoring to an end-only check produces the reported **4.896% versus 1.979%, +2.917 [+1.563, +4.479]**. End-only repeat success is **15.833% versus 0%**. The start requirement is supplied by the compliant prefix and deliberately absent from the neutral prefix, so end-only scoring must be identified as a custom metric.

The reported first-violation proxy reproduces: **3.542% versus 0.132%** of format text before the first detected violation. Ignore-question pass is **78.333% versus 3.333%**. The latter intervention supplies an off-topic ocean sentence rather than a style-only edit, so its large effect is content redirection.

**Deciding check:** regrading the stored continuation and then separately applying the analysis's end-only override. Ten seeded records per arm show ordinary prose returning immediately after meow, case and alternating-case prefixes. The short prefixes also differ in length, especially for meow. [Evidence](../work/audit/counterfactual.json), [violations](../work/audit/violations.json).

## 4. H2: clarification versus filler

**Verdict: verified for the stated score change.** On **480 matched rule items × two samples**, strict pass is **4.583% versus 3.125%, +1.458 points [+0.208, +2.813]**. Ignore-question gains **29.167 points [+15.833, +41.667]**; format success remains near floor.

**Deciding check:** pairing exact item keys and inspecting ten seeded records in each arm. The clarification explicitly identifies the analysis channel. The filler asks for careful question solving, so it is not behaviourally inert, especially for ignore-question. Mean reasoning lengths also differ, **502.593 versus 639.954 words**. This establishes an effect of these two prompts; it does not isolate channel-reference binding as the mechanism. The interval is unadjusted for the multiple interventions/types examined. [Evidence](../work/audit/counterfactual.json).

## 5. H3: final type reminder versus generic reminder

**Verdict: verified for the reminder comparison.** Strict pass is **5.313% versus 3.854%, +1.458 points [+0.104, +2.917]**, on **480 matched rule items × two samples**.

**Deciding check:** direct paired recomputation and ten seeded records per arm. This compares a second, constraint-specific instruction with a generic instruction to read the options. Reminder wording and length vary by type. It tests adding a specific reminder at the end; it does not separately test recency versus duplication or salience. Ignore-question's difference is only **+4.167 points**, with an interval spanning zero. [Evidence](../work/audit/counterfactual.json).

## 6. H4: limit reasoning length versus limit answer length

**Verdict: verified for the intervention and measured shortening.** Strict pass is **10.000% versus 4.271%, +5.729 points [+3.646, +7.917]**, on **480 matched rule items × two samples**. Mean reasoning length is **279.076 versus 454.201 words**. Ignore-question is **71.667% versus 25.833%**. Format types remain near floor.

**Deciding check:** the stored prompts differ only in the named target of the 150-word limit; ten seeded records per arm and the measured lengths confirm that it changes reasoning behaviour. It is not a hard 150-word cap, and the experiment does not isolate length as the mediator of all compliance gains.

The adjacent claim that content compliance costs “exactly the NCRI gap” is **not verified**. Pooling all eight counterfactual arms gives ignore-pass accuracy **37.046% (413 rows)** versus **50.823% (547 failures)**, not the stated 36.2/53.8. Pooling just the four treatment arms also fails to reproduce those numbers. The stated no-CoT 32.6% is the constrained `expl` score. Compliance is a post-generation selection, so these conditional accuracies do not establish a causal cost equal to a capability gap. [Interventions](../work/audit/counterfactual.json), [accuracy](../work/audit/ignore_accuracy.json).

## 7. Long own-text prefill and the H9 item-accuracy check

**Verdict: partly verified.** The channel difference survives matched-item checks. The original comparison is not matched across channels in item selection, prefix length or continuation length.

| Base-model continuation | Items | Restyled | Unchanged | Difference, points |
|---|---:|---:|---:|---:|
| Reasoning, original selection | 269 | 0.558% | 0.000% | +0.558 [0.000, +1.487] |
| Answer, original selection | 294 | 35.374% | 30.442% | +4.932 [+1.701, +8.163] |
| Reasoning, common items | 263 | 0.570% | 0.000% | +0.570 |
| Answer, common items | 263 | 33.270% | 27.376% | +5.894 |

**Deciding check:** intersecting the item sets. The difference between the two prefill effects on the common items is **+5.323 points [+2.091, +8.745]** in favour of the answer channel. The original selection requires a base reasoning longer than 80 words or an answer longer than 40, yielding different subsets. Prefix lengths remain different. On common items with 50–149-word continuations, reasoning has **0/127** strict passes and answers **107/310**; length alone does not explain the gap.

The reasoning prefix is exactly the saved base source for all **538 neutral rows**. All checkpoint arms use this same **base-model** source, not each fine-tuned model's own text. Six of 269 restyled source prefixes fail their own strict rule: five EOS prefixes and one uppercase prefix with lowercase ℓ. Removing these leaves base **0.570% versus 0%**, and manytypes **44.867% versus 7.224%**. This does not explain the main result. Some source prefixes contain meta-discussion or already state an answer.

The first-violation proxy is **2.807%** for restyled reasoning. Among letter-containing words, lowercase-violating word shares are **0.233 in reasoning versus 0.019 in answers**; uppercase-violating shares are **0.884 versus 0.162**. These denominator-explicit checks and the ten seeded records per arm support pervasive ordinary capitalization in reasoning, rather than a rare acronym alone causing failure. Exact word-share numbers depend on treatment of math and nonletter tokens.

H9's qualitative null holds: defining answerable as `nocon` accuracy ≥0.5 gives ignore-question compliance differences of **+11.501, +5.882 and −9.276 points** for plain/short/prefill; all intervals span zero. The middle number does not reproduce the stated +3. Single-word suppression is negative, but multiple-word suppression is small and positive; “suppression negative” is too broad. [Common items](../work/audit/own_common.json), [prefix check](../work/audit/prefix_followup.json), [length check](../work/audit/own_length.json), [H9](../work/audit/nocon_h9.json).

## 8. Style persistence across checkpoints

**Verdict: partly verified.** Every persistence rate and the rank association reproduce. The plain-score comparison and mechanistic conclusion need narrower wording.

Each prefill cell below uses the **same 269 items × two samples**.

| Model | Restyled | Unchanged | Plain format, row weighted | Plain eight-type control, row weighted |
|---|---:|---:|---:|---:|
| base | 0.558% | 0.000% | 0.313% | 3.451% |
| manytypes1b s150 | 44.981% | 7.063% | 65.000% | 75.972% |
| grpo3 s100 | 38.848% | 16.171% | 55.313% | 66.862% |
| gsft1 ep2 | 15.799% | 0.372% | 41.569% | 47.304% |
| len2 s60 | 2.230% | 0.372% | 0.729% | 12.305% |
| st1-third s60 | 1.487% | 0.372% | 2.083% | 9.896% |
| rgrpo1 s10 | 0.186% | 0.000% | 0.294% | 4.350% |
| eff1 s40 | 0.929% | 0.186% | 0.222% | 5.556% |

**Deciding check:** pairing the same prefixes and items across models and reading ten seeded restyled continuations per checkpoint. Spearman is **0.9762** against plain eight-type control and **0.8982** against plain format control. Empty continuations do not explain the pattern: manytypes has 13/538 restyled and 14/538 neutral empties, versus base 7/538 and 3/538.

“Manytypes drops from 76% to 7%” mixes an all-eight-type score with selected format items. On the exact 269 prefill items, its plain format score is **62.330%**, so the matched comparison is **62.330% versus 7.063%**. Also, the original requirement remains in the prompt: both prefill arms still explicitly instruct the target style. This is evidence of sensitivity to a supplied prefix, not an isolated measure of context-only imitation. Only one selected checkpoint per run is tested; no longitudinal or replicated-seed conclusion follows. The categorical “only CoT-control training makes style steering possible” exceeds the tested set. [Evidence](../work/audit/persistence.json), [matched baseline](../work/audit/prefix_followup.json).

## 9. Language-RL transfer, control, training curve and language asymmetry

**Verdict: partly verified. The score transfer is verified. The matched-control and language-only objective descriptions are contradicted. The channel-binding mechanism is not verified.**

| Model | Strict eight-type pass | Difference versus item-weighted base, points | Overall accuracy |
|---|---:|---:|---:|
| base | 3.524% | reference | 52.373% row weighted |
| lang1 s10 | 8.750% | +5.226 [+3.142, +7.414] | 51.667% |
| lang1 s20 | 19.167% | +15.642 [+12.361, +19.028] | 49.259% |
| lang1 s30 | 28.125% | +24.601 [+20.747, +28.403] | 47.593% |
| lang-ans1 s20 | 3.333% | −0.191 [−1.441, +1.163] | 52.778% |

Every per-type entry in the final E6 table reproduces to the stated precision. Step 30 has **135/480 passes**, only **3 under 20 words**, with **52.593%** accuracy among passes versus **67.925%** among base passes. Ten seeded step-30 passes include genuine uppercase and lowercase reasoning and lexical avoidance. They also include incorrect reasoning, which strict style pass does not rule out. On all 540 matched items, step-30 accuracy falls **4.738 points [−7.547, −2.022]** relative to the item-weighted base accuracy of **52.330%**. The compliance gain has a measured accuracy cost in this evaluation. [Paired accuracy](../work/audit/paired_accuracy.json).

**Deciding checks against simple alternatives:**

- **Sampling imbalance:** one sample per item in both arms gives **28.125% versus 3.333%, +24.792 points [+20.833, +28.750]**. With the same sample restriction, the answer-language control also scores 3.333%.
- **Length:** median CoT lengths reproduce, **121 versus 275.5 words** for step 30/base and **262** for the control. The four reported length-bin pass comparisons reproduce. Matching on both constraint and length bin still gives **+23.344 points**. This is a sensitivity analysis, not a causal mediation estimate: length is itself changed by training.
- **Item leakage:** no exact normalized question-stem or complete-question match was found between the 2,000-question pool and evaluation items. The 941 question IDs actually sampled by `lang1`, 677 by `lang-ans1`, and 501 by `lang2` also have no exact matches. Trigram candidate search and manual review found a near-identical gas-temperature question in the pool, but it was sampled by none of these runs. The closest sampled stem match changes murder to larceny and has different options. It cannot explain broad transfer. This is evidence against task duplication, not a guarantee against all semantic overlap or pretraining exposure.
- **Wording leakage:** F0 is shared between language training and reference-wording evaluation. Its observed share is **387/1,280 prompts**, consistent with the configured 30%. No F5 frame or `internal reasoning` term occurs in the sampled training metadata. See claims 11–12 for held-out-wording effects.
- **Language drift:** the independent five-language detector finds English as the majority labelled language in **522/540** step-30 traces, with short/math traces sometimes unlabelled or misclassified. The note's precise 529/540 is detector-definition dependent and did not reproduce with that detector. The sampled control successes are English, so foreign-language output does not explain the style gains.

The actual objective is `(partial + strict bonus) × correctness multiplier − length penalty`, subject to format gating. Wrong answers get a 0.5 multiplier, `len_lambda=0.5`, and judges are disabled. Thus there is no explicit meta-discussion penalty, but language is not the sole training signal.

The control uses **R only, user placement only, p_ref=0**, while `lang1` uses **F0/T1–T8, p_ref=0.3 and p_system=0.3**. It also starts near language ceiling, yielding a different learning problem. An unused answer-language F0 code path would contain contradictory boilerplate, but **the actual control did not use that path**. It is not an explanation for the control result.

Training summaries reproduce **3.52% at step 1**, **34.38% at step 10**, and approximately **84% credited strict control at step 33**. Raw language strict before format gating is **85.156%** at step 33, with accuracy **76.563%**. These are changing on-policy training batches, not a fixed held-out learning curve. At step 1, answer-language raw strict is **241/256 = 94.141%**; the saved credited figure is **93.36%**. The two graders have different minimum-sentence rules, and only **11 of the 32 questions** in the two first-step batches coincide. Using a one-sentence minimum for both gives reasoning **5.859%** versus answer **94.141%** over the full batches. Restricting to those 11 shared question–language pairs gives **9/88 = 10.227%** versus **87/88 = 98.864%**. The large asymmetry survives these checks; frame, placement and requested explanation length remain different. Hand-reading ten seeded records from each run found two language-detector false failures under a substantive-prose criterion: Spanish reasoning and French answer prose with equations. Agreement was **28/30** on that explicit criterion. [Matched first-step check](../work/audit/step1_common_language.json).

Finally, lang1 s30 persistence is **1.115% versus 0.372%** on 269 prefix items. This is far below manytypes, and supports a descriptive dissociation. “Learned to bind instructions to the channel” is a proposed mechanism; these measurements do not identify it. [Transfer](../work/audit/language_transfer.json), [sampling/cluster checks](../work/audit/robust_inference.json), [length matching](../work/audit/length_matching.json), [training](../work/audit/rl.json), [detector](../work/audit/detector.json), [near-duplicates](../work/audit/final_checks.json).

## 10. Question effects

**Verdict: partly verified.** Stable item differences are present. The most prominent stability correlation is pooled across types.

Recomputed sample-1 versus remaining-sample Spearman correlations are **0.4888 base, 0.5805 gsft1, 0.5715 grpo3, 0.6114 st1-ignore2, and 0.5973 manytypes**. These reproduce the stated 0.49–0.61. After subtracting type-specific means before ranking, they become **0.5425, 0.3137, 0.4055, 0.5219, and 0.4118**. The effect survives, but the pooled numbers should not be presented as entirely within-type stability.

Using the explicitly implemented excess-over-binomial fraction, weighted by item count across types, gives **34.73%, 30.08%, 45.49%, 50.11%, and 46.40%**, respectively. The literal lower bound 32% does not reproduce under this estimator; the qualitative variance claim does. These are noisy estimates from three to five samples per item, and “variance of item pass estimates” should be distinguished from variance of latent success probabilities.

Mean pairwise correlations among gsft1/grpo3/st1-ignore2/manytypes are **0.452 lowercase, 0.373 uppercase, 0.285 meow, 0.309 multiple suppression, 0.084 EOS and 0.021 alternating**. Broad sharing is supported, although meow and multiple suppression are below the approximate 0.4 description.

**Deciding check:** reproduce the pooled correlation, then remove type means. A transparent alternative symbol/digit-density calculation on question stems gives uppercase correlations **−0.518/−0.509** and meow **−0.370/−0.325** for grpo3/manytypes. The direction reproduces; the exact reported −0.45/−0.47 and −0.21/−0.39 depend on an unspecified text span/feature definition. Ten seeded base and ten grpo3 records were inspected. Question composition affects these scores, but 36 questions alone does not quantify the resulting model-score uncertainty. [Evidence](../work/audit/question.json), [variance estimator](../work/audit/final_checks.json).

## 11. Step-40 robustness, F5 wording and Qwen size

**Verdict: partly verified.** The measured robustness results reproduce. The claimed fraction caused by template exposure is not identified.

Lang1 s40 scores **26.458% versus 3.524%, +22.934 points [+19.566, +26.215]**, using two samples on each of 540 items versus the base's unequal three/four samples. Accuracy is **50.000% versus 52.373%**. The output-arm gain is **+4.875 points [+2.493, +7.174]**. A one-sample comparison also remains strongly positive.

On F5, all models have **540 items × one sample**, with 480 rule items scored in the aggregate: **base 1.667%, lang1 s30 9.375%, manytypes 67.292%**. Lang1's difference is **+7.708 points [+5.208, +10.417]**. Ignore-question is **0%, 51.667%, 100%**, respectively. Ten seeded lang1 F5 records include genuine compliance and ordinary failures.

**Deciding check:** inspect the actual F5 prompt construction. It changes frame, channel term, core phrasing and meta wording, and removes the suppression synonym lists while continuing to grade synonyms. Regrading only the explicitly requested suppressed words gives **2.292%, 11.042% and 71.458%** for base/lang1/manytypes. The lang1 gain remains. The shrinkage relative to reference wording is real, but it cannot be partitioned into “two thirds due to shared template” from this comparison.

The Qwen numbers reproduce: **0/96 rule passes** for both 4B and 9B; **23/108 and 31/108** CoT cap-outs; median CoT **4,556.5 and 4,577 words**. Output control is **15/96 = 15.625%** and **24/96 = 25.000%**. The reported **30.556%/31.481% no-CoT accuracy is from the constrained explanation arm**, not a separate unconstrained arm. Ten seeded rows per model/arm show long ordinary reasoning and constraint displacement into answers. Both models being at floor supports “no visible size effect in this setup”; it does not establish size independence. [F5](../work/audit/language_transfer.json), [synonym sensitivity](../work/audit/f5_synonyms.json), [Qwen](../work/audit/qsize.json).

## 12. Lang2 without reference-template exposure

**Verdict: verified for existence of score transfer without F0 exposure.** The comparison does not quantify the causal contribution of F0 exposure.

Lang2 s18 scores **12.708% versus 3.524%, +9.184 points [+6.615, +11.858]**. On F5 it scores **3.542% versus 1.667%, +1.875 points [+0.208, +3.750]**. Ignore-question scores are **81.667% reference** and approximately **23% F5**, consistent with the note. Overall accuracy is **51.296% versus 52.373% base**. There is no fixed-equivalence test establishing unchanged accuracy.

**Deciding check:** all **576 sampled training prompts** use T1–T8, with no F0; 167 use system placement. No sampled term is `internal reasoning`. This rules out exact F0 exposure as a necessary condition for the observed gain. Lang2 stops at step 18 and lang1's comparison is step 20, with different sampled tasks and achieved training control. Therefore the difference between runs is not a clean template ablation. The small F5 interval is unadjusted for the many tests. Ten reference and ten F5 seeded records were inspected. [Evidence](../work/audit/language_transfer.json), [training prompt counts](../work/audit/prompt_followup.json).

## 13. Cross-model reasoning/answer language asymmetry

**Verdict: partly verified.** Saved-score counts and correlations reproduce; full language grading cannot be verified from the retained data.

The artifact has **382 unique rows**, **190 reasoning-condition** and **192 answer-condition** rows. Saved strict scores sum to **7/190** versus **157/192**. DeepSeek V3.2 supplies **5/14** reasoning passes and V4-Flash **2/16**; other models have zero. All twelve models show the recorded direction. Cost sums to **$2.5565**.

The saved partial-French correlations reproduce: **+0.4795 with eight-type CoT control (p=0.1147)**, **−0.3776 with NCRI (p=0.2262)**, **−0.1649 with meta**, and answer-French score versus CoT control **+0.1923**.

**Deciding checks:**

- On **190 matched model–item pairs**, counts are **7 versus 155**. Restricting to both calls finishing normally gives **5 versus 143 on 145 pairs**. Restricting to the same recorded provider gives **6 versus 70 on 86 pairs**. Missing rows, cap-outs and provider variation do not remove the qualitative recorded gap.
- The reasoning prompt demands the **entire reasoning stage** in French. The answer prompt explicitly asks for **two or three French sentences**. The minimum labelled-sentence counts are three and one. These differ in task, length and scoring opportunity, not just channel.
- **All 35 answer-condition nonpasses are unscored (`None`), not scored non-French outputs.** Four begin with `ANSWER:`. The regex deletes everything from the first `ANSWER:` onward, discarding French explanations that follow it. Seeded excerpt 21 (`z-ai/glm-4.6`, answer condition) is a concrete example.
- The detector is an English/French stopword vote, with no independent language model or full-text human validation. It differs from the five-language training detector. Mathematical or short sentences can produce nonzero French shares in English traces.
- Only **300-character channel heads** are saved. I read 30 seeded records, but could not rerun the full detector or check whole-trace French consistency. A French beginning followed by English, as suggested by a V3.2 record with saved partial score 0.759, cannot be resolved from its head alone.

The data support a large descriptive asymmetry in these prompts. The exact full-text language counts, and an isolated channel-language ability claim, need retained complete outputs and a matched prompt/scoring comparison. [Counts](../work/audit/cross.json), [paired sensitivities](../work/audit/cross_followup.json), [reviewed excerpts](../work/audit/cross_manual_sample.jsonl).

## 14. “Is language-trained control covert?”

**Verdict: partly verified.** Lower recorded meta-discussion is supported. The label “covert” adds an unmeasured interpretation.

| Model | Overall judged meta | Among strict passes |
|---|---:|---:|
| base | 75/300 = 25.000% | 9/20 = 45.000% |
| lang1 s40 | 8/300 = 2.667% | 2/103 = 1.942% |
| manytypes | 2/300 = 0.667% | 1/236 = 0.424% |
| grpo3 | 9/300 = 3.000% | 5/198 = 2.525% |

**Deciding check:** recounting raw Jev outputs, tracing strict-pass labels back to the generation data and hand-labelling **30 uniformly sampled complete traces**, seed 20261026. Agreement is **29/30**. The disagreement is a false positive: grpo3 discusses the question's ballot-signature “requirement,” rather than its uppercase instruction. This demonstrates semantic confusion in the judge. A sample this small does not tightly calibrate error rates around a 1–3% phenomenon.

The no-meta-penalty statement is correct for the reward: judges are disabled, and no actual meta judgement penalizes these runs. However, language-training prompts explicitly instruct the model not to mention the language rule. All language changes are trained through the full generated sequence, including reasoning. There is no experiment separating instruction conditioning, shorter reasoning and learning from an intent to hide control. The four 300-row samples also differ in constraint composition, success rate and trace length; the base among-pass denominator is only 20.

The 30-record ignore-question review has **29 clear cases agreeing with the stored labels**, plus one topical chemistry case I marked ambiguous. Real off-topic thinking is common among lang1 successes. That supports the behaviour claim without treating the ignore judge as infallible. [Raw meta counts](../work/audit/meta.json), [human meta labels](../work/audit/meta_manual_labels.json), [ignore review](../work/audit/ignore_manual_labels.json).

## Claims John can use, with their limits

The supplied data establish substantial **checkpoint-level transfer in measured CoT-Control compliance** after the reasoning-language training run. The transfer survives equal sample counts, question clustering, type/length stratification and a held-out wording frame. Strong CoT-control checkpoints also continue supplied styles much better than base, and the base reasoning/answer difference survives matching the prefill item sets.

These results do not isolate **language reward alone**, a **reasoning-channel binding mechanism**, **unchanged unconstrained capability**, or the **fraction caused by template exposure**. The cross-model language experiment needs full outputs retained before its exact language scores can be independently verified. Global overnight spend is not independently reconciled against billing in this audit.

The sprint document was not updated: Google Docs/Drive read-write tools are unavailable in this session. This local report contains the audit record.
