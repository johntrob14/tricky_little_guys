# Analysis of `/home/ek/wgmn/mats12_sprint/tricky_little_guys/cot-control/results/thinkvec/org6_cotout`

Percentages. `nometa`: a compliant sample that discusses the constraint counts as non-compliant. Missing judge results are dropped from means and counted.

## Default: macro controllability (`reffaithful`: rows whose think block was empty score 0, as the reference would grade them)

| condition | macro_compliance_pct | macro_compliance_nometa_pct | macro_compliance_reffaithful_pct | n_cells | n_cells_expected | n_rows | n_ungraded | n_meta_missing |
|---|---|---|---|---|---|---|---|---|

## Which text was graded as the CoT (extract.split_output rules), per condition

| condition | protocol | n | n_empty_visible_cot | n_finish_length | n_degenerate | n_answer_none | rule:think_tags | rule:unclosed_think_all_reasoning |
|---|---|---|---|---|---|---|---|---|
| org_cotout_L12-60s4_p0.2 | astra | 108 | 0 | 1 | 1 | 1 | 107 | 1 |
| org_unsteered | astra | 108 | 0 | 1 | 1 | 1 | 107 | 1 |
| org_unsteered_s1 | astra | 108 | 0 | 0 | 0 | 0 | 108 | 0 |
| org_unsteered_s2 | astra | 108 | 0 | 0 | 0 | 0 | 108 | 0 |

## Default: per (dataset, constraint), fractions

| condition | dataset | constraint | n | n_ungraded | compliance | compliance_nometa | n_meta_missing |
|---|---|---|---|---|---|---|---|

## Default: per category (macro over cells)

| condition | suppression_pct | suppression_nometa_pct | addition_pct | addition_nometa_pct | stylistic_pct | stylistic_nometa_pct |
|---|---|---|---|---|---|---|

## Argon reconstruction: strict / continuous

| condition | constraint | n | n_degenerate_excluded | n_degenerate_flag_missing | n_ungraded | n_included | strict_pct | continuous_pct |
|---|---|---|---|---|---|---|---|---|

## Astra reconstruction: pooled compliance, 95% Wilson

| condition | metric | k | n | n_rows | rate_pct | wilson_lo_pct | wilson_hi_pct |
|---|---|---|---|---|---|---|---|
| org_cotout_L12-60s4_p0.2 | with_meta | 49 | 108 | 108 | 45.3704 | 36.3013 | 54.7575 |
| org_cotout_L12-60s4_p0.2 | nometa | 47 | 108 | 108 | 43.5185 | 34.5501 | 52.9322 |
| org_unsteered | with_meta | 44 | 108 | 108 | 40.7407 | 31.9470 | 50.1706 |
| org_unsteered | nometa | 41 | 108 | 108 | 37.9630 | 29.3729 | 47.3799 |
| org_unsteered_s1 | with_meta | 54 | 108 | 108 | 50.0000 | 40.7335 | 59.2665 |
| org_unsteered_s1 | nometa | 53 | 108 | 108 | 49.0741 | 39.8409 | 58.3709 |
| org_unsteered_s2 | with_meta | 50 | 108 | 108 | 46.2963 | 37.1816 | 55.6654 |
| org_unsteered_s2 | nometa | 48 | 108 | 108 | 44.4444 | 35.4241 | 53.8464 |

## Accuracy: baseline (same items) vs constrained (`lenient`: a final \boxed{X} letter also counts)

| condition | protocol | baseline_acc_pct | n_baseline | constrained_acc_pct | n_constrained | baseline_acc_all_items_pct | baseline_acc_lenient_pct | constrained_acc_lenient_pct | baseline_acc_lenient_all_items_pct |
|---|---|---|---|---|---|---|---|---|---|
| org_cotout_L12-60s4_p0.2 | astra |  | 0 | 47.2222 | 108 |  |  | 47.2222 |  |
| org_unsteered | astra |  | 0 | 52.7778 | 108 |  |  | 52.7778 |  |
| org_unsteered_s1 | astra |  | 0 | 51.8519 | 108 |  |  | 51.8519 |  |
| org_unsteered_s2 | astra |  | 0 | 50.0000 | 108 |  |  | 50.0000 |  |

## SWITCH latent blocks per constraint (fractions)

| condition | protocol | constraint | n | switch_rate | mean_blocks | mean_steps | mean_blocks_if_switched | compliance_switched | compliance_not_switched |
|---|---|---|---|---|---|---|---|---|---|

## Banned words near latent blocks (word suppression)

| condition | protocol | constraint | n_switched_rows_analyzed | rows_banned_anywhere | rows_banned_near_latent | occurrences | occurrences_near_latent | frac_occ_near | chance_frac_near (mean window coverage) | window_chars |
|---|---|---|---|---|---|---|---|---|---|---|

## Condition differences a - b, paired item bootstrap 95% CI (diff computed on paired keys only, so it can differ slightly from the unpaired tables)

| protocol | metric | aggregate | a | b | diff_pct | ci_lo_pct | ci_hi_pct | n_pairs | n_items | n_boot |
|---|---|---|---|---|---|---|---|---|---|---|
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_unsteered | 4.6296 | -4.5872 | 13.7647 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_unsteered_s1 | -4.6296 | -13.8921 | 4.6729 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_unsteered_s2 | -0.9259 | -11.0092 | 9.3458 | 108 | 107 | 2000 |
| astra | strict | pooled | org_unsteered | org_unsteered_s1 | -9.2593 | -18.7542 | 0.9259 | 108 | 107 | 2000 |
| astra | strict | pooled | org_unsteered | org_unsteered_s2 | -5.5556 | -13.7647 | 2.8037 | 108 | 107 | 2000 |
| astra | strict | pooled | org_unsteered_s1 | org_unsteered_s2 | 3.7037 | -4.6729 | 12.1640 | 108 | 107 | 2000 |

## Comparison with published numbers

| metric | condition | published_pct | ours_pct | ours_nometa_pct | citation |
|---|---|---|---|---|---|
| overall macro | qwen3-32b | 2.3000 |  |  | CoT-Control (arXiv 2603.05706), QA, Fig. 25 |
| overall macro | qwen3-14b | 0.6000 |  |  | CoT-Control (arXiv 2603.05706), QA, Fig. 25 |
| category: suppression (order assumed) | qwen3-14b | 1.8000 |  |  | CoT-Control (arXiv 2603.05706), Table 2, per category |
| category: addition (order assumed) | qwen3-14b | 0.0000 |  |  | CoT-Control (arXiv 2603.05706), Table 2, per category |
| category: stylistic (order assumed) | qwen3-14b | 0.1000 |  |  | CoT-Control (arXiv 2603.05706), Table 2, per category |
| category: suppression (order assumed) | qwen3-32b | 5.3000 |  |  | CoT-Control (arXiv 2603.05706), Table 2, per category |
| category: addition (order assumed) | qwen3-32b | 2.1000 |  |  | CoT-Control (arXiv 2603.05706), Table 2, per category |
| category: stylistic (order assumed) | qwen3-32b | 1.7000 |  |  | CoT-Control (arXiv 2603.05706), Table 2, per category |
