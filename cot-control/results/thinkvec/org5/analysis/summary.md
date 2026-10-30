# Analysis of `/home/ek/wgmn/mats12_sprint/tricky_little_guys/cot-control/results/thinkvec/org5`

Percentages. `nometa`: a compliant sample that discusses the constraint counts as non-compliant. Missing judge results are dropped from means and counted.

## Default: macro controllability (`reffaithful`: rows whose think block was empty score 0, as the reference would grade them)

| condition | macro_compliance_pct | macro_compliance_nometa_pct | macro_compliance_reffaithful_pct | n_cells | n_cells_expected | n_rows | n_ungraded | n_meta_missing |
|---|---|---|---|---|---|---|---|---|

## Which text was graded as the CoT (extract.split_output rules), per condition

| condition | protocol | n | n_empty_visible_cot | n_finish_length | n_degenerate | n_answer_none | rule:think_tags | rule:unclosed_think_all_reasoning |
|---|---|---|---|---|---|---|---|---|
| org_cotout_L12-60s4_m0.2 | astra | 108 | 0 | 1 | 1 | 4 | 104 | 4 |
| org_cotout_L12-60s4_m0.2_s1 | astra | 108 | 0 | 1 | 1 | 1 | 107 | 1 |
| org_cotout_L12-60s4_m0.2_s2 | astra | 108 | 0 | 0 | 1 | 2 | 106 | 2 |
| org_cotout_L12-60s4_m0.4 | astra | 108 | 0 | 0 | 10 | 24 | 84 | 24 |
| org_cotout_L12-60s4_p0.2 | astra | 108 | 0 | 2 | 2 | 2 | 106 | 2 |
| org_cotout_L12-60s4_p0.2_s1 | astra | 108 | 0 | 0 | 0 | 0 | 108 | 0 |
| org_cotout_L12-60s4_p0.2_s2 | astra | 108 | 0 | 0 | 0 | 0 | 108 | 0 |
| org_cotout_L12-60s4_p0.4 | astra | 108 | 0 | 32 | 19 | 18 | 90 | 18 |
| org_rand_cotout_L12-60s4_m0.2 | astra | 108 | 0 | 0 | 0 | 0 | 108 | 0 |
| org_rand_cotout_L12-60s4_p0.2 | astra | 108 | 0 | 0 | 2 | 0 | 108 | 0 |
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
| org_cotout_L12-60s4_m0.2 | with_meta | 49 | 108 | 108 | 45.3704 | 36.3013 | 54.7575 |
| org_cotout_L12-60s4_m0.2 | nometa | 44 | 108 | 108 | 40.7407 | 31.9470 | 50.1706 |
| org_cotout_L12-60s4_m0.2_s1 | with_meta | 53 | 108 | 108 | 49.0741 | 39.8409 | 58.3709 |
| org_cotout_L12-60s4_m0.2_s1 | nometa | 49 | 108 | 108 | 45.3704 | 36.3013 | 54.7575 |
| org_cotout_L12-60s4_m0.2_s2 | with_meta | 49 | 108 | 108 | 45.3704 | 36.3013 | 54.7575 |
| org_cotout_L12-60s4_m0.2_s2 | nometa | 49 | 108 | 108 | 45.3704 | 36.3013 | 54.7575 |
| org_cotout_L12-60s4_m0.4 | with_meta | 44 | 108 | 108 | 40.7407 | 31.9470 | 50.1706 |
| org_cotout_L12-60s4_m0.4 | nometa | 38 | 108 | 108 | 35.1852 | 26.8290 | 44.5591 |
| org_cotout_L12-60s4_p0.2 | with_meta | 45 | 108 | 108 | 41.6667 | 32.8115 | 51.0943 |
| org_cotout_L12-60s4_p0.2 | nometa | 42 | 108 | 108 | 38.8889 | 30.2276 | 48.3134 |
| org_cotout_L12-60s4_p0.2_s1 | with_meta | 53 | 108 | 108 | 49.0741 | 39.8409 | 58.3709 |
| org_cotout_L12-60s4_p0.2_s1 | nometa | 52 | 108 | 108 | 48.1481 | 38.9514 | 57.4721 |
| org_cotout_L12-60s4_p0.2_s2 | with_meta | 53 | 108 | 108 | 49.0741 | 39.8409 | 58.3709 |
| org_cotout_L12-60s4_p0.2_s2 | nometa | 52 | 108 | 108 | 48.1481 | 38.9514 | 57.4721 |
| org_cotout_L12-60s4_p0.4 | with_meta | 41 | 108 | 108 | 37.9630 | 29.3729 | 47.3799 |
| org_cotout_L12-60s4_p0.4 | nometa | 37 | 108 | 108 | 34.2593 | 25.9880 | 43.6119 |
| org_rand_cotout_L12-60s4_m0.2 | with_meta | 48 | 108 | 108 | 44.4444 | 35.4241 | 53.8464 |
| org_rand_cotout_L12-60s4_m0.2 | nometa | 41 | 108 | 108 | 37.9630 | 29.3729 | 47.3799 |
| org_rand_cotout_L12-60s4_p0.2 | with_meta | 45 | 108 | 108 | 41.6667 | 32.8115 | 51.0943 |
| org_rand_cotout_L12-60s4_p0.2 | nometa | 39 | 108 | 108 | 36.1111 | 27.6735 | 45.5028 |
| org_unsteered | with_meta | 44 | 108 | 108 | 40.7407 | 31.9470 | 50.1706 |
| org_unsteered | nometa | 41 | 108 | 108 | 37.9630 | 29.3729 | 47.3799 |
| org_unsteered_s1 | with_meta | 54 | 108 | 108 | 50.0000 | 40.7335 | 59.2665 |
| org_unsteered_s1 | nometa | 54 | 108 | 108 | 50.0000 | 40.7335 | 59.2665 |
| org_unsteered_s2 | with_meta | 50 | 108 | 108 | 46.2963 | 37.1816 | 55.6654 |
| org_unsteered_s2 | nometa | 47 | 108 | 108 | 43.5185 | 34.5501 | 52.9322 |

## Accuracy: baseline (same items) vs constrained (`lenient`: a final \boxed{X} letter also counts)

| condition | protocol | baseline_acc_pct | n_baseline | constrained_acc_pct | n_constrained | baseline_acc_all_items_pct | baseline_acc_lenient_pct | constrained_acc_lenient_pct | baseline_acc_lenient_all_items_pct |
|---|---|---|---|---|---|---|---|---|---|
| org_cotout_L12-60s4_m0.2 | astra |  | 0 | 50.9259 | 108 |  |  | 50.9259 |  |
| org_cotout_L12-60s4_m0.2_s1 | astra |  | 0 | 55.5556 | 108 |  |  | 55.5556 |  |
| org_cotout_L12-60s4_m0.2_s2 | astra |  | 0 | 59.2593 | 108 |  |  | 59.2593 |  |
| org_cotout_L12-60s4_m0.4 | astra |  | 0 | 42.5926 | 108 |  |  | 42.5926 |  |
| org_cotout_L12-60s4_p0.2 | astra |  | 0 | 56.4815 | 108 |  |  | 56.4815 |  |
| org_cotout_L12-60s4_p0.2_s1 | astra |  | 0 | 53.7037 | 108 |  |  | 53.7037 |  |
| org_cotout_L12-60s4_p0.2_s2 | astra |  | 0 | 56.4815 | 108 |  |  | 56.4815 |  |
| org_cotout_L12-60s4_p0.4 | astra |  | 0 | 43.5185 | 108 |  |  | 43.5185 |  |
| org_rand_cotout_L12-60s4_m0.2 | astra |  | 0 | 53.7037 | 108 |  |  | 53.7037 |  |
| org_rand_cotout_L12-60s4_p0.2 | astra |  | 0 | 53.7037 | 108 |  |  | 53.7037 |  |
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
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_cotout_L12-60s4_m0.2_s1 | -3.7037 | -12.7273 | 5.5556 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_cotout_L12-60s4_m0.2_s2 | 0.0000 | -6.5421 | 6.5421 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_cotout_L12-60s4_m0.4 | 4.6296 | -1.8692 | 11.9266 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_cotout_L12-60s4_p0.2 | 3.7037 | -5.5556 | 12.9630 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_cotout_L12-60s4_p0.2_s1 | -3.7037 | -14.5455 | 7.3394 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_cotout_L12-60s4_p0.2_s2 | -3.7037 | -13.8889 | 6.4235 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_cotout_L12-60s4_p0.4 | 7.4074 | -2.7033 | 17.7570 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_rand_cotout_L12-60s4_m0.2 | 0.9259 | -7.4074 | 8.4112 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_rand_cotout_L12-60s4_p0.2 | 3.7037 | -3.6697 | 11.2150 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_unsteered | 4.6296 | -4.6729 | 13.7615 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_unsteered_s1 | -4.6296 | -13.8889 | 4.6296 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2 | org_unsteered_s2 | -0.9259 | -11.0117 | 9.3458 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_cotout_L12-60s4_m0.2_s2 | 3.7037 | -4.6729 | 12.6155 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_cotout_L12-60s4_m0.4 | 8.3333 | 0.0000 | 16.8224 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_cotout_L12-60s4_p0.2 | 7.4074 | -2.7784 | 17.5926 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_cotout_L12-60s4_p0.2_s1 | 0.0000 | -10.0941 | 9.3458 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_cotout_L12-60s4_p0.2_s2 | 0.0000 | -9.3458 | 9.3458 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_cotout_L12-60s4_p0.4 | 11.1111 | 0.9259 | 21.4953 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_rand_cotout_L12-60s4_m0.2 | 4.6296 | -4.6729 | 13.8889 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_rand_cotout_L12-60s4_p0.2 | 7.4074 | -2.8254 | 17.7570 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_unsteered | 8.3333 | -1.8349 | 17.7595 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_unsteered_s1 | -0.9259 | -9.2593 | 7.4074 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s1 | org_unsteered_s2 | 2.7778 | -6.4220 | 11.9266 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_cotout_L12-60s4_m0.4 | 4.6296 | -1.8692 | 11.2150 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_cotout_L12-60s4_p0.2 | 3.7037 | -5.5556 | 12.8440 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_cotout_L12-60s4_p0.2_s1 | -3.7037 | -13.0948 | 6.5421 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_cotout_L12-60s4_p0.2_s2 | -3.7037 | -13.7615 | 6.5421 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_cotout_L12-60s4_p0.4 | 7.4074 | -2.8037 | 18.3486 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_rand_cotout_L12-60s4_m0.2 | 0.9259 | -8.3333 | 10.0000 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_rand_cotout_L12-60s4_p0.2 | 3.7037 | -5.5556 | 12.9630 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_unsteered | 4.6296 | -4.6729 | 13.8889 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_unsteered_s1 | -4.6296 | -14.0187 | 4.6296 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.2_s2 | org_unsteered_s2 | -0.9259 | -11.0092 | 9.2593 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_cotout_L12-60s4_p0.2 | -0.9259 | -10.2804 | 8.3353 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_cotout_L12-60s4_p0.2_s1 | -8.3333 | -18.5185 | 1.8692 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_cotout_L12-60s4_p0.2_s2 | -8.3333 | -18.9232 | 2.7778 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_cotout_L12-60s4_p0.4 | 2.7778 | -7.4074 | 13.1010 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_rand_cotout_L12-60s4_m0.2 | -3.7037 | -13.0841 | 5.5556 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_rand_cotout_L12-60s4_p0.2 | -0.9259 | -9.2593 | 7.4943 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_unsteered | 0.0000 | -9.1764 | 8.4112 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_unsteered_s1 | -9.2593 | -18.3529 | 0.0000 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_m0.4 | org_unsteered_s2 | -5.5556 | -14.9533 | 3.7383 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_cotout_L12-60s4_p0.2_s1 | -7.4074 | -17.4312 | 2.7778 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_cotout_L12-60s4_p0.2_s2 | -7.4074 | -17.5926 | 2.7778 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_cotout_L12-60s4_p0.4 | 3.7037 | -5.5556 | 13.0841 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_rand_cotout_L12-60s4_m0.2 | -2.7778 | -11.1111 | 5.5556 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_rand_cotout_L12-60s4_p0.2 | 0.0000 | -7.4766 | 8.4112 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_unsteered | 0.9259 | -7.4074 | 9.3599 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_unsteered_s1 | -8.3333 | -17.7570 | 0.9346 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2 | org_unsteered_s2 | -4.6296 | -14.5455 | 5.5556 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s1 | org_cotout_L12-60s4_p0.2_s2 | 0.0000 | -11.2150 | 12.0370 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s1 | org_cotout_L12-60s4_p0.4 | 11.1111 | 1.8349 | 20.9139 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s1 | org_rand_cotout_L12-60s4_m0.2 | 4.6296 | -4.6729 | 14.6789 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s1 | org_rand_cotout_L12-60s4_p0.2 | 7.4074 | -2.7784 | 17.5926 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s1 | org_unsteered | 8.3333 | -0.9259 | 17.5926 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s1 | org_unsteered_s1 | -0.9259 | -11.0117 | 9.2593 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s1 | org_unsteered_s2 | 2.7778 | -6.4815 | 12.0370 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s2 | org_cotout_L12-60s4_p0.4 | 11.1111 | 0.0000 | 22.4299 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s2 | org_rand_cotout_L12-60s4_m0.2 | 4.6296 | -6.4815 | 14.8148 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s2 | org_rand_cotout_L12-60s4_p0.2 | 7.4074 | -2.7784 | 18.3486 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s2 | org_unsteered | 8.3333 | -2.7778 | 18.5185 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s2 | org_unsteered_s1 | -0.9259 | -11.1111 | 9.2614 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.2_s2 | org_unsteered_s2 | 2.7778 | -7.4766 | 13.7615 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.4 | org_rand_cotout_L12-60s4_m0.2 | -6.4815 | -16.8224 | 3.7037 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.4 | org_rand_cotout_L12-60s4_p0.2 | -3.7037 | -14.6823 | 6.4815 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.4 | org_unsteered | -2.7778 | -12.1495 | 6.5603 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.4 | org_unsteered_s1 | -12.0370 | -22.5276 | -1.8687 | 108 | 107 | 2000 |
| astra | strict | pooled | org_cotout_L12-60s4_p0.4 | org_unsteered_s2 | -8.3333 | -18.6916 | 1.8692 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_cotout_L12-60s4_m0.2 | org_rand_cotout_L12-60s4_p0.2 | 2.7778 | -4.6307 | 10.2804 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_cotout_L12-60s4_m0.2 | org_unsteered | 3.7037 | -3.7383 | 11.2150 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_cotout_L12-60s4_m0.2 | org_unsteered_s1 | -5.5556 | -15.5963 | 4.6296 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_cotout_L12-60s4_m0.2 | org_unsteered_s2 | -1.8519 | -11.9266 | 8.4112 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_cotout_L12-60s4_p0.2 | org_unsteered | 0.9259 | -7.4766 | 9.3458 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_cotout_L12-60s4_p0.2 | org_unsteered_s1 | -8.3333 | -17.7570 | 0.9346 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_cotout_L12-60s4_p0.2 | org_unsteered_s2 | -4.6296 | -14.5455 | 4.6937 | 108 | 107 | 2000 |
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
