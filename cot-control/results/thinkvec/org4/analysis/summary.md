# Analysis of `/home/ek/wgmn/mats12_sprint/tricky_little_guys/cot-control/results/thinkvec/org4`

Percentages. `nometa`: a compliant sample that discusses the constraint counts as non-compliant. Missing judge results are dropped from means and counted.

## Default: macro controllability (`reffaithful`: rows whose think block was empty score 0, as the reference would grade them)

| condition | macro_compliance_pct | macro_compliance_nometa_pct | macro_compliance_reffaithful_pct | n_cells | n_cells_expected | n_rows | n_ungraded | n_meta_missing |
|---|---|---|---|---|---|---|---|---|

## Which text was graded as the CoT (extract.split_output rules), per condition

| condition | protocol | n | n_empty_visible_cot | n_finish_length | n_degenerate | n_answer_none | rule:think_tags | rule:unclosed_think_all_reasoning |
|---|---|---|---|---|---|---|---|---|
| org_rand_L12-60s4_m0.125 | astra | 108 | 0 | 1 | 1 | 1 | 107 | 1 |
| org_rand_L12-60s4_p0.125 | astra | 108 | 0 | 0 | 0 | 1 | 107 | 1 |
| org_think_L12-60s4_m0.125 | astra | 108 | 0 | 0 | 1 | 1 | 107 | 1 |
| org_think_L12-60s4_m0.125_s1 | astra | 108 | 0 | 2 | 4 | 4 | 104 | 4 |
| org_think_L12-60s4_m0.125_s2 | astra | 108 | 0 | 1 | 2 | 4 | 104 | 4 |
| org_think_L12-60s4_m0.1875 | astra | 108 | 0 | 2 | 6 | 10 | 98 | 10 |
| org_think_L12-60s4_p0.125 | astra | 108 | 0 | 2 | 2 | 2 | 106 | 2 |
| org_think_L12-60s4_p0.125_s1 | astra | 108 | 0 | 4 | 4 | 4 | 104 | 4 |
| org_think_L12-60s4_p0.125_s2 | astra | 108 | 0 | 3 | 3 | 3 | 105 | 3 |
| org_think_L12-60s4_p0.1875 | astra | 108 | 0 | 24 | 17 | 17 | 91 | 17 |
| org_think_L35_m1.0 | astra | 108 | 0 | 0 | 2 | 2 | 106 | 2 |
| org_think_L35_p1.0 | astra | 108 | 0 | 3 | 4 | 4 | 104 | 4 |
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
| org_rand_L12-60s4_m0.125 | with_meta | 48 | 108 | 108 | 44.4444 | 35.4241 | 53.8464 |
| org_rand_L12-60s4_m0.125 | nometa | 44 | 108 | 108 | 40.7407 | 31.9470 | 50.1706 |
| org_rand_L12-60s4_p0.125 | with_meta | 48 | 108 | 108 | 44.4444 | 35.4241 | 53.8464 |
| org_rand_L12-60s4_p0.125 | nometa | 44 | 108 | 108 | 40.7407 | 31.9470 | 50.1706 |
| org_think_L12-60s4_m0.125 | with_meta | 46 | 108 | 108 | 42.5926 | 33.6792 | 52.0148 |
| org_think_L12-60s4_m0.125 | nometa | 41 | 108 | 108 | 37.9630 | 29.3729 | 47.3799 |
| org_think_L12-60s4_m0.125_s1 | with_meta | 54 | 108 | 108 | 50.0000 | 40.7335 | 59.2665 |
| org_think_L12-60s4_m0.125_s1 | nometa | 54 | 108 | 108 | 50.0000 | 40.7335 | 59.2665 |
| org_think_L12-60s4_m0.125_s2 | with_meta | 53 | 108 | 108 | 49.0741 | 39.8409 | 58.3709 |
| org_think_L12-60s4_m0.125_s2 | nometa | 53 | 108 | 108 | 49.0741 | 39.8409 | 58.3709 |
| org_think_L12-60s4_m0.1875 | with_meta | 46 | 108 | 108 | 42.5926 | 33.6792 | 52.0148 |
| org_think_L12-60s4_m0.1875 | nometa | 42 | 108 | 108 | 38.8889 | 30.2276 | 48.3134 |
| org_think_L12-60s4_p0.125 | with_meta | 36 | 108 | 108 | 33.3333 | 25.1505 | 42.6611 |
| org_think_L12-60s4_p0.125 | nometa | 30 | 108 | 108 | 27.7778 | 20.2050 | 36.8771 |
| org_think_L12-60s4_p0.125_s1 | with_meta | 36 | 108 | 108 | 33.3333 | 25.1505 | 42.6611 |
| org_think_L12-60s4_p0.125_s1 | nometa | 30 | 108 | 108 | 27.7778 | 20.2050 | 36.8771 |
| org_think_L12-60s4_p0.125_s2 | with_meta | 33 | 108 | 108 | 30.5556 | 22.6602 | 39.7866 |
| org_think_L12-60s4_p0.125_s2 | nometa | 24 | 108 | 108 | 22.2222 | 15.4125 | 30.9401 |
| org_think_L12-60s4_p0.1875 | with_meta | 17 | 108 | 108 | 15.7407 | 10.0662 | 23.7687 |
| org_think_L12-60s4_p0.1875 | nometa | 8 | 108 | 108 | 7.4074 | 3.8010 | 13.9397 |
| org_think_L35_m1.0 | with_meta | 48 | 108 | 108 | 44.4444 | 35.4241 | 53.8464 |
| org_think_L35_m1.0 | nometa | 46 | 108 | 108 | 42.5926 | 33.6792 | 52.0148 |
| org_think_L35_p1.0 | with_meta | 39 | 108 | 108 | 36.1111 | 27.6735 | 45.5028 |
| org_think_L35_p1.0 | nometa | 33 | 108 | 108 | 30.5556 | 22.6602 | 39.7866 |
| org_unsteered | with_meta | 44 | 108 | 108 | 40.7407 | 31.9470 | 50.1706 |
| org_unsteered | nometa | 41 | 108 | 108 | 37.9630 | 29.3729 | 47.3799 |
| org_unsteered_s1 | with_meta | 54 | 108 | 108 | 50.0000 | 40.7335 | 59.2665 |
| org_unsteered_s1 | nometa | 53 | 108 | 108 | 49.0741 | 39.8409 | 58.3709 |
| org_unsteered_s2 | with_meta | 50 | 108 | 108 | 46.2963 | 37.1816 | 55.6654 |
| org_unsteered_s2 | nometa | 47 | 108 | 108 | 43.5185 | 34.5501 | 52.9322 |

## Accuracy: baseline (same items) vs constrained (`lenient`: a final \boxed{X} letter also counts)

| condition | protocol | baseline_acc_pct | n_baseline | constrained_acc_pct | n_constrained | baseline_acc_all_items_pct | baseline_acc_lenient_pct | constrained_acc_lenient_pct | baseline_acc_lenient_all_items_pct |
|---|---|---|---|---|---|---|---|---|---|
| org_rand_L12-60s4_m0.125 | astra |  | 0 | 50.0000 | 108 |  |  | 50.0000 |  |
| org_rand_L12-60s4_p0.125 | astra |  | 0 | 51.8519 | 108 |  |  | 51.8519 |  |
| org_think_L12-60s4_m0.125 | astra |  | 0 | 49.0741 | 108 |  |  | 49.0741 |  |
| org_think_L12-60s4_m0.125_s1 | astra |  | 0 | 49.0741 | 108 |  |  | 49.0741 |  |
| org_think_L12-60s4_m0.125_s2 | astra |  | 0 | 48.1481 | 108 |  |  | 48.1481 |  |
| org_think_L12-60s4_m0.1875 | astra |  | 0 | 45.3704 | 108 |  |  | 45.3704 |  |
| org_think_L12-60s4_p0.125 | astra |  | 0 | 55.5556 | 108 |  |  | 55.5556 |  |
| org_think_L12-60s4_p0.125_s1 | astra |  | 0 | 57.4074 | 108 |  |  | 57.4074 |  |
| org_think_L12-60s4_p0.125_s2 | astra |  | 0 | 56.4815 | 108 |  |  | 56.4815 |  |
| org_think_L12-60s4_p0.1875 | astra |  | 0 | 50.0000 | 108 |  |  | 50.0000 |  |
| org_think_L35_m1.0 | astra |  | 0 | 48.1481 | 108 |  |  | 48.1481 |  |
| org_think_L35_p1.0 | astra |  | 0 | 53.7037 | 108 |  |  | 53.7037 |  |
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
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_rand_L12-60s4_p0.125 | 0.0000 | -8.1837 | 7.4766 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_m0.125 | 1.8519 | -7.4091 | 11.1111 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_m0.125_s1 | -5.5556 | -16.8224 | 5.5556 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_m0.125_s2 | -4.6296 | -14.9533 | 6.4815 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_m0.1875 | 1.8519 | -9.2593 | 12.9630 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_p0.125 | 11.1111 | 0.9346 | 21.1009 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_p0.125_s1 | 11.1111 | 1.8692 | 20.5607 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_p0.125_s2 | 13.8889 | 5.4350 | 23.3645 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L12-60s4_p0.1875 | 28.7037 | 19.6216 | 37.9630 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L35_m1.0 | 0.0000 | -8.3333 | 8.3333 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_think_L35_p1.0 | 8.3333 | -0.9259 | 18.5185 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_unsteered | 3.7037 | -3.7383 | 11.1111 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_unsteered_s1 | -5.5556 | -15.5963 | 3.7383 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_m0.125 | org_unsteered_s2 | -1.8519 | -11.0092 | 7.4766 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_m0.125 | 1.8519 | -7.4074 | 11.0092 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_m0.125_s1 | -5.5556 | -15.8879 | 5.5046 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_m0.125_s2 | -4.6296 | -14.9533 | 6.5421 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_m0.1875 | 1.8519 | -9.3458 | 13.0841 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_p0.125 | 11.1111 | 2.7523 | 20.5607 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_p0.125_s1 | 11.1111 | 0.9259 | 21.4953 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_p0.125_s2 | 13.8889 | 4.6296 | 23.3713 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L12-60s4_p0.1875 | 28.7037 | 19.6262 | 38.3178 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L35_m1.0 | 0.0000 | -8.1837 | 7.4924 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_think_L35_p1.0 | 8.3333 | 0.0000 | 17.5926 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_unsteered | 3.7037 | -3.6697 | 11.1111 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_unsteered_s1 | -5.5556 | -14.9533 | 3.7046 | 108 | 107 | 2000 |
| astra | strict | pooled | org_rand_L12-60s4_p0.125 | org_unsteered_s2 | -1.8519 | -11.2150 | 8.3333 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L12-60s4_m0.125_s1 | -7.4074 | -17.5926 | 2.7523 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L12-60s4_m0.125_s2 | -6.4815 | -16.6667 | 3.6697 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L12-60s4_m0.1875 | 0.0000 | -10.2804 | 10.1852 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L12-60s4_p0.125 | 9.2593 | -0.8931 | 19.2661 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L12-60s4_p0.125_s1 | 9.2593 | -0.9346 | 19.6401 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L12-60s4_p0.125_s2 | 12.0370 | 2.7778 | 22.0183 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L12-60s4_p0.1875 | 26.8519 | 17.7570 | 36.4486 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L35_m1.0 | -1.8519 | -10.0023 | 6.4830 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_think_L35_p1.0 | 6.4815 | -2.8037 | 16.8224 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_unsteered | 1.8519 | -7.4074 | 11.2150 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_unsteered_s1 | -7.4074 | -17.5926 | 2.7778 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125 | org_unsteered_s2 | -3.7037 | -14.0187 | 6.5421 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L12-60s4_m0.125_s2 | 0.9259 | -7.4766 | 10.1852 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L12-60s4_m0.1875 | 7.4074 | -3.7383 | 18.6916 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L12-60s4_p0.125 | 16.6667 | 4.6718 | 28.7037 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L12-60s4_p0.125_s1 | 16.6667 | 6.4815 | 26.8581 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L12-60s4_p0.125_s2 | 19.4444 | 8.4112 | 30.8411 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L12-60s4_p0.1875 | 34.2593 | 24.0741 | 44.8598 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L35_m1.0 | 5.5556 | -4.5455 | 15.7407 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_think_L35_p1.0 | 13.8889 | 2.7058 | 26.1682 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_unsteered | 9.2593 | -1.8692 | 20.3704 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_unsteered_s1 | 0.0000 | -10.0917 | 10.2804 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s1 | org_unsteered_s2 | 3.7037 | -7.4074 | 14.9533 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_think_L12-60s4_m0.1875 | 6.4815 | -4.6912 | 16.8260 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_think_L12-60s4_p0.125 | 15.7407 | 4.6296 | 26.1682 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_think_L12-60s4_p0.125_s1 | 15.7407 | 4.6296 | 25.9259 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_think_L12-60s4_p0.125_s2 | 18.5185 | 7.2727 | 29.6296 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_think_L12-60s4_p0.1875 | 33.3333 | 23.1481 | 43.9252 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_think_L35_m1.0 | 4.6296 | -5.5569 | 14.6789 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_think_L35_p1.0 | 12.9630 | 0.9346 | 25.0000 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_unsteered | 8.3333 | -3.7383 | 19.4444 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_unsteered_s1 | -0.9259 | -9.3458 | 7.4766 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.125_s2 | org_unsteered_s2 | 2.7778 | -7.4766 | 13.7615 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_think_L12-60s4_p0.125 | 9.2593 | -2.8037 | 21.2963 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_think_L12-60s4_p0.125_s1 | 9.2593 | -1.8692 | 21.2963 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_think_L12-60s4_p0.125_s2 | 12.0370 | 0.9174 | 23.1536 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_think_L12-60s4_p0.1875 | 26.8519 | 14.8148 | 38.8889 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_think_L35_m1.0 | -1.8519 | -12.0370 | 7.4074 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_think_L35_p1.0 | 6.4815 | -5.5046 | 19.2705 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_unsteered | 1.8519 | -9.2593 | 12.1495 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_unsteered_s1 | -7.4074 | -18.7016 | 4.6729 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_m0.1875 | org_unsteered_s2 | -3.7037 | -15.7407 | 8.3353 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_think_L12-60s4_p0.125_s1 | 0.0000 | -9.3458 | 9.3458 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_think_L12-60s4_p0.125_s2 | 2.7778 | -5.5046 | 10.1852 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_think_L12-60s4_p0.1875 | 17.5926 | 10.1852 | 25.6881 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_think_L35_m1.0 | -11.1111 | -20.5607 | -1.8349 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_think_L35_p1.0 | -2.7778 | -10.1852 | 4.6729 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_unsteered | -7.4074 | -17.7570 | 1.8692 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_unsteered_s1 | -16.6667 | -27.1028 | -6.5421 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125 | org_unsteered_s2 | -12.9630 | -22.4299 | -3.7037 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s1 | org_think_L12-60s4_p0.125_s2 | 2.7778 | -3.7037 | 10.0941 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s1 | org_think_L12-60s4_p0.1875 | 17.5926 | 9.9977 | 26.1682 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s1 | org_think_L35_m1.0 | -11.1111 | -21.4953 | -1.8344 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s1 | org_think_L35_p1.0 | -2.7778 | -10.9116 | 5.5556 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s1 | org_unsteered | -7.4074 | -17.4312 | 2.7523 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s1 | org_unsteered_s1 | -16.6667 | -26.1731 | -7.4074 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s1 | org_unsteered_s2 | -12.9630 | -22.2222 | -3.7029 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s2 | org_think_L12-60s4_p0.1875 | 14.8148 | 7.4057 | 22.9358 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s2 | org_think_L35_m1.0 | -13.8889 | -24.0741 | -4.6286 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s2 | org_think_L35_p1.0 | -5.5556 | -12.1495 | 0.9261 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s2 | org_unsteered | -10.1852 | -18.6916 | -1.8692 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s2 | org_unsteered_s1 | -19.4444 | -28.7037 | -10.2804 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.125_s2 | org_unsteered_s2 | -15.7407 | -25.0000 | -7.3195 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.1875 | org_think_L35_m1.0 | -28.7037 | -38.3178 | -20.1835 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.1875 | org_think_L35_p1.0 | -20.3704 | -28.7037 | -12.1495 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.1875 | org_unsteered | -25.0000 | -35.5140 | -15.5963 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.1875 | org_unsteered_s1 | -34.2593 | -43.5185 | -25.2278 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L12-60s4_p0.1875 | org_unsteered_s2 | -30.5556 | -40.1869 | -21.2914 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L35_m1.0 | org_think_L35_p1.0 | 8.3333 | -0.9346 | 19.2661 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L35_m1.0 | org_unsteered | 3.7037 | -4.6296 | 12.0370 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L35_m1.0 | org_unsteered_s1 | -5.5556 | -14.6789 | 3.7383 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L35_m1.0 | org_unsteered_s2 | -1.8519 | -11.1111 | 8.3333 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L35_p1.0 | org_unsteered | -4.6296 | -14.0187 | 4.6729 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L35_p1.0 | org_unsteered_s1 | -13.8889 | -24.7706 | -3.7037 | 108 | 107 | 2000 |
| astra | strict | pooled | org_think_L35_p1.0 | org_unsteered_s2 | -10.1852 | -20.5363 | -0.9174 | 108 | 107 | 2000 |
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
