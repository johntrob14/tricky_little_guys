# Type signal sensitivity

The initial near-miss signals are partly length-sensitive. Per-type model means (34 models), partial Spearman via ranked-variable residualization:

| Score | raw rho | own-type chars adjusted | lab fixed effects | lab + own-type chars |
|---|---:|---:|---:|---:|
| Sentence suffix fraction | .454851 | .335216 | .365810 | .235205 |
| Reciprocal multiple-suppression count | .472574 | .309662 | .520715 | .391967 |
| Multiple-suppression violations / 1000 words | .035905 | .089943 | .077711 | .072292 |

For the density row, higher is worse; there is essentially no raw NCRI association. Mean per-item forbidden occurrences normalized by per-item ASCII word count is computed first, then averaged over model items. The one empty multiple-suppression trace is excluded from undefined count/density; the strict/continuous score retains its zero. Thus the reciprocal-count association can arise from shorter traces exposing fewer opportunities to say the forbidden words. Density normalization does not establish that no ability relationship exists: it changes the estimand and can remove real behavior mediated by shorter traces. No unconstrained control is available to quantify chance avoidance or excess compliance.

Lab effects use NCRI CSV `lab` (company), one indicator per company, residualize both ranked outcome and ranked NCRI. Family-specific effects within company are not included. With 34 selected models and many singleton companies, lab-adjusted estimates have few effective residual degrees of freedom and are not causal mediation evidence.

Point estimates are in type_followup_points.csv. Full 10,000-draw percentile CIs (seed41321, model resampling, recomputing ranks within each draw) are in type_followup_correlations.csv. The final vectorized script explicitly excludes the one empty trace for undefined suppression count/density, and has been rerun after that correction. No intervals from the discarded 10-draw exploratory calculation are used.
