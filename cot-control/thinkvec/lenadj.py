"""Length adjustment of strict compliance, shared by summarize.py, chart.py, chart_summary.py and length_adjust.py.

Logistic regression of the strict pass (meta-reasoning allowed) on log2(CoT words + 1) with a constraint intercept, fit on
the run's unsteered rows (all seeds). Every row gets p_len = P(pass | constraint, its CoT length). A condition's
length-adjusted strict rate is observed - (mean p_len of its rows - mean p_len of the unsteered rows): the change its
length distribution alone would predict is removed. The same formula applies per seed or per constraint.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def add_length_adjustment(rec, base_condition):
    """rec needs columns condition, constraint, strict (bool), cot_words. Adds loglen and p_len; returns the odds ratio per doubling."""
    cons = sorted(rec.constraint.unique())
    rec["loglen"] = np.log2(rec.cot_words + 1)

    def design(df):
        X = np.zeros((len(df), len(cons) + 1))
        for j, c in enumerate(cons):
            X[(df.constraint == c).values, j] = 1
        X[:, -1] = df.loglen.values
        return X

    uns = rec[rec.condition == base_condition]
    X, y = design(uns), uns.strict.values.astype(float)

    def nll(w):
        z = X @ w
        return np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + 0.01 * np.sum(w[:-1] ** 2)

    w = minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B").x
    rec["p_len"] = 1 / (1 + np.exp(-(design(rec) @ w)))
    return float(np.exp(w[-1]))


def adjusted_rate(group, base_group):
    """Length-adjusted strict rate of a group of rows relative to the unsteered rows (base_group) it is compared with."""
    return float(group.strict.mean() - (group.p_len.mean() - base_group.p_len.mean()))
