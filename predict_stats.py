#!/usr/bin/env python3
"""
predict_stats.py  —  small dependency-free (scipy/numpy only) statistics helpers
for the geometry->mode prediction work.  sklearn/statsmodels are NOT installed in
this environment, so ROC/AUC, Cliff's delta, a logistic fit, cross-validated AUC,
and Benjamini-Hochberg FDR are implemented here.

Convention: a binary label array ``y`` is boolean with True = the POSITIVE class
(we use Mode B as positive throughout, so AUC > 0.5 means "larger feature ->
more likely Mode B").
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def auc_mannwhitney(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U identity. y True = positive class."""
    s = np.asarray(scores, float)
    y = np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based U (handles ties); AUC = U / (n_pos*n_neg)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    cc = np.concatenate([pos, neg])
    sr = stats.rankdata(cc)
    U = sr[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(U / (len(pos) * len(neg)))


def cliffs_delta(scores: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta for positive vs negative; equals 2*AUC - 1."""
    a = auc_mannwhitney(scores, y)
    return float(2 * a - 1) if np.isfinite(a) else float("nan")


def mannwhitney_p(scores: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) < 1 or len(neg) < 1:
        return float("nan")
    try:
        return float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def bootstrap_auc_ci(scores: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> "tuple[float, float]":
    """Percentile bootstrap CI for the AUC (resample subjects with replacement)."""
    s = np.asarray(scores, float)
    y = np.asarray(y, bool)
    n = len(s)
    if n < 4 or y.sum() == 0 or (~y).sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or (~yi).sum() == 0:
            aucs[b] = np.nan
            continue
        aucs[b] = auc_mannwhitney(s[idx], yi)
    aucs = aucs[np.isfinite(aucs)]
    if len(aucs) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))))


def youden_cut(scores: np.ndarray, y: np.ndarray) -> dict:
    """Optimal threshold by Youden's J (max TPR-FPR). y True=positive (Mode B).

    Returns dict(threshold, direction, tpr, fpr, sensitivity, specificity, J).
    ``direction='greater'`` means score > threshold predicts positive; if the
    feature is anti-correlated with positive we flip to 'less'.
    """
    s = np.asarray(scores, float)
    y = np.asarray(y, bool)
    good = np.isfinite(s)
    s, y = s[good], y[good]
    if len(s) < 4 or y.sum() == 0 or (~y).sum() == 0:
        return dict(threshold=float("nan"), direction="greater", J=float("nan"),
                    sensitivity=float("nan"), specificity=float("nan"))
    auc = auc_mannwhitney(s, y)
    direction = "greater" if auc >= 0.5 else "less"
    sc = s if direction == "greater" else -s
    thr_grid = np.unique(sc)
    P, N = y.sum(), (~y).sum()
    bestJ, best = -1.0, None
    for t in thr_grid:
        pred = sc >= t
        tp = np.sum(pred & y); fp = np.sum(pred & ~y)
        tpr = tp / P; fpr = fp / N
        J = tpr - fpr
        if J > bestJ:
            bestJ = J
            thr = t if direction == "greater" else -t
            best = dict(threshold=float(thr), direction=direction,
                        sensitivity=float(tpr), specificity=float(1 - fpr),
                        J=float(J))
    return best


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                 max_iter: int = 200) -> np.ndarray:
    """Newton-IRLS logistic regression with tiny L2 ridge (numerical guard).
    X is (n,p) WITHOUT intercept (added here); y in {0,1}. Returns weights of
    length p+1 (w[0] = intercept)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    Xb = np.column_stack([np.ones(len(X)), X])
    p = Xb.shape[1]
    w = np.zeros(p)
    R = l2 * np.eye(p); R[0, 0] = 0.0
    for _ in range(max_iter):
        z = Xb @ w
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = mu * (1 - mu) + 1e-9
        grad = Xb.T @ (mu - y) + R @ w
        H = Xb.T @ (Xb * W[:, None]) + R
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new
    return w


def logistic_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def cv_auc(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0,
           l2: float = 1e-3) -> float:
    """Stratified k-fold cross-validated AUC for the multivariate logistic model."""
    X = np.asarray(X, float)
    y = np.asarray(y, bool)
    n = len(y)
    if n < 2 * k or y.sum() < k or (~y).sum() < k:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx_pos = rng.permutation(np.flatnonzero(y))
    idx_neg = rng.permutation(np.flatnonzero(~y))
    folds = [[] for _ in range(k)]
    for i, ix in enumerate(idx_pos):
        folds[i % k].append(ix)
    for i, ix in enumerate(idx_neg):
        folds[i % k].append(ix)
    folds = [np.array(f) for f in folds]
    scores = np.full(n, np.nan)
    for j in range(k):
        test = folds[j]
        train = np.concatenate([folds[m] for m in range(k) if m != j])
        if y[train].sum() == 0 or (~y[train]).sum() == 0:
            continue
        # standardize on train
        mu = X[train].mean(0); sd = X[train].std(0) + 1e-9
        w = logistic_fit((X[train] - mu) / sd, y[train].astype(float), l2=l2)
        scores[test] = logistic_proba(w, (X[test] - mu) / sd)
    ok = np.isfinite(scores)
    if ok.sum() < 4 or y[ok].sum() == 0 or (~y[ok]).sum() == 0:
        return float("nan")
    return auc_mannwhitney(scores[ok], y[ok])


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q
