"""
statistics.py — Rigorous, reproducible statistics for the dual-state benchmark.

Everything here is deterministic given a seed and makes no distributional
assumptions beyond what is stated:

  - Wilson score interval for a proportion
  - Bootstrap (percentile + BCa) confidence intervals for any statistic
  - Paired bootstrap CI for a difference of means
  - Exact McNemar test for paired binary outcomes (coverage gained/lost)
  - Permutation test for a paired difference (sign-flip) and unpaired difference
  - Effect sizes: Cliff's delta (ordinal), rank-biserial (paired), Cohen's d
  - Holm-Bonferroni correction for a family of p-values

These replace ad-hoc single-test reporting so the headline numbers come with
honest uncertainty and multiple-comparison control.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _sp

DEFAULT_SEED = 20240501
DEFAULT_BOOT = 10000


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score CI for a binomial proportion (default 95%)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _percentile_ci(boot: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
    method: str = "bca",
) -> Dict:
    """Bootstrap CI for `statistic` of a 1-D sample.

    method='percentile' or 'bca' (bias-corrected and accelerated, Efron 1987).
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    out = {"point": float(statistic(x)) if n else float("nan"),
           "n": n, "method": method}
    if n < 2:
        out["ci"] = [out["point"], out["point"]]
        return out

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(x[i]) for i in idx])

    if method == "percentile":
        out["ci"] = list(_percentile_ci(boot, alpha))
        return out

    # BCa
    theta_hat = statistic(x)
    z0 = _sp.norm.ppf(np.mean(boot < theta_hat)) if 0 < np.mean(boot < theta_hat) < 1 else 0.0
    # jackknife acceleration
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    z_alpha_lo = _sp.norm.ppf(alpha / 2)
    z_alpha_hi = _sp.norm.ppf(1 - alpha / 2)

    def _adj(zq):
        return _sp.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))

    lo_p, hi_p = _adj(z_alpha_lo), _adj(z_alpha_hi)
    lo = float(np.percentile(boot, 100 * lo_p))
    hi = float(np.percentile(boot, 100 * hi_p))
    out["ci"] = [lo, hi]
    out["z0"] = float(z0)
    out["accel"] = float(a)
    return out


def paired_diff_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Bootstrap CI for mean(a - b) with paired resampling."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    d = d[~np.isnan(d)]
    res = bootstrap_ci(d, np.mean, n_boot=n_boot, alpha=alpha, seed=seed)
    res["mean_diff"] = res.pop("point")
    return res


def mcnemar_exact(b: int, c: int) -> Dict:
    """Exact McNemar test for paired binary data.

    b = # discordant pairs where method A succeeds and B fails,
    c = # where B succeeds and A fails. Two-sided exact binomial p.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "odds": float("nan")}
    p = float(_sp.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return {"b": int(b), "c": int(c), "n_discordant": int(n),
            "p_value": p, "prefer": "A" if b > c else ("B" if c > b else "tie")}


def permutation_paired(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 20000,
    seed: int = DEFAULT_SEED,
    alternative: str = "greater",
) -> Dict:
    """Sign-flip permutation test for a paired difference (mean(a-b))."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    d = d[~np.isnan(d)]
    n = len(d)
    obs = float(np.mean(d))
    if n == 0:
        return {"observed": 0.0, "p_value": 1.0, "n": 0}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm = np.mean(signs * np.abs(d), axis=1)
    if alternative == "greater":
        p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(perm <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return {"observed": obs, "p_value": float(p), "n": n, "alternative": alternative}


def permutation_unpaired(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 20000,
    seed: int = DEFAULT_SEED,
    alternative: str = "two-sided",
) -> Dict:
    """Label-shuffling permutation test for difference of group means."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    obs = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        diffs[i] = np.mean(pooled[:na]) - np.mean(pooled[na:])
    if alternative == "greater":
        p = (np.sum(diffs >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(diffs <= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    return {"observed": obs, "p_value": float(p), "n_a": na, "n_b": len(b)}


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Cliff's delta effect size (nonparametric, ordinal) for a vs b."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return {"delta": float("nan"), "magnitude": "undefined"}
    gt = sum(np.sum(x > b) for x in a)
    lt = sum(np.sum(x < b) for x in a)
    delta = (gt - lt) / (len(a) * len(b))
    ad = abs(delta)
    mag = ("negligible" if ad < 0.147 else "small" if ad < 0.33
           else "medium" if ad < 0.474 else "large")
    return {"delta": float(delta), "magnitude": mag}


def rank_biserial_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (Wilcoxon effect size)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = _sp.rankdata(np.abs(d))
    r_pos = np.sum(ranks[d > 0])
    r_neg = np.sum(ranks[d < 0])
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total > 0 else 0.0


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    d = d[~np.isnan(d)]
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else 0.0


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict:
    """Holm-Bonferroni step-down correction for a family of named p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        adj = min(1.0, max(prev, p * (m - rank)))
        prev = adj
        out[name] = {"p_raw": float(p), "p_adjusted": float(adj),
                     "threshold": float(thresh), "reject_h0": bool(adj < alpha)}
    return out


def proportion_diff_test(k1: int, n1: int, k2: int, n2: int) -> Dict:
    """Two-proportion comparison via Fisher's exact test (paired-independent)."""
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    odds, p = _sp.fisher_exact(table, alternative="two-sided")
    return {"rate1": k1 / n1 if n1 else float("nan"),
            "rate2": k2 / n2 if n2 else float("nan"),
            "odds_ratio": float(odds), "p_value": float(p)}
