"""Core experiment statistics.

Every estimator here works either on raw arrays (for tests and small data) or on
sufficient statistics (counts, sums, cross-products) so that the 14M-row readout
never has to materialise the full table in memory.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------- SRM
@dataclass
class SRMResult:
    n_treat: int
    n_control: int
    expected_share: float
    observed_share: float
    chi2: float
    p_value: float

    def to_dict(self):
        return asdict(self)


def srm_test(n_treat: int, n_control: int, expected_share: float) -> SRMResult:
    """Chi-square goodness-of-fit test of the observed split vs the design split."""
    n = n_treat + n_control
    exp_t, exp_c = n * expected_share, n * (1 - expected_share)
    chi2 = (n_treat - exp_t) ** 2 / exp_t + (n_control - exp_c) ** 2 / exp_c
    p = float(sps.chi2.sf(chi2, df=1))
    return SRMResult(int(n_treat), int(n_control), expected_share, n_treat / n, float(chi2), p)


# --------------------------------------------------------------------------- ATE
@dataclass
class Effect:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float

    def to_dict(self):
        return asdict(self)


def _effect(est: float, se: float, alpha: float = 0.05) -> Effect:
    z = sps.norm.ppf(1 - alpha / 2)
    p = float(2 * sps.norm.sf(abs(est / se))) if se > 0 else float("nan")
    return Effect(float(est), float(se), float(est - z * se), float(est + z * se), p)


def diff_in_means(mean_t, var_t, n_t, mean_c, var_c, n_c, alpha=0.05) -> Effect:
    """Welch/Neyman difference in means from per-arm moments."""
    se = np.sqrt(var_t / n_t + var_c / n_c)
    return _effect(mean_t - mean_c, se, alpha)


def relative_lift(mean_t, var_t, n_t, mean_c, var_c, n_c, alpha=0.05) -> Effect:
    """Relative lift (mean_t / mean_c - 1) with a delta-method SE on the log ratio.

    The CI is built on log(mean_t/mean_c) and exponentiated, which keeps it
    asymmetric and inside (-1, inf) as it should be for a ratio.
    """
    se_log = np.sqrt(var_t / (n_t * mean_t**2) + var_c / (n_c * mean_c**2))
    log_r = np.log(mean_t / mean_c)
    z = sps.norm.ppf(1 - alpha / 2)
    p = float(2 * sps.norm.sf(abs(log_r / se_log)))
    return Effect(float(np.exp(log_r) - 1), float(se_log * mean_t / mean_c),
                  float(np.exp(log_r - z * se_log) - 1), float(np.exp(log_r + z * se_log) - 1), p)


def bootstrap_binary_ate(k_t: int, n_t: int, k_c: int, n_c: int, n_boot: int = 10_000,
                         alpha: float = 0.05, seed: int = 0) -> dict:
    """Nonparametric bootstrap of a difference in proportions.

    Resampling n Bernoulli rows with replacement from an arm with k successes
    gives a success count distributed exactly Binomial(n, k/n), so the row-level
    bootstrap can be drawn without touching the rows.
    """
    rng = np.random.default_rng(seed)
    pt = rng.binomial(n_t, k_t / n_t, size=n_boot) / n_t
    pc = rng.binomial(n_c, k_c / n_c, size=n_boot) / n_c
    diff = pt - pc
    rel = pt / pc - 1
    q = [alpha / 2, 1 - alpha / 2]
    return {
        "ate_ci": [float(x) for x in np.quantile(diff, q)],
        "ate_se": float(diff.std(ddof=1)),
        "rel_ci": [float(x) for x in np.quantile(rel, q)],
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- CUPED / regression adjustment
def cuped_theta(cov_xx: np.ndarray, cov_xy: np.ndarray) -> np.ndarray:
    """Multivariate CUPED coefficient: theta = Sigma_xx^{-1} Sigma_xy (pooled over arms)."""
    return np.linalg.solve(np.atleast_2d(cov_xx), np.atleast_1d(cov_xy))


def cuped_from_moments(arms: dict, alpha: float = 0.05, per_arm_theta: bool = False) -> Effect:
    """Variance-reduced ATE from per-arm moments.

    arms = {"t": {...}, "c": {...}} with keys n, mean_y, var_y, mean_x (p,),
    cov_xx (p,p), cov_xy (p,). With per_arm_theta=False this is classic CUPED
    (Deng et al. 2013) with a pooled theta. With per_arm_theta=True it is the
    fully interacted regression estimator of Lin (2013): each arm gets its own
    slope and is re-centred on the pooled covariate mean.
    """
    t, c = arms["t"], arms["c"]
    n = t["n"] + c["n"]
    xbar = (t["n"] * t["mean_x"] + c["n"] * c["mean_x"]) / n
    if per_arm_theta:
        th_t = cuped_theta(t["cov_xx"], t["cov_xy"])
        th_c = cuped_theta(c["cov_xx"], c["cov_xy"])
    else:
        # pooled within-arm covariance, so the treatment shift in means does not leak in
        cov_xx = ((t["n"] - 1) * t["cov_xx"] + (c["n"] - 1) * c["cov_xx"]) / (n - 2)
        cov_xy = ((t["n"] - 1) * t["cov_xy"] + (c["n"] - 1) * c["cov_xy"]) / (n - 2)
        th_t = th_c = cuped_theta(cov_xx, cov_xy)

    def adj(a, th):
        m = a["mean_y"] - th @ (a["mean_x"] - xbar)
        v = a["var_y"] - 2 * th @ a["cov_xy"] + th @ a["cov_xx"] @ th
        return m, max(v, 0.0)

    mt, vt = adj(t, th_t)
    mc, vc = adj(c, th_c)
    return diff_in_means(mt, vt, t["n"], mc, vc, c["n"], alpha)


def moments_from_arrays(y: np.ndarray, X: np.ndarray) -> dict:
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    return {
        "n": len(y),
        "mean_y": float(y.mean()),
        "var_y": float(y.var(ddof=1)),
        "mean_x": X.mean(axis=0),
        "cov_xx": np.atleast_2d(np.cov(X, rowvar=False)),
        "cov_xy": np.array([np.cov(X[:, j], y)[0, 1] for j in range(X.shape[1])]),
    }


def cuped_ate(y: np.ndarray, t: np.ndarray, X: np.ndarray, alpha=0.05, per_arm_theta=False) -> Effect:
    """Array front-end to cuped_from_moments (used by the tests)."""
    t = t.astype(bool)
    X = X.reshape(len(y), -1)
    arms = {"t": moments_from_arrays(y[t], X[t]), "c": moments_from_arrays(y[~t], X[~t])}
    return cuped_from_moments(arms, alpha, per_arm_theta)


# --------------------------------------------------------------------------- multiple comparisons
def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


# --------------------------------------------------------------------------- power
def sample_size_two_prop(p_c: float, rel_mde: float, share_t: float = 0.5,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Total N to detect a relative lift rel_mde on baseline p_c (two-sided z-test,
    unpooled variance, allocation share_t to treatment)."""
    p_t = p_c * (1 + rel_mde)
    za, zb = sps.norm.ppf(1 - alpha / 2), sps.norm.ppf(power)
    var_unit = p_t * (1 - p_t) / share_t + p_c * (1 - p_c) / (1 - share_t)
    return int(np.ceil((za + zb) ** 2 * var_unit / (p_t - p_c) ** 2))


def mde_two_prop(p_c: float, n_total: int, share_t: float = 0.5,
                 alpha: float = 0.05, power: float = 0.8) -> float:
    """Absolute MDE for a given total N (variance evaluated at the baseline)."""
    za, zb = sps.norm.ppf(1 - alpha / 2), sps.norm.ppf(power)
    var_unit = p_c * (1 - p_c) * (1 / share_t + 1 / (1 - share_t))
    return float((za + zb) * np.sqrt(var_unit / n_total))
