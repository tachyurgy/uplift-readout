"""Peeking, alpha spending and mSPRT on simulated experiments.

Experiments are simulated as the z-statistic process of a two-arm test observed
at K equally spaced looks. Under the normal approximation the running
z-statistic at information fraction t_k is Z_k = S_k / sqrt(k) with S_k a
Gaussian random walk plus a drift, which is exact for the difference in means
of two arms with known variance and a very good approximation for the
proportions used here (thousands of conversions per look).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def simulate_z_paths(n_sims: int, looks: int, drift_at_end: float = 0.0, seed: int = 0) -> np.ndarray:
    """Return an (n_sims, looks) array of running z-statistics.

    drift_at_end is the expected z at the final look (0 = A/A test).
    """
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((n_sims, looks)).astype(np.float64)
    inc += drift_at_end / np.sqrt(looks)
    s = np.cumsum(inc, axis=1)
    k = np.arange(1, looks + 1)
    return s / np.sqrt(k)


def naive_peeking_reject(z: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Stop at the first look where |z| crosses the fixed-horizon critical value."""
    crit = sps.norm.ppf(1 - alpha / 2)
    return (np.abs(z) >= crit).any(axis=1)


def obrien_fleming_spending(t: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Lan-DeMets O'Brien-Fleming-type cumulative alpha spent at information fraction t."""
    return 2 * (1 - sps.norm.cdf(sps.norm.ppf(1 - alpha / 2) / np.sqrt(t)))


def spending_boundaries(looks: int, alpha: float = 0.05, n_calib: int = 400_000, seed: int = 12345) -> np.ndarray:
    """Two-sided critical values c_k for the Lan-DeMets OBF spending function.

    Calibrated on an independent set of null paths: at each look c_k is the
    quantile that makes P(first crossing at k) equal the alpha increment.
    """
    t = np.arange(1, looks + 1) / looks
    spend = np.diff(np.concatenate([[0.0], obrien_fleming_spending(t, alpha)]))
    z = np.abs(simulate_z_paths(n_calib, looks, 0.0, seed))
    alive = np.ones(n_calib, bool)
    bounds = np.empty(looks)
    for k in range(looks):
        target = spend[k] * n_calib            # rejections allowed at this look
        zk = np.where(alive, z[:, k], -np.inf)
        if target < 1:
            bounds[k] = np.inf if k < looks - 1 else zk.max()
        else:
            bounds[k] = np.sort(zk)[::-1][int(round(target)) - 1]
        alive &= ~(z[:, k] >= bounds[k])
    return bounds


def spending_reject(z: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return (np.abs(z) >= bounds[None, :]).any(axis=1)


def msprt_reject(z: np.ndarray, tau2: float = 1.0, alpha: float = 0.05) -> np.ndarray:
    """Mixture SPRT (Johari et al. 2017) with a N(0, tau2) mixing prior on the
    standardized effect per look-unit.

    With S_k = sqrt(k) Z_k the sum of k unit-variance increments, the mixture
    likelihood ratio is
        Lambda_k = sqrt(1/(1 + k tau2)) * exp(tau2 S_k^2 / (2 (1 + k tau2)))
    and rejecting when Lambda_k >= 1/alpha controls type I error at alpha under
    continuous monitoring (Ville's inequality).
    """
    k = np.arange(1, z.shape[1] + 1)[None, :]
    s = z * np.sqrt(k)
    log_lr = 0.5 * np.log(1 / (1 + k * tau2)) + tau2 * s**2 / (2 * (1 + k * tau2))
    return (log_lr >= np.log(1 / alpha)).any(axis=1)


def first_crossing_look(z: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Look index (1-based) of the first crossing, or 0 for none."""
    hit = np.abs(z) >= bounds[None, :]
    idx = hit.argmax(axis=1) + 1
    idx[~hit.any(axis=1)] = 0
    return idx
