"""Meta-learners for conditional average treatment effects and their evaluation.

Treatment assignment in the Criteo test is randomized with a known propensity,
so the X-learner and the transformed-outcome learner use that constant instead
of an estimated propensity model.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=500,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
    n_jobs=4, verbose=-1,
)


def _clf(seed):
    return lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)


def _reg(seed):
    return lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)


def s_learner(X, t, y, X_new, seed=0):
    m = _clf(seed).fit(np.column_stack([X, t]), y)
    p1 = m.predict_proba(np.column_stack([X_new, np.ones(len(X_new))]))[:, 1]
    p0 = m.predict_proba(np.column_stack([X_new, np.zeros(len(X_new))]))[:, 1]
    return p1 - p0


def t_learner(X, t, y, X_new, seed=0, return_models=False):
    tb = t.astype(bool)
    m1 = _clf(seed).fit(X[tb], y[tb])
    m0 = _clf(seed).fit(X[~tb], y[~tb])
    tau = m1.predict_proba(X_new)[:, 1] - m0.predict_proba(X_new)[:, 1]
    return (tau, m1, m0) if return_models else tau


def x_learner(X, t, y, X_new, propensity, seed=0, outcome_models=None):
    """Kunzel et al. (2019). Stage 1 = T-learner outcome models (pass them in to
    reuse a fitted T-learner); stage 2 regresses imputed individual effects in
    each arm; the two are blended with the known propensity."""
    tb = t.astype(bool)
    if outcome_models is None:
        _, m1, m0 = t_learner(X, t, y, X[:1], seed, return_models=True)
    else:
        m1, m0 = outcome_models
    d1 = y[tb] - m0.predict_proba(X[tb])[:, 1]
    d0 = m1.predict_proba(X[~tb])[:, 1] - y[~tb]
    g1 = _reg(seed).fit(X[tb], d1)
    g0 = _reg(seed).fit(X[~tb], d0)
    # g(x) = e g0(x) + (1-e) g1(x): with e = 0.85 most weight goes to g0, whose
    # imputations use m1, the outcome model fitted on the large treated arm
    return propensity * g0.predict(X_new) + (1 - propensity) * g1.predict(X_new)


def transformed_outcome(X, t, y, X_new, propensity, seed=0):
    """Class-transformation / transformed-outcome learner (Athey & Imbens 2015):
    Z = y * (t/e - (1-t)/(1-e)) has E[Z|X] = tau(X) under randomization."""
    z = y * (t / propensity - (1 - t) / (1 - propensity))
    return _reg(seed).fit(X, z).predict(X_new)


# --------------------------------------------------------------------------- evaluation
def uplift_curve(tau_hat, t, y, grid=None):
    """Held-out uplift curve.

    Sort by predicted uplift (descending). For the top fraction phi of users,
    incremental outcomes per user in the population = phi * (mean_y_t - mean_y_c)
    within that top slice. Returns grid and curve (per-user scale, i.e. divided by N).
    Random targeting gives phi * ATE.
    """
    if grid is None:
        grid = np.linspace(0, 1, 101)
    order = np.argsort(-tau_hat, kind="stable")
    t_s, y_s = t[order].astype(float), y[order].astype(float)
    ct, cc = np.cumsum(t_s), np.cumsum(1 - t_s)
    yt, yc = np.cumsum(y_s * t_s), np.cumsum(y_s * (1 - t_s))
    n = len(t)
    out = np.zeros(len(grid))
    for i, phi in enumerate(grid):
        k = int(round(phi * n))
        if k == 0 or ct[k - 1] == 0 or cc[k - 1] == 0:
            continue
        out[i] = phi * (yt[k - 1] / ct[k - 1] - yc[k - 1] / cc[k - 1])
    return grid, out


def auuc(grid, curve):
    return float(np.trapezoid(curve, grid))


def qini_coefficient(grid, curve):
    """Area between the model's uplift curve and the random-targeting diagonal,
    normalised by the area of the diagonal (so 0 = random, 1 = twice as good)."""
    random_curve = grid * curve[-1]
    a_rand = np.trapezoid(random_curve, grid)
    return float((np.trapezoid(curve, grid) - a_rand) / a_rand)


def policy_value(tau_hat, t, y, fractions, value_per_outcome, cost_per_treated):
    """Net value per user in the population of treating the top-k fraction by tau_hat.

    net(phi) = phi * (value * uplift_in_slice - cost), estimated on held-out
    randomized data (difference in means inside the slice).
    """
    grid, curve = uplift_curve(tau_hat, t, y, np.asarray(fractions))
    return value_per_outcome * curve - cost_per_treated * grid
