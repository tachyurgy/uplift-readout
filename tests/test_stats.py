import numpy as np
import pytest

from uplift_readout import stats, sequential, uplift


def test_srm_detects_mismatch_and_passes_design():
    ok = stats.srm_test(850_000, 150_000, 0.85)
    assert ok.p_value > 0.99
    bad = stats.srm_test(851_500, 148_500, 0.85)  # 0.15pp off on 1M users
    assert bad.p_value < 1e-3


def test_srm_false_positive_rate_is_nominal():
    rng = np.random.default_rng(1)
    n = 100_000
    nt = rng.binomial(n, 0.85, size=4000)
    p = np.array([stats.srm_test(int(a), int(n - a), 0.85).p_value for a in nt])
    assert abs((p < 0.05).mean() - 0.05) < 0.012


def _sim(n, rng, effect=0.3):
    x = rng.normal(size=(n, 2))
    t = rng.random(n) < 0.85
    y = 1.0 + 2.0 * x[:, 0] - 1.0 * x[:, 1] + effect * t + rng.normal(size=n)
    return x, t, y


@pytest.mark.parametrize("per_arm", [False, True])
def test_cuped_is_unbiased_and_covers(per_arm):
    rng = np.random.default_rng(7)
    ests, cover = [], []
    for _ in range(400):
        x, t, y = _sim(2000, rng)
        e = stats.cuped_ate(y, t, x, per_arm_theta=per_arm)
        ests.append(e.estimate)
        cover.append(e.ci_low <= 0.3 <= e.ci_high)
    ests = np.array(ests)
    # bias well inside Monte Carlo error
    assert abs(ests.mean() - 0.3) < 4 * ests.std() / np.sqrt(len(ests))
    assert 0.92 <= np.mean(cover) <= 0.98


def test_cuped_shrinks_variance():
    rng = np.random.default_rng(3)
    x, t, y = _sim(20_000, rng)
    raw = stats.diff_in_means(y[t].mean(), y[t].var(ddof=1), t.sum(), y[~t].mean(), y[~t].var(ddof=1), (~t).sum())
    adj = stats.cuped_ate(y, t, x)
    # R^2 of x on y is 5/6, so SE should shrink by ~sqrt(1/6)
    assert adj.se / raw.se == pytest.approx(np.sqrt(1 / 6), rel=0.05)


def test_binomial_bootstrap_matches_analytic_se():
    b = stats.bootstrap_binary_ate(4_000, 100_000, 3_000, 100_000, n_boot=20_000)
    analytic = np.sqrt(0.04 * 0.96 / 1e5 + 0.03 * 0.97 / 1e5)
    assert b["ate_se"] == pytest.approx(analytic, rel=0.03)


def test_holm_and_bh():
    p = [0.01, 0.04, 0.03, 0.5]
    assert np.allclose(stats.holm(p), [0.04, 0.09, 0.09, 0.5])
    bh = stats.benjamini_hochberg(p)
    assert np.allclose(bh, [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert all(bh <= stats.holm(p) + 1e-12)


def test_sample_size_matches_textbook():
    # 10% -> 12% at 50/50, alpha .05 power .8 needs ~3,840 per arm (unpooled)
    n = stats.sample_size_two_prop(0.10, 0.20, 0.5)
    assert 7_500 <= n <= 7_800


def test_peeking_inflates_and_sequential_controls():
    z = sequential.simulate_z_paths(40_000, 20, 0.0, seed=11)
    naive = sequential.naive_peeking_reject(z).mean()
    bounds = sequential.spending_boundaries(20, n_calib=200_000)
    spend = sequential.spending_reject(z, bounds).mean()
    msprt = sequential.msprt_reject(z, tau2=1.0).mean()
    assert naive > 0.2
    assert abs(spend - 0.05) < 0.006
    assert msprt <= 0.05


def test_uplift_curve_random_is_diagonal():
    rng = np.random.default_rng(0)
    n = 200_000
    t = rng.random(n) < 0.5
    y = (rng.random(n) < 0.1 + 0.02 * t).astype(float)
    grid, curve = uplift.uplift_curve(rng.random(n), t, y)
    assert curve[-1] == pytest.approx(0.02, abs=0.004)
    assert abs(uplift.qini_coefficient(grid, curve)) < 0.25


def test_uplift_curve_rewards_oracle():
    rng = np.random.default_rng(1)
    n = 200_000
    x = rng.random(n)
    t = rng.random(n) < 0.5
    y = (rng.random(n) < 0.05 + 0.08 * x * t).astype(float)
    g, oracle = uplift.uplift_curve(x, t, y)
    assert uplift.qini_coefficient(g, oracle) > 0.3
