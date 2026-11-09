import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import cqr


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_offset_order_statistic():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # n=5, alpha=0.2 -> k = ceil(0.8*6) = 5 -> the 5th smallest
    assert close(cqr.conformal_offset(s, 0.2), 5.0)

    # alpha=0.5 -> k = ceil(0.5*6) = 3 -> the 3rd smallest
    assert close(cqr.conformal_offset(s, 0.5), 3.0)

    # order must not matter
    assert close(cqr.conformal_offset(s[::-1], 0.5), 3.0)
    print("offset order statistic ok")


def test_offset_edges():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # alpha <= 0 -> k > n -> whole real line
    assert cqr.conformal_offset(s, 0.0) == np.inf
    assert cqr.conformal_offset(s, -0.3) == np.inf

    # alpha >= 1 -> k < 1 -> empty set
    assert cqr.conformal_offset(s, 1.0) == -np.inf
    assert cqr.conformal_offset(s, 1.5) == -np.inf
    print("offset edges ok")


def test_score_sign():
    # inside the interval the score is negative
    y = np.array([[[5.0]]])
    lo = np.array([[[0.0]]])
    hi = np.array([[[10.0]]])
    e = cqr.scores_raw(y, lo, hi)
    assert e[0, 0, 0] == -5.0, e

    # above the interval it is positive
    y = np.array([[[13.0]]])
    e = cqr.scores_raw(y, lo, hi)
    assert e[0, 0, 0] == 3.0, e
    print("score sign ok")


def test_shrinks_too_wide_interval():
    """An over-wide interval must come back narrower, not wider."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=(50, 20, 4))
    lo = np.full(y.shape, -10.0)
    hi = np.full(y.shape, 10.0)

    q = cqr.fit_pooled(y, lo, hi, 0.2)
    assert q < 0, q

    lo2, hi2 = cqr.apply_pooled(lo, hi, q)
    assert (hi2 - lo2).mean() < (hi - lo).mean()
    print("shrink ok, offset", round(float(q), 3))


def test_coverage_on_exchangeable_data():
    """With exchangeable data the calibrated interval should cover about
    1-alpha. Averaged over repeats it must not fall below it."""
    alpha = 0.2
    rng = np.random.default_rng(1)
    covs = []

    for rep in range(200):
        cal = rng.normal(0, 1, size=(200, 1, 1))
        test = rng.normal(0, 1, size=(200, 1, 1))

        lo_c = np.zeros_like(cal)
        hi_c = np.zeros_like(cal)
        lo_t = np.zeros_like(test)
        hi_t = np.zeros_like(test)

        q = cqr.fit_pooled(cal, lo_c, hi_c, alpha)
        l, h = cqr.apply_pooled(lo_t, hi_t, q)
        covs.append(((test >= l) & (test <= h)).mean())

    m = float(np.mean(covs))
    assert m >= 1.0 - alpha - 0.01, m
    assert m <= 1.0 - alpha + 0.05, m
    print("exchangeable coverage ok, mean", round(m, 4))


def test_horizon_offsets_grow():
    """If the error grows with horizon, the per-horizon offsets should too."""
    rng = np.random.default_rng(2)
    n_o, n_s, h = 40, 10, 6
    y = np.zeros((n_o, n_s, h))
    for w in range(h):
        y[:, :, w] = rng.normal(0, 1.0 + w, size=(n_o, n_s))

    lo = np.full(y.shape, -1.0)
    hi = np.full(y.shape, 1.0)
    scale = np.ones(n_s)

    q = cqr.fit_horizon(y, lo, hi, scale, 0.2)
    assert q[-1] > q[0], q
    print("horizon offsets grow ok", np.round(q, 2))


if __name__ == "__main__":
    test_offset_order_statistic()
    test_offset_edges()
    test_score_sign()
    test_shrinks_too_wide_interval()
    test_coverage_on_exchangeable_data()
    test_horizon_offsets_grow()
    print("all cqr tests passed")
