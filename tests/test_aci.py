import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import aci


def setup(n_orig=30, n_series=20, h=4, seed=0, test_sd=1.0):
    rng = np.random.default_rng(seed)
    cal = rng.normal(0, 1.0, size=(40, n_series, h))
    y = rng.normal(0, test_sd, size=(n_orig, n_series, h))
    lo = np.zeros_like(y)
    hi = np.zeros_like(y)
    scale = np.ones(n_series)
    return y, lo, hi, cal, scale


def test_no_look_ahead():
    """Changing outcomes at late origins must not move early alphas."""
    y, lo, hi, cal, scale = setup()
    delay = aci.make_delay(y.shape[2], "hourly")

    _, _, a1, _ = aci.run_aci(y, lo, hi, cal, scale, 0.2, 0.05, delay)

    y2 = y.copy()
    y2[20:] = y2[20:] + 100.0  # make every late origin a miss

    _, _, a2, _ = aci.run_aci(y2, lo, hi, cal, scale, 0.2, 0.05, delay)

    # alpha at origin t, horizon w uses errors up to t - delay[w],
    # so origins up to 20 must be untouched
    assert np.allclose(a1[:21], a2[:21]), "look-ahead leak"
    print("no look-ahead ok")


def test_first_origins_use_base_alpha():
    y, lo, hi, cal, scale = setup()
    h = y.shape[2]
    delay = aci.make_delay(h, "hourly")
    alpha = 0.2

    _, _, a, _ = aci.run_aci(y, lo, hi, cal, scale, alpha, 0.05, delay)

    # nothing is known yet when t < delay[w]
    for w in range(h):
        for t in range(int(delay[w])):
            assert abs(a[t, w] - alpha) < 1e-12, (t, w, a[t, w])
    print("first origins ok")


def test_edges_give_infinite_sets():
    """alpha <= 0 must cover everything, alpha >= 1 must cover nothing."""
    y, lo, hi, cal, scale = setup(n_orig=5)
    delay = aci.make_delay(y.shape[2], "daily")

    # a huge gamma with every origin a hit drives alpha above 1 quickly
    y_hit = np.zeros_like(y)
    l, u, a, e = aci.run_aci(y_hit, lo, hi, cal, scale, 0.2, 5.0, delay)
    assert (a > 1.0).any(), a.max()
    bad = (l == np.inf) & (u == -np.inf)
    assert bad.any(), "expected an empty interval"
    assert e[a > 1.0].min() == 1.0, "empty set must count as a miss"

    # every origin a miss drives alpha below 0
    y_miss = np.full_like(y, 1e9)
    l, u, a, e = aci.run_aci(y_miss, lo, hi, cal, scale, 0.2, 5.0, delay)
    assert (a < 0.0).any(), a.min()
    wide = (l == -np.inf) & (u == np.inf)
    assert wide.any(), "expected the whole line"
    assert e[a < 0.0].max() == 0.0, "whole line must count as a hit"
    print("edge conventions ok")


def test_bound_holds_on_adversarial_sequences():
    """The long-run bound must hold whatever the data does."""
    alpha = 0.2
    for sd in [0.2, 1.0, 5.0, 50.0]:
        for gamma in [0.01, 0.05, 0.2]:
            y, lo, hi, cal, scale = setup(n_orig=60, test_sd=sd)
            delay = aci.make_delay(y.shape[2], "hourly")
            _, _, _, e = aci.run_aci(
                y, lo, hi, cal, scale, alpha, gamma, delay
            )
            bound = aci.coverage_bound(alpha, gamma, delay, e.shape[0])
            gap = abs(e.mean(axis=0) - alpha).max()
            assert gap <= bound + 1e-9, (sd, gamma, gap, bound)
    print("bound holds ok")


def test_reacts_to_a_harder_test_period():
    """If the test period is much noisier than calibration, alpha must fall,
    which is what widens the interval."""
    y, lo, hi, cal, scale = setup(n_orig=60, test_sd=4.0)
    delay = aci.make_delay(y.shape[2], "daily")
    _, _, a, e = aci.run_aci(y, lo, hi, cal, scale, 0.2, 0.05, delay)

    assert a[-1].mean() < a[0].mean(), (a[0].mean(), a[-1].mean())
    print(
        "reacts ok, alpha",
        round(float(a[0].mean()), 3),
        "->",
        round(float(a[-1].mean()), 3),
    )


def test_daily_bound_is_vacuous():
    """With 7 daily origins and the literature gamma the bound says nothing.
    This is the number that justifies the hourly protocol."""
    delay = aci.make_delay(24, "daily")
    b = aci.coverage_bound(0.2, 0.005, delay, 7)
    assert b > 1.0, b
    print("daily bound", round(b, 2), "which is vacuous, as expected")

    b2 = aci.coverage_bound(0.2, 0.05, aci.make_delay(24, "hourly"), 168)
    print("hourly bound at gamma 0.05", round(b2, 3))


if __name__ == "__main__":
    test_no_look_ahead()
    test_first_origins_use_base_alpha()
    test_edges_give_infinite_sets()
    test_bound_holds_on_adversarial_sequences()
    test_reacts_to_a_harder_test_period()
    test_daily_bound_is_vacuous()
    print("all aci tests passed")
