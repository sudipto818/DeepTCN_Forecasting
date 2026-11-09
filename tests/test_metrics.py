import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import metrics


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_pinball():
    # y below the forecast: loss = (1-q) * (yhat - y) = 0.1 * 2 = 0.2
    v = float(metrics.pinball(np.array(3.0), np.array(5.0), 0.9))
    assert close(v, 0.2), v

    # y above the forecast: loss = q * (y - yhat) = 0.9 * 2 = 1.8
    v = float(metrics.pinball(np.array(7.0), np.array(5.0), 0.9))
    assert close(v, 1.8), v

    # exact hit
    v = float(metrics.pinball(np.array(5.0), np.array(5.0), 0.5))
    assert close(v, 0.0), v
    print("pinball ok")


def test_quantile_loss():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    yhat = np.array([[1.0, 1.0], [3.0, 3.0]])
    # q=0.5: errors 0, 1, 0, 1 -> pinball 0, 0.5, 0, 0.5 = 1.0; sum|y| = 10
    v = metrics.quantile_loss(y, yhat, 0.5)
    assert close(v, 0.1), v
    print("quantile_loss ok")


def test_smape():
    y = np.array([4.0, 0.0])
    yhat = np.array([6.0, 0.0])
    # first: |2*(4-6)| / (4+6) = 4/10 = 0.4; second: 0/0 -> 0
    v = metrics.smape(y, yhat)
    assert close(v, 0.2), v
    print("smape ok")


def test_nrmse():
    y = np.array([2.0, 4.0])
    yhat = np.array([1.0, 5.0])
    # rmse = sqrt((1+1)/2) = 1; mean|y| = 3
    v = metrics.nrmse(y, yhat)
    assert close(v, 1.0 / 3.0), v
    print("nrmse ok")


def test_mase():
    # a series that repeats every 2 steps has zero seasonal error, so use
    # period 1 on a ramp: |y[t]-y[t-1]| = 1 everywhere
    y_train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    s = metrics.mase_scale(y_train, period=1)
    assert close(float(s[0]), 1.0), s

    y = np.array([[[10.0, 10.0]]])
    yhat = np.array([[[8.0, 12.0]]])
    # errors 2 and 2, scale 1 -> 2.0
    v = metrics.mase(y, yhat, s)
    assert close(v, 2.0), v
    print("mase ok")


def test_mase_scale_is_seasonal_naive():
    """The denominator is the in-sample seasonal naive error, so a series
    that repeats exactly every period gets the floor, and a level shift of
    3 between days gives 3 on the shifted steps."""
    day = np.arange(24, dtype=float)
    flat = np.tile(day, 4)[None, :]
    assert float(metrics.mase_scale(flat, 24)[0]) <= 1e-6

    shifted = np.concatenate([day, day + 3.0])[None, :]
    assert close(float(metrics.mase_scale(shifted, 24)[0]), 3.0)
    print("mase scale ok")


def test_coverage_width():
    y = np.array([[[1.0, 5.0, 9.0]]])
    lo = np.array([[[0.0, 0.0, 0.0]]])
    hi = np.array([[[2.0, 2.0, 2.0]]])
    # only the first is inside
    assert close(metrics.coverage(y, lo, hi), 1.0 / 3.0)
    # width 2 each, mean|y| = 5
    assert close(metrics.width(y, lo, hi), 2.0 / 5.0)
    print("coverage/width ok")


def test_winkler():
    alpha = 0.2
    # inside: just the width
    y = np.array([[[5.0]]])
    lo = np.array([[[0.0]]])
    hi = np.array([[[10.0]]])
    v = metrics.winkler(y, lo, hi, alpha)
    assert close(v, 10.0 / 5.0), v

    # below: width + (2/alpha)*(lo-y) = 10 + 10*1 = 20, mean|y| = 1
    y = np.array([[[-1.0]]])
    v = metrics.winkler(y, lo, hi, alpha)
    assert close(v, 20.0 / 1.0), v

    # above: width + (2/alpha)*(y-hi) = 10 + 10*2 = 30, mean|y| = 12
    y = np.array([[[12.0]]])
    v = metrics.winkler(y, lo, hi, alpha)
    assert close(v, 30.0 / 12.0), v
    print("winkler ok")


def test_crossing():
    # one forecast with 0.5 above 0.9
    p = np.zeros((1, 1, 1, 3))
    p[0, 0, 0] = [1.0, 5.0, 3.0]
    assert close(metrics.crossing_rate(p), 1.0)

    p[0, 0, 0] = [1.0, 2.0, 3.0]
    assert close(metrics.crossing_rate(p), 0.0)
    print("crossing ok")


if __name__ == "__main__":
    test_pinball()
    test_quantile_loss()
    test_smape()
    test_nrmse()
    test_mase()
    test_mase_scale_is_seasonal_naive()
    test_coverage_width()
    test_winkler()
    test_crossing()
    print("all metrics tests passed")
