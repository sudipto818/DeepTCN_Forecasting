import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from scipy import stats

# Needs torch and scipy, like test_model.py.

import torch
import torch.nn.functional as F

import config
import distributions as D
from train import build_model, make_batch, batch_loss


def inv_softplus(x):
    return float(np.log(np.expm1(x)))


def raw_for(dist, a, b, nu):
    """Raw network output that params() maps to (a, b) at scale nu."""
    if dist == "gaussian":
        r = [a / nu, inv_softplus(b / nu)]
    elif dist == "negbin":
        r = [inv_softplus(a / nu), inv_softplus(b * np.sqrt(nu))]
    else:
        r = [inv_softplus(a / nu), float(np.log(b / (1 - b)))]
    return torch.tensor([[r]], dtype=torch.float64), torch.tensor([nu], dtype=torch.float64)


def test_nll_matches_scipy():
    nu = 3.0
    for y in [0.0, 1.0, 4.0, 17.0]:
        raw, sc = raw_for("gaussian", 5.0, 2.0, nu)
        v = float(D.nll("gaussian", torch.tensor([[y]]), raw, sc))
        assert abs(v + stats.norm.logpdf(y, 5.0, 2.0)) < 1e-4, ("gaussian", y)

        mu, alpha = 6.0, 0.4
        raw, sc = raw_for("negbin", mu, alpha, nu)
        v = float(D.nll("negbin", torch.tensor([[y]]), raw, sc))
        ref = stats.nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu))
        assert abs(v + ref) < 1e-4, ("negbin", y, v, ref)

        lam, pi = 4.0, 0.3
        raw, sc = raw_for("zip", lam, pi, nu)
        v = float(D.nll("zip", torch.tensor([[y]]), raw, sc))
        ref = np.log((pi if y == 0 else 0.0) + (1 - pi) * stats.poisson.pmf(y, lam))
        assert abs(v + ref) < 1e-4, ("zip", y, v, ref)
    print("nll matches scipy ok")


def brute_quantile(pmf, q):
    k = np.arange(0, 2000)
    return float(k[np.searchsorted(np.cumsum(pmf(k)), q - 1e-12)])


def test_quantiles_invert_the_cdf():
    qs = config.QUANTILES
    nu = 2.0
    for mu, alpha in [(0.5, 0.1), (8.0, 0.5), (60.0, 0.05)]:
        raw, sc = raw_for("negbin", mu, alpha, nu)
        got = D.quantiles("negbin", raw.numpy(), sc.numpy(), qs)[0, 0]
        n, p = 1 / alpha, 1 / (1 + alpha * mu)
        for j, q in enumerate(qs):
            ref = brute_quantile(lambda k: stats.nbinom.pmf(k, n, p), q)
            assert abs(got[j] - ref) < 1e-6, ("negbin", mu, alpha, q, got[j], ref)

    for lam, pi in [(3.0, 0.05), (20.0, 0.3), (5.0, 0.6)]:
        raw, sc = raw_for("zip", lam, pi, nu)
        got = D.quantiles("zip", raw.numpy(), sc.numpy(), qs)[0, 0]
        pmf = lambda k: (k == 0) * pi + (1 - pi) * stats.poisson.pmf(k, lam)
        for j, q in enumerate(qs):
            ref = brute_quantile(pmf, q)
            assert abs(got[j] - ref) < 1e-6, ("zip", lam, pi, q, got[j], ref)

    raw, sc = raw_for("gaussian", 5.0, 2.0, nu)
    got = D.quantiles("gaussian", raw.numpy(), sc.numpy(), qs)[0, 0]
    assert np.allclose(got, stats.norm.ppf(qs, 5.0, 2.0), atol=1e-4), got
    print("quantiles invert the cdf ok")


def test_mle_recovers_parameters():
    """Fit one constant raw output to samples by minimising nll. If the
    likelihood or the parameter maps were wrong this would land elsewhere."""
    rng = np.random.default_rng(0)
    n = 20000
    nu = torch.full((n,), 4.0)

    mu, alpha = 12.0, 0.3
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu), size=(n, 1))
    lam, pi = 7.0, 0.25
    yz = np.where(rng.random((n, 1)) < pi, 0, rng.poisson(lam, size=(n, 1)))

    for dist, data, truth in [
        ("negbin", y, (mu, alpha)),
        ("zip", yz, (lam, pi)),
        ("gaussian", rng.normal(3.0, 1.5, size=(n, 1)), (3.0, 1.5)),
    ]:
        w = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        yt = torch.tensor(data, dtype=torch.float32)
        for _ in range(800):
            raw = w.expand(n, 1, 2)
            loss = D.nll(dist, yt, raw, nu)
            opt.zero_grad()
            loss.backward()
            opt.step()
        a, b = D.params(dist, w.detach().expand(1, 1, 2), nu[:1])
        a, b = float(a), float(b)
        if dist == "zip":
            b = float(torch.sigmoid(torch.tensor(b)))
        assert abs(a - truth[0]) / truth[0] < 0.05, (dist, a, truth)
        assert abs(b - truth[1]) / truth[1] < 0.1, (dist, b, truth)
        print("  ", dist, "fit", round(a, 3), round(b, 3), "truth", truth)
    print("mle recovers parameters ok")


def test_each_head_trains_on_one_batch():
    rng = np.random.default_rng(3)
    n_series, t = 5, config.IN_LEN + config.OUT_LEN + 50
    y = (2.0 + rng.gamma(2.0, 1.0, size=(n_series, t))).astype(np.float32)
    y[0, :] = 0.0   # an all-zero series must not blow up the scale
    cal = np.zeros((t, 3), dtype=np.int64)
    si = np.arange(n_series)
    oi = np.full(n_series, config.IN_LEN + 10)

    for dist in D.NAMES:
        torch.manual_seed(0)
        m = build_model("synthetic", n_series, dist)
        m.train()
        yh, yt, s, ch, cf, sc = make_batch(
            y, cal, si, oi, "cpu", False, dist, 10.0
        )
        if D.is_count(dist):
            assert torch.equal(yt, torch.round(yt)), "count target not integer"
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        first = None
        for i in range(60):
            loss = batch_loss(dist, yt, m(yh, s, ch, cf), sc)
            assert torch.isfinite(loss), (dist, i)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i == 0:
                first = float(loss)
        assert float(loss) < first, (dist, first, float(loss))
        print("  ", dist, round(first, 4), "->", round(float(loss), 4))
    print("each head trains ok")


def test_old_quantile_checkpoint_still_loads():
    ckpt = os.path.join(config.ARRAYS, "traffic_seed1.pt")
    if not os.path.exists(ckpt):
        print("no traffic checkpoint, skipped")
        return
    m = build_model("traffic", 963, "quantile")
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    print("old quantile checkpoint loads ok")


if __name__ == "__main__":
    test_nll_matches_scipy()
    test_quantiles_invert_the_cdf()
    test_mle_recovers_parameters()
    test_each_head_trains_on_one_batch()
    test_old_quantile_checkpoint_still_loads()
    print("all distribution tests passed")
