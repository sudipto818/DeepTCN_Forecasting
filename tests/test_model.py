import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

# This file needs torch. Run it once on the machine that will train, before
# starting a long run. Everything it checks is cheap.

import torch

import config
from losses import pinball_loss
from model import DeepTCN


def build(n_series=6, channels=8, dilations=None):
    if dilations is None:
        dilations = [1, 2, 4]
    torch.manual_seed(0)
    return DeepTCN(n_series, dilations, channels, 2, len(config.QUANTILES))


def fake_batch(n_series=6, b=4):
    rng = np.random.default_rng(0)
    y = torch.tensor(
        rng.random((b, config.IN_LEN)).astype(np.float32)
    )
    s = torch.tensor(rng.integers(0, n_series, size=b), dtype=torch.long)
    ch = torch.tensor(
        rng.integers(0, [24, 7, 12], size=(b, config.IN_LEN, 3)), dtype=torch.long
    )
    cf = torch.tensor(
        rng.integers(0, [24, 7, 12], size=(b, config.OUT_LEN, 3)), dtype=torch.long
    )
    return y, s, ch, cf


def test_output_shape():
    m = build()
    m.eval()
    y, s, ch, cf = fake_batch()
    with torch.no_grad():
        out = m(y, s, ch, cf)
    assert out.shape == (4, config.OUT_LEN, len(config.QUANTILES)), out.shape
    print("output shape ok", tuple(out.shape))


def test_output_non_negative():
    m = build()
    m.eval()
    y, s, ch, cf = fake_batch()
    with torch.no_grad():
        out = m(y, s, ch, cf)
    assert (out >= 0).all(), out.min()
    print("non-negative ok")


def test_encoder_is_causal():
    """Changing the LAST input value must change the output, but changing
    nothing else about the future may leak backwards. The real check: the
    encoder summary must not depend on anything after the origin, and there
    is nothing after the origin in y_hist, so instead we check that the
    convolution stack keeps its length and that a change at position t only
    affects positions >= t."""
    m = build()
    m.eval()

    b = 1
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.random((b, 8, config.IN_LEN)).astype(np.float32))

    with torch.no_grad():
        a = x
        for blk in m.blocks:
            a = blk(a)

        x2 = x.clone()
        x2[:, :, 100] += 5.0
        c = x2
        for blk in m.blocks:
            c = blk(c)

    assert a.shape == x.shape, (a.shape, x.shape)

    d = (a - c).abs().sum(dim=1)[0]
    # nothing before position 100 may move
    assert d[:100].max() < 1e-5, d[:100].max()
    # position 100 itself must move
    assert d[100] > 1e-5, d[100]
    print("encoder causal ok, length kept", a.shape[2])


def test_future_covariates_matter():
    m = build()
    m.eval()
    y, s, ch, cf = fake_batch()
    with torch.no_grad():
        o1 = m(y, s, ch, cf)
        cf2 = cf.clone()
        cf2[:, :, 0] = (cf2[:, :, 0] + 7) % 24
        o2 = m(y, s, ch, cf2)
    assert (o1 - o2).abs().max() > 1e-6, "future covariates are being ignored"
    print("future covariates matter ok")


def test_loss_decreases_on_one_batch():
    """Overfit a single batch. If the loss does not fall, something in the
    graph is broken."""
    m = build()
    m.train()
    y, s, ch, cf = fake_batch()
    tgt = torch.tensor(
        np.random.default_rng(2)
        .random((4, config.OUT_LEN))
        .astype(np.float32)
    )

    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    first = None
    for i in range(60):
        pred = m(y, s, ch, cf)
        loss = pinball_loss(tgt, pred, config.QUANTILES)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
    last = float(loss)
    assert last < first, (first, last)
    print("loss decreases ok", round(first, 4), "->", round(last, 4))


def test_pinball_matches_hand_value():
    y = torch.tensor([[3.0]])
    pred = torch.tensor([[[5.0]]])
    v = float(pinball_loss(y, pred, [0.9]))
    assert abs(v - 0.2) < 1e-6, v
    print("pinball ok")


def test_quantiles_come_out_roughly_ordered():
    """Not guaranteed by the architecture, so this only reports the rate.
    A high number here is the argument for Paper 3."""
    m = build()
    m.eval()
    y, s, ch, cf = fake_batch(b=64)
    with torch.no_grad():
        out = m(y, s, ch, cf).numpy()
    bad = (out[:, :, 0] > out[:, :, 1]) | (out[:, :, 1] > out[:, :, 2])
    print("crossing rate on an untrained model:", round(float(bad.mean()), 4))


if __name__ == "__main__":
    test_output_shape()
    test_output_non_negative()
    test_encoder_is_causal()
    test_future_covariates_matter()
    test_pinball_matches_hand_value()
    test_loss_decreases_on_one_batch()
    test_quantiles_come_out_roughly_ordered()
    print("all model tests passed")
