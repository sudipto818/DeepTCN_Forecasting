import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import itertools

import torch

from xseries import XTCN

# python3 -m pytest tests/test_xseries.py   (needs torch)

N, L, H = 12, 168, 24
VARIANTS = list(itertools.product([False, True], [False, True]))   # (attn, tau)


def build(attn, tau, seed=0):
    torch.manual_seed(seed)
    return XTCN(N, L, channels=8, attn=attn, tau=tau)


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(N, L, generator=g)
    s = torch.arange(N)
    ch = torch.stack([torch.arange(L) % 24, (torch.arange(L) // 24) % 7], 1)
    cf = torch.stack([torch.arange(H) % 24, (torch.arange(H) // 24) % 7], 1)
    return y, s, ch.expand(N, -1, -1), cf.expand(N, -1, -1)


def test_causal_by_autograd_every_variant():
    """d enc[:, :, t] / d y[:, t+k] == 0 for all k >= 1, every variant."""
    for attn, tau in VARIANTS:
        m = build(attn, tau)
        m.eval()
        y, s, ch, _ = batch()
        y.requires_grad_(True)
        out = m.encode(y, s, ch)
        for t in range(L):
            g, = torch.autograd.grad(out[:, :, t].sum(), y, retain_graph=True)
            if t < L - 1:
                assert g[:, t + 1 :].abs().max() == 0, (attn, tau, t)
            assert g[:, t].abs().max() > 0, (attn, tau, t)


def test_baseline_equivalence_at_init():
    """alpha = 0 and mask ~ all ones -> same output as the baseline with the
    same weights."""
    base = build(False, False)
    base.eval()
    y, s, ch, cf = batch()
    want = base(y, s, ch, cf)
    for attn, tau in VARIANTS[1:]:
        m = build(attn, tau)
        sd = base.state_dict()
        mine = m.state_dict()
        for k, v in sd.items():
            if k == "inp.weight" and tau:
                # mask channel sits at index 1 and starts at zero
                mine[k] = torch.cat([v[:, :1], mine[k][:, 1:2], v[:, 1:]], 1)
            else:
                mine[k] = v
        m.load_state_dict(mine)
        if tau:
            with torch.no_grad():
                m.tau_mlp[2].bias.fill_(30.0)   # mask exactly one
                m.s.fill_(0.5)
            assert m.mask(s).min() > 1 - 1e-6
        m.eval()
        assert torch.allclose(m(y, s, ch, cf), want, atol=1e-5), (attn, tau)


def test_initial_mask_is_open_and_alpha_zero():
    m = build(True, True)
    assert m.alpha.item() == 0.0
    assert m.mask(torch.arange(N)).min() > 0.97


def test_no_cross_series_leak_when_alpha_zero():
    m = build(True, False)
    m.eval()
    y, s, ch, cf = batch()
    y.requires_grad_(True)
    g, = torch.autograd.grad(m(y, s, ch, cf)[0].sum(), y)
    assert g[1:].abs().max() == 0


def test_attention_rows_and_tau_range():
    m = build(True, True)
    s = torch.arange(N)
    A = m.attention(s)
    assert torch.allclose(A.sum(1), torch.ones(N))
    assert A.diagonal().abs().max() == 0
    t = m.tau(s)
    assert (t > 0).all() and (t < m.tau_max).all()


def test_gradient_reaches_alpha_and_tau():
    m = build(True, True)
    m.train()
    y, s, ch, cf = batch()
    m(y, s, ch, cf).pow(2).mean().backward()
    assert m.alpha.grad.abs() > 0
    assert m.tau_mlp[2].bias.grad.abs() > 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
    print("all xseries tests passed")
