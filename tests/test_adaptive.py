import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

# Needs torch. Runs under pytest or directly:
#   python3 -m pytest tests/test_adaptive.py
#   python3 tests/test_adaptive.py

import torch

import data_panel
from adaptive import MODES, GatedTCN, MemoryGate, context
from model import DeepTCN

L, H, NS = 64, 12, 10
DIL = [1, 2, 4, 8, 16]


def build(mode, fixed_len=None, seed=0):
    torch.manual_seed(seed)
    return GatedTCN(mode, NS, DIL, 8, 2, 3, L, positive=False,
                    fixed_len=fixed_len)


def batch(b=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(b, L, generator=g)
    s = torch.randint(0, NS, (b,), generator=g)
    return y, s, torch.zeros(b, L, 3, dtype=torch.long), torch.zeros(b, H, 3, dtype=torch.long)


def encoder_seq(tcn, x):
    """Encoder output for every time step, (B, C, L)."""
    a = tcn.inp(x)
    for blk in tcn.blocks:
        a = blk(a)
    return a


def test_encoder_causal_by_autograd():
    """d enc[:, :, t] / d x[:, :, t+k] == 0 for every t and every k >= 1."""
    tcn = build("series").tcn
    tcn.eval()
    x = torch.randn(2, tcn.inp.in_channels, L, requires_grad=True)
    out = encoder_seq(tcn, x)
    for t in range(L):
        g, = torch.autograd.grad(out[:, :, t].sum(), x, retain_graph=True)
        if t < L - 1:
            assert g[:, :, t + 1 :].abs().max() == 0, t
        # and it does depend on its own step
        assert g[:, :, t].abs().max() > 0, t


def test_gated_path_causal_by_autograd():
    """With a context-free gate (everything except series), the encoder output
    at step t still depends only on y up to t, mask included."""
    for mode in ["none", "fixed", "global"]:
        m = build(mode, fixed_len=20)
        m.eval()
        y, s, _, _ = batch(2)
        y.requires_grad_(True)
        mask, _ = m.gate(y)
        parts = [(y * mask).unsqueeze(1)]
        if m.tcn.n_extra:
            parts.append(mask.unsqueeze(1))
        cov = torch.zeros(2, m.tcn.inp.in_channels - len(parts), L)
        out = encoder_seq(m.tcn, torch.cat(parts + [cov], dim=1))
        for t in range(0, L - 1, 7):
            g, = torch.autograd.grad(out[:, :, t].sum(), y, retain_graph=True)
            assert g[:, t + 1 :].abs().max() == 0, (mode, t)


def test_starts_as_baseline():
    """At initialisation the mask is ~1 everywhere."""
    for mode in ["global", "series"]:
        m = build(mode)
        y, s, _, _ = batch()
        mask, tau = m.gate(y)
        assert mask.min() > 0.97, (mode, float(mask.min()))


def test_baseline_equivalence_when_mask_is_ones():
    """Gated model with an all-ones mask == plain DeepTCN with the same
    weights, to float precision."""
    for mode in ["global", "series"]:
        g = build(mode)
        g.eval()
        # push the mask to exactly one
        with torch.no_grad():
            if mode == "global":
                g.gate.logit.fill_(30.0)
            else:
                g.gate.mlp[2].weight.zero_()
                g.gate.mlp[2].bias.fill_(30.0)
            g.gate.s.fill_(0.5)

        plain = DeepTCN(NS, DIL, 8, 2, 3, positive=False)
        sd = {k: v.clone() for k, v in g.tcn.state_dict().items()}
        w = sd["inp.weight"]
        sd["inp.weight"] = torch.cat([w[:, :1], w[:, 2:]], dim=1)
        plain.load_state_dict(sd)
        plain.eval()

        y, s, ch, cf = batch()
        assert torch.allclose(g(y, s, ch, cf), plain(y, s, ch, cf), atol=1e-6), mode


def test_none_mode_is_plain_deeptcn():
    g = build("none")
    assert g.tcn.n_extra == 0
    assert g.tcn.inp.in_channels == 1 + 17


def test_tau_in_range_and_mask_in_unit_interval():
    for mode in ["global", "series"]:
        m = build(mode)
        for scale in [0.0, 1.0, 100.0]:
            y, s, _, _ = batch(32)
            y = y * scale
            mask, tau = m.gate(y)
            assert (tau > 0).all() and (tau < m.gate.tau_max).all(), mode
            assert (mask >= 0).all() and (mask <= 1).all(), mode
        # also after the gate has been pushed hard either way
        for b in [-50.0, 50.0]:
            with torch.no_grad():
                if mode == "global":
                    m.gate.logit.fill_(b)
                else:
                    m.gate.mlp[2].bias.fill_(b)
            _, tau = m.gate(batch(8)[0])
            assert (tau >= 0).all() and (tau <= m.gate.tau_max).all(), (mode, b)


def test_fixed_mask_keeps_newest_steps():
    m = build("fixed", fixed_len=5)
    mask, tau = m.gate(batch(3)[0])
    assert tau is None
    assert mask[:, -5:].eq(1).all() and mask[:, :-5].eq(0).all()


def test_gradient_reaches_tau():
    for mode in ["global", "series"]:
        m = build(mode)
        m.train()
        y, s, ch, cf = batch(16)
        loss = m(y, s, ch, cf).pow(2).mean() + m.penalty(1e-3)
        loss.backward()
        p = m.gate.logit if mode == "global" else m.gate.mlp[2].bias
        assert p.grad is not None and p.grad.abs() > 0, mode


def test_anneal_moves_temperature():
    g = MemoryGate("global", L)
    g.anneal(0.0, 1.0)
    assert abs(float(g.s) - L / 10) < 1e-6
    g.anneal(1.0, 1.0)
    assert abs(float(g.s) - 1.0) < 1e-6


def test_context_counts_observed_steps():
    y = torch.zeros(2, L)
    y[0, -3:] = torch.tensor([1.0, -2.0, 0.5])
    y[1] = 1.0
    c = context(y)
    n0 = np.expm1(float(c[0, 0]) * np.log1p(L))
    assert abs(n0 - 3) < 1e-4, n0
    assert abs(float(c[0, 1]) - (L - 3) / L) < 1e-6
    assert abs(float(c[1, 1])) < 1e-6


def test_panel_shapes_and_type_d_padding():
    d, types = data_panel.panel(n_per_type=5, in_len=L, out_len=H, seed=0)
    x, y, sid, kind = d["train"]
    assert x.shape == (4 * 5 * 6, L) and y.shape == (4 * 5 * 6, H)
    xd = x[kind == "D"]
    assert (xd[:, :-3] == 0).all() and (xd[:, -3:] != 0).all()
    assert list(types).count("D") == 5


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
    print("all adaptive tests passed")
