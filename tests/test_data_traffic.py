import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import data_traffic


# Builds a small file in the PEMS-SF layout, scrambles the days with a known
# permutation, and checks the loader can both parse it and work out which way
# round the permutation goes.


def make_days(n_days=30, n_sensors=12, n_cols=144, seed=0):
    """Days with a daily shape that drifts slowly, so neighbouring days
    really are more alike than distant ones."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_cols)
    days = []
    for d in range(n_days):
        phase = 0.12 * d
        base = 1.0 + 0.8 * np.sin(2 * np.pi * t / n_cols + phase)
        day = np.zeros((n_sensors, n_cols), dtype=np.float32)
        for s in range(n_sensors):
            day[s] = base * (0.5 + rng.random()) + rng.normal(0, 0.02, n_cols)
        days.append(day)
    return days


def write_pems(path, days):
    f = open(path, "w")
    for d in days:
        rows = []
        for r in range(d.shape[0]):
            rows.append(" ".join(str(round(float(v), 4)) for v in d[r]))
        f.write("[" + ";".join(rows) + "]\n")
    f.close()


def test_parse_line_shape():
    days = make_days(n_days=3, n_sensors=5, n_cols=144)
    p = os.path.join(TMP, "PEMS_tiny")
    write_pems(p, days)

    back = data_traffic.read_days(p)
    assert len(back) == 3, len(back)
    assert back[0].shape == (5, 144), back[0].shape
    assert np.allclose(back[0], days[0], atol=1e-3)
    print("parse ok", back[0].shape)


def test_randperm_direction_is_detected():
    n = 30
    true_days = make_days(n_days=n, n_sensors=12)

    rng = np.random.default_rng(5)
    perm = rng.permutation(n) + 1  # 1-based, like the real file

    # scrambled[i] = true[perm[i]-1], so putting file entry i back at
    # position perm[i]-1 restores the truth. That is the "forward" rule.
    scrambled = [None] * n
    for i in range(n):
        scrambled[i] = true_days[perm[i] - 1]

    c_true = data_traffic.day_continuity(true_days)
    c_scr = data_traffic.day_continuity(scrambled)
    assert c_true > c_scr, (c_true, c_scr)

    fwd = data_traffic.apply_perm(scrambled, list(perm), "forward")
    bwd = data_traffic.apply_perm(scrambled, list(perm), "backward")

    c_f = data_traffic.day_continuity(fwd)
    c_b = data_traffic.day_continuity(bwd)

    print(
        "continuity: true", round(c_true, 4),
        "scrambled", round(c_scr, 4),
        "forward", round(c_f, 4),
        "backward", round(c_b, 4),
    )

    # forward is the correct undo here, and the loader picks the higher one
    assert c_f > c_b, (c_f, c_b)
    assert abs(c_f - c_true) < 1e-9, (c_f, c_true)
    print("randperm direction detected ok")


def test_hourly_aggregation_shape():
    days = make_days(n_days=4, n_sensors=7, n_cols=144)
    hourly = []
    for d in days:
        hourly.append(d.reshape(d.shape[0], 24, 6).sum(axis=2))
    y = np.concatenate(hourly, axis=1)
    assert y.shape == (7, 4 * 24), y.shape
    # summing 6 readings must give roughly 6x one reading
    assert y.max() > days[0].max() * 3
    print("hourly aggregation ok", y.shape)


if __name__ == "__main__":
    TMP = os.path.join(ROOT, "data", "raw", "_test_tmp")
    os.makedirs(TMP, exist_ok=True)
    test_parse_line_shape()
    test_randperm_direction_is_detected()
    test_hourly_aggregation_shape()
    import shutil

    shutil.rmtree(TMP, ignore_errors=True)
    print("all traffic loader tests passed")
