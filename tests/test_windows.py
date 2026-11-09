import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

import config
import windows


def test_split_counts():
    sp = windows.split_origins(2400, "daily")
    assert len(sp["test"]) == config.TEST_DAYS, len(sp["test"])
    assert len(sp["calib"]) == config.CALIB_DAYS, len(sp["calib"])
    assert len(sp["val"]) == config.VAL_DAYS, len(sp["val"])
    assert len(sp["train"]) > 0
    print("split counts ok")


def test_test_block_ends_at_series_end():
    """The last test target must be the very last hour of the series."""
    n = 2400
    sp = windows.split_origins(n, "daily")
    last_target = int(sp["test"][-1]) + config.OUT_LEN
    assert last_target == n - 1, last_target

    # and the first test target starts exactly TEST_DAYS back
    first_target = int(sp["test"][0]) + 1
    assert first_target == n - config.TEST_DAYS * 24, first_target
    print("test block ends at series end ok")


def test_no_overlap():
    sp = windows.split_origins(2400, "daily")
    assert windows.check_splits(sp)

    # a deliberately broken split must be caught
    bad = dict(sp)
    bad["calib"] = np.concatenate([sp["calib"], sp["test"][:1]])
    try:
        windows.check_splits(bad)
        raise AssertionError("overlap was not detected")
    except AssertionError as e:
        if "was not detected" in str(e):
            raise
    print("overlap detection ok")


def test_hourly_gives_more_origins():
    sp_d = windows.split_origins(2400, "daily")
    sp_h = windows.split_origins(2400, "hourly")
    assert len(sp_h["test"]) > len(sp_d["test"])
    # 7 days of hourly origins, minus the tail where the target runs past
    assert len(sp_h["test"]) == 7 * 24 - config.OUT_LEN + 1, len(sp_h["test"])
    print("hourly origins ok", len(sp_h["test"]))


def test_batch_indexing():
    n_series, n_steps = 5, 600
    y = np.arange(n_series * n_steps, dtype=np.float32).reshape(n_series, n_steps)
    cal = windows.calendar(n_steps)

    si = np.array([0, 3])
    oi = np.array([300, 400])

    yh, yt, ch, cf = windows.build_batch(y, cal, si, oi)

    assert yh.shape == (2, config.IN_LEN)
    assert yt.shape == (2, config.OUT_LEN)
    assert ch.shape == (2, config.IN_LEN, 3)
    assert cf.shape == (2, config.OUT_LEN, 3)

    # the input window must end exactly at the origin
    assert yh[0, -1] == y[0, 300]
    # the target must start one step after it
    assert yt[0, 0] == y[0, 301]
    assert yt[1, 0] == y[3, 401]
    print("batch indexing ok")


def test_no_target_leaks_into_input():
    """The last input value must be strictly before the first target value."""
    n_series, n_steps = 3, 600
    y = np.arange(n_series * n_steps, dtype=np.float32).reshape(n_series, n_steps)
    cal = windows.calendar(n_steps)

    for t0 in [200, 350, 500]:
        yh, yt, _, _ = windows.build_batch(
            y, cal, np.array([1]), np.array([t0])
        )
        assert yh[0, -1] < yt[0, 0]
        assert len(set(yh[0]) & set(yt[0])) == 0
    print("no leak into input ok")


def test_calendar_repeats():
    cal = windows.calendar(500)
    assert cal[0, 0] == 0 and cal[23, 0] == 23 and cal[24, 0] == 0
    assert cal[0, 1] == 0 and cal[24, 1] == 1
    assert cal[:, 0].max() == 23
    assert cal[:, 1].max() <= 6
    print("calendar ok")


def test_series_scale_positive():
    y = np.zeros((4, 500), dtype=np.float32)
    y[0] = 10.0
    y[1] = 1.0
    sp = windows.split_origins(500, "daily")
    s = windows.series_scale(y, sp["train"])
    assert (s > 0).all(), s
    assert s[0] > s[1], s
    print("series scale ok", np.round(s, 3))


def test_insample_stops_before_test():
    n = 2400
    y = np.arange(n, dtype=float)[None, :]
    ins = windows.insample(y)
    sp = windows.split_origins(n, "daily")
    first_test = int(sp["test"][0]) + 1
    assert ins.shape[1] == first_test == n - config.TEST_DAYS * 24, ins.shape
    print("insample ok")


if __name__ == "__main__":
    test_split_counts()
    test_test_block_ends_at_series_end()
    test_no_overlap()
    test_hourly_gives_more_origins()
    test_batch_indexing()
    test_no_target_leaks_into_input()
    test_calendar_repeats()
    test_series_scale_positive()
    test_insample_stops_before_test()
    print("all window tests passed")
