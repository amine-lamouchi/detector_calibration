import numpy as np
import pytest

from jf_twin.calib import calib, dark_process, gain_range
from jf_twin.geometry import Geometry
from jf_twin.jungfrau import Jungfrau, JungfrauParams


@pytest.fixture
def jf():
    return Jungfrau(Geometry(0.1, (16, 32)), JungfrauParams(dead_fraction=0), seed=1)


def test_gain_bits():
    raw = np.array([5, (1 << 14) | 5, (3 << 14) | 5, (2 << 14) | 5], dtype=np.uint16)
    assert list(gain_range(raw)) == [0, 1, 2, -1]


def test_stages_follow_thresholds(jf):
    t01, t12 = jf.threshold
    for deposit, stage in ((0.5 * t01, 0), (1.01 * t01, 1), (0.99 * t12, 1), (1.01 * t12, 2)):
        assert (gain_range(jf.frame(deposit)) == stage).all()


def test_gains_have_psana_signs(jf):
    assert (jf.gain[0] > 0).all() and (jf.gain[1] < 0).all() and (jf.gain[2] < 0).all()
    assert np.mean(jf.gain[0] / -jf.gain[1]) == pytest.approx(30.1, rel=0.02)


def dark(jf, n=200):
    rng = np.random.default_rng(0)
    zero = np.zeros(jf.geometry.shape)
    return [np.array([jf.frame(zero, mode, rng=rng) for _ in range(n)])
            for mode in ("dynamic", "forceswitchg1", "forceswitchg2")]


def test_dark_gives_forced_switch_pedestals(jf):
    peds, rms, status = dark_process(dark(jf))
    assert np.abs(peds - jf.pedestal).max() < 3
    assert rms[0].mean() == pytest.approx(8, rel=0.1)
    assert (status == 0).all()


def test_offsets_are_the_gap(jf):
    peds, _, _ = dark_process(dark(jf))
    t01 = jf.threshold[0]
    above = 1.2 * t01
    raw = jf.frame(above, rng=np.random.default_rng(1))
    energy = calib(raw, peds, jf.gain)
    # with forced-switch pedestals, G1 reads off by switch_offset_keV
    assert np.median(energy - above) == pytest.approx(np.median(jf.switch_offset_keV[1]), abs=1)
    offsets = np.array([0 * peds[0], jf.gain[1] * jf.switch_offset_keV[1], jf.gain[2] * jf.switch_offset_keV[2]])
    assert np.median(calib(raw, peds, jf.gain, offsets) - above) == pytest.approx(0, abs=1)
