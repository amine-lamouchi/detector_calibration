import numpy as np
import pytest
from pcdsdevices.attenuator import _att_classes

from jf_twin.attenuator import MFX_FILTERS, _att_class, blade_mu, make_att_sim, real_pvs


@pytest.fixture
def att():
    return make_att_sim(energy_eV=9500)


def expected(blades, inserted, energy):
    return np.exp(-sum(blade_mu(m, energy) * t for (m, t), i in zip(blades, inserted) if i))


def test_no_real_pvs():
    assert "setpoint" in real_pvs(_att_classes[11])  # the check does catch them
    assert real_pvs(_att_class(11)) == []


def test_starts_out(att):
    assert att.position == 1.0 and not att.inserted


def test_call_like_hutch_python(att):
    att(0.1, wait=True)
    inserted = [f.position == "IN" for f in att.filters]
    assert att.position == pytest.approx(expected(MFX_FILTERS, inserted, 9500))
    floor, ceil = att.trans_floor.get(), att.trans_ceil.get()
    assert floor <= 0.1 <= ceil
    assert att.position == (floor if abs(0.1 - ceil) > abs(0.1 - floor) else ceil)
    assert att() == att.position  # att() with no argument = wm()


def test_blade_moves_update_readback(att):
    att.filter4.insert(wait=True)
    assert att.position == pytest.approx(np.exp(-blade_mu("Si", 9500) * 160))
    att.remove(wait=True)
    assert att.position == 1.0
    att.insert(wait=True)
    assert att.position < 1e-50


def test_energy(att):
    att(0.01, wait=True)
    low = att.position
    att.set_energy(12000)
    assert att.position > low  # same blades, harder x-rays
    assert att.readback_3rd.get() > att.position  # 3rd harmonic goes through more easily


def test_thickness_error_is_hidden():
    att = make_att_sim(energy_eV=9500, thickness_error=0.05, seed=1)
    att(1e-3, wait=True)
    assert att._true_transmission(9500) != pytest.approx(att.position, rel=1e-3)
    assert make_att_sim(energy_eV=9500)._true_transmission(9500) == 1.0
