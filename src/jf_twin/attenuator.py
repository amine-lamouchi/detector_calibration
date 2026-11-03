"""
Offline MFX hutch attenuator: the real pcdsdevices class with every PV faked,
and the IOC's job (pick blades, compute the transmission) done here with xraydb.

    att = make_att_sim()      # MFX:ATT, pcdsdevices Attenuator(n_filters=11) (happi: mfx_attenuator)

    att(0.1, wait=True)       # same calls as in hutch-python
    att.position              # transmission readback

Blade thicknesses are what the IOC believes. The real blades can be given a
thickness error (`thickness_error`), so the readback is not exactly the
transmission the beam sees, as on the instrument. `_true_transmission()` is the
hidden truth; don't use it in a protocol.
"""

import numpy as np
import xraydb
from ophyd import Device
from ophyd.signal import EpicsSignalBase
from ophyd.sim import make_fake_device
from pcdsdevices.attenuator import _att_classes


# MFX:ATT has 11 pneumatic filters (happi n_filters; pcdshub/ioc-mfx-lusiAtt). Thicknesses are runtime
# PVs (MFX:ATT:NN:THICK), not in public config. Assumed: Si, doubling from 20 um, as the same IOC is
# configured at XPP, XCS and MEC (10 Si blades named 20UM ... 10240UM). Blade 11 is the transfocator's
# protection shutter (pcdshub/ioc-mfx-tfs-lens inserts MFX:ATT:11 on a lens fault); its material and
# thickness are unknown, here a 20.48 mm Si blade, which blocks the beam. TO CHECK on the instrument.
MFX_FILTERS = [("Si", 20.0 * 2**i) for i in range(11)]
DENSITY = {"Si": 2.33, "C": 3.51, "Al": 2.70, "Be": 1.85, "Cu": 8.96, "Mo": 10.22}
IN, OUT = 1, 2          # MFX:ATT Filter :STATE values


def blade_mu(material, energy_eV):
    """Linear attenuation coefficient [1/um]."""
    return xraydb.material_mu(material, energy_eV, density=DENSITY.get(material)) * 1e-4


def real_pvs(device_cls, path=""):
    """Components of `device_cls` that would connect to EPICS. Checks the class only."""
    found = []
    for attr, cpt in device_cls._sig_attrs.items():
        if issubclass(cpt.cls, EpicsSignalBase):
            found.append(path + attr)
        elif issubclass(cpt.cls, Device):
            found += real_pvs(cpt.cls, f"{path}{attr}.")
    return found


def combinations(n):
    """Every in/out combination of n blades, (2**n, n)."""
    return (np.arange(2**n)[:, None] >> np.arange(n)) & 1


class _SimBlades:
    """Blade physics: nominal and real thicknesses, transmission."""

    def _blade_setup(self, blades, thickness_error, rng):
        self._blades = [(material, float(thickness)) for material, thickness in blades]
        nominal = np.array([t for _, t in self._blades])
        rng = np.random.default_rng(rng)
        self._true_thickness = nominal * (1 + thickness_error * rng.standard_normal(len(nominal)))
        self._live_energy = None

    def _log_t(self, energy_eV, thickness=None):
        thickness = np.array([t for _, t in self._blades]) if thickness is None else thickness
        return -np.array([blade_mu(m, energy_eV) for m, _ in self._blades]) * thickness

    def _follow(self, energy_source):
        """Use `energy_source()` [eV] as the live beam energy."""
        self._live_energy = energy_source
        self._sim_update()

    def _true_transmission(self, energy_eV):
        """Transmission the beam actually sees (real blade thicknesses). Hidden truth."""
        return float(np.exp(self._log_t(energy_eV, self._true_thickness) @ self._inserted()))


class _SimAtt(_SimBlades):
    """IOC logic for a faked MFX:ATT (AttBaseWith3rdHarmonic)."""

    def _sim_setup(self, blades, energy_eV, thickness_error, rng):
        self._blade_setup(blades, thickness_error, rng)
        self._configs = combinations(len(self._blades))
        for filt, (material, thickness) in zip(self.filters, self._blades):
            filt.thickness.sim_put(thickness)
            filt.material.sim_put(material)
            filt.state.sim_put(OUT)
            filt.stuck.sim_put(0)
            filt.state.subscribe(self._sim_update, run=False)
        self.done.sim_put(self.done_value)
        self.calcpend.sim_put(0)
        self.set_energy(energy_eV)

    def _sim_energy(self):
        return self.user_energy.get() if self._live_energy is None else self._live_energy()

    def _inserted(self):
        return np.array([filt.state.get() == IN for filt in self.filters])

    def _sim_update(self, *args, **kwargs):
        energy = self._sim_energy()
        inserted = self._inserted()
        self.readback.sim_put(float(np.exp(self._log_t(energy) @ inserted)))
        self.readback_3rd.sim_put(float(np.exp(self._log_t(3 * energy) @ inserted)))
        self.energy.sim_put(energy)
        goal = self.setpoint.get()
        transmissions = np.exp(self._configs @ self._log_t(energy))
        below, above = transmissions[transmissions <= goal], transmissions[transmissions >= goal]
        self.trans_floor.sim_put(float(below.max() if below.size else transmissions.min()))
        self.trans_ceil.sim_put(float(above.min() if above.size else transmissions.max()))

    def _setup_move(self, position):
        self.setpoint.sim_put(position)
        self._sim_update()
        # Same choice as AttBase.actuate_value: the closer of floor and ceiling (ceiling on a tie)
        goal, ceil, floor = position, self.trans_ceil.get(), self.trans_floor.get()
        target = floor if abs(goal - ceil) > abs(goal - floor) else ceil
        transmissions = np.exp(self._configs @ self._log_t(self._sim_energy()))
        config = self._configs[np.argmin(np.abs(transmissions - target))]
        self.done.sim_put(1 - self.done_value)
        for filt, inserted in zip(self.filters, config):
            filt.state.sim_put(IN if inserted else OUT)
        self.done.sim_put(self.done_value)

    def set_energy(self, energy=None):
        """Energy [eV] for the transmission calculation. None: follow the beam energy."""
        if energy is None:
            self.eget_cmd.sim_put(6)
            if self._live_energy is None:
                raise ValueError(f"{self.name} has no live beam energy; pass energy=")
        else:
            self.eget_cmd.sim_put(0)
            self.user_energy.sim_put(float(energy))
            self._live_energy = None
        self._sim_update()

    def _follow(self, energy_source):
        self.eget_cmd.sim_put(6)
        super()._follow(energy_source)


def _checked(cls):
    # Never build a simulator that can talk to real hardware.
    assert not real_pvs(cls), real_pvs(cls)
    return cls


def _att_class(n_filters):
    return _checked(type(f"SimAttenuator{n_filters}", (_SimAtt, make_fake_device(_att_classes[n_filters])), {}))


def make_att_sim(blades=None, energy_eV=9500.0, thickness_error=0.0, seed=None,
                 prefix="SIM:ATT", name="mfx_attenuator"):
    """Offline MFX:ATT. `blades`: [(material, thickness_um), ...], default MFX_FILTERS."""
    blades = blades or MFX_FILTERS
    att = _att_class(len(blades))(prefix, name=name)
    att._sim_setup(blades, energy_eV, thickness_error, seed)
    return att
