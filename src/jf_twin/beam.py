"""
SASE beam and the beam monitor at MFX DG2 (MfxDg2BmMon), after the attenuator.

The beam monitor reads the pulse energy that gets through the attenuator
(totalIntensityJoules), so it is the intensity reference a protocol uses. It
has reading noise and a calibration factor. The 3rd harmonic goes through the
blades much more easily than the fundamental.
"""
from dataclasses import dataclass

import numpy as np

EV = 1.602176634e-19  # J


@dataclass
class Beam:
    energy_eV: float = 9500.0
    pulse_energy_mJ: float = 2.0       # mean SASE pulse energy in the FEE
    sase_jitter: float = 0.15          # rms relative shot-to-shot fluctuation (gamma distributed)
    monitor_noise: float = 0.02        # rms relative noise of the beam monitor reading
    monitor_scale: float = 1.0         # beam monitor reading / true pulse energy at the monitor
    third_harmonic: float = 0.0        # fraction of the pulse energy at 3 * energy_eV (~1e-3..1e-2 at LCLS)
    beamline_transmission: float = 0.3  # mirrors, windows, lenses (not the attenuator)
    rate_hz: float = 120.0
    on: bool = True                    # False: beam blocked upstream (stopper, pulse picker)

    def pulse(self, rng):
        """True pulse energy [mJ] of one shot in the FEE."""
        if not self.on:
            return 0.0
        k = 1 / self.sase_jitter**2
        return self.pulse_energy_mJ * rng.gamma(k, 1 / k)

    def monitor(self, energy_J, rng):
        """Beam monitor reading [J] for `energy_J` reaching it."""
        return energy_J * self.monitor_scale * (1 + self.monitor_noise * rng.standard_normal())

    def photons(self, pulse_mJ, energy_eV=None):
        """Number of photons in a pulse of `pulse_mJ` at `energy_eV`."""
        return pulse_mJ * 1e-3 / ((energy_eV or self.energy_eV) * EV)
