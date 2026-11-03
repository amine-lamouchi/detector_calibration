"""
JUNGFRAU: charge-integrating pixels with dynamic gain switching.

Each pixel integrates the charge of one pulse. Below a threshold it stays in
high gain (G0); above it the preamplifier switches in a larger feedback
capacitor (G1, medium), then a larger one again (G2, low). Raw data are 16-bit
words, as in psana's det.raw.raw(evt):

    bits 15-14  gain bits 00 (G0), 01 (G1), 11 (G2)
    bits 13-0   ADC value 0..16383

In G1 and G2 the ADC value goes *down* with signal (the CDS stage is bypassed
after a switch), so the gains in ADU/keV are negative there, as in psana's
pixel_gain constants (~41.5, -1.39, -0.11).

The thing to calibrate: after a *dynamic* switch the G1/G2 pedestal is not
the one measured in the dark with forced switching (forceswitchg1/g2), which
is what the pedestal runs give. Corrected with those pedestals, G1 and G2
read too low or too high, and the energy jumps at the switching points
(Sikorski et al. 2023, Ramilli et al. 2022). `switch_offset_keV` sets how far
off they read, per pixel, with structure per ASIC.

Charge sharing is crude: a fraction of the photons leave a uniform random
part of their charge in the pixel and spread the rest over the 4 neighbours.

Not modelled: memory cells (single-cell mode, 120 Hz), common mode,
double-size pixels, temperature.
"""
from dataclasses import dataclass, field

import numpy as np
import xraydb

ADC_MAX = 0x3FFF
GAIN_BITS = np.array([0b00, 0b01, 0b11], dtype=np.uint16)
GAIN_MODES = ("dynamic", "forceswitchg1", "forceswitchg2")


@dataclass
class JungfrauParams:
    """Truth of the simulated detector. Means and pixel-to-pixel spreads."""
    gain0: float = 41.3                 # G0 gain [ADU/keV]
    ratio01: float = 30.1               # G0/|G1|   -> G1 ~ -1.37 ADU/keV
    ratio12: float = 12.6               # |G1|/|G2| -> G2 ~ -0.109 ADU/keV
    gain_spread: float = 0.02           # relative rms of each gain over pixels
    pedestal: tuple = (2000.0, 14000.0, 14500.0)  # dark ADC value per gain [ADU]
    pedestal_spread: tuple = (300.0, 400.0, 400.0)
    pedestal_drift: tuple = (0.0, 0.0, 0.0)       # [ADU/hour]
    noise: tuple = (8.0, 4.0, 3.0)      # rms [ADU]; TO CHECK against real pixel_rms
    threshold_keV: float = 240.0        # G0 -> G1 switch, deposited energy (~25 photons at 9.5 keV)
    threshold_spread: float = 0.03      # relative; G1 -> G2 switches at threshold * ratio01
    # How far G1, G2 read off in energy with forced-switch pedestals: mean, ASIC rms, pixel rms [keV]
    switch_offset_keV: tuple = (-10.0, -100.0)
    switch_offset_asic_spread_keV: tuple = (5.0, 40.0)
    switch_offset_pixel_spread_keV: tuple = (2.0, 15.0)
    g0_nonlinearity: float = 0.0        # G0 response compressed by this fraction at the threshold
    gain_calib_error: float = 0.0       # rms relative error of the deployed pixel_gain vs truth (0: offsets are the only error)
    sensor_um: float = 320.0            # Si sensor thickness
    charge_sharing: float = 0.25        # fraction of photons that split their charge with a neighbour (TO CHECK)
    dead_fraction: float = 1e-4         # pixels that always read 0


@dataclass
class Jungfrau:
    """The simulated detector: `frame(deposit_keV)` -> raw uint16."""
    geometry: object
    params: JungfrauParams = field(default_factory=JungfrauParams)
    seed: int = 0

    def __post_init__(self):
        p, shape = self.params, self.geometry.shape
        rng = np.random.default_rng(self.seed)
        n_asics = self.geometry.n_modules * 8
        asic = (np.arange(self.geometry.n_modules)[:, None, None] * 8 + self.geometry.asic_index())

        def spread(mean, rel):
            return mean * (1 + rel * rng.standard_normal(shape))

        g0 = spread(p.gain0, p.gain_spread)
        g1 = -g0 / spread(p.ratio01, p.gain_spread)
        g2 = g1 / spread(p.ratio12, p.gain_spread)
        self.gain = np.array([g0, g1, g2])  # true ADU/keV, signed
        self.pedestal = np.array([m + s * rng.standard_normal(shape) for m, s in zip(p.pedestal, p.pedestal_spread)])
        self.noise = np.array(p.noise)[:, None, None, None] * np.ones((3, *shape))
        self.threshold = np.array([spread(p.threshold_keV, p.threshold_spread)])
        self.threshold = np.concatenate([self.threshold, self.threshold * (-g0 / g1)])  # keV, (2, ...)
        offset = np.zeros((3, *shape))
        for s in (1, 2):
            per_asic = p.switch_offset_asic_spread_keV[s - 1] * rng.standard_normal(n_asics)
            offset[s] = (p.switch_offset_keV[s - 1] + per_asic[asic]
                         + p.switch_offset_pixel_spread_keV[s - 1] * rng.standard_normal(shape))
        self.switch_offset_keV = offset
        self.dead = rng.random(shape) < p.dead_fraction
        # What the detector group would deploy as pixel_gain (truth + calibration error)
        self.deployed_gain = self.gain * (1 + p.gain_calib_error * rng.standard_normal((3, *shape)))
        self._rng = rng

    def efficiency(self, energy_eV):
        """Fraction of photons absorbed in the sensor."""
        return 1 - np.exp(-xraydb.material_mu("Si", energy_eV, density=2.33) * self.params.sensor_um * 1e-4)

    def absorb(self, photons, photon_energy_eV, rng=None):
        """Energy [keV] deposited by Poisson(`photons`) photons per pixel, with charge sharing."""
        rng = rng or self._rng
        e = photon_energy_eV / 1000
        n = rng.poisson(photons)
        share = self.params.charge_sharing
        if not share:
            return n * e
        k = rng.binomial(n, share)  # photons that share
        kept = np.clip(k / 2 + np.sqrt(k / 12) * rng.standard_normal(n.shape), 0, k)  # sum of k U(0,1)
        spill = (k - kept) * e / 4
        deposit = (n - k + kept) * e
        deposit[..., 1:, :] += spill[..., :-1, :]
        deposit[..., :-1, :] += spill[..., 1:, :]
        deposit[..., :, 1:] += spill[..., :, :-1]
        deposit[..., :, :-1] += spill[..., :, 1:]
        return deposit

    def pedestal_at(self, t_hours):
        return self.pedestal + np.array(self.params.pedestal_drift)[:, None, None, None] * t_hours

    def adc(self, deposit_keV, stage, t_hours=0.0, dynamic=True):
        """Mean ADC value (no noise, no clipping) of pixels in gain `stage` (0/1/2)."""
        ped = self.pedestal_at(t_hours)
        gain = np.choose(stage, self.gain)
        energy = deposit_keV
        if self.params.g0_nonlinearity:
            compress = 1 - self.params.g0_nonlinearity * (deposit_keV / self.threshold[0]) ** 2
            energy = np.where(stage == 0, deposit_keV * compress, deposit_keV)
        value = np.choose(stage, ped) + gain * energy
        if dynamic:
            value = value + gain * np.choose(stage, self.switch_offset_keV)
        return value

    def frame(self, deposit_keV, mode="dynamic", t_hours=0.0, rng=None):
        """Raw frame (n_modules, rows, cols) uint16 for the energy deposited in each pixel."""
        rng = rng or self._rng
        if mode == "dynamic":
            stage = (deposit_keV >= self.threshold[0]).astype(np.intp) + (deposit_keV >= self.threshold[1])
        elif mode in GAIN_MODES:
            stage = np.full(self.geometry.shape, GAIN_MODES.index(mode), dtype=np.intp)
        else:
            raise ValueError(f"mode must be one of {GAIN_MODES}")
        value = self.adc(deposit_keV, stage, t_hours, dynamic=(mode == "dynamic"))
        value = value + np.choose(stage, self.noise) * rng.standard_normal(value.shape)
        adc = np.clip(np.rint(value), 0, ADC_MAX).astype(np.uint16)
        adc[self.dead] = 0
        return adc | (GAIN_BITS[stage] << 14)
