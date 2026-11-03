"""
The sample: what lands on each pixel per photon reaching the sample.

`components(geometry, energy_eV)` gives a list of (photon_energy_eV, map); `map`
is the mean number of photons hitting each pixel per incident photon, shaped like
the detector. The twin multiplies it by the photons on the sample in each shot
and draws Poisson counts. Absolute intensities are estimates (factor ~2).
"""
import numpy as np
import xraydb


class Sample:
    name = "sample"
    jitter = 0.0  # shot-to-shot signal fluctuation the beam monitor doesn't see

    def components(self, geometry, energy_eV):
        raise NotImplementedError

    def shot_scale(self, rng):
        return max(0.0, 1 + self.jitter * rng.standard_normal()) if self.jitter else 1.0

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in vars(self).items())})"


class Foil(Sample):
    """K-alpha fluorescence of a `thickness_um` foil, isotropic. Self-absorption ignored."""

    def __init__(self, element="Cu", thickness_um=5.0):
        self.element = element
        self.thickness_um = thickness_um

    @property
    def name(self):
        return f"{self.element} foil"

    def components(self, geometry, energy_eV):
        edge = xraydb.xray_edge(self.element, "K")
        if energy_eV < edge.energy:
            return []
        lines = xraydb.xray_lines(self.element, "K")
        ka = {k: v for k, v in lines.items() if k.startswith("Ka")}
        ka_fraction = sum(v.intensity for v in ka.values()) / sum(v.intensity for v in lines.values())
        ka_energy = sum(v.energy * v.intensity for v in ka.values()) / sum(v.intensity for v in ka.values())
        absorbed = 1 - np.exp(-xraydb.material_mu(self.element, energy_eV, kind="photo") * self.thickness_um * 1e-4)
        k_shell = (edge.jump_ratio - 1) / edge.jump_ratio
        per_sr = absorbed * k_shell * edge.fyield * ka_fraction / (4 * np.pi)
        return [(ka_energy, per_sr * geometry.solid_angle)]
