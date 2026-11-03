"""
The README figure of the analysis: energy / intensity for each gain stage
along an attenuator scan on the simulator, before and after min/max offsets.
Same scan and plot as the quickstart notebook.

    python scripts/response_figure.py          # -> docs/img/response.png (about a minute)
"""
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jf_twin import Beam, DetectorTwin, gain_range
from jf_twin.psana_sim import DataSource
from protocols import minmax
from jf_twin.mfx_sim import attenuator_scan, get_run

REPO = Path(__file__).resolve().parents[1]
COLORS, INK, MUTED = ["#2a78c8", "#c77700", "#1f9e6e"], "#222", "#888"


def response(twin, run, per_t):
    """(intensity, gain, energy / intensity) per transmission and gain stage, median over pixels that stayed in it.
    The scan records `per_t` events at each transmission, in order."""
    myrun = next(DataSource(exp=twin.exp, run=run, dir=twin.workdir).runs())
    det, bmmon = myrun.Detector("jungfrau"), myrun.Detector("MfxDg2BmMon")
    all_events = list(myrun.events())
    points = []
    for start in range(0, len(all_events), per_t):
        events = all_events[start:start + per_t]
        energy = np.mean([det.raw.calib(e) for e in events], axis=0)
        gains = np.array([gain_range(det.raw.raw(e)) for e in events])
        intensity = np.mean([bmmon.raw.totalIntensityJoules(e) for e in events])
        for g in range(3):
            stayed = (gains == g).all(axis=0)
            if stayed.mean() > 0.5:
                points.append((intensity, g, np.median(energy[stayed]) / intensity))
    return np.array(points).T


def main(path=REPO / "docs" / "img" / "response.png"):
    twin = DetectorTwin(9500, module_shape=(64, 128), beam=Beam(monitor_noise=0), att_thickness_error=0,
                        exp="figure", workdir=Path(tempfile.mkdtemp()))
    twin.take_pedestals()
    attenuator_scan(transmissions=list(np.geomspace(3e-4, 1, 25)), duration=1, record=True, runs=1)
    run, per_t = get_run(), int(twin.beam.rate_hz)
    before = response(twin, run, per_t)
    twin.calib.deploy("pixel_offset", minmax.pixel_offset(twin.exp, run, dir=twin.workdir), run)
    after = response(twin, run, per_t)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, sharey=True, constrained_layout=True)
    for ax, (x, g, r), title in zip(axes, (before, after), ("Pedestals and gains only", "With pixel_offset from min/max")):
        norm = np.median(r[g == 0])
        ax.axhline(1, color=MUTED, lw=0.8)
        for k in range(3):
            ax.semilogx(x[g == k], r[g == k] / norm, "o", color=COLORS[k], ms=6, label=f"G{k}")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("beam monitor I [J]", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    axes[0].set_ylabel("E / I (G0 = 1)", color=INK)
    axes[0].set_ylim(0.97, 1.01)
    axes[0].legend(frameon=False, loc="lower left")
    fig.savefig(path, facecolor="white")
    print(path)


if __name__ == "__main__":
    main()
