"""
The README figure: why the forced-switch pedestal is off, for one pixel with
the simulator's average numbers (JungfrauParams).

    python scripts/baseline_figure.py          # -> docs/img/baselines.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jf_twin.jungfrau import JungfrauParams

REPO = Path(__file__).resolve().parents[1]
G0_COLOR, G1_COLOR, INK, MUTED = "#2a78c8", "#c77700", "#222", "#888"


def main(path=REPO / "docs" / "img" / "baselines.png"):
    p = JungfrauParams()
    g0, g1 = p.gain0, -p.gain0 / p.ratio01               # ADU/keV
    ped0, ped1 = p.pedestal[:2]                          # dark-run pedestals; G1 forced
    switch, offset = p.threshold_keV, p.switch_offset_keV[0]
    real1 = ped1 + g1 * offset                           # where G1 really starts after an X-ray switch
    e0, e1 = np.linspace(0, switch, 50), np.linspace(switch, 800, 50)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150, constrained_layout=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
        ax.axvline(switch, color=MUTED, lw=0.8, ls=":")

    ax = axes[0]
    ax.set_title("1. What one pixel reads", loc="left", fontsize=11, color=INK)
    ax.plot(e0, ped0 + g0 * e0, color=G0_COLOR, lw=2)
    ax.plot(e1, real1 + g1 * e1, color=G1_COLOR, lw=2)
    for ped, color, text in ((ped0, G0_COLOR, "G0 pedestal (dark run)"),
                             (ped1, G1_COLOR, "G1 pedestal (dark run, switch forced)")):
        ax.axhline(ped, color=color, lw=1, ls="--")
        ax.text(790, ped + 250, text, color=color, fontsize=9, ha="right")
    ax.text(switch * 0.5, ped0 + g0 * switch * 0.5 + 900, "G0", color=G0_COLOR, fontsize=11, ha="right")
    ax.text(600, real1 + g1 * 600 - 900, "G1", color=G1_COLOR, fontsize=11, va="top")
    ax.text(switch + 8, 6000, "switch", color=MUTED, fontsize=9)
    ax.set_ylim(0, 16384); ax.set_xlim(0, 800)
    ax.set_xlabel("energy deposited in the pixel [keV]", color=INK); ax.set_ylabel("reading [ADU]", color=INK)

    ax = axes[1]
    ax.set_title("2. Zoom on G1, just after the switch", loc="left", fontsize=11, color=INK)
    lo, hi = switch - 10, switch + 50
    ez = np.linspace(lo, hi, 50)
    ax.plot(ez, ped1 + g1 * ez, color=G1_COLOR, lw=1.5, ls="--")
    ax.plot(ez[ez >= switch], (real1 + g1 * ez)[ez >= switch], color=G1_COLOR, lw=2.5)
    ax.text(hi - 1, real1 + g1 * lo - 12, "the pixel after an X-ray switch", color=G1_COLOR, fontsize=9,
            ha="right", va="top")
    ax.text(lo + 1, ped1 + g1 * hi + 14, "what psana expects: the forced pedestal\n"
            f"+ G1 gain × energy ({g1:.2f} ADU/keV)".replace("-", "−"), color=G1_COLOR, fontsize=9, va="top")
    x = switch + 30
    ax.annotate("", xy=(x, ped1 + g1 * x), xytext=(x, real1 + g1 * x),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1, shrinkA=0, shrinkB=0))
    ax.text(x + 1, real1 + g1 * x + 3, f"{real1 - ped1:.0f} ADU = {abs(offset):.0f} keV",
            color=INK, fontsize=9, va="bottom")
    ax.text(switch + 1, real1 + g1 * lo + 2, "switch", color=MUTED, fontsize=9, va="top")
    ax.set_xlim(lo, hi); ax.set_ylim(ped1 + g1 * hi - 25, real1 + g1 * lo + 5)
    ax.set_xlabel("energy deposited in the pixel [keV]", color=INK)

    ax = axes[2]
    ax.set_title("3. Energy psana reports − energy deposited", loc="left", fontsize=11, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(e0, 0 * e0, color=G0_COLOR, lw=2)
    ax.plot(e1, 0 * e1 + offset, color=G1_COLOR, lw=2)
    ax.plot([switch, switch], [0, offset], color=INK, lw=1)
    ax.text(switch + 15, offset / 2, "the jump at the switch", color=INK, fontsize=9, va="center")
    ax.text(500, offset - 1.5, "G1 reads too low by the same amount at every energy", color=G1_COLOR,
            fontsize=9, ha="center", va="top")
    ax.text(20, 1.5, "G0", color=G0_COLOR, fontsize=11)
    ax.set_xlim(0, 800); ax.set_ylim(offset * 2, -offset * 0.6)
    ax.set_xlabel("energy deposited in the pixel [keV]", color=INK); ax.set_ylabel("keV", color=INK)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    print(path)


if __name__ == "__main__":
    main()
