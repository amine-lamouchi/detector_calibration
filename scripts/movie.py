"""
Animated walk-through of the measurement on the simulator: the dark run, the
attenuator scan, and the min/max correction.

    pip install -e ".[movie]"
    python scripts/movie.py                    # -> docs/img/scan.mp4 and scan.png (a few minutes)
    python scripts/movie.py --out my.mp4 --steps 20

Left, the beam line in 3D (xrt, see beamline3d.py): blades move, the beam dims
or brightens, the copper foil lights the detector, each pixel coloured by its
gain stage. Right, 300 pixels' raw output (ADU) and the error of their
reading, both against the energy deposited in them (simulator truth, log scale). A window opens while it records; it needs a screen.
"""
import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba_array  # noqa: E402
from PIL import Image  # noqa: E402

from jf_twin import Beam, DetectorTwin, gain_range  # noqa: E402
from jf_twin.calib import calib  # noqa: E402
from jf_twin.psana_sim import DataSource  # noqa: E402
from protocols import minmax  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GAIN_COLORS = ["#4c9be8", "#f5a623", "#3ecf8e"]  # G0, G1, G2
GAIN_RGBA = to_rgba_array(GAIN_COLORS)
BG = "black"
KEV_XLIM, ERR_YLIM = (3, 6e4), (-300, 40)
SIZE, DPI, FPS = (16, 9), 90, 12
VIEW_3D = (0.0, 0.0, 0.62, 0.9)      # where the 3D view sits in the frame (figure fraction)
CANVAS_3D = (860, 700)               # its size on screen while recording [px], same aspect


def ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))


# --- data ------------------------------------------------------------------

def truth_shots(twin, pix, n=3):
    """A few extra shots at the current settings, with the energy each pixel really got (simulator only)."""
    peds, gain = twin.calib.get("pedestals")[0], twin.calib.get("pixel_gain")[0]
    out = {key: [] for key in ("deposit", "adu", "err", "stage", "pixel")}
    for _ in range(n):
        true_mj = twin.beam.pulse(twin.rng)
        deposit = twin.deposit(true_mj)
        raw = twin.jungfrau.frame(deposit, "dynamic", twin.time_s / 3600, twin.rng)
        out["deposit"].append(deposit[pix])
        out["adu"].append(raw[pix] & 0x3FFF)
        out["err"].append(calib(raw, peds, gain)[pix] - deposit[pix])
        out["stage"].append(gain_range(raw)[pix])
        out["pixel"].append(np.arange(len(pix[0])))
    return {key: np.concatenate(v) for key, v in out.items()}


def take_data(twin, steps, events):
    """Dark run and an attenuator scan (low to high intensity). Returns what the frames need."""
    att, daq = twin.att, twin.daq
    twin.take_pedestals(events=100)
    tpix = tuple(np.random.default_rng(2).integers(0, n, 300) for n in twin.geometry.shape)
    pedestals = [float(np.median(p[tpix])) for p in twin.calib.get("pedestals")[0]]   # ADU, per gain
    gains = [float(np.median(g[tpix])) for g in twin.calib.get("pixel_gain")[0]]       # ADU/keV, per gain

    scan = []
    for t in np.geomspace(3e-4, 1, steps):                      # weak to strong
        att(t, wait=True)
        scan.append({"att": np.array([f.position == "IN" for f in att.filters], float), "t": att.position})
        daq.begin(events=events, record=True, controls=[att], wait=True)
        scan[-1]["truth"] = truth_shots(twin, tpix)
    daq.end_run()
    run = daq.run_number()

    myrun = next(DataSource(exp=twin.exp, run=run, dir=twin.workdir).runs())
    det = myrun.Detector("jungfrau")
    for step, s in zip(myrun.steps(), scan):                    # gain stage of each pixel, first shot of the step
        s["gains"] = gain_range(det.raw.raw(next(iter(step.events()))))[0]

    offsets = minmax.pixel_offset(twin.exp, run, dir=twin.workdir)
    gain = twin.calib.get("pixel_gain")[0]
    offset_kev = (offsets / gain)[:, tpix[0], tpix[1], tpix[2]]   # (3, 300): what pixel_offset removes, keV
    for s in scan:
        truth = s["truth"]
        truth["err_after"] = truth["err"] - offset_kev[truth["stage"], truth["pixel"]]
        truth["keep"] = truth["deposit"] > KEV_XLIM[0]
        s["medians"] = []                                       # (gain, deposit, err before, err after)
        for g in range(3):
            sel = truth["keep"] & (truth["stage"] == g)
            if sel.sum() > 30:
                s["medians"].append((g, np.median(truth["deposit"][sel]), np.median(truth["err"][sel]),
                                     np.median(truth["err_after"][sel])))
    return scan, list(zip(pedestals, gains))


# --- drawing -----------------------------------------------------------------

def style(ax, title=None):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#ccc", labelsize=9)
    for side in ax.spines.values():
        side.set_color("#555")
    if title:
        ax.set_title(title, color="w", fontsize=12)


def percent(t):
    return f"{t * 100:.0f}%" if t >= 0.1 else f"{t * 100:.2g}%"


def scatter_steps(ax, steps, x, y, rgba, fade, size):
    """Scatter all steps, the last one faded in by `fade`."""
    for chunk, alpha in ((steps[:-1], 1.0), (steps[-1:], fade)):
        if chunk:
            colors = np.concatenate([rgba(s) for s in chunk]).copy()
            colors[:, 3] *= alpha
            ax.scatter(np.concatenate([x(s) for s in chunk]), np.concatenate([y(s) for s in chunk]), s=size,
                       c=colors, lw=0)


def truth_rgba(s, alpha):
    keep = s["truth"]["keep"]
    return np.c_[GAIN_RGBA[s["truth"]["stage"][keep], :3], np.full(keep.sum(), alpha)]


def draw_adc(ax, scan, upto, levels, fade=1.0, ped_sweep=None):
    """
    Raw output (ADU) vs energy deposited, with the pedestal of each gain from the dark run.
    levels: (pedestal, gain) per gain stage.
    """
    ax.clear(); style(ax, "raw output")
    pedestals = [ped for ped, _ in levels]
    shown = len(pedestals) if ped_sweep is None else len(ped_sweep)
    labels = ["G0 pedestal", "G1 pedestal (forced)", "G2 pedestal (forced)"]
    for g in range(shown):
        sweep = 1.0 if ped_sweep is None else ped_sweep[g]
        xb = np.exp(np.log(KEV_XLIM[0]) + (np.log(KEV_XLIM[1]) - np.log(KEV_XLIM[0])) * ease(sweep))
        ax.plot([KEV_XLIM[0], xb], [pedestals[g]] * 2, color=GAIN_COLORS[g], ls="--", lw=1.2)
        x, ha = (KEV_XLIM[1] / 1.2, "right") if g == 0 else (KEV_XLIM[0] * 1.2, "left")   # away from the data
        ax.text(x, pedestals[g] + (350 if g != 1 else -350), labels[g], color=GAIN_COLORS[g], fontsize=9, ha=ha,
                va="bottom" if g != 1 else "top", alpha=ease(sweep))
    scatter_steps(ax, scan[:upto], lambda s: s["truth"]["deposit"][s["truth"]["keep"]],
                  lambda s: s["truth"]["adu"][s["truth"]["keep"]], lambda s: truth_rgba(s, 0.5), fade, 3)
    ax.set_xscale("log"); ax.set_xlim(*KEV_XLIM); ax.set_ylim(0, 16384)
    ax.set_ylabel("ADU", color="#ccc")
    ax.tick_params(labelbottom=False)


def draw_errors(ax, scan, upto, fade=1.0, morph=0.0):
    """Error of the reading (psana energy − energy deposited) vs energy deposited; morph: 0 before, 1 after correction."""
    ax.clear(); style(ax, "error of the energy reading")
    ax.axhline(0, color="#777", lw=0.8)
    m = ease(morph)
    err = lambda s: (s["truth"]["err"] + (s["truth"]["err_after"] - s["truth"]["err"]) * m)[s["truth"]["keep"]]  # noqa
    scatter_steps(ax, scan[:upto], lambda s: s["truth"]["deposit"][s["truth"]["keep"]], err,
                  lambda s: truth_rgba(s, 0.15), fade, 3)
    for k, s in enumerate(scan[:upto]):
        alpha = fade if k == upto - 1 else 1.0
        for g, x, before, after in s["medians"]:
            ax.plot(x, before + (after - before) * m, "o", color=GAIN_COLORS[g], ms=7, mec="w", mew=0.8,
                    alpha=alpha)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=20)
    ax.set_xlim(*KEV_XLIM); ax.set_ylim(*ERR_YLIM)
    ticks = [20, 0, -10, -20, -50, -100, -200]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:d}".replace("-", "−") for t in ticks])
    ax.set_xlabel("energy deposited in the pixel [keV], log scale", color="#ccc")
    ax.set_ylabel("keV", color="#ccc")
    ax.text(0.99, 0.02, "axis compressed below −20", transform=ax.transAxes, ha="right", va="bottom",
            color="#777", fontsize=8)


class Frames:
    """Collects frames from figures of the same size, then writes an MP4 or a GIF."""

    def __init__(self):
        self.frames, self.durations = [], []

    def snap(self, fig, seconds=1 / FPS):
        fig.canvas.draw()
        image = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        self.frames.append(image.convert("RGB"))
        self.durations.append(seconds)

    def save(self, path):
        if Path(path).suffix == ".mp4":
            import imageio_ffmpeg
            writer = imageio_ffmpeg.write_frames(str(path), self.frames[0].size, fps=FPS, quality=8,
                                                 macro_block_size=2)
            writer.send(None)
            for frame, seconds in zip(self.frames, self.durations):
                for _ in range(max(1, round(seconds * FPS))):
                    writer.send(np.asarray(frame).tobytes())
            writer.close()
        else:  # smaller and at a lower frame rate, so the README page stays light
            fps, scale = 6, 0.5
            starts = np.cumsum([0, *self.durations])
            picked, durations = [], []
            for t in np.arange(0, starts[-1], 1 / fps):
                i = int(np.searchsorted(starts, t, side="right") - 1)
                if picked and picked[-1] == i:
                    durations[-1] += 1000 / fps
                else:
                    picked.append(i); durations.append(1000 / fps)
            size = tuple(int(v * scale) for v in self.frames[0].size)
            frames = [self.frames[i].resize(size, Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=255)
                      for i in picked]
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=[int(d) for d in durations],
                           loop=0, optimize=True)


class Measurement:
    """One frame: the 3D beam line on the left, the two plots on the right."""

    def __init__(self, scene, frames):
        self.scene, self.frames = scene, frames
        self.fig = plt.figure(figsize=SIZE, dpi=DPI, facecolor=BG)
        self.ax_3d = self.fig.add_axes(VIEW_3D)
        self.ax_3d.axis("off")
        self.image = None
        self.ax_adc = self.fig.add_axes([0.68, 0.52, 0.30, 0.34])
        self.ax_err = self.fig.add_axes([0.68, 0.1, 0.30, 0.34])
        self.title = self.fig.text(0.03, 0.955, "", color="w", fontsize=18, va="center")
        self.legend = [self.fig.text(0.0, 0.0, f"G{g}", color=c, fontsize=11, fontweight="bold")
                       for g, c in enumerate(GAIN_COLORS)]
        self.labels = {}

    def snap(self, head, state, seconds=1 / FPS):
        """Set the beam line to `state`, grab the 3D view, add a frame (the caller draws the plots)."""
        self.scene.show(**state)
        view = np.asarray(self.scene.grab())
        if self.image is None:
            self.image = self.ax_3d.imshow(view, extent=(0, 1, 0, 1), aspect="auto")
        self.image.set_data(view)
        beam_on = state["beam"] > 0
        notes = {"attenuator": f"lets through {percent(state['t'])}" if beam_on else "",
                 "X-ray beam": "9.5 keV",
                 "detector": "one module: 8 chips of 256 × 256 pixels\ncolour: gain stage of each pixel"}
        for label, (x, y) in self.scene.label_xy.items():
            if label not in self.labels:
                below = label in ("X-ray beam", "Cu foil")          # hang under the element, not on it
                self.labels[label] = self.ax_3d.text(0, 0, "", color="w", fontsize=11, ha="center",
                                                     va="top" if below else "bottom", linespacing=1.4)
            self.labels[label].set_position((x, 1 - y))
            self.labels[label].set_text(label + (f"\n{notes[label]}" if notes.get(label) else ""))
            self.labels[label].set_visible(beam_on or label != "X-ray beam")
        if "detector" in self.scene.label_xy:          # gain colours, under the detector label
            x, y = self.scene.label_xy["detector"]
            for g, text in enumerate(self.legend):
                text.set_position((VIEW_3D[0] + VIEW_3D[2] * x + (g - 1.2) * 0.03, VIEW_3D[3] * (1 - y) - 0.03))
        self.title.set_text(head)
        self.frames.snap(self.fig, seconds)


def beam_state(s, prev=None, u=1.0):
    """Beam line at fraction u of the move from step `prev` to step `s`."""
    first = prev is None
    if first:
        prev = {"att": np.zeros_like(s["att"]), "t": 1.0, "gains": s["gains"]}
    e = ease(u)
    return {"att": prev["att"] + (s["att"] - prev["att"]) * e,
            "t": float(np.exp(np.log(prev["t"]) + (np.log(s["t"]) - np.log(prev["t"])) * e)),
            "beam": e if first else 1.0, "gains": s["gains"] if u > 0.5 else prev["gains"]}


def measurement(scene, frames, scan, levels, step_frames=10):
    m = Measurement(scene, frames)
    off = {"att": np.zeros(len(scan[0]["att"])), "t": 1.0, "beam": 0.0}

    # 1. dark run: no beam; each gain's pedestal is recorded in turn
    head = "Dark run, beam off"
    sweep = 18
    for g in range(3):
        draw_adc(m.ax_adc, scan, 0, levels, ped_sweep=[1.0] * g + [0.0])
        draw_errors(m.ax_err, scan, 0)
        m.snap(head, {**off, "detector": GAIN_COLORS[g]}, 4.0 if g == 0 else 2.0)   # read the step first
        for k in range(sweep):
            draw_adc(m.ax_adc, scan, 0, levels, ped_sweep=[1.0] * g + [(k + 1) / sweep])
            m.snap(head, {**off, "detector": GAIN_COLORS[g]}, 2.0 if k == sweep - 1 else 1 / FPS)

    # 2. scan: blades move, the beam brightens, the new step fades in
    head = "Attenuator scan"
    m.snap(head, {**off, "detector": GAIN_COLORS[0]}, 3.0)
    for i, s in enumerate(scan):
        for k in range(1, step_frames + 1):
            u = k / step_frames
            draw_adc(m.ax_adc, scan, i + 1, levels, fade=ease(u))
            draw_errors(m.ax_err, scan, i + 1, fade=ease(u))
            m.snap(head, beam_state(s, scan[i - 1] if i else None, u))
    last = beam_state(scan[-1], scan[-2])
    m.snap(head, last, 2.5)
    frames.still = len(frames.frames) - 1         # the README image: end of the scan

    # 3. correction from min/max
    head = "Correction"
    m.snap(head, last, 3.0)
    n = 36
    for k in range(n + 1):
        draw_errors(m.ax_err, scan, len(scan), morph=k / n)
        m.snap(head, last, 6 if k == n else 1 / FPS)
    plt.close(m.fig)


def make_movie(twin, paths, steps=30, events=60):
    print("taking data on the simulator ...")
    scan, levels = take_data(twin, steps, events)
    print("drawing ...")
    plt.style.use("dark_background")
    frames = Frames()

    import beamline3d  # xrt and Qt, only needed here
    from xrt.gui.commons import qt

    scene = beamline3d.Scene(twin.geometry, GAIN_COLORS, size=CANVAS_3D)

    def record():
        try:
            measurement(scene, frames, scan, levels)
        finally:
            scene.glow.close()
            scene.app.quit()

    qt.QTimer.singleShot(500, record)
    scene.app.exec_()

    total = sum(frames.durations)
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        frames.save(p)
        if Path(p).suffix == ".mp4":
            frames.frames[frames.still].save(Path(p).with_suffix(".png"))
        print(f"saved {total:.0f} s, {Path(p).stat().st_size / 1e6:.1f} MB: {p}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", nargs="+", default=[REPO / "docs" / "img" / "scan.mp4"],
                        help="MP4 or GIF files to write")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--events", type=int, default=120, help="shots per step (as in compare.py)")
    args = parser.parse_args()
    # one full module; perfect intensity reference so the effect is clear (as in the notebook)
    twin = DetectorTwin(9500, module_shape=(512, 1024), beam=Beam(monitor_noise=0), att_thickness_error=0,
                        exp="movie", workdir=Path(tempfile.mkdtemp()))
    make_movie(twin, args.out, args.steps, args.events)


if __name__ == "__main__":
    main()
