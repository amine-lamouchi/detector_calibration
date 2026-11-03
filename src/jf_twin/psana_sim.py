"""
A small stand-in for psana2, reading the twin's runs (or its live shots).

    try:
        from psana import DataSource          # on S3DF
    except ImportError:
        from jf_twin.psana_sim import DataSource

    ds = DataSource(exp="mfxsim", run=12)
    myrun = next(ds.runs())
    det = myrun.Detector("jungfrau")
    bmmon = myrun.Detector("MfxDg2BmMon")          # beam monitor after the attenuator
    for evt in myrun.events():
        raw = det.raw.raw(evt)                     # (n_modules, rows, cols) uint16
        energy = det.raw.calib(evt)                # keV
        joules = bmmon.raw.totalIntensityJoules(evt)
    det.calibconst["pedestals"][0]              # (3, n_modules, rows, cols)

A run taken with daq.begin(controls=[att]) also has steps, and the controls as
scan variables: myrun.Detector("mfx_attenuator")(step). MFX's attenuator_scan
runs the DAQ freely, so its runs have one step and no scan variables.

Only this subset of psana2 exists here. Check the names on real data with
`detnames exp=...,run=...` (-s for scan variables): the detector is "jungfrau"
at MFX, and MFX's own scripts read the beam monitor as "MfxDg2BmMon".
"""
import json
from pathlib import Path

import numpy as np

from jf_twin.calib import CalibStore, calib

_BLD = {"MfxDg2BmMon": ("totalIntensityJoules", "bmmon_J")}   # detector: (field, stored key)


def _default_dir(exp):
    from jf_twin._paths import REPO_ROOT
    return REPO_ROOT / "runs" / exp


class DataSource:
    """DataSource(exp=, run=, dir=, max_events=)."""

    def __init__(self, exp="mfxsim", run=None, dir=None, max_events=0, detectors=None, shmem=None, **kwargs):
        self.exp = exp
        self.max_events = max_events
        if run is None:
            raise ValueError("give run= (live data, shmem=, is not simulated)")
        self._workdir = Path(dir) if dir else _default_dir(exp)
        self._runs = [run] if np.isscalar(run) else list(run)

    def runs(self):
        for run in self._runs:
            yield Run(self._workdir, self.exp, run, self.max_events)


class Event:
    def __init__(self, data, index):
        self._data = data      # dict of per-event arrays / values
        self._index = index
        seconds = data["timestamp"]     # as in psana2: seconds in the upper 32 bits, nanoseconds in the lower
        self.timestamp = np.uint64((int(seconds) << 32) | int(round((seconds % 1) * 1e9)))


class Step:
    def __init__(self, run, index, info, max_events):
        self._run = run
        self._index = index
        self._info = info
        self._max_events = max_events

    def events(self):
        stem = self._run._folder / f"s{self._index:03d}"
        raw = np.load(f"{stem}.npy", mmap_mode="r")
        meta = np.load(f"{stem}.npz")
        n = len(raw) if not self._max_events else min(len(raw), self._max_events - self._run._count)
        for i in range(max(n, 0)):
            self._run._count += 1
            data = {key: float(meta[key][i]) for key in meta.files}
            data["raw"] = np.asarray(raw[i])
            data["step"] = self
            yield Event(data, i)


class Run:
    def __init__(self, workdir, exp, run, max_events=0):
        self._folder = Path(workdir) / "xtc" / f"r{run:04d}"
        if not self._folder.exists():
            raise FileNotFoundError(f"no run {run} in {self._folder.parent}")
        self._info = json.loads((self._folder / "run.json").read_text())
        self.runnum = run
        self.expt = exp
        self._max_events = max_events
        self._count = 0
        self._calib = CalibStore(Path(workdir) / "calib")
        self.detnames = {"jungfrau", *_BLD}
        self.epicsinfo = {}
        self.scannames = set()
        for f in self._folder.glob("s[0-9][0-9][0-9].json"):
            info = json.loads(f.read_text())
            self.scannames |= set(info["controls"]) | {k for k in ("step_value", "step_docstring") if k in info}

    def steps(self):
        files = sorted(self._folder.glob("s[0-9][0-9][0-9].json"))
        for f in files:
            if self._max_events and self._count >= self._max_events:
                return
            yield Step(self, int(f.stem[1:]), json.loads(f.read_text()), self._max_events)

    def events(self):
        for step in self.steps():
            yield from step.events()

    def Detector(self, name):
        if name == "jungfrau":
            return JungfrauDetector(self._calib.calibconst(self.runnum), self._info["geometry"])
        if name in _BLD:
            return BldDetector(*_BLD[name])
        if name in self.scannames:
            return StepDetector(name)
        raise KeyError(f"no detector {name!r}; this run has {sorted(self.detnames)} + scan {sorted(self.scannames)}")


class _JungfrauRaw:
    def __init__(self, det):
        self._det = det

    def raw(self, evt):
        return evt._data["raw"]

    def calib(self, evt):
        """As psana2: (raw - pedestal - pixel_offset) / pixel_gain, times the pixel_status mask."""
        cc = self._det.calibconst
        if "pedestals" not in cc:
            raise RuntimeError("no pedestals deployed: take a dark run first")
        offset = cc["pixel_offset"][0] if "pixel_offset" in cc else None
        mask = None
        if "pixel_status" in cc:
            good = cc["pixel_status"][0] == 0
            mask = good if good.ndim == 3 else good.all(axis=0)     # one status per pixel, or per gain and pixel
        return calib(self.raw(evt), cc["pedestals"][0], cc["pixel_gain"][0], offset, mask)


class JungfrauDetector:
    def __init__(self, calibconst, geometry):
        self.calibconst = calibconst
        self._geometry = geometry
        self.raw = _JungfrauRaw(self)


class _BldRaw:
    def __init__(self, field, key):
        setattr(self, field, lambda evt: evt._data[key])


class BldDetector:
    def __init__(self, field, key):
        self.raw = _BldRaw(field, key)


class StepDetector:
    def __init__(self, name):
        self._name = name

    def __call__(self, step_or_evt):
        step = step_or_evt._data["step"] if isinstance(step_or_evt, Event) else step_or_evt
        info = step._info
        return info[self._name] if self._name in info else info["controls"].get(self._name)
