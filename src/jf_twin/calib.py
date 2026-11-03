"""
JUNGFRAU corrections as psana does them, and a small calibration database.

psana2 (psana/detector/UtilsJungfrau.py, calib_jungfrau_single_panel):

    calib = (raw & 0x3fff - pedestals[g] - pixel_offset[g]) / pixel_gain[g]

with g the gain range from the gain bits (00 -> 0, 01 -> 1, 11 -> 2; 10 is a
bad pixel and gives 0). The result is in keV. Constants are (3, n_modules,
rows, cols). Common mode is not applied here.

CalibStore stands in for the LCLS calibration database: constants are saved
under calib/<det>/<ctype>/, each valid from a run on, and psana_sim hands
them out as det.calibconst[ctype] = (array, metadata).
"""
import json
from pathlib import Path

import numpy as np

MSK = 0x3FFF
CTYPES = ("pedestals", "pixel_rms", "pixel_status", "pixel_gain", "pixel_offset")

# pixel_status bits set by dark_process
STATUS_DEAD = 1        # mean 0 or 16383
STATUS_NOISY = 2       # rms far above the module median
STATUS_BAD_BITS = 4    # wrong gain bits in the dark


def gain_range(raw):
    """0, 1, 2 from the gain bits; -1 for the bad combination 10."""
    bits = np.asarray(raw) >> 14
    return np.select([bits == 0, bits == 1, bits == 3], [0, 1, 2], default=-1)


def calib(raw, pedestals, pixel_gain, pixel_offset=None, mask=None):
    """Energy [keV] per pixel, same formula as psana's det.raw.calib (no common mode)."""
    g = gain_range(raw)
    good = g >= 0
    gi = np.where(good, g, 0)
    pedoff = np.take_along_axis(pedestals, gi[None], 0)[0]
    if pixel_offset is not None:
        pedoff = pedoff + np.take_along_axis(pixel_offset, gi[None], 0)[0]
    gain = np.take_along_axis(pixel_gain, gi[None], 0)[0]
    energy = ((np.asarray(raw) & MSK) - pedoff) / gain
    energy[~good] = 0
    return energy if mask is None else energy * mask


def dark_process(frames_by_mode, noisy=5.0):
    """
    Pedestals from a dark run with three steps: dynamic (G0), forceswitchg1,
    forceswitchg2. `frames_by_mode`: list of 3 arrays (events, n_modules, rows, cols).
    Returns pedestals, pixel_rms, pixel_status, each (3, n_modules, rows, cols) or status (n_modules, rows, cols).
    """
    peds, rms = [], []
    status = np.zeros(frames_by_mode[0].shape[1:], dtype=np.uint16)
    for g, frames in enumerate(frames_by_mode):
        adc = (frames & MSK).astype(np.float64)
        peds.append(adc.mean(axis=0))
        rms.append(adc.std(axis=0))
        status[(gain_range(frames) != g).any(axis=0)] |= STATUS_BAD_BITS
        status[(peds[-1] == 0) | (peds[-1] == MSK)] |= STATUS_DEAD
        median = np.median(rms[-1].reshape(len(rms[-1]), -1), axis=1)[:, None, None]
        status[rms[-1] > noisy * median] |= STATUS_NOISY
    return np.array(peds), np.array(rms), status


class CalibStore:
    """calib/<det>/<ctype>/r<run>-end.npy: constants valid from <run> on, like the calib DB."""

    def __init__(self, root, detname="jungfrau"):
        self.root = Path(root) / detname
        self.detname = detname

    def deploy(self, ctype, array, run, note=""):
        if ctype not in CTYPES:
            raise ValueError(f"ctype must be one of {CTYPES}")
        folder = self.root / ctype
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"r{run:04d}-end.npy", np.asarray(array))
        (folder / f"r{run:04d}-end.json").write_text(json.dumps({"ctype": ctype, "run": run, "note": note}))

    def valid_runs(self, ctype):
        folder = self.root / ctype
        return sorted(int(f.name[1:5]) for f in folder.glob("r*-end.npy")) if folder.exists() else []

    def get(self, ctype, run=None):
        """(array, metadata) valid for `run` (default: latest), or None."""
        runs = [r for r in self.valid_runs(ctype) if run is None or r <= run]
        if not runs:
            return None
        stem = self.root / ctype / f"r{runs[-1]:04d}-end"
        return np.load(f"{stem}.npy"), json.loads(Path(f"{stem}.json").read_text())

    def calibconst(self, run=None):
        """{ctype: (array, metadata)} for the constants valid for `run`, as det.calibconst."""
        out = {}
        for ctype in CTYPES:
            entry = self.get(ctype, run)
            if entry is not None:
                out[ctype] = entry
        return out
