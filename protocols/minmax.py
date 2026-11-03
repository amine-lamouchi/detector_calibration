"""
The NoGainNoPain offset calibration (Philip Hart / Chris O'Grady), ported to
run on one machine against psana or psana_sim. Same three steps:

1. run_minmax      (jfMinMax.py)       per run: per pixel, max ADU in G0, min and max in G1,
                                       max in G2, pedestal subtracted
2. combine_minmax  (combineMinMaxz.py) across runs, then fill pixels that never reached a gain
                                       with the module median
3. make_offsets    (makeOffsetsz.py)   pixel_offset [ADU] from the extremes, using pixel_gain

The idea: in a scan that crosses a switching point, the largest energy seen in
the lower gain and the smallest energy seen in the higher gain both approach
the switching energy, so their difference is the offset of the higher gain.

The original splits events over MPI ranks (rank 0, 1 reserved) and reduces
with MPI.MAX / MPI.MIN; here it is one loop. The arithmetic is unchanged.
"""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

try:
    from psana import DataSource
except ImportError:
    from jf_twin.psana_sim import DataSource

MAX_RAW = 0x3FFF
MIN_RAW = 0
TOO_HIGH = 0x3FFF
TOO_LOW = -0x3FFF
G0CUT, G1CUT, G2CUT = 1 << 14, 2 << 14, 3 << 14


def run_minmax(exp, run, detname="jungfrau", filter_size=1, max_events=0, **ds_kwargs):
    """jfMinMax.py for one run. Returns {high_max, med_min, med_max, low_max}, ADU above pedestal."""
    ds = DataSource(exp=exp, run=run, max_events=max_events, **ds_kwargs)
    myrun = next(ds.runs())
    det = myrun.Detector(detname)
    size = (1, filter_size, filter_size)
    hmax = mmin = mmax = lmax = None
    for evt in myrun.events():
        raw = det.raw.raw(evt)
        if raw is None:
            continue
        amplitudes = raw & 0x3FFF
        high = np.where(raw < G0CUT, amplitudes, 0)
        detarr_hmax = maximum_filter(high, size=size, mode="constant", cval=0)
        med_mask = (raw >= G0CUT) & (raw < G1CUT)
        detarr_mmin = minimum_filter(np.where(med_mask, amplitudes, MAX_RAW), size=size, mode="constant", cval=MAX_RAW)
        detarr_mmax = np.where(med_mask, amplitudes, MIN_RAW)
        detarr_lmax = np.where(raw < G2CUT, MIN_RAW, amplitudes)
        if hmax is None:
            hmax, mmin, mmax, lmax = (a.astype(np.float64) for a in (detarr_hmax, detarr_mmin, detarr_mmax, detarr_lmax))
        else:
            np.maximum(hmax, detarr_hmax, out=hmax)
            np.minimum(mmin, detarr_mmin, out=mmin)
            np.maximum(mmax, detarr_mmax, out=mmax)
            np.maximum(lmax, detarr_lmax, out=lmax)

    ped = det.calibconst["pedestals"][0]
    out = {}
    for key, arr, g, empty, flag in (("high_max", hmax, 0, MIN_RAW, TOO_LOW), ("med_min", mmin, 1, MIN_RAW, TOO_HIGH),
                                     ("med_max", mmax, 1, MIN_RAW, TOO_LOW), ("low_max", lmax, 2, MIN_RAW, TOO_LOW)):
        arr = arr.copy()
        arr[arr == empty] = flag
        arr[arr != flag] -= ped[g][arr != flag]
        out[key] = arr
    return out


def combine_minmax(results, fill=True):
    """combineMinMaxz.py: combine run_minmax results. fill: replace flagged pixels by the module median."""
    combined = None
    for d in results:
        da = np.array([d["high_max"], -1 * d["med_min"], d["med_max"], d["low_max"]])
        combined = da if combined is None else np.maximum.reduce([combined, da])
    combined[1] *= -1
    if fill:
        for m in range(combined.shape[1]):
            for kind in range(4):
                flag = TOO_HIGH if kind == 1 else TOO_LOW
                values = combined[kind, m]
                values[values == flag] = np.median(values[values != flag])
    return dict(zip(("high_max", "med_min", "med_max", "low_max"), combined))


def make_offsets(combined, gain, fix_g2_sign=True):
    """
    makeOffsetsz.py: pixel_offset (3, ...) [ADU] from combined extremes and pixel_gain.

    fix_g2_sign: the original uses (lowMinE - medMaxE) - (medMinE - highMaxE) for
    the G2 offset in keV. Correcting G1 by its offset first gives
    (lowMinE - medMaxE) + (medMinE - highMaxE); the two differ by twice the G1
    offset. Fixed by default; fix_g2_sign=False reproduces the original.
    """
    high_max_e = combined["high_max"] / gain[0]
    med_max_e = combined["med_min"] / gain[1]  # gains 1 and 2 are negative
    med_min_e = combined["med_max"] / gain[1]
    low_min_e = combined["low_max"] / gain[2]
    med_gap = med_min_e - high_max_e  # keV
    med_offset = med_gap * gain[1]
    sign = 1 if fix_g2_sign else -1
    low_offset = ((low_min_e - med_max_e) + sign * med_gap) * gain[2]
    return np.array([med_offset * 0.0, med_offset, low_offset])


def pixel_offset(exp, runs, detname="jungfrau", fix_g2_sign=True, **ds_kwargs):
    """The whole method: runs of a scan -> pixel_offset (3, n_modules, rows, cols) [ADU]."""
    runs = [runs] if np.isscalar(runs) else list(runs)
    combined = combine_minmax([run_minmax(exp, run, detname, **ds_kwargs) for run in runs])
    gain = next(DataSource(exp=exp, run=runs[0], **ds_kwargs).runs()).Detector(detname).calibconst["pixel_gain"][0]
    return make_offsets(combined, gain, fix_g2_sign)
