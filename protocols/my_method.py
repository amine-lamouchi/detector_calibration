"""
Your method. Copy minmax.py, or start here.

Rules, so the same file runs on the twin and at the beamtime:
* read data only through DataSource (psana when available, else the twin);
* return pixel_offset [ADU], shape (3, n_modules, rows, cols), G0 row 0;
* never use the twin's truth (twin.score, twin.jungfrau, ...) in here.

Compare with the baselines: python scripts/compare.py
"""
import numpy as np

try:
    from psana import DataSource
except ImportError:
    from jf_twin.psana_sim import DataSource


def pixel_offset(exp, runs, detname="jungfrau", **ds_kwargs):
    runs = [runs] if np.isscalar(runs) else list(runs)
    myrun = next(DataSource(exp=exp, run=runs[0], **ds_kwargs).runs())
    det = myrun.Detector(detname)
    offsets = np.zeros_like(det.calibconst["pedestals"][0])
    # ... your analysis ...
    return offsets
