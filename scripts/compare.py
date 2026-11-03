"""
Score offset methods on the twin: one dark run and one attenuator scan, then
every protocols/<name>.py that has pixel_offset(exp, runs, **ds_kwargs).

    python scripts/compare.py                  # beam monitor noise and blade thickness errors
    python scripts/compare.py --ideal          # perfect intensity reference
    python scripts/compare.py my_method minmax

The score is the rms over pixels of the energy jump left at each switching
point, divided by the switching energy and averaged over the two switches, in %.
The columns after it give the jump itself (median and rms over pixels, keV).
Both come from the twin's hidden truth. 0 is perfect.
"""
import argparse
import importlib
import pkgutil
import tempfile
from pathlib import Path

import numpy as np

import protocols
from jf_twin import Beam, DetectorTwin
from jf_twin.mfx_sim import attenuator_scan, get_run

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("methods", nargs="*", help="modules in protocols/ (default: all with pixel_offset)")
parser.add_argument("--ideal", action="store_true", help="no beam monitor noise, exact blade thicknesses")
parser.add_argument("--steps", type=int, default=30, help="number of transmissions")
parser.add_argument("--duration", type=float, default=1.0, help="seconds per transmission")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

names = args.methods or [m.name for m in pkgutil.iter_modules(protocols.__path__)]
methods = {n: getattr(importlib.import_module(f"protocols.{n}"), "pixel_offset", None) for n in names}
methods = {n: f for n, f in methods.items() if f is not None}

extra = dict(beam=Beam(monitor_noise=0), att_thickness_error=0) if args.ideal else {}
twin = DetectorTwin(module_shape=(64, 128), seed=args.seed, exp="compare",
                    workdir=Path(tempfile.mkdtemp(prefix="jf_compare_")), **extra)
twin.take_pedestals(events=200)
attenuator_scan(transmissions=list(np.geomspace(3e-4, 1, args.steps)), duration=args.duration, record=True, runs=1)
run = get_run()

print(f"{'perfect intensity measurement' if args.ideal else 'beam monitor noise and blade errors'}, "
      f"{args.steps} transmissions x {args.duration:g} s")
print(f"{'':16s}{'score %':>8s}{'G0->G1 median':>15s}{'rms':>8s}{'G1->G2 median':>16s}{'rms':>8s}   keV")
for name, offsets in [("no offsets", None)] + [(n, f(twin.exp, run, dir=twin.workdir)) for n, f in methods.items()]:
    s = twin.score(offsets, verbose=False)
    row = [f(s[k]) for k in ("gap01", "gap12") for f in (np.nanmedian, lambda g: np.sqrt(np.nanmean(g ** 2)))]
    print(f"{name:16s}{s['score']:8.2f}{row[0]:+15.1f}{row[1]:8.1f}{row[2]:+16.1f}{row[3]:8.1f}")
