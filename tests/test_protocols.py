import numpy as np
import pytest

from jf_twin import Beam, DetectorTwin
from jf_twin import gain_range
from jf_twin.mfx_sim import attenuator_scan, get_run
from jf_twin.psana_sim import DataSource
from protocols.minmax import combine_minmax, make_offsets, run_minmax


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    twin = DetectorTwin(module_shape=(16, 32), beam=Beam(monitor_noise=0), att_thickness_error=0,
                        workdir=tmp_path_factory.mktemp("runs"), exp="scan")
    twin.take_pedestals(events=100)
    attenuator_scan(transmissions=list(np.geomspace(3e-4, 1, 30)), duration=80 / twin.beam.rate_hz,
                    record=True, runs=1)
    return twin, get_run()


def median_gaps(twin, offsets=None):
    score = twin.score(offsets, verbose=False)
    return abs(np.nanmedian(score["gap01"])), abs(np.nanmedian(score["gap12"]))


def test_scan_crosses_both_switching_points(scanned):
    twin, run = scanned
    myrun = next(DataSource(exp="scan", run=run, dir=twin.workdir).runs())
    det, events = myrun.Detector("jungfrau"), list(myrun.events())
    assert len(events) == 30 * 80
    first, last = (gain_range(det.raw.raw(e)) for e in (events[0], events[-1]))
    assert (first == 0).mean() > 0.99 and (last == 2).mean() > 0.99   # transmissions run low to high


def test_minmax(scanned):
    twin, run = scanned
    combined = combine_minmax([run_minmax("scan", run, dir=twin.workdir)])
    gain = twin.calib.get("pixel_gain")[0]
    none = median_gaps(twin)
    original = median_gaps(twin, make_offsets(combined, gain, fix_g2_sign=False))
    fixed = median_gaps(twin, make_offsets(combined, gain))
    assert original[0] < 0.5 * none[0] and original[1] < none[1]
    assert fixed[1] < original[1]  # the G2 sign in makeOffsetsz.py


def test_every_method_has_the_same_entry_point(scanned):
    from protocols import minmax, my_method
    twin, run = scanned
    shape = twin.calib.get("pedestals")[0].shape
    for method in (minmax, my_method):
        assert method.pixel_offset("scan", run, dir=twin.workdir).shape == shape
