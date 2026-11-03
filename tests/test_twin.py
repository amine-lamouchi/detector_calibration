import numpy as np
import pytest

from jf_twin import DetectorTwin
from jf_twin.psana_sim import DataSource


@pytest.fixture
def twin(tmp_path):
    return DetectorTwin(module_shape=(16, 32), workdir=tmp_path, exp="test")


def test_workdir_belongs_to_one_detector(twin, tmp_path):
    DetectorTwin(module_shape=(16, 32), workdir=tmp_path, exp="test")  # same detector: fine
    with pytest.raises(RuntimeError):
        DetectorTwin(module_shape=(16, 32), workdir=tmp_path, exp="test", seed=1)
    DetectorTwin(module_shape=(16, 32), workdir=tmp_path, exp="test", seed=1, overwrite=True)


def test_attenuator_controls_the_signal(twin):
    twin.att(1.0, wait=True)
    bright = twin.deposit(2.0).mean()
    twin.att(0.01, wait=True)
    assert twin.deposit(2.0).mean() == pytest.approx(0.01 * bright, rel=0.2)


def test_pedestal_run(twin):
    run = twin.take_pedestals(events=50)
    myrun = next(DataSource(exp="test", run=run, dir=twin.workdir).runs())
    assert len(list(myrun.steps())) == 3  # dynamic (G0), forceswitchg1, forceswitchg2
    cc = myrun.Detector("jungfrau").calibconst
    assert cc["pedestals"][0].shape == (3, 1, 16, 32)
    assert cc["pedestals"][1]["run"] == run and cc["pixel_gain"][1]["run"] == 0


def test_scan_run_reads_like_psana(twin):
    twin.take_pedestals(events=50)
    for t in (0.5, 0.05):
        twin.att(t, wait=True)
        twin.daq.begin(events=10, record=True, controls=[twin.att], wait=True)
    twin.daq.end_run()
    myrun = next(DataSource(exp="test", run=twin.daq.run_number(), dir=twin.workdir).runs())
    det, bmmon = myrun.Detector("jungfrau"), myrun.Detector("MfxDg2BmMon")
    assert "mfx_attenuator" in myrun.scannames
    means = []
    for step in myrun.steps():
        events = list(step.events())
        assert len(events) == 10
        assert det.raw.raw(events[0]).dtype == np.uint16
        means.append(np.mean([det.raw.calib(e).mean() / bmmon.raw.totalIntensityJoules(e) for e in events]))
    assert means[0] == pytest.approx(means[1], rel=0.1)  # energy per unit reference is constant


def test_not_recording_makes_no_run(twin):
    twin.daq.begin(events=10, record=False)
    twin.daq.end_run()
    assert twin.daq.run_number() is None and twin.time_s > 0


def test_moving_during_a_step_warns(twin):
    twin.daq.begin(events=10, record=False)  # wait=False: returns while acquiring, as on the real DAQ
    with pytest.warns(UserWarning, match="moved while the DAQ was acquiring"):
        twin.att(0.1, wait=True)
    with pytest.raises(RuntimeError):
        twin.daq.begin(events=10)
    twin.daq.wait()
    twin.daq.end_run()


def test_score(twin):
    twin.take_pedestals(events=50)
    none = twin.score(verbose=False)
    assert np.nanmedian(none["gap01"]) == pytest.approx(np.median(twin.jungfrau.switch_offset_keV[1]), abs=0.5)
    ideal = twin.score(twin.ideal_offsets(), verbose=False)
    assert np.sqrt(np.nanmean(ideal["gap01"] ** 2)) < 2  # left: pixel_gain error over the G0 range
    assert np.nanmax(np.abs(ideal["error1"])) < 1



def test_daq_rules_match_daqlcls2(twin):
    twin.take_pedestals(events=5)
    twin.daq.begin(events=2, record=True, controls=[twin.att], wait=True)
    with pytest.raises(RuntimeError):            # the real DAQ refuses new controls in an open run
        twin.daq.begin(events=2, wait=True)
    twin.daq.end_run()
    with pytest.raises(TypeError):               # no arguments the real DAQ lacks
        twin.daq.begin(events=1, gain_mode="forceswitchg1")
    myrun = next(DataSource(exp="test", run=twin.daq.run_number(), dir=twin.workdir).runs())
    assert myrun.Detector("step_value")(next(myrun.steps())) == 1      # steps count from 1


def test_mfx_attenuator_scan(twin):
    from jf_twin.mfx_sim import attenuator_scan, get_run
    twin.take_pedestals(events=5)
    attenuator_scan(transmissions=[1.0, 0.01], duration=0.1, runs=2)     # record=False, as at MFX
    assert get_run() == 1                                                  # only the dark run
    attenuator_scan(transmissions=[1.0, 0.01], duration=0.1, record=True, runs=2)
    assert get_run() == 3
    myrun = next(DataSource(exp="test", run=3, dir=twin.workdir).runs())
    assert len(list(myrun.steps())) == 1 and myrun.scannames == set()      # one free-running step, no scan variables
    assert len(list(myrun.events())) == 2 * 12
    with pytest.raises(ValueError):
        attenuator_scan(transmissions=[2.0])
