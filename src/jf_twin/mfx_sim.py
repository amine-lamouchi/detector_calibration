"""
The MFX hutch-python functions this project uses, run on the simulator. Same
names, signatures and defaults as in pcdshub/mfx, acting on the most recently
created DetectorTwin.

    from jf_twin.mfx_sim import attenuator_scan, get_run
    # at MFX: from mfx.attenuator_scan import attenuator_scan
    #         from mfx.macros import get_run

    attenuator_scan(sample="Cu foil", transmissions=[1.0, 0.1, 0.01], duration=1, record=True, runs=1)
    run = get_run()

As at MFX with the LCLS-II DAQ, each scan is one run with one step: the DAQ
runs while the attenuator moves, and the transmission is not recorded.
Differences: no events are recorded while the blades move, only daq_num=2 is
simulated, and the elog, quotes and pulse picker do nothing.
"""
_twin = None   # set by DetectorTwin


def get_run(hutch="mfx", station=0):
    """Current (last) DAQ run number."""
    return _twin.daq.run_number() or 0


def attenuator_scan(sample="?", tag=None, transmissions=None, inspire=False, duration=10.0, record=False,
                    use_daq=True, runs=5, daq_delay=5, picker=None, daq_num=2):
    """mfx.attenuator_scan.attenuator_scan: `duration` seconds at each transmission, `runs` runs. Returns None."""
    if transmissions is None:
        transmissions = [1.0, 0.5, 0.1, 0.05, 0.01]
    if any(not 0.0 <= t <= 1.0 for t in transmissions):
        raise ValueError("Invalid transmission value")
    if daq_num not in (1, 2):
        raise ValueError("Invalid daq_num")
    if daq_num == 1:
        raise NotImplementedError("only the LCLS-II DAQ (daq_num=2) is simulated")
    att, daq, rate = _twin.att, _twin.daq, _twin.beam.rate_hz
    n = int(round(duration * rate))
    for run_idx in range(runs):
        chunks = [(lambda t=t: att(t, wait=True), n) for t in transmissions]
        if use_daq:
            daq._free_run(record, chunks)
        else:
            for move, events in chunks:
                move()
                _twin.time_s += events / rate
        if run_idx < runs - 1:
            _twin.time_s += daq_delay
