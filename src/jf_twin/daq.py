"""
Simulated DAQ with the calls of hutch-python's `daq` at MFX (pcdsdaq DaqLCLS2):

    daq.configure(events=240, record=True, controls=[att])
    daq.begin(wait=True)                  # one step; begin(events=..., record=..., controls=...) also works
    daq.begin(events=240, wait=True)      # next step, same run
    daq.end_run()
    daq.run_number()

As on the real DAQ, begin() returns immediately unless wait=True, and the run
stays open (one step per begin) until end_run(). Moving an attenuator while a
step is still acquiring gives a warning: on the instrument that step would
mix two transmissions. `controls` are recorded once per step, under their
names (att -> "mfx_attenuator"), and psana reads them as
scan variables.

Recorded runs go to <workdir>/xtc/r<run>/: run.json, and per step sNNN.npy
(raw frames, uint16), sNNN.npz (per event: timestamp, bmmon_J,
photon_energy_eV) and sNNN.json (step_value, step_docstring, controls).
With record=False nothing is simulated or saved; only the clock moves on.
MFX's attenuator_scan does not step the DAQ: see jf_twin.mfx_sim.
"""
import json
import warnings

import numpy as np

STEP_FIELDS = ("timestamp", "bmmon_J", "photon_energy_eV")
_UNSET = object()


class SimDAQ:
    def __init__(self, twin):
        self._twin = twin
        self.config = {"events": None, "duration": None, "record": False, "controls": [],
                       "detname": "scan", "scantype": "scan"}
        self._run = None        # open run number (None if no run or not recording)
        self._open = False
        self._step = 0
        self._pending = None    # step started with wait=False, not acquired yet
        self._last_run = None
        self._run_controls = None   # control names of the open run

    @property
    def _xtc_dir(self):
        return self._twin.workdir / "xtc"

    @property
    def record(self):
        return self.config["record"]

    @record.setter
    def record(self, record):
        self.configure(record=record)

    @property
    def state(self):
        if self._pending:
            return "running"
        return "starting" if self._open else "configured"

    def configure(self, events=_UNSET, duration=_UNSET, record=_UNSET, controls=_UNSET, detname=_UNSET,
                  scantype=_UNSET):
        """Change the defaults used by begin(). events/duration are per step."""
        if self._open and record is not _UNSET and record != self.config["record"]:
            raise RuntimeError("cannot change record during a run: end_run() first")
        new = {k: v for k, v in dict(events=events, duration=duration, record=record, controls=controls).items()
               if v is not _UNSET}
        new.update({k: v for k, v in dict(detname=detname, scantype=scantype).items() if v is not _UNSET})
        if new.get("events") is not None:
            self.config["duration"] = None
        if new.get("duration") is not None:
            self.config["events"] = None
        self.config.update(new)
        return self.config, self.config

    def begin(self, wait=False, end_run=False, events=_UNSET, duration=_UNSET, record=_UNSET,
              controls=_UNSET):
        self._begin(wait, end_run, events, duration, record, controls, "dynamic")

    def _begin(self, wait, end_run, events, duration, record, controls, mode):
        if self._pending:
            raise RuntimeError("the DAQ is still running the previous step: call daq.wait() or "
                               "begin(..., wait=True)")
        config = dict(self.config)
        config.update({k: v for k, v in dict(events=events, duration=duration, record=record,
                                              controls=controls).items() if v is not _UNSET})
        if config["events"] is not None:
            n = int(config["events"])
        elif config["duration"] is not None:
            n = int(round(config["duration"] * self._twin.beam.rate_hz))
        else:
            raise ValueError("endless runs are not simulated: give events= or duration=")
        names = [c.name for c in config["controls"] or []]
        if self._open and names != self._run_controls:
            # as on DaqLCLS2: changing controls needs a configure transition, refused in an open run
            raise RuntimeError("Cannot configure transition during an open run! "
                               "Pass the same controls to every begin() of a run, or end_run() first.")
        if not self._open:
            self._open, self._step, self._run_controls = True, 0, names
            self._run = self._next_run() if config["record"] else None
            if self._run is not None:
                folder = self._xtc_dir / f"r{self._run:04d}"
                folder.mkdir(parents=True)
                (folder / "run.json").write_text(json.dumps(self._twin.run_info(self._run), indent=2))
        controls = config["controls"] or []
        step_value = self._step + 1     # DaqLCLS2 counts steps from 1
        self._pending = {"chunks": [(None, n)], "mode": mode,
                         "controls": {c.name: float(c.position) for c in controls},
                         "step_value": step_value,
                         # the DaqLCLS2 format, which is not JSON
                         "docstring": (f'{{"detname": "{config["detname"]}", }}'
                                       f'{{"scantype": "{config["scantype"]}", }}{{"step": {step_value}}}')}
        if wait or end_run:
            self.wait()
        if end_run:
            self.end_run()

    def _next_run(self):
        runs = [int(p.name[1:]) for p in self._xtc_dir.glob("r[0-9]*")] if self._xtc_dir.exists() else []
        return max(runs, default=0) + 1

    def _free_run(self, record, chunks):
        """
        One run with one step while a script moves things between chunks of events, as when MFX's
        attenuator_scan sets the LCLS-II DAQ running (DaqControl) and moves the attenuator meanwhile.
        `chunks`: list of (move, events), `move()` called before its events. Returns the run number or None.
        Unlike the instrument, no events are recorded while the attenuator moves.
        """
        if self._open or self._pending:
            raise RuntimeError("the DAQ is busy: end_run() first")
        step = {"chunks": chunks, "mode": "dynamic", "controls": {}}
        if not record:
            for move, n in chunks:
                move()
                self._twin.time_s += n / self._twin.beam.rate_hz
            return None
        self._run, self._step = self._next_run(), 0
        folder = self._xtc_dir / f"r{self._run:04d}"
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps(self._twin.run_info(self._run), indent=2))
        self._write_step(step)
        self._last_run, self._run = self._run, None
        return self._last_run

    def _device_moved(self, device):
        """Called by the twin when an attenuator moves."""
        if self._pending:
            warnings.warn(f"{device.name} moved while the DAQ was acquiring (begin without wait=True): "
                          "on the real DAQ this step would mix two transmissions", stacklevel=4)

    def wait(self, timeout=None, end_run=False):
        """Wait for the current step to finish (here: acquire it)."""
        step, self._pending = self._pending, None
        if step is not None:
            if self._run is None:
                self._twin.time_s += sum(n for _, n in step["chunks"]) / self._twin.beam.rate_hz
            else:
                self._write_step(step)
            self._step += 1
        if end_run:
            self.end_run()

    def _write_step(self, step):
        stem = self._xtc_dir / f"r{self._run:04d}" / f"s{self._step:03d}"
        total = sum(n for _, n in step["chunks"])
        raw = np.lib.format.open_memmap(f"{stem}.npy", mode="w+", dtype=np.uint16,
                                        shape=(total, *self._twin.geometry.shape))
        meta = {key: np.zeros(total) for key in STEP_FIELDS}
        i = 0
        for move, n in step["chunks"]:
            if move is not None:
                move()
            for _ in range(n):
                shot = self._twin.shot(step["mode"])
                raw[i] = shot.pop("raw")
                for key in STEP_FIELDS:
                    meta[key][i] = shot[key]
                i += 1
        raw.flush()
        del raw
        np.savez(f"{stem}.npz", **meta)
        info = {"controls": step["controls"], "gain_mode": step["mode"]}
        if "step_value" in step:        # a free-running DAQ records no scan variables
            info.update(step_value=step["step_value"], step_docstring=step["docstring"])
        (stem.parent / f"{stem.name}.json").write_text(json.dumps(info))

    def stop(self, success=False):
        self.wait()

    def end_run(self, timeout=10.0, wait=True):
        self.wait()
        if self._run is not None:
            self._last_run = self._run
        self._open, self._run, self._run_controls = False, None, None

    def run_number(self):
        """Current run if one is open and recording, else the last recorded run."""
        return self._run if self._run is not None else self._last_run
