"""
MFX detector twin: beam -> att -> beam monitor (DG2) -> sample -> JUNGFRAU -> DAQ.

    twin.att        offline MFX:ATT (same API as `att` in hutch-python)
    twin.daq        simulated DAQ (same calls as `daq` in hutch-python)
    twin.beam       SASE beam + beam monitor settings (energy, pulse energy, 3rd harmonic, ...)
    twin.sample     what scatters onto the detector (a Cu foil)
    twin.jungfrau   the detector, including its hidden truth
    twin.calib      calibration constants (pixel_gain deployed at run 0; pedestals from take_pedestals())

Runs and constants go to runs/<exp>/ and are read back with
jf_twin.psana_sim.DataSource(exp=<exp>, run=...), as psana reads real runs.
Grade with twin.score(offsets) / twin.ideal_offsets(); never use them in a protocol.
"""
import json
import shutil
from dataclasses import asdict

import numpy as np

from jf_twin._paths import REPO_ROOT
from jf_twin.attenuator import make_att_sim
from jf_twin.beam import Beam
from jf_twin.calib import CalibStore, dark_process
from jf_twin import mfx_sim
from jf_twin.daq import SimDAQ
from jf_twin.geometry import Geometry
from jf_twin.jungfrau import GAIN_MODES, Jungfrau, JungfrauParams
from jf_twin.sample import Foil

class DetectorTwin:
    """
    Parameters
    ----------
    energy_eV : float
        Photon energy.
    sample : jf_twin.sample.Sample
        Default Foil("Cu"): a flat field that reaches all three gains in an attenuator scan.
    module_shape, distance :
        One JUNGFRAU module, cropped to `module_shape` (default 256 x 512, 4 ASICs), at
        `distance` [m] from the sample.
    params : JungfrauParams
        Detector truth (gains, thresholds, switching offsets, noise ...).
    seed : int
        Seed of the detector truth and of the shots.
    exp : str
        Name of the simulated experiment: runs go to <workdir>/<exp>.
    overwrite : bool
        Start from an empty <workdir>/<exp> (erase earlier runs and constants).
    """

    def __init__(self, energy_eV=9500.0, sample=None, module_shape=(256, 512), distance=0.1,
                 params=None, beam=None, seed=0, att_thickness_error=0.01, exp="mfxsim", workdir=None,
                 overwrite=False):
        self.exp = exp
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.beam = beam or Beam(energy_eV=energy_eV)
        self.beam.energy_eV = energy_eV
        self.geometry = Geometry(distance, tuple(module_shape))
        self.sample = sample or Foil("Cu")
        self.jungfrau = Jungfrau(self.geometry, params or JungfrauParams(), seed=seed)
        # The beam monitor (MfxDg2BmMon) is after the attenuator: it reads the intensity directly
        self.att = make_att_sim(energy_eV=energy_eV, thickness_error=att_thickness_error, seed=seed)
        self.att._follow(lambda: self.beam.energy_eV)
        self.att.subscribe(lambda *a, **k: self.daq._device_moved(self.att), event_type="done_moving", run=False)
        self.time_s = 0.0
        self._components = {}

        self.workdir = (REPO_ROOT / "runs" if workdir is None else workdir) / exp
        self._init_workdir(overwrite)
        self.calib = CalibStore(self.workdir / "calib")
        self.daq = SimDAQ(self)
        mfx_sim._twin = self       # mfx_sim.attenuator_scan and get_run act on this twin

    # --- setup -------------------------------------------------------------

    def _signature(self):
        return {"seed": self.seed, "module_shape": list(self.geometry.module_shape),
                "distance": self.geometry.distance, "params": json.loads(json.dumps(asdict(self.jungfrau.params)))}

    def _init_workdir(self, overwrite):
        sig_file = self.workdir / "twin.json"
        if overwrite and self.workdir.exists():
            shutil.rmtree(self.workdir)
        if sig_file.exists():
            if json.loads(sig_file.read_text()) == self._signature():
                return  # same detector: keep its runs and constants
            raise RuntimeError(f"{self.workdir} holds runs of a different detector. "
                               "Pick another exp= or pass overwrite=True to erase it.")
        self.workdir.mkdir(parents=True, exist_ok=True)
        sig_file.write_text(json.dumps(self._signature(), indent=2))
        # constants from the detector group (like pixel_gain from PSI), valid from run 0
        CalibStore(self.workdir / "calib").deploy("pixel_gain", self.jungfrau.deployed_gain, 0, "detector group")

    def run_info(self, run):
        return {"exp": self.exp, "run": run, "sample": repr(self.sample), "energy_eV": self.beam.energy_eV,
                "start_time_s": self.time_s,
                "geometry": {"distance": self.geometry.distance,
                             "module_shape": list(self.geometry.module_shape)}}

    def set_energy(self, energy_eV):
        self.beam.energy_eV = energy_eV
        self.att._sim_update()  # the readback follows the beam unless set to a fixed energy

    # --- shots -------------------------------------------------------------

    def _sample_components(self, energy_eV):
        sample, components = self._components.get(energy_eV, (None, None))
        if sample is not self.sample:
            components = [(e, m * self.jungfrau.efficiency(e))
                          for e, m in self.sample.components(self.geometry, energy_eV)]
            self._components[energy_eV] = (self.sample, components)
        return components

    def deposit(self, true_mJ):
        """Energy [keV] deposited in each pixel by one pulse of `true_mJ` (FEE)."""
        beam, rng = self.beam, self.rng
        deposit = np.zeros(self.geometry.shape)
        scale = self.sample.shot_scale(rng)
        for energy, fraction in ((beam.energy_eV, 1 - beam.third_harmonic), (3 * beam.energy_eV, beam.third_harmonic)):
            photons = (beam.photons(true_mJ * fraction, energy) * beam.beamline_transmission
                       * self.att._true_transmission(energy) * scale)
            if photons <= 0:
                continue
            for photon_energy, per_photon in self._sample_components(energy):
                deposit += self.jungfrau.absorb(per_photon * photons, photon_energy, rng)
        return deposit

    def shot(self, mode="dynamic"):
        """One event: raw frame and what the DAQ records with it."""
        beam = self.beam
        true_mJ = beam.pulse(self.rng)
        at_monitor_J = true_mJ * 1e-3 * beam.beamline_transmission * self.att._true_transmission(beam.energy_eV)
        bmmon_J = beam.monitor(at_monitor_J, self.rng)
        raw = self.jungfrau.frame(self.deposit(true_mJ), mode, self.time_s / 3600, self.rng)
        event = {"raw": raw, "timestamp": self.time_s, "bmmon_J": bmmon_J, "photon_energy_eV": beam.energy_eV}
        self.time_s += 1 / self.beam.rate_hz
        return event

    # --- dark runs ---------------------------------------------------------

    def take_pedestals(self, events=200):
        """
        Dark run with 3 steps (dynamic -> G0, forceswitchg1, forceswitchg2), beam
        blocked, then process it and deploy pedestals, pixel_rms, pixel_status from
        that run on. On the instrument the DAQ/POC does this (jungfrau dark processing).
        """
        was_on, self.beam.on = self.beam.on, False
        try:
            for mode in GAIN_MODES:
                self.daq._begin(True, False, events, None, True, [], mode)
            self.daq.end_run()
        finally:
            self.beam.on = was_on
        run = self.daq.run_number()
        self.process_dark(run)
        return run

    def process_dark(self, run):
        from jf_twin.psana_sim import Run
        frames = [np.array([evt._data["raw"] for evt in step.events()])
                  for step in Run(self.workdir, self.exp, run).steps()]
        pedestals, rms, status = dark_process(frames)
        for ctype, array in (("pedestals", pedestals), ("pixel_rms", rms), ("pixel_status", status)):
            self.calib.deploy(ctype, array, run, f"dark run {run}")

    # --- truth, for grading only -----------------------------------------

    def _deployed(self, run):
        cc = self.calib.calibconst(run)
        if "pedestals" not in cc:
            raise RuntimeError("no pedestals deployed: run twin.take_pedestals() first")
        return cc["pedestals"][0], cc["pixel_gain"][0]

    def ideal_offsets(self, run=None):
        """pixel_offset [ADU] that makes the calibrated energy exact at each switching point. Hidden truth."""
        peds, gains = self._deployed(run)
        jf, t = self.jungfrau, self.time_s / 3600
        offsets = np.zeros_like(peds)
        offsets[0] = jf.pedestal_at(t)[0] - peds[0]
        for s in (1, 2):
            threshold = jf.threshold[s - 1]
            offsets[s] = jf.adc(threshold, s, t) - peds[s] - gains[s] * threshold
        return offsets

    def score(self, pixel_offset=None, run=None, verbose=True):
        """
        Energy jump [keV] at the G0->G1 and G1->G2 switching points with the deployed
        pedestals and gains and `pixel_offset` (None: no offsets). 0 is perfect.

        result["score"] is the number to optimise: the rms over pixels of the jump at each
        switch divided by the switching energy, averaged over the two switches, in %.
        """
        peds, gains = self._deployed(run)
        offsets = np.zeros_like(peds) if pixel_offset is None else np.nan_to_num(np.asarray(pixel_offset))
        jf, t = self.jungfrau, self.time_s / 3600
        status = self.calib.get("pixel_status", run)
        good = ~jf.dead & ((status[0] == 0) if status else True)
        result = {}
        for s in (1, 2):
            threshold = jf.threshold[s - 1]
            below = (jf.adc(threshold, s - 1, t) - peds[s - 1] - offsets[s - 1]) / gains[s - 1]
            above = (jf.adc(threshold, s, t) - peds[s] - offsets[s]) / gains[s]
            result[f"gap{s - 1}{s}"] = np.where(good, above - below, np.nan)
            result[f"error{s}"] = np.where(good, above - threshold, np.nan)
        relative = [np.sqrt(np.nanmean((result[f"gap{s - 1}{s}"] / jf.threshold[s - 1]) ** 2)) for s in (1, 2)]
        result["score"] = 100 * float(np.mean(relative))
        if verbose:
            e_ph = self.beam.energy_eV / 1000
            for key in ("gap01", "gap12"):
                gap = result[key]
                print(f"{key}: median {np.nanmedian(gap):+8.2f} keV, rms {np.sqrt(np.nanmean(gap**2)):7.2f} keV "
                      f"({np.sqrt(np.nanmean(gap**2)) / e_ph:6.2f} photons of {e_ph:.1f} keV)")
            print(f"score: {result['score']:.2f} % (0 is perfect)")
        return result

    def report(self):
        """Print the hidden truth of the detector."""
        jf = self.jungfrau
        print(f"E0 = {self.beam.energy_eV:.0f} eV, sample = {self.sample!r}, t = {self.time_s:.0f} s")
        print(f"G0->G1 threshold: {np.mean(jf.threshold[0]):.0f} keV, G1->G2: {np.mean(jf.threshold[1]):.0f} keV")
        for s in (1, 2):
            off = jf.switch_offset_keV[s]
            print(f"G{s} reads off by {np.mean(off):+.1f} keV (pixel rms {np.std(off):.1f} keV) with forced-switch pedestals")
