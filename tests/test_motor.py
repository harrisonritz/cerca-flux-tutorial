"""Cropping, the motor-response stages and their caching, end to end on synthetic data.

The synthetic recording carries a 6 Hz evoked deflection on the central radial
sensors after every event, so a response-locked central-RMS ERF must peak just
after time zero, and the left-M1 label exists in the synthetic FreeSurfer subject.
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from cerca_flux.config import Config
from cerca_flux.context import SubjectContext
from cerca_flux.motor import load_motor_result
from cerca_flux.paths import SubjectPaths, discover_recordings, raw_bids_path
from cerca_flux.pipeline import resolve_stages, run_subject
from cerca_flux.preprocess import load_raw
from cerca_flux.utils import read_json, setup_logging

from ._motor_support import motor_config
from ._synthetic_data import CUE_TIMES, DURATION, SFREQ


def _context(cfg: Config) -> SubjectContext:
    rec = discover_recordings(cfg)[0]
    return SubjectContext(cfg=cfg, rec=rec, paths=SubjectPaths(cfg, rec), logger=setup_logging())


# -- cropping ---------------------------------------------------------------- #


def test_crop_start_and_duration_select_the_analysed_span(response_study):
    cfg = motor_config(response_study, study={"crop_start": 60.0, "crop_duration": 100.0,
                                              "variant": "crop-a"})
    ctx = _context(cfg)
    raw = load_raw(ctx)
    assert raw.times[-1] == pytest.approx(100.0, abs=3 / SFREQ)
    # Only events inside [60, 160) survive the crop.
    onsets = raw.annotations.onset - raw.first_time
    assert onsets.min() >= -1e-6 and onsets.max() < 100.0
    assert len(raw.annotations) == int(((CUE_TIMES >= 60) & (CUE_TIMES < 160)).sum())
    assert ctx.metrics["crop_start_s"] == 60.0
    assert ctx.metrics["duration_full_s"] == pytest.approx(DURATION, abs=1 / SFREQ)


def test_the_span_is_capped_at_the_reference_recording(response_study, tmp_path):
    """The notebook keeps only as much of Princeton as Oxford lasts."""
    cfg = motor_config(response_study, study={"variant": "crop-b"})
    reference = tmp_path / "reference_raw.fif"
    full = mne.io.read_raw_fif(raw_bids_path(cfg, discover_recordings(cfg)[0]).fpath,
                               preload=True, verbose="ERROR")
    full.crop(0, 50.0).save(reference, verbose="ERROR")
    capped = motor_config(response_study, study={"variant": "crop-b", "crop_start": 100.0,
                                                 "match_duration_to": str(reference)})
    ctx = _context(capped)
    raw = load_raw(ctx)
    assert raw.times[-1] == pytest.approx(50.0, abs=3 / SFREQ)
    assert ctx.metrics["reference_duration_s"] == pytest.approx(50.0, abs=1 / SFREQ)


def test_a_reference_longer_than_the_data_does_not_extend_the_span(response_study):
    capped = motor_config(response_study, study={
        "variant": "crop-c", "crop_start": 200.0, "crop_duration": 1000.0})
    raw = load_raw(_context(capped))
    assert raw.times[-1] == pytest.approx(DURATION - 200.0, abs=3 / SFREQ)


def test_crop_beyond_the_recording_is_a_clear_error(response_study):
    cfg = motor_config(response_study, study={"variant": "crop-d", "crop_start": 500.0})
    with pytest.raises(RuntimeError, match="does not reach"):
        load_raw(_context(cfg))


def test_event_codes_do_not_depend_on_what_the_crop_keeps(response_study):
    """A label absent from the analysed span must not shift the codes of the others."""
    cfg = motor_config(response_study, study={"variant": "crop-e", "crop_start": 229.0})
    ctx = _context(cfg)
    raw = load_raw(ctx)
    assert {str(d) for d in raw.annotations.description} == {"resp_T"}  # resp_L was cropped away
    assert ctx.state["task_event_labels"] == ["resp_L", "resp_T"]
    assert ctx.state["task_event_id"] == {"resp_L": 1, "resp_T": 2}


# -- the motor stages -------------------------------------------------------- #


def test_the_motor_preset_runs_clean(motor_run):
    _, _, row = motor_run
    assert row["status"] == "ok", row["error"]


def test_the_motor_preset_skips_everything_else(motor_run):
    cfg, rec, _ = motor_run
    states = read_json(SubjectPaths(cfg, rec).metrics)["stages"]
    assert states["motor"] == "ok" and states["motor_source"] == "ok"
    paths = SubjectPaths(cfg, rec)
    assert paths.motor_sensor.exists() and paths.motor_source.exists()
    assert not paths.evoked.exists() and not paths.decoding.exists()


def test_central_rms_erf_peaks_just_after_the_response(motor_run):
    cfg, rec, _ = motor_run
    result = load_motor_result(SubjectPaths(cfg, rec).motor_sensor)
    times, rms = result["erf_times"], result["erf_rms_ft"]
    assert times[0] == pytest.approx(-1.0, abs=1e-6) and times[-1] == pytest.approx(1.2, abs=1e-2)
    assert len(result["channels"]) == 6                      # C1-C6, radial only
    assert all(ch.endswith(" Z") for ch in result["channels"])
    window = (times >= -0.1) & (times <= 0.5)
    assert 0.0 <= times[window][np.argmax(rms[window])] <= 0.45
    assert result["metrics"]["motor_peak_erf_rms_fT"] > result["metrics"]["motor_baseline_rms_fT"]
    # The baseline-normalised curve is centred and scaled on its own baseline.
    baseline = (times >= cfg.motor.sensor.erf_baseline[0]) & (times <= cfg.motor.sensor.erf_baseline[1])
    assert result["erf_z"][baseline].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["erf_z"][baseline].std() == pytest.approx(1.0, rel=1e-6)


def test_beta_has_the_notebook_frequency_grid(motor_run):
    cfg, rec, _ = motor_run
    result = load_motor_result(SubjectPaths(cfg, rec).motor_sensor)
    assert list(result["tfr_freqs"]) == list(np.arange(15, 30, 2))
    assert result["tfr_percent"].shape == (8, len(result["tfr_times"]))
    assert np.isfinite(result["beta_percent"]).all()
    # Percent change is zero on average over the baseline window by construction.
    base = (result["tfr_times"] >= -0.9) & (result["tfr_times"] <= -0.6)
    assert result["tfr_percent"][:, base].mean() == pytest.approx(0.0, abs=1e-6)


def test_left_m1_time_courses(motor_run):
    cfg, rec, _ = motor_run
    result = load_motor_result(SubjectPaths(cfg, rec).motor_source)
    assert result["label"] == "precentral-lh"
    # Stored arrays are real: beamformer output is complex-typed, and plotting it would
    # have silently dropped an imaginary part.
    assert all(not np.iscomplexobj(v) for v in result.values() if isinstance(v, np.ndarray))
    assert np.allclose(result["dics_times"], np.arange(-0.75, 1.0001, 0.1))
    assert len(result["dics_db"]) == len(result["dics_times"])
    base = (result["lcmv_times"] >= -0.8) & (result["lcmv_times"] <= -0.5)
    assert result["lcmv_z"][base].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["lcmv_z"][base].std() == pytest.approx(1.0, rel=1e-6)
    # Parcel power is a magnitude: pca_flip's sign is arbitrary and must not leak into dB
    # (a negative sign once floored to about -3000 dB).
    assert (result["dics_power"] > 0).all()
    assert np.abs(result["dics_db"]).max() < 30
    base_windows = (result["dics_times"] >= -0.75) & (result["dics_times"] <= -0.5)
    assert 10 * np.log10(
        result["dics_power"][base_windows] / result["dics_power"][base_windows].mean()
    ).mean() == pytest.approx(0.0, abs=1.0)


def test_a_second_run_reuses_the_cache_and_keeps_the_metrics(motor_run):
    """A cached stage must still contribute its numbers to the group table."""
    cfg, rec, first = motor_run
    paths = SubjectPaths(cfg, rec)
    before = (paths.motor_sensor.stat().st_mtime_ns, paths.motor_source.stat().st_mtime_ns)
    second = run_subject(cfg, rec, resolve_stages("motor", None))
    assert second["status"] == "ok"
    assert (paths.motor_sensor.stat().st_mtime_ns, paths.motor_source.stat().st_mtime_ns) == before
    for key in ("motor_peak_erf_snr", "motor_rebound_beta_percent", "motor_lcmv_peak_abs_z"):
        assert second[key] == pytest.approx(first[key])


def test_motor_stages_are_off_unless_the_config_enables_them(response_study):
    cfg = motor_config(response_study, study={"variant": "off"}, motor={"enabled": False})
    rec = discover_recordings(cfg)[0]
    row = run_subject(cfg, rec, resolve_stages(None, ["qc", "motor", "motor_source"]))
    assert row["status"] == "ok"
    paths = SubjectPaths(cfg, rec)
    assert not paths.motor_sensor.exists() and not paths.motor_source.exists()
    stages = read_json(paths.preprocessing("state", ".json"))["stages"]
    assert stages["motor"] == "disabled" and stages["motor_source"] == "disabled"


# -- provenance of a single-recording run (one SLURM array task) -------------- #


def test_a_single_recording_run_records_the_configuration_it_used(response_study, tmp_path):
    """Array tasks run one recording each, so each must leave its own resolved config."""
    from cerca_flux.cli import main
    from cerca_flux.config import dump_config, load_config

    cfg = motor_config(response_study, study={"variant": "provenance"})
    config_path = dump_config(cfg, tmp_path / "study.yaml")
    code = main(["run", "--config", str(config_path), "--subjects", "01", "--stages", "qc",
                 "--set", "hfc.order=3", "--no-figures"])
    assert code == 0
    rec = discover_recordings(cfg)[0]
    recorded = cfg.deriv_root / f"config_{rec.key}.yaml"
    assert recorded.exists()
    resolved = load_config(recorded)
    assert resolved.hfc.order == 3                       # the override that was in force
    assert resolved.study.variant == "provenance"
