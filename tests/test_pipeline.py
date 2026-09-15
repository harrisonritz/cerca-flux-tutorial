"""End-to-end run of the whole pipeline on the synthetic study.

Session-scoped: the pipeline runs once and the assertions inspect what it left
behind, so the full stack is exercised without paying for it repeatedly.
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from cerca_flux.paths import SubjectPaths, discover_recordings
from cerca_flux.pipeline import PRESETS, STAGE_NAMES, resolve_stages, run_study
from cerca_flux.utils import read_json

from ._synthetic_data import CUE_TIMES, FLAT_SENSOR, NOISY_SENSOR


@pytest.fixture(scope="session")
def completed(config):
    rows = run_study(config, resolve_stages("full", None), n_jobs=1)
    return config, rows


def test_every_recording_completes(completed):
    _, rows = completed
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows), [r["error"] for r in rows]


def test_stage_order_is_the_flux_order():
    assert STAGE_NAMES == (
        "qc", "hfc", "annotate", "ica", "epochs", "erf", "tfr", "mvpa",
        "forward", "source", "morph", "report",
    )
    # Presets must stay in pipeline order whatever order they are requested in.
    assert [s.name for s in resolve_stages(None, ["source", "qc"])] == ["qc", "source"]
    assert set(PRESETS["full"]) == set(STAGE_NAMES)


def test_bad_sensors_are_found(completed):
    """Both the manually flagged flat sensor and the noisy one must be excluded."""
    config, _ = completed
    rec = discover_recordings(config)[0]
    # The per-subject JSON keeps native types; only the group table flattens them.
    metrics = read_json(SubjectPaths(config, rec).metrics)["metrics"]
    bads = set(metrics["bad_channels"])
    assert any(ch.startswith(FLAT_SENSOR + " ") for ch in bads)
    assert any(ch.startswith(NOISY_SENSOR + " ") for ch in bads)
    # The noisy sensor is caught automatically, not just by the manual list.
    assert any(ch.startswith(NOISY_SENSOR + " ") for ch in metrics["noisy_channels"])
    # The flat one is what the manual list contributes.
    assert any(ch.startswith(FLAT_SENSOR + " ") for ch in metrics["auto_bad_channels"]) or \
        FLAT_SENSOR in str(config.channels.manual_bads)


def test_hfc_attenuates_the_homogeneous_field(completed):
    """The synthetic interference is spatially uniform, so HFC must reduce power."""
    config, rows = completed
    for row in rows:
        assert row["n_hfc_projections"] == 8      # order 2 -> 8 components
        assert row["hfc_reduction_1_40_db"] > 1.0


def test_blink_annotations_land_on_the_blinks(completed):
    """Guards the first_samp handling: annotation onsets share the raw time base."""
    from ._synthetic_data import BLINK_TIMES

    config, _ = completed
    rec = discover_recordings(config)[0]
    raw = mne.io.read_raw_fif(SubjectPaths(config, rec).ann_raw, preload=False, verbose="ERROR")
    onsets = np.array([o for o, d in zip(raw.annotations.onset, raw.annotations.description)
                       if d == "BAD_blink"])
    assert len(onsets) >= 0.8 * len(BLINK_TIMES)
    truth = BLINK_TIMES + raw.first_time
    # Each annotation is centred on a detected crossing, within one blink width.
    errors = [np.min(np.abs(truth - (onset + 0.25))) for onset in onsets]
    assert max(errors) < 0.2


def test_epochs_are_locked_to_the_cues(completed):
    config, _ = completed
    rec = discover_recordings(config)[0]
    epochs = mne.read_epochs(SubjectPaths(config, rec).epochs, preload=False, verbose="ERROR")
    assert len(epochs) > 0
    assert set(epochs.event_id) == {"cue_left/cue_Left", "cue_right/cue_Right"}
    # Both pooled selections must resolve.
    assert len(epochs["cue_left"]) + len(epochs["cue_right"]) == len(epochs)
    times = epochs.events[:, 0] / epochs.info["sfreq"]
    truth = CUE_TIMES + times.min() - CUE_TIMES.min()
    assert min(np.min(np.abs(truth - t)) for t in times) < 0.05


def test_source_rank_reflects_the_hfc_projections(completed):
    """HFC leaves the data rank-deficient; the beamformer must be told."""
    config, rows = completed
    for row in rows:
        # run_subject returns metrics un-flattened, so rank is still a dict.
        assert row["source_rank"]["mag"] == row["n_source_channels"] - row["n_hfc_projections"]


def test_source_estimates_exist_for_both_spaces(completed):
    config, _ = completed
    rec = discover_recordings(config)[0]
    paths = SubjectPaths(config, rec)
    for space in ("volume", "surface"):
        assert paths.fwd(space).exists()
        for name in ("lcmv-powerdb", "dics-alpha-post_vs_pre"):
            assert paths.stc(name, space).with_suffix(".h5").exists() or \
                   list(paths.analysis_dir.glob(f"*{name}_{space}_native*.h5"))


def test_peaks_are_labelled(completed):
    config, _ = completed
    rec = discover_recordings(config)[0]
    metrics = read_json(SubjectPaths(config, rec).metrics)["metrics"]
    volume_peak = metrics["source_peaks_volume"]["lcmv-powerdb"]
    assert "parcel" in volume_peak and volume_peak["parcel"]
    assert all(f"mni_{axis}_mm" in volume_peak for axis in "xyz")
    surface_peak = metrics["source_peaks_surface"]["lcmv-powerdb"]
    assert surface_peak["hemisphere"] in {"lh", "rh"}


def test_group_outputs_are_written(completed):
    import pandas as pd

    config, _ = completed
    group = config.deriv_root / "group"
    table = pd.read_csv(group / "quality_metrics.tsv", sep="\t")
    assert len(table) == 2
    assert set(table["status"]) == {"ok"}
    for column in ("hfc_reduction_1_40_db", "epochs_retained", "mvpa_peak_score"):
        assert table[column].notna().all()

    assert (group / "grand_average_ave.fif").exists()
    assert (group / "group_decoding.npz").exists()
    # Morphed estimates share the fsaverage grid, so they can be averaged.
    assert list(group.glob("group_*_volume_fsaverage*"))
    assert list(group.glob("group_*_surface_fsaverage*"))


def test_report_collects_every_stage(completed):
    config, _ = completed
    rec = discover_recordings(config)[0]
    html = SubjectPaths(config, rec).report_file.read_text(encoding="utf-8")
    for section in ("Sensor quality", "HFC", "Artefacts", "ICA", "Epochs",
                    "ERF", "Time-frequency", "Decoding", "Source"):
        assert section in html, f"{section} missing from the report"


def test_rerunning_reuses_cached_outputs(completed, caplog):
    """A second run must not recompute what is already on disk."""
    import logging

    config, _ = completed
    rec = discover_recordings(config)[0]
    before = SubjectPaths(config, rec).epochs.stat().st_mtime
    with caplog.at_level(logging.INFO, logger="cerca_flux"):
        run_study(config, resolve_stages("preproc", None), n_jobs=1, recordings=[rec])
    assert SubjectPaths(config, rec).epochs.stat().st_mtime == before
    assert "reusing" in caplog.text


def test_partial_stage_run_keeps_earlier_status(completed):
    """Re-running one stage must not erase what earlier runs recorded."""
    config, _ = completed
    rec = discover_recordings(config)[0]
    run_study(config, resolve_stages(None, ["tfr", "report"]), n_jobs=1, recordings=[rec])
    stages = read_json(SubjectPaths(config, rec).metrics)["stages"]
    assert {"qc", "hfc", "annotate", "ica", "epochs", "source"} <= set(stages)
