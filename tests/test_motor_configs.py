"""The shipped Oxford and Princeton configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cerca_flux.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "motor"
#: The full TSX sample, as defined by della_mne_batch.sh / local_channel.sh.
FULL_SAMPLE = [f"{i:03d}" for i in [*range(7, 16), *range(18, 40), *range(41, 45)]]


@pytest.fixture
def tsx_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TSX_DIR", str(tmp_path / "TSX"))
    monkeypatch.delenv("OXFORD_BIDS", raising=False)
    return tmp_path / "TSX"


@pytest.fixture
def sites(tsx_root):
    return load_config(CONFIG_DIR / "oxford.yaml"), load_config(CONFIG_DIR / "princeton.yaml")


def test_princeton_uses_the_tsx_opm_data_layout(sites, tsx_root):
    _, princeton = sites
    assert princeton.bids_root == tsx_root / "data" / "TSX" / "bids"
    assert princeton.fs_subjects_dir == tsx_root / "data" / "TSX" / "freesurfer"
    assert princeton.study.fs_subject_template == "sub-{subject}_ses-{session}"
    assert princeton.study.tasks == ["TSX"]            # not the task-noise empty-room file
    assert princeton.study.line_freq == 60.0


def test_princeton_is_the_full_35_subject_sample(sites):
    _, princeton = sites
    assert princeton.study.subjects == FULL_SAMPLE and len(FULL_SAMPLE) == 35
    assert all(isinstance(s, str) for s in princeton.study.subjects)
    assert princeton.epochs.conditions == {"response": ["response/right"]}


def test_princeton_is_cropped_and_matched_to_the_oxford_recording(sites, tsx_root):
    _, princeton = sites
    assert princeton.study.crop_start == 400.0
    assert princeton.reference_recording == (
        tsx_root / "data" / "oxford_sub-01" / "Data" / "Cerca_Spatt_BIDS" / "sub-01"
        / "ses-01" / "meg" / "sub-01_ses-01_task-SpAtt_run-01_meg.fif"
    )


def test_oxford_pools_both_presses_and_reads_its_transform_from_the_forward(sites, tsx_root):
    oxford, _ = sites
    assert oxford.bids_root == tsx_root / "data" / "oxford_sub-01" / "Data" / "Cerca_Spatt_BIDS"
    assert oxford.epochs.conditions == {"response": ["resp_T", "resp_L"]}
    assert oxford.fs_subjects_dir == oxford.bids_root / "derivatives" / "Freesurfer"
    assert oxford.study.fs_subject_template == "T1s"
    assert oxford.forward.trans.endswith("_fwd.fif") and oxford.forward.bem.endswith("_bem-sol.fif")
    assert oxford.channels.manual_bads == {"01": ["B4", "H6"]}
    assert oxford.channels.bad_sensor_match == "substring"
    assert oxford.study.crop_start == 0.0 and oxford.study.match_duration_to is None


def test_oxford_bids_can_be_relocated_with_one_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("OXFORD_BIDS", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("TSX_DIR", str(tmp_path / "TSX"))
    oxford = load_config(CONFIG_DIR / "oxford.yaml")
    princeton = load_config(CONFIG_DIR / "princeton.yaml")
    assert oxford.bids_root == tmp_path / "elsewhere"
    assert str(princeton.reference_recording).startswith(str(tmp_path / "elsewhere"))


def test_the_two_sites_differ_only_where_they_must(sites):
    """Every shared analysis choice is identical, so a preprocessing option moves both sites."""
    oxford, princeton = (copy.deepcopy(c.to_dict()) for c in sites)
    for cfg in (oxford, princeton):
        for key in ("bids_root", "fs_subjects_dir", "fs_subject_template", "subjects",
                    "sessions", "tasks", "line_freq", "site", "crop_start",
                    "match_duration_to"):
            cfg["study"].pop(key)
        cfg["epochs"].pop("conditions")
        cfg["channels"].pop("manual_bads")
        cfg["channels"].pop("bad_sensor_match")
        cfg["forward"].pop("trans")
        cfg["forward"].pop("bem")
    assert oxford == princeton


def test_the_notebook_matched_analysis_choices(sites):
    for cfg in sites:
        assert cfg.hfc.order == 2 and cfg.hfc.resample_sfreq == 300.0
        assert cfg.annotate.eog.prefer_native is False        # shared frontal-OPM blink detector
        assert cfg.annotate.muscle.filter_freq == [110.0, 130.0]
        assert (cfg.ica.detect_ecg, cfg.ica.max_exclude) == (False, 3)
        assert (cfg.epochs.tmin, cfg.epochs.tmax) == (-1.0, 1.2)
        assert cfg.epochs.continuous_h_freq == 45.0
        assert cfg.epochs.reject == {"mag": pytest.approx(5e-11)}
        assert cfg.motor.enabled and cfg.motor.sensor.sensors == ["C1", "C2", "C3", "C4", "C5", "C6"]
        assert cfg.source.spaces == ["surface"] and cfg.forward.volume.enabled is False


def test_motor_outputs_never_share_a_folder_with_a_plain_run(sites):
    for cfg in sites:
        assert cfg.deriv_root.name == "cerca-flux-motor"
    variant = load_config(CONFIG_DIR / "princeton.yaml", variant="hfc3")
    assert variant.deriv_root.name == "cerca-flux-motor_hfc3"
