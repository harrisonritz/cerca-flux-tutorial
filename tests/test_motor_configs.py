"""The shipped Oxford and Princeton configurations."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

from cerca_flux.config import Config, load_config
from cerca_flux.provenance import (
    PREPROCESSING_SECTIONS, fingerprints, flat_settings, settings_differences,
)

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


def test_oxford_pools_both_presses_and_uses_only_the_raw_recording_and_freesurfer(sites, tsx_root):
    oxford, _ = sites
    assert oxford.bids_root == tsx_root / "data" / "oxford_sub-01" / "Data" / "Cerca_Spatt_BIDS"
    assert oxford.epochs.conditions == {"response": ["resp_T", "resp_L"]}
    assert oxford.fs_subjects_dir == oxford.bids_root / "derivatives" / "Freesurfer"
    assert oxford.study.fs_subject_template == "T1s"
    # Oxford's own forward model and BEM are NOT used: both are rebuilt like Princeton's.
    assert oxford.forward.trans is None and oxford.forward.bem is None
    assert oxford.channels.manual_bads == {"01": ["B4", "H6"]}
    assert oxford.channels.bad_sensor_match == "substring"
    assert oxford.study.crop_start == 0.0 and oxford.study.match_duration_to is None


def test_both_sites_run_in_strict_provenance_mode_and_never_name_a_precomputed_input(sites):
    for cfg in sites:
        assert cfg.provenance.strict is True
        assert cfg.forward.bem is None and cfg.forward.trans is None


def test_the_data_folder_can_be_relocated_with_one_variable(monkeypatch, tmp_path):
    """E.g. a mounted volume that is not under the TSX folder."""
    monkeypatch.setenv("TSX_DIR", str(tmp_path / "TSX"))
    monkeypatch.setenv("TSX_DATA", str(tmp_path / "mount" / "data"))
    monkeypatch.delenv("OXFORD_BIDS", raising=False)
    princeton = load_config(CONFIG_DIR / "princeton.yaml")
    oxford = load_config(CONFIG_DIR / "oxford.yaml")
    assert princeton.bids_root == tmp_path / "mount" / "data" / "TSX" / "bids"
    assert princeton.fs_subjects_dir == tmp_path / "mount" / "data" / "TSX" / "freesurfer"
    assert oxford.bids_root == (tmp_path / "mount" / "data" / "oxford_sub-01" / "Data"
                                / "Cerca_Spatt_BIDS")


def test_oxford_bids_can_be_relocated_with_one_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("OXFORD_BIDS", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("TSX_DIR", str(tmp_path / "TSX"))
    oxford = load_config(CONFIG_DIR / "oxford.yaml")
    princeton = load_config(CONFIG_DIR / "princeton.yaml")
    assert oxford.bids_root == tmp_path / "elsewhere"
    assert str(princeton.reference_recording).startswith(str(tmp_path / "elsewhere"))


def test_the_two_sites_differ_only_where_they_must(sites):
    """Every shared analysis choice is identical, so a preprocessing option moves both sites."""
    oxford_cfg, princeton_cfg = sites
    assert settings_differences(oxford_cfg, princeton_cfg) == []
    assert (fingerprints(oxford_cfg)["fingerprint_shared"]
            == fingerprints(princeton_cfg)["fingerprint_shared"])
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


# -- preprocessing.yaml: every setting listed, every marker true ------------------- #

MARKER = re.compile(r"\[(?:= Cerca default|CHANGED from Cerca default: (?P<default>.*?))\](?=\s|$)")


def _marked_settings(path: Path) -> dict[str, tuple[bool, object]]:
    """``{"hfc.order": (changed, stated_default)}`` read from the comments of a YAML file."""
    stack: list[tuple[int, str]] = []
    marks: dict[str, tuple[bool, object]] = {}
    for line in path.read_text().splitlines():
        code = line.split("#", 1)[0].rstrip()
        if not code.strip():
            continue
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, code.strip().split(":", 1)[0]))
        found = MARKER.search(line.split("#", 1)[1]) if "#" in line else None
        if found:
            marks[".".join(key for _, key in stack)] = (
                found.group("default") is not None,
                yaml.safe_load(found.group("default")) if found.group("default") is not None else None,
            )
    return marks


def test_preprocessing_yaml_lists_every_shared_preprocessing_setting():
    """A new preprocessing option added to the code must be added to the file."""
    listed = set(_marked_settings(CONFIG_DIR / "preprocessing.yaml"))
    existing = set(flat_settings(Config(), PREPROCESSING_SECTIONS))
    assert listed == existing, (
        f"missing from preprocessing.yaml: {sorted(existing - listed)}; "
        f"not a setting: {sorted(listed - existing)}"
    )


def test_every_marker_in_preprocessing_yaml_is_true():
    """'[= Cerca default]' really is the default, and '[CHANGED ...]' really differs."""
    path = CONFIG_DIR / "preprocessing.yaml"
    data = yaml.safe_load(path.read_text())
    values = flat_settings(_build_from(data), PREPROCESSING_SECTIONS)
    defaults = flat_settings(Config(), PREPROCESSING_SECTIONS)
    for key, (changed, stated) in _marked_settings(path).items():
        if changed:
            assert values[key] != defaults[key], f"{key} is marked CHANGED but equals the default"
            assert stated == defaults[key], f"{key}: the stated default {stated!r} is not {defaults[key]!r}"
        else:
            assert values[key] == defaults[key], f"{key} is marked as the default but is {values[key]!r}"


def _build_from(data: dict) -> Config:
    from cerca_flux.config import _build
    return _build(Config, {**data, "study": {"bids_root": "/x"}})


def test_the_changes_from_the_cerca_defaults_are_exactly_the_documented_ones():
    changed = {key for key, (flag, _) in _marked_settings(CONFIG_DIR / "preprocessing.yaml").items()
               if flag}
    assert changed == {
        "qc.psd_n_fft_seconds", "qc.flag_flat", "hfc.resample_sfreq",
        "annotate.eog.prefer_native", "annotate.muscle.filter_freq",
        "ica.random_state", "ica.detect_ecg", "ica.max_exclude",
        "epochs.continuous_h_freq", "epochs.tmin", "epochs.tmax", "epochs.reject.mag",
    }


def test_site_files_do_not_restate_preprocessing_settings(sites):
    """Preprocessing lives in one file; the site files hold only what is per-site."""
    allowed = {"channels": {"manual_bads", "bad_sensor_match"}, "epochs": {"conditions"},
               "study": None, "forward": None, "extends": None}
    for name in ("oxford.yaml", "princeton.yaml"):
        data = yaml.safe_load((CONFIG_DIR / name).read_text())
        assert data["extends"] == ["preprocessing.yaml", "motor.yaml"]
        for section, body in data.items():
            assert section in allowed, f"{name} sets {section}, which belongs in preprocessing.yaml/motor.yaml"
            if allowed[section]:
                assert set(body) <= allowed[section], (name, section, sorted(body))
