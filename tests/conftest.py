"""Session-scoped synthetic study, built once and reused by every test."""

from __future__ import annotations

import pytest

from cerca_flux.config import Config, _build
from cerca_flux.utils import use_headless_backend

from ._synthetic_data import make_synthetic_bids
from ._synthetic_fs import make_synthetic_freesurfer

use_headless_backend()


@pytest.fixture(scope="session")
def synthetic_study(tmp_path_factory):
    """A two-subject synthetic BIDS dataset plus matching FreeSurfer subjects."""
    root = tmp_path_factory.mktemp("study")
    bids_root = make_synthetic_bids(root / "bids", subjects=("01", "02"))
    fs_dir = root / "freesurfer"
    for subject in ("sub-01", "sub-02", "fsaverage"):
        make_synthetic_freesurfer(fs_dir, subject)
    return bids_root, fs_dir


@pytest.fixture(scope="session")
def config(synthetic_study) -> Config:
    """A configuration sized down so the whole pipeline runs in a test."""
    bids_root, fs_dir = synthetic_study
    return _build(Config, {
        "study": {
            "bids_root": str(bids_root),
            "fs_subjects_dir": str(fs_dir),
            "fs_subject_template": "sub-{subject}",
            "line_freq": 60,
        },
        "channels": {"manual_bads": {"*": ["B4"]}},
        "qc": {"psd_tmin": 20.0, "psd_span": 100.0, "first_look_tmin": 30.0},
        "hfc": {"order": 2, "resample_sfreq": 300.0},
        "ica": {"n_components": 15, "max_exclude": 3},
        "epochs": {
            "conditions": {"cue_left": ["cue_Left"], "cue_right": ["cue_Right"]},
            "tmin": -0.75, "tmax": 2.0, "reject": {"mag": 5e-11},
        },
        "tfr": {
            "bands": [{"name": "slow", "fmin": 4.0, "fmax": 30.0, "fstep": 2.0,
                       "n_cycles_divisor": 2.0, "time_bandwidth": 2.0,
                       "baseline": [-0.5, -0.25]}],
            "contrasts": [{"name": "alpha_lat", "band": "slow", "a": "cue_right",
                           "b": "cue_left", "kind": "normalised_difference"}],
        },
        "mvpa": {"conditions": ["cue_left", "cue_right"], "cv": 5, "step_samples": 10},
        "forward": {"volume": {"pos": 10.0}, "surface": {"spacing": "oct5"}},
        "source": {
            "spaces": ["volume", "surface"],
            "lcmv": {"cov_tmin": -0.7, "cov_tmax": 1.5,
                     "baseline_window": [-0.7, -0.3], "active_window": [0.05, 0.45]},
            "dics": {"bands": [{
                "name": "alpha", "fmin": 8.0, "fmax": 12.0, "bandwidth": 3.0,
                "windows": {"pre": [-0.7, -0.2], "post": [0.2, 0.7]},
                "contrasts": [{"name": "post_vs_pre", "a": "post", "b": "pre",
                               "kind": "relative_change"}],
            }]},
        },
        "morph": {"subject_to": "fsaverage", "volume_zooms": 10.0,
                  "surface_spacing": 4, "fetch_fsaverage": False},
    })
