"""Shared construction of the synthetic response-locked study used by the motor tests."""

from __future__ import annotations

from cerca_flux.config import Config, _build, _deep_merge

from ._synthetic_data import CUE_TIMES

#: Alternating labels, as Oxford's two button presses are pooled into one response.
RESPONSE_LABELS = ["resp_T" if i % 2 == 0 else "resp_L" for i in range(len(CUE_TIMES))]


def motor_config(response_study, **sections) -> Config:
    """A small, fast configuration for the response-locked synthetic dataset."""
    bids_root, fs_dir = response_study
    base = {
        "study": {"bids_root": str(bids_root), "fs_subjects_dir": str(fs_dir),
                  "fs_subject_template": "sub-{subject}", "line_freq": 60,
                  "variant": "motor-test"},
        "channels": {"manual_bads": {"*": ["B4"]}},
        "qc": {"psd_tmin": 20.0, "psd_span": 100.0, "first_look_tmin": 30.0},
        "hfc": {"order": 2, "resample_sfreq": 300.0},
        "annotate": {"eog": {"prefer_native": False}, "muscle": {"filter_freq": [110, 130]}},
        "ica": {"n_components": 15, "max_exclude": 3, "detect_ecg": False},
        "epochs": {"conditions": {"response": ["resp_T", "resp_L"]}, "tmin": -1.0, "tmax": 1.2,
                   "continuous_h_freq": 45.0, "reject": {"mag": 5e-11}},
        "erf": {"enabled": False}, "tfr": {"enabled": False}, "mvpa": {"enabled": False},
        "source": {"enabled": False, "spaces": ["surface"]}, "morph": {"enabled": False},
        "forward": {"volume": {"enabled": False}, "surface": {"spacing": "oct5"}},
        "motor": {"enabled": True},
        "provenance": {"strict": True},   # every pipeline test runs under enforcement
        "group": {"enabled": False},
    }
    return _build(Config, _deep_merge(base, sections))
