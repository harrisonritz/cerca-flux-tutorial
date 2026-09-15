"""Synthetic triaxial-OPM BIDS dataset mimicking the Cerca/QuSpin layout.

Not real anatomy or real brain activity: just enough structure - triaxial
channel naming, sensor geometry, a homogeneous interference field, blinks, a
cardiac component, two conditions and a bad sensor of each kind - for the
pipeline's behaviour to be exercised without any real data.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import mne
import mne_bids

SFREQ = 400.0
DURATION = 240.0
SLOTS = "ABCDEFGH"
#: Sensor ids echo the Cerca naming: F9/F10 frontal, C1-C6 central, O6/O7/Iz occipital.
SENSORS = ["F9", "F10", "C1", "C2", "C3", "C4", "C5", "C6", "O6", "O7", "Iz", "B4",
           "H6", "P1", "P2", "T1", "T2", "FZ", "CZ", "PZ", "OZ", "A1", "A2", "A3"]
#: Ground truth the tests assert against.
FLAT_SENSOR = "B4"
NOISY_SENSOR = "H6"
BLINK_TIMES = np.arange(12.0, DURATION - 5.0, 9.0)
CUE_TIMES = np.arange(6.0, DURATION - 6.0, 4.0)
CUE_LABELS = ["cue_Left" if i % 2 == 0 else "cue_Right" for i in range(len(CUE_TIMES))]


def _sensor_geometry() -> tuple[list[str], list[np.ndarray]]:
    """Channel names and MNE ``loc`` arrays for a triaxial array on a 9 cm sphere."""
    golden = np.pi * (3 - np.sqrt(5))
    ch_names, locs = [], []
    for index, sensor in enumerate(SENSORS):
        z = 1 - index / max(len(SENSORS) - 1, 1)
        radius = np.sqrt(max(1 - z * z, 1e-6))
        theta = golden * index
        unit = np.array([radius * np.cos(theta), radius * np.sin(theta), max(z, 0.05)])
        unit /= np.linalg.norm(unit)
        position = unit * 0.09

        ez = unit
        reference = np.array([0.0, 0.0, 1.0]) if abs(ez[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        ex = np.cross(reference, ez)
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)

        slot = f"{SLOTS[index % len(SLOTS)]}{index // len(SLOTS) + 1}"
        for axis, normal in (("X", ex), ("Y", ey), ("Z", ez)):
            ch_names.append(f"{sensor} {slot} {axis}")
            others = [v for v in (ex, ey, ez) if not np.allclose(v, normal)]
            # loc = position, two in-plane vectors, then the coil normal.
            locs.append(np.concatenate([position, others[0], others[1], normal]))
    return ch_names, locs


def _build_raw(rng: np.random.Generator) -> mne.io.RawArray:
    ch_names, locs = _sensor_geometry()
    info = mne.create_info(ch_names, SFREQ, ch_types="mag")
    with info._unlock():
        info["line_freq"] = 60.0
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc.astype(float)
        ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
        ch["coord_frame"] = mne.io.constants.FIFF.FIFFV_COORD_HEAD
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))

    n = int(DURATION * SFREQ)
    t = np.arange(n) / SFREQ
    data = rng.normal(0, 8e-14, (len(ch_names), n))

    # Spatially homogeneous interference, which HFC should attenuate.
    homogeneous = (2e-11 * np.sin(2 * np.pi * 0.2 * t)
                   + 4e-12 * np.sin(2 * np.pi * 60 * t)
                   + 6e-12 * np.cumsum(rng.normal(0, 1e-3, n)) / np.sqrt(n))
    for i, ch in enumerate(info["chs"]):
        data[i] += homogeneous * float(ch["loc"][9:12] @ np.array([0.3, 0.2, 0.9]))
        data[i] += rng.normal(0, 2e-13)  # static per-channel DC offset

    # Blinks of opposite sign on the frontal pair, so the bipolar surrogate sees them.
    half = int(0.15 * SFREQ)
    kernel = np.exp(-0.5 * ((np.arange(-half, half) / (0.05 * SFREQ)) ** 2))
    for i, ch in enumerate(ch_names):
        sensor = ch.split()[0]
        if sensor not in ("F9", "F10"):
            continue
        sign = 1.0 if sensor == "F9" else -1.0
        for onset in BLINK_TIMES:
            start = int(onset * SFREQ)
            usable = max(0, min(len(kernel), n - start))
            data[i, start:start + usable] += sign * 9e-12 * kernel[:usable]

    # Cardiac-like component on every channel.
    for beat in np.arange(1.0, DURATION, 0.9):
        start = int(beat * SFREQ)
        if start + 20 < n:
            data[:, start:start + 20] += 3e-13 * np.hanning(20)[None, :]

    # Evoked response per condition, plus alpha suppressed after each cue.
    span = int(0.4 * SFREQ)
    shape = np.sin(2 * np.pi * 6 * np.arange(span) / SFREQ) * np.hanning(span)
    for onset, label in zip(CUE_TIMES, CUE_LABELS):
        start = int(onset * SFREQ)
        if start + span >= n:
            continue
        for i, ch in enumerate(ch_names):
            sensor = ch.split()[0]
            if not ch.endswith(" Z"):
                continue
            if sensor in ("O6", "O7", "Iz", "OZ"):
                data[i, start:start + span] += 6e-13 * shape
            elif sensor in ("C1", "C2", "C3", "C4", "C5", "C6"):
                data[i, start:start + span] += 3e-13 * shape * (1.4 if label == "cue_Left" else 0.6)

    gain = np.ones(n)
    for onset in CUE_TIMES:
        gain[int((onset + 0.2) * SFREQ):min(int((onset + 1.0) * SFREQ), n)] = 0.45
    alpha = 5e-13 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi)) * gain
    for i, ch in enumerate(ch_names):
        if ch.split()[0] in ("O6", "O7", "Iz", "OZ", "PZ"):
            data[i] += alpha

    # One dead-quiet sensor and one very noisy one for the QC stage to catch.
    for i, ch in enumerate(ch_names):
        if ch.split()[0] == FLAT_SENSOR:
            data[i] *= 0.001
        if ch.split()[0] == NOISY_SENSOR:
            data[i] += rng.normal(0, 3e-11, n)

    raw = mne.io.RawArray(data, info, first_samp=1234, verbose="ERROR")
    raw.set_meas_date(1_600_000_000.0)
    raw.set_annotations(mne.Annotations(
        onset=CUE_TIMES + raw.first_time,
        duration=np.zeros(len(CUE_TIMES)),
        description=CUE_LABELS,
        orig_time=raw.info["meas_date"],
    ))
    return raw


def make_synthetic_bids(out, subjects=("01", "02"), labels=None) -> Path:
    """Write the synthetic recording as a BIDS dataset and return its root.

    ``labels`` overrides the event descriptions, which lets a test build a
    recording whose ``trial_type`` values do not match a configuration.
    """
    mne.set_log_level("ERROR")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)

    raw = _build_raw(np.random.default_rng(7))
    if labels is not None:
        annotations = raw.annotations
        raw.set_annotations(mne.Annotations(
            annotations.onset, annotations.duration,
            list(labels) * len(annotations) if len(labels) == 1 else list(labels),
            orig_time=annotations.orig_time,
        ))

    for subject in subjects:
        bids_path = mne_bids.BIDSPath(subject=subject, session="01", task="SpAtt",
                                      run="01", datatype="meg", root=out)
        mne_bids.write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True,
                                format="FIF", verbose="ERROR")
    return out


if __name__ == "__main__":
    print(make_synthetic_bids(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth_bids"))
