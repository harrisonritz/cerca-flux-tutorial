"""Preprocessing stages, in the order recommended by the OPM-FLUX pipeline.

    1. :func:`load_raw`        - read the BIDS recording (A First Look)
    2. :func:`check_sensors`   - bad-sensor identification (Sensor Quality checking)
    3. :func:`apply_hfc`       - homogeneous field correction (ArtefactSuppressionHFC)
    4. :func:`annotate_artefacts` - blink and muscle annotation (ArtefactAnnotation)
    5. :func:`run_ica`         - ocular/cardiac component removal (ICA)
    6. :func:`make_epochs`     - condition-specific trials (ConditionSpecificTrials)

Each function is independently callable on a :class:`~cerca_flux.context.SubjectContext`
and caches its output in the BIDS derivatives tree, so a stage can be re-run in
isolation without repeating the ones before it.
"""

from __future__ import annotations

import numpy as np
import mne
import mne_bids
from scipy.stats import median_abs_deviation

from .context import SubjectContext
from .paths import raw_bids_path
from .utils import (
    axis_of,
    expand_bad_sensors,
    find_channel_by_sensor,
    manual_bads_for,
    psd_db,
    radial_channels,
    save_figure,
    timed,
)


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #


def load_raw(ctx: SubjectContext, preload: bool = False) -> mne.io.BaseRaw:
    """Read the BIDS recording and freeze the task event labels.

    ``read_raw_bids`` turns ``events.tsv`` into annotations.  Their descriptions
    are recorded now, before blink/muscle annotations are added, so that
    epoching can tell task events from artefact markers.
    """
    cfg = ctx.cfg
    bids_path = raw_bids_path(cfg, ctx.rec)
    raw = mne_bids.read_raw_bids(
        bids_path=bids_path, extra_params={"preload": preload}, verbose="ERROR"
    )

    if raw.info.get("line_freq") is None and cfg.study.line_freq is not None:
        raw.info["line_freq"] = cfg.study.line_freq
    if raw.info.get("line_freq") is None:
        ctx.logger.warning(
            "no line_freq in the BIDS sidecar and study.line_freq is unset; "
            "mains diagnostics will be skipped"
        )

    labels = sorted({str(d) for d in raw.annotations.description})
    if not labels:
        raise RuntimeError(
            f"{ctx.rec.key}: no annotations were read from the BIDS events.tsv. "
            "Epoching needs trial_type labels."
        )
    # A deterministic label -> code map, identical across subjects, so that
    # saved epochs can be compared between recordings.
    ctx.state.setdefault("task_event_labels", labels)
    ctx.state.setdefault(
        "task_event_id", {label: i + 1 for i, label in enumerate(ctx.state["task_event_labels"])}
    )

    types = raw.get_channel_types()
    ctx.record(
        sfreq_raw=float(raw.info["sfreq"]),
        duration_s=float(raw.times[-1]),
        n_mag=int(sum(t == "mag" for t in types)),
        n_eog=int(sum(t == "eog" for t in types)),
        line_freq=raw.info.get("line_freq"),
    )
    ctx.logger.info(
        "loaded %s: %.1f s at %.0f Hz, %d magnetometers",
        ctx.rec.key, raw.times[-1], raw.info["sfreq"], ctx.metrics["n_mag"],
    )
    return raw


# --------------------------------------------------------------------------- #
# 2. Sensor quality check
# --------------------------------------------------------------------------- #


def _quality_spectrum(raw: mne.io.BaseRaw, picks: list[str], cfg) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD in dB(fT^2/Hz) over a fixed, artefact-light stretch."""
    qc = cfg.qc
    nyquist = raw.info["sfreq"] / 2
    fmax = min(qc.psd_fmax, nyquist - 1)
    tmax = min(raw.times[-1], qc.psd_tmin + qc.psd_span)
    tmin = min(qc.psd_tmin, max(0.0, tmax - 1.0))
    n_fft = int(round(qc.psd_n_fft_seconds * raw.info["sfreq"]))
    n_fft = min(n_fft, int((tmax - tmin) * raw.info["sfreq"]))
    spectrum = raw.compute_psd(
        method="welch", fmin=qc.psd_fmin, fmax=fmax, tmin=tmin, tmax=tmax,
        picks=picks, n_fft=n_fft, n_overlap=n_fft // 2,
        reject_by_annotation=False, verbose="ERROR",
    )
    return spectrum.freqs, psd_db(spectrum.get_data(picks=picks, exclude=[]))


def check_sensors(ctx: SubjectContext, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Flag faulty and unusually noisy sensors before they enter the HFC model.

    HFC is a linear projection across the array, so a noisy sensor left in would
    spread its noise everywhere.  Two sources of bad channels are combined:
    sensors noted during acquisition (``channels.manual_bads``) and a
    power-spectral outlier rule.

    ``qc.method='robust_mad'`` flags channels above median + k.MAD *within each
    measurement axis*.  This data-adaptive rule is preferable to transferring
    one site's absolute dB threshold to another acquisition; ``'fixed_db'``
    reproduces the tutorial's 39.1 dB cut when you need exactly that.
    """
    cfg = ctx.cfg
    channels = cfg.channels

    manual = manual_bads_for(ctx.rec.subject, channels)
    manual_channels = expand_bad_sensors(raw.ch_names, manual, channels)
    metadata_bads = list(raw.info["bads"])
    raw.info["bads"] = sorted(set(metadata_bads) | set(manual_channels))

    if not cfg.qc.enabled or cfg.qc.method == "none":
        ctx.record(bad_channels=list(raw.info["bads"]), auto_bad_channels=[])
        ctx.state["bads"] = list(raw.info["bads"])
        return raw

    good_mag = [
        ch for ch, typ in zip(raw.ch_names, raw.get_channel_types())
        if typ == "mag" and ch not in raw.info["bads"]
    ]
    if not good_mag:
        raise RuntimeError(f"{ctx.rec.key}: every magnetometer is already marked bad")

    freqs, db = _quality_spectrum(raw, good_mag, cfg)
    mean_db = db.mean(axis=1)

    noisy: list[str] = []
    flat: list[str] = []
    thresholds: dict[str, float] = {}
    robust_z = np.zeros_like(mean_db)
    if cfg.qc.method == "robust_mad":
        for axis in channels.axes:
            idx = np.array(
                [i for i, ch in enumerate(good_mag) if axis_of(ch, channels) == axis.upper()],
                dtype=int,
            )
            if idx.size < 3:
                continue
            median = float(np.median(mean_db[idx]))
            scale = max(1.4826 * float(median_abs_deviation(mean_db[idx], scale=1.0)), 1e-6)
            robust_z[idx] = (mean_db[idx] - median) / scale
            thresholds[axis] = median + cfg.qc.mad_threshold * scale
        noisy = [ch for ch, z in zip(good_mag, robust_z) if z > cfg.qc.mad_threshold]
        if cfg.qc.flag_flat:
            flat = [ch for ch, z in zip(good_mag, robust_z) if z < -cfg.qc.flat_mad_threshold]
    else:  # fixed_db
        thresholds = {axis: cfg.qc.fixed_db_threshold for axis in channels.axes}
        noisy = [ch for ch, value in zip(good_mag, mean_db) if value > cfg.qc.fixed_db_threshold]
    auto_bad = sorted(set(noisy) | set(flat))

    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(auto_bad))
    ctx.state["bads"] = list(raw.info["bads"])
    ctx.record(
        bad_channels=list(raw.info["bads"]),
        auto_bad_channels=sorted(auto_bad),
        noisy_channels=sorted(noisy),
        flat_channels=sorted(flat),
        n_bad_channels=len(raw.info["bads"]),
        n_auto_bad_channels=len(auto_bad),
        n_manual_bad_channels=len(set(raw.info["bads"]) - set(auto_bad)),
        qc_threshold_db=thresholds,
    )
    ctx.logger.info(
        "bad channels: %d total (%d automatic by %s: %d noisy, %d flat)",
        len(raw.info["bads"]), len(auto_bad), cfg.qc.method, len(noisy), len(flat),
    )

    _plot_sensor_quality(ctx, raw, good_mag, freqs, db, mean_db, thresholds)
    return raw


def _plot_sensor_quality(ctx, raw, names, freqs, db, mean_db, thresholds) -> None:
    import matplotlib.pyplot as plt

    cfg = ctx.cfg
    channels = cfg.channels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    for axis in channels.axes:
        values = [v for v, ch in zip(mean_db, names) if axis_of(ch, channels) == axis.upper()]
        if not values:
            continue
        axes[0].hist(values, bins=18, alpha=0.55, label=axis)
        if axis in thresholds:
            axes[0].axvline(thresholds[axis], color="k", lw=0.8, ls="--")
    axes[0].set(
        title=f"{ctx.rec.key}: sensor power",
        xlabel=r"Mean PSD (dB fT$^2$/Hz)", ylabel="Channels",
    )
    axes[0].legend(title="Axis", fontsize=8)

    axes[1].plot(freqs, db.T, lw=0.4, alpha=0.5)
    axes[1].plot(freqs, db.mean(axis=0), color="k", lw=1.6, label="Mean")
    line = raw.info.get("line_freq")
    if line:
        axes[1].axvline(float(line), color="0.3", ls=":", lw=1, label=f"{float(line):g} Hz mains")
    axes[1].set(
        title="Power spectra (good channels)",
        xlabel="Frequency (Hz)", ylabel=r"PSD (dB fT$^2$/Hz)",
    )
    axes[1].legend(fontsize=8)

    path = save_figure(fig, ctx.figure_path("01_sensor_quality"), cfg)
    ctx.add_figure("Sensor quality", "Sensor power and spectra", path)

    # Channel-centred radial RMS: the tutorial's "first look" trace, with each
    # channel's static DC offset removed so that dynamic fields are compared.
    radial = radial_channels(raw, channels)
    if not radial:
        return
    sfreq = raw.info["sfreq"]
    start = int(min(cfg.qc.first_look_tmin, max(0.0, raw.times[-1] - cfg.qc.first_look_span)) * sfreq)
    stop = min(int(start + cfg.qc.first_look_span * sfreq), len(raw.times))
    picks = [raw.ch_names.index(ch) for ch in radial]
    data, times = raw[picks, start:stop]
    centered = data - data.mean(axis=1, keepdims=True)
    dynamic_rms = float(np.sqrt(np.mean(centered**2)) * 1e12)
    absolute_rms = float(np.sqrt(np.mean(data**2)) * 1e12)
    ctx.record(
        first_look_dynamic_rms_pT=dynamic_rms,
        first_look_dc_inclusive_rms_pT=absolute_rms,
        first_look_dc_inflation=absolute_rms / max(dynamic_rms, 1e-15),
    )

    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    ax.plot(times - times[0], np.sqrt(np.mean(centered**2, axis=0)) * 1e12, lw=0.8)
    ax.axhline(dynamic_rms, color="k", ls="--", lw=0.8, label="window RMS")
    ax.set(
        title=f"{ctx.rec.key}: channel-centred radial field",
        xlabel="Time in excerpt (s)", ylabel="Dynamic RMS (pT)",
    )
    ax.legend(fontsize=8)
    path = save_figure(fig, ctx.figure_path("02_first_look"), cfg)
    ctx.add_figure("Sensor quality", "Channel-centred radial field", path)


# --------------------------------------------------------------------------- #
# 3. Homogeneous field correction
# --------------------------------------------------------------------------- #


def apply_hfc(ctx: SubjectContext, raw: mne.io.BaseRaw | None = None) -> mne.io.BaseRaw:
    """Project out the spatially homogeneous interference field.

    Environmental interference and movement of the array through the room's
    residual field are approximately uniform across the sensors, whereas
    neuronal sources are close enough to produce strong spatial gradients.  HFC
    models the uniform part with spherical harmonics up to ``hfc.order`` and
    removes it as a projection.  Bad sensors are excluded first, otherwise the
    projection would smear their noise across the whole array.

    The triaxial array is kept intact through the projection (``hfc.keep_axes:
    all``), because the X and Y channels constrain the homogeneous model; the
    reduction to radial channels happens afterwards if requested.
    """
    cfg = ctx.cfg
    out_file = ctx.paths.hfc_raw

    if not ctx.needs(out_file):
        ctx.logger.info("HFC: reusing %s", out_file.name)
        return mne.io.read_raw_fif(out_file, preload=True, verbose="ERROR")

    if raw is None:
        raw = check_sensors(ctx, load_raw(ctx))

    keep = [
        ch for ch, typ in zip(raw.ch_names, raw.get_channel_types())
        if typ in {"mag", "eog"} and ch not in raw.info["bads"]
    ]
    work = raw.copy().pick(keep).load_data(verbose="ERROR")

    if cfg.hfc.resample_sfreq and cfg.hfc.resample_sfreq < work.info["sfreq"]:
        muscle_high = max(cfg.annotate.muscle.filter_freq)
        if cfg.annotate.enabled and cfg.annotate.muscle.enabled and cfg.hfc.resample_sfreq <= 2 * muscle_high:
            raise ValueError(
                f"hfc.resample_sfreq={cfg.hfc.resample_sfreq} Hz is at or below twice the "
                f"muscle band upper edge ({muscle_high} Hz). Raise it above "
                f"{2 * muscle_high} Hz or narrow annotate.muscle.filter_freq."
            )
        with timed(f"resampling to {cfg.hfc.resample_sfreq:g} Hz", ctx.logger):
            work.resample(cfg.hfc.resample_sfreq, npad="auto", verbose="ERROR")

    radial_before = radial_channels(work, cfg.channels)
    freqs, before_db = _quality_spectrum(work, radial_before, cfg)

    if cfg.hfc.enabled:
        projs = mne.preprocessing.compute_proj_hfc(
            work.info, order=cfg.hfc.order, exclude="bads", verbose="ERROR"
        )
        work.add_proj(projs).apply_proj(verbose="ERROR")
        ctx.record(n_hfc_projections=len(projs), hfc_order=cfg.hfc.order)
        ctx.logger.info("HFC: applied %d projections (order %d)", len(projs), cfg.hfc.order)
    else:
        ctx.record(n_hfc_projections=0)
        ctx.logger.info("HFC: disabled by config")

    _, after_db = _quality_spectrum(work, radial_before, cfg)
    _record_hfc_diagnostics(ctx, work, freqs, before_db, after_db, radial_before)

    if cfg.hfc.keep_axes == "radial":
        keep_radial = radial_channels(work, cfg.channels) + [
            ch for ch, typ in zip(work.ch_names, work.get_channel_types()) if typ == "eog"
        ]
        work.pick(keep_radial)
        work.info.normalize_proj()

    work.save(out_file, overwrite=True, verbose="ERROR")
    ctx.logger.info("HFC: wrote %s", out_file.name)
    return work


def _record_hfc_diagnostics(ctx, raw, freqs, before_db, after_db, radial) -> None:
    import matplotlib.pyplot as plt

    before = before_db.mean(axis=0)
    after = after_db.mean(axis=0)
    band = (freqs >= 1) & (freqs <= 40)
    metrics = {
        "n_radial_channels": len(radial),
        "psd_before_hfc_db": float(np.mean(before[band])) if band.any() else None,
        "psd_after_hfc_db": float(np.mean(after[band])) if band.any() else None,
        "hfc_reduction_1_40_db": float(np.median(before[band] - after[band])) if band.any() else None,
    }
    line = raw.info.get("line_freq")
    if line:
        line = float(line)
        at_line = np.abs(freqs - line) <= 1
        flank = (np.abs(freqs - line) >= 3) & (np.abs(freqs - line) <= 8)
        if at_line.any() and flank.any():
            metrics["line_excess_before_hfc_db"] = float(np.mean(before[at_line]) - np.mean(before[flank]))
            metrics["line_excess_after_hfc_db"] = float(np.mean(after[at_line]) - np.mean(after[flank]))
    ctx.record(**metrics)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(freqs, before, alpha=0.5, label="Before HFC")
    ax.plot(freqs, after, lw=1.8, label="After HFC")
    if line:
        ax.axvline(line, color="0.3", ls=":", lw=1, label=f"{line:g} Hz mains")
    ax.set(
        title=f"{ctx.rec.key}: HFC effect (n={len(radial)} radial channels)",
        xlabel="Frequency (Hz)", ylabel=r"Mean PSD (dB fT$^2$/Hz)",
    )
    ax.legend(fontsize=8)
    path = save_figure(fig, ctx.figure_path("03_hfc_psd"), ctx.cfg)
    ctx.add_figure("HFC", "Spectra before and after HFC", path)


# --------------------------------------------------------------------------- #
# 4. Artefact annotation
# --------------------------------------------------------------------------- #


def _annotation_onsets(raw: mne.io.BaseRaw, samples: np.ndarray) -> np.ndarray:
    """Convert absolute event samples to the time base of ``raw.annotations``.

    ``mne`` event samples include ``first_samp``, and annotation onsets are
    expressed in that same acquisition time base, so dividing by the sampling
    rate is the whole conversion.
    """
    return np.asarray(samples, dtype=float) / raw.info["sfreq"]


def _reset_cached_projector(raw: mne.io.BaseRaw) -> None:
    """Drop the projection matrix MNE cached before a channel was added.

    ``Raw.add_channels`` does not resize the cached projector, so a later
    ``pick`` indexes past the end of it.  The HFC projections have already been
    applied to the data at this point and stay described in ``info['projs']``
    (so the rank they remove is still reported correctly); only the stale cache
    has to go, and MNE rebuilds it on demand.
    """
    if getattr(raw, "_projector", None) is not None:
        raw._projector = None


def _build_eog_surrogate(ctx: SubjectContext, raw: mne.io.BaseRaw) -> str | None:
    """Create a bipolar frontal OPM channel that behaves like an EOG electrode."""
    cfg = ctx.cfg.annotate.eog
    channels = ctx.cfg.channels
    left = find_channel_by_sensor(raw, cfg.surrogate_left, channels)
    right = find_channel_by_sensor(raw, cfg.surrogate_right, channels)
    if left is None or right is None:
        ctx.logger.warning(
            "cannot build the frontal EOG surrogate: sensors %s/%s not both present as "
            "radial channels; ocular detection will be skipped",
            cfg.surrogate_left, cfg.surrogate_right,
        )
        return None
    signal = raw.get_data(picks=[left])[0] - raw.get_data(picks=[right])[0]
    info = mne.create_info([cfg.surrogate_name], raw.info["sfreq"], ["eog"])
    proxy = mne.io.RawArray(signal[np.newaxis], info, first_samp=raw.first_samp, verbose="ERROR")
    raw.add_channels([proxy], force_update_info=True)
    _reset_cached_projector(raw)
    ctx.logger.info("built EOG surrogate %s = %s - %s", cfg.surrogate_name, left, right)
    return cfg.surrogate_name


def annotate_artefacts(ctx: SubjectContext, raw: mne.io.BaseRaw | None = None) -> mne.io.BaseRaw:
    """Mark ocular and muscle segments without yet rejecting anything.

    Blinks are detected from a 1-10 Hz ocular signal - a recorded EOG channel if
    the system has one, otherwise the difference between two frontal radial OPMs.
    Muscle artefacts are detected as z-scored 110-140 Hz power, since
    magnetomyographic activity is broadband and high-frequency.  Both are stored
    as ``BAD_*`` annotations so that epoching can drop the affected trials.
    """
    cfg = ctx.cfg
    out_file = ctx.paths.ann_raw

    if not ctx.needs(out_file):
        ctx.logger.info("annotation: reusing %s", out_file.name)
        return mne.io.read_raw_fif(out_file, preload=True, verbose="ERROR")

    if raw is None:
        raw = apply_hfc(ctx)
    raw = raw.copy().load_data(verbose="ERROR")

    if not cfg.annotate.enabled:
        raw.save(out_file, overwrite=True, verbose="ERROR")
        return raw

    # -- ocular ----------------------------------------------------------- #
    eog_cfg = cfg.annotate.eog
    eog_name: str | None = None
    if eog_cfg.enabled:
        native = [
            ch for ch, typ in zip(raw.ch_names, raw.get_channel_types())
            if typ == "eog" and ch != eog_cfg.surrogate_name
        ]
        surrogate = _build_eog_surrogate(ctx, raw)
        if eog_cfg.prefer_native and native:
            eog_name = _most_ocular_channel(raw, native, surrogate)
        else:
            eog_name = surrogate or (native[0] if native else None)

    n_blinks = 0
    if eog_name is not None:
        eog_events = mne.preprocessing.find_eog_events(
            raw, ch_name=eog_name, l_freq=eog_cfg.l_freq, h_freq=eog_cfg.h_freq,
            thresh=eog_cfg.threshold, verbose="ERROR",
        )
        n_blinks = len(eog_events)
        if n_blinks:
            half = eog_cfg.window / 2
            raw.annotations.append(
                onset=_annotation_onsets(raw, eog_events[:, 0]) - half,
                duration=np.full(n_blinks, eog_cfg.window),
                description=[eog_cfg.description] * n_blinks,
            )
    ctx.state["eog_channel"] = eog_name
    ctx.record(
        eog_channel=eog_name,
        n_blinks=n_blinks,
        blinks_per_min=n_blinks / max(raw.times[-1] / 60, 1e-9),
    )

    # -- muscle ----------------------------------------------------------- #
    muscle_seconds = 0.0
    scores = None
    muscle_cfg = cfg.annotate.muscle
    if muscle_cfg.enabled:
        fmin, fmax = float(min(muscle_cfg.filter_freq)), float(max(muscle_cfg.filter_freq))
        nyquist = raw.info["sfreq"] / 2
        if fmax >= nyquist:
            ctx.logger.warning(
                "muscle band %.0f-%.0f Hz exceeds Nyquist (%.0f Hz); skipping muscle annotation",
                fmin, fmax, nyquist,
            )
        else:
            clean = raw.copy().drop_channels(raw.info["bads"]) if raw.info["bads"] else raw
            muscle_ann, scores = mne.preprocessing.annotate_muscle_zscore(
                clean, ch_type="mag", threshold=muscle_cfg.threshold,
                min_length_good=muscle_cfg.min_length_good, filter_freq=(fmin, fmax),
                n_jobs=1, verbose="ERROR",
            )
            raw.set_annotations(raw.annotations + muscle_ann)
            muscle_seconds = float(
                sum(d for d, desc in zip(raw.annotations.duration, raw.annotations.description)
                    if str(desc).startswith("BAD_muscle"))
            )
    ctx.record(
        muscle_seconds=muscle_seconds,
        muscle_percent=100 * muscle_seconds / max(raw.times[-1], 1e-9),
    )
    ctx.logger.info(
        "annotated %d blinks (%.1f/min) and %.1f s of muscle artefact (%.2f%%)",
        n_blinks, ctx.metrics["blinks_per_min"], muscle_seconds, ctx.metrics["muscle_percent"],
    )

    _plot_artefacts(ctx, raw, eog_name, scores)

    raw.save(out_file, overwrite=True, verbose="ERROR")
    raw.annotations.save(str(ctx.paths.ann_csv), overwrite=True)
    return raw


def _most_ocular_channel(raw, native: list[str], surrogate: str | None) -> str:
    """Pick the recorded EOG channel that best matches the frontal OPM surrogate."""
    if surrogate is None or len(native) == 1:
        return native[0]
    stop = int(min(180.0, raw.times[-1]) * raw.info["sfreq"])
    proxy = raw.get_data(picks=[surrogate], stop=stop)[0]
    proxy = (proxy - proxy.mean()) / max(proxy.std(), 1e-30)
    best, best_r = native[0], -np.inf
    for ch in native:
        signal = raw.get_data(picks=[ch], stop=stop)[0]
        signal = (signal - signal.mean()) / max(signal.std(), 1e-30)
        r = abs(float(np.corrcoef(proxy, signal)[0, 1]))
        if r > best_r:
            best, best_r = ch, r
    return best


def _plot_artefacts(ctx, raw, eog_name, scores) -> None:
    import matplotlib.pyplot as plt

    if eog_name is None and scores is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
    sfreq = raw.info["sfreq"]

    if eog_name is not None:
        t0 = min(60.0, max(0.0, raw.times[-1] - 30))
        t1 = min(t0 + 30, raw.times[-1])
        signal = raw.get_data(picks=[eog_name], start=int(t0 * sfreq), stop=int(t1 * sfreq))[0]
        signal = (signal - signal.mean()) / max(signal.std(), 1e-30)
        axes[0].plot(np.arange(len(signal)) / sfreq, signal, lw=0.7)
        axes[0].set(title=f"Ocular signal ({eog_name})", xlabel="Time in excerpt (s)", ylabel="z score")
    else:
        axes[0].set_axis_off()

    if scores is not None:
        times = np.arange(len(scores)) / sfreq
        axes[1].plot(times[::10], scores[::10], lw=0.55)
        axes[1].axhline(ctx.cfg.annotate.muscle.threshold, color="r", ls="--", lw=1)
        axes[1].set(title="Muscle z score", xlabel="Time (s)", ylabel="z score")
    else:
        axes[1].set_axis_off()

    path = save_figure(fig, ctx.figure_path("04_artefact_annotation"), ctx.cfg)
    ctx.add_figure("Artefacts", "Ocular signal and muscle z score", path)


# --------------------------------------------------------------------------- #
# 5. ICA
# --------------------------------------------------------------------------- #


def run_ica(ctx: SubjectContext, raw: mne.io.BaseRaw | None = None) -> mne.io.BaseRaw:
    """Estimate and remove ocular and cardiac components.

    The decomposition is estimated on a 3-30 Hz copy downsampled to 250 Hz (the
    high-pass matters most: removing slow drifts makes the decomposition far
    more effective) and the unmixing is then applied to the full-rate data.

    The tutorial selects components by eye.  That does not scale to a whole
    study, so components are selected automatically by their correlation with
    the ocular channel and by CTPS against a synthesised cardiac signal; the
    number removed is capped at ``ica.max_exclude`` and the component figures
    are written out so every subject's selection remains auditable.
    """
    cfg = ctx.cfg
    out_file, ica_file = ctx.paths.ica_raw, ctx.paths.ica_solution

    if not ctx.needs(out_file, ica_file):
        ctx.logger.info("ICA: reusing %s", out_file.name)
        return mne.io.read_raw_fif(out_file, preload=True, verbose="ERROR")

    if raw is None:
        raw = annotate_artefacts(ctx)

    if not cfg.ica.enabled:
        raw = raw.copy().load_data(verbose="ERROR")
        raw.save(out_file, overwrite=True, verbose="ERROR")
        ctx.record(n_ica_components=0, ica_excluded=[])
        return raw

    # Keep the ocular channel alongside the magnetometers: the decomposition is
    # fitted on "mag" only, but find_bads_eog needs the EOG channel present.
    fit_picks = [
        ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ in {"mag", "eog"}
    ]
    fit_raw = raw.copy().pick(fit_picks).load_data(verbose="ERROR")
    fit_raw.filter(cfg.ica.fit_l_freq, cfg.ica.fit_h_freq, picks=["mag", "eog"], verbose="ERROR")
    if cfg.ica.fit_sfreq and cfg.ica.fit_sfreq < fit_raw.info["sfreq"]:
        fit_raw.resample(cfg.ica.fit_sfreq, verbose="ERROR")

    n_components = _clamp_components(ctx, fit_raw)
    ica = mne.preprocessing.ICA(
        method=cfg.ica.method, n_components=n_components,
        random_state=cfg.ica.random_state, max_iter=cfg.ica.max_iter, verbose="ERROR",
    )
    with timed("fitting ICA", ctx.logger):
        ica.fit(fit_raw, picks="mag", reject_by_annotation=True, verbose="ERROR")

    exclude, detail = _select_components(ctx, ica, fit_raw)
    ica.exclude = exclude

    clean = raw.copy().load_data(verbose="ERROR")
    ica.apply(clean, verbose="ERROR")

    ica.save(ica_file, overwrite=True, verbose="ERROR")
    clean.save(out_file, overwrite=True, verbose="ERROR")

    ctx.state["ica_excluded"] = exclude
    ctx.record(n_ica_components=int(ica.n_components_), ica_excluded=exclude,
               n_ica_excluded=len(exclude), ica_selection=detail)
    ctx.logger.info("ICA: %d components, excluded %s", ica.n_components_, exclude or "none")

    _plot_ica(ctx, ica, detail)
    return clean


def _clamp_components(ctx: SubjectContext, fit_raw: mne.io.BaseRaw) -> float | int:
    """Keep ``n_components`` within the rank that HFC has left in the data."""
    requested = ctx.cfg.ica.n_components
    if isinstance(requested, float) and 0 < requested < 1:
        return requested
    requested = int(requested)
    mag_only = fit_raw.copy().pick("mag")
    try:
        rank = int(min(mne.compute_rank(mag_only, rank="info", verbose="ERROR").values()))
    except (ValueError, RuntimeError):
        rank = len(mag_only.ch_names)
    allowed = max(1, min(requested, rank))
    if allowed < requested:
        ctx.logger.warning(
            "ICA: reducing n_components from %d to %d, the rank left after HFC",
            requested, allowed,
        )
    return allowed


def _select_components(ctx: SubjectContext, ica, fit_raw) -> tuple[list[int], dict]:
    """Rank artefact components by evidence and keep at most ``max_exclude``."""
    cfg = ctx.cfg.ica
    detail: dict[str, object] = {}
    ranked: list[tuple[float, int, str]] = []

    eog_name = ctx.state.get("eog_channel")
    if cfg.detect_eog and eog_name:
        try:
            idx, scores = ica.find_bads_eog(
                fit_raw, ch_name=eog_name, threshold=cfg.eog_threshold, verbose="ERROR"
            )
            scores = np.atleast_2d(np.asarray(scores))
            strength = np.abs(scores).max(axis=0)
            detail["eog_components"] = [int(i) for i in idx]
            detail["eog_scores"] = strength.tolist()
            ranked += [(float(strength[i]), int(i), "eog") for i in idx]
        except (RuntimeError, ValueError) as exc:
            ctx.logger.warning("ICA: ocular detection failed (%s)", exc)

    if cfg.detect_ecg:
        try:
            # No ECG electrode on an OPM array, so MNE synthesises one from the
            # magnetometers; CTPS then scores phase locking to the R peaks.
            threshold = cfg.ecg_threshold
            threshold = threshold if threshold == "auto" else float(threshold)
            idx, scores = ica.find_bads_ecg(
                fit_raw, ch_name=None, threshold=threshold, method="ctps", verbose="ERROR"
            )
            scores = np.atleast_2d(np.asarray(scores))
            strength = np.abs(scores).max(axis=0)
            detail["ecg_components"] = [int(i) for i in idx]
            detail["ecg_scores"] = strength.tolist()
            ranked += [(float(strength[i]), int(i), "ecg") for i in idx]
        except (RuntimeError, ValueError) as exc:
            ctx.logger.warning("ICA: cardiac detection failed (%s)", exc)

    # Strongest evidence first, one entry per component.
    best: dict[int, float] = {}
    for strength, index, _kind in ranked:
        best[index] = max(best.get(index, -np.inf), strength)
    exclude = [i for i, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]
    if len(exclude) > cfg.max_exclude:
        ctx.logger.warning(
            "ICA: %d artefact components found, keeping the %d strongest",
            len(exclude), cfg.max_exclude,
        )
        exclude = exclude[: cfg.max_exclude]
    return sorted(exclude), detail


def _plot_ica(ctx, ica, detail) -> None:
    import matplotlib.pyplot as plt

    if not ctx.cfg.output.figures:
        return
    try:
        figs = ica.plot_components(show=False)
        figs = figs if isinstance(figs, list) else [figs]
        for i, fig in enumerate(figs):
            path = save_figure(fig, ctx.figure_path(f"05_ica_topographies_{i:02d}"), ctx.cfg)
            ctx.add_figure("ICA", f"Component topographies {i + 1}", path)
    except (RuntimeError, ValueError) as exc:
        ctx.logger.warning("ICA: could not plot component topographies (%s)", exc)

    scores = detail.get("eog_scores") or detail.get("ecg_scores")
    if not scores:
        return
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    bars = ax.bar(np.arange(len(scores)), scores)
    for index in ica.exclude:
        if index < len(bars):
            bars[index].set_color("#B91C1C")
    ax.set(
        title=f"{ctx.rec.key}: artefact association per component",
        xlabel="ICA component", ylabel="|score|",
    )
    path = save_figure(fig, ctx.figure_path("05_ica_scores"), ctx.cfg)
    ctx.add_figure("ICA", "Component artefact scores", path)


# --------------------------------------------------------------------------- #
# 6. Epoching
# --------------------------------------------------------------------------- #


def make_epochs(ctx: SubjectContext, raw: mne.io.BaseRaw | None = None) -> mne.Epochs:
    """Cut condition-specific trials from the cleaned continuous data.

    The continuous data are filtered once here rather than per epoch, so that
    filter edge effects fall outside the epochs.  ``epochs.continuous_h_freq``
    defaults to ``None`` because a low-pass would destroy the gamma band that
    the FLUX time-frequency analysis targets.

    Epochs are saved un-baselined and broadband, with only a per-epoch linear
    detrend applied, so that every downstream analysis can choose its own
    baseline and filter.
    """
    cfg = ctx.cfg
    out_file = ctx.paths.epochs

    if not ctx.needs(out_file):
        ctx.logger.info("epochs: reusing %s", out_file.name)
        return mne.read_epochs(out_file, preload=True, verbose="ERROR")

    if raw is None:
        raw = run_ica(ctx)
    raw = raw.copy().load_data(verbose="ERROR")

    if cfg.epochs.continuous_l_freq is not None or cfg.epochs.continuous_h_freq is not None:
        raw.filter(
            cfg.epochs.continuous_l_freq, cfg.epochs.continuous_h_freq,
            picks="mag", verbose="ERROR",
        )

    events, event_id = _task_events(ctx, raw)
    condition_id = _condition_event_id(ctx, event_id)

    reject = {k: float(v) for k, v in cfg.epochs.reject.items()} or None
    baseline = tuple(cfg.epochs.baseline) if cfg.epochs.baseline else None
    epochs = mne.Epochs(
        raw, events, condition_id,
        tmin=cfg.epochs.tmin, tmax=cfg.epochs.tmax,
        baseline=baseline, picks=cfg.epochs.picks, detrend=cfg.epochs.detrend,
        reject=reject, reject_by_annotation=cfg.epochs.reject_by_annotation,
        proj=False, preload=True, verbose="ERROR",
    )
    epochs.info.normalize_proj()

    requested = int(np.isin(events[:, 2], list(condition_id.values())).sum())
    retained = len(epochs)
    counts = {name: int(len(epochs[name])) for name in sorted(cfg.epochs.conditions)
              if any(key.split("/")[0] == name for key in condition_id)}
    if retained == 0:
        raise RuntimeError(
            f"{ctx.rec.key}: every epoch was rejected. Check epochs.reject "
            f"(currently {cfg.epochs.reject}) and the artefact annotations."
        )
    ctx.record(
        epochs_requested=requested, epochs_retained=retained,
        epochs_rejected_percent=100 * (1 - retained / max(requested, 1)),
        epochs_per_condition=counts, epochs_sfreq=float(epochs.info["sfreq"]),
    )
    ctx.logger.info(
        "epochs: kept %d/%d (%.1f%% rejected); %s",
        retained, requested, ctx.metrics["epochs_rejected_percent"], counts,
    )

    epochs.save(out_file, overwrite=True, verbose="ERROR")
    _plot_drop_log(ctx, epochs)
    return epochs


def _task_events(ctx: SubjectContext, raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict[str, int]]:
    """Decode task events, ignoring the ``BAD_*`` artefact annotations."""
    stored = ctx.state.get("task_event_id")
    present = {str(d) for d in raw.annotations.description}
    if not stored:
        labels = sorted(d for d in present if not d.startswith("BAD_"))
        stored = {label: i + 1 for i, label in enumerate(labels)}
        ctx.state["task_event_id"] = stored
    usable = {label: code for label, code in stored.items() if label in present}
    if not usable:
        raise RuntimeError(
            f"{ctx.rec.key}: none of the task event labels {sorted(stored)} survive in the "
            "cleaned data. Were the BIDS annotations dropped during preprocessing?"
        )
    events, event_id = mne.events_from_annotations(raw, event_id=usable, verbose="ERROR")
    return events, event_id


def _condition_event_id(ctx: SubjectContext, event_id: dict[str, int]) -> dict[str, int]:
    """Map each configured condition onto the event codes pooled into it.

    Conditions that pool several labels (e.g. ``resp_T`` and ``resp_L`` into one
    ``response``) are expressed as MNE hierarchical names, so ``epochs['response']``
    selects the union.
    """
    conditions = ctx.cfg.epochs.conditions
    bad_names = [name for name in conditions if "/" in name]
    if bad_names:
        raise ValueError(
            f"condition name(s) {bad_names} contain '/', which MNE reserves for "
            "hierarchical event selection. Rename them."
        )
    selected: dict[str, int] = {}
    missing: dict[str, list[str]] = {}
    for name, labels in conditions.items():
        found = [label for label in labels if label in event_id]
        absent = [label for label in labels if label not in event_id]
        if absent:
            missing[name] = absent
        for label in found:
            # "condition/label" lets MNE select either the pooled condition or
            # one of its constituent labels; a one-label condition that already
            # carries the label's name needs no suffix.
            key = name if label == name else f"{name}/{label}"
            selected[key] = event_id[label]
    if not selected:
        raise RuntimeError(
            f"{ctx.rec.key}: no configured condition matched the recording's event labels. "
            f"Configured: {conditions}. Available: {sorted(event_id)}."
        )
    if missing:
        ctx.logger.warning("event labels absent from this recording: %s", missing)
        ctx.record(missing_event_labels=missing)
    return selected


def _plot_drop_log(ctx, epochs) -> None:
    if not ctx.cfg.output.figures:
        return
    try:
        fig = epochs.plot_drop_log(show=False)
        path = save_figure(fig, ctx.figure_path("06_epoch_drop_log"), ctx.cfg)
        ctx.add_figure("Epochs", "Epoch drop log", path)
    except (RuntimeError, ValueError) as exc:
        ctx.logger.warning("could not plot the drop log (%s)", exc)
