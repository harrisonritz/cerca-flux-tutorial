"""Response-locked motor summaries for the Oxford-vs-Princeton comparison.

    13. :func:`compute_motor_sensor`  - central-RMS ERF and central motor beta
    14. :func:`compute_motor_source`  - left primary motor cortex LCMV and DICS time courses

These are the quantities behind figures 06, 07 and 11 of the
``FLUX_Oxford_Princeton_Comparison`` notebook, computed per recording and cached as
small ``.npz`` files so that a whole study can be overlaid afterwards without
touching the epochs again (see :mod:`cerca_flux.compare`).

Corresponding OPM sensors do not share location or polarity across participants,
so the sensor-level measures use the polarity-invariant RMS over a fixed set of
central radial sensors.  Everything is returned in physical units next to a
baseline-normalised version, because amplitude also depends on anatomy, sensor
placement, task and trial count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mne
from mne.beamformer import apply_dics_csd, apply_lcmv, make_dics, make_lcmv
from mne.time_frequency import csd_multitaper

from .context import SubjectContext
from .provenance import fingerprints
from .utils import axis_of, resolve_picks, save_figure, sensor_id, timed

_TINY = np.finfo(float).tiny


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _window_mask(times: np.ndarray, window, what: str) -> np.ndarray:
    mask = (times >= window[0]) & (times <= window[1])
    if not mask.any():
        raise ValueError(
            f"{what} window {list(window)} lies outside the data "
            f"({times[0]:.2f} to {times[-1]:.2f} s)"
        )
    return mask


def _as_real(values, what: str, tol: float = 1e-8) -> np.ndarray:
    """A real array from a possibly complex-typed one, refusing to discard real information.

    Beamformer output can be complex-typed although its imaginary part is zero.  A
    silent cast would hide it if that ever stopped being true.
    """
    values = np.asarray(values)
    if not np.iscomplexobj(values):
        return values
    scale = max(float(np.abs(values.real).max()), np.finfo(float).tiny)
    if float(np.abs(values.imag).max()) > tol * scale:
        raise RuntimeError(f"{what} has a non-negligible imaginary part; refusing to discard it")
    return values.real.copy()


def _pool_conditions(ctx: SubjectContext, epochs: mne.Epochs) -> mne.Epochs:
    """The union of the configured motor conditions that have surviving epochs."""
    present = []
    for name in ctx.cfg.motor.conditions:
        try:
            if len(epochs[name]):
                present.append(name)
        except KeyError:
            continue
    if not present:
        raise RuntimeError(
            f"{ctx.rec.key}: none of the motor conditions {ctx.cfg.motor.conditions} has "
            "surviving epochs"
        )
    return epochs[present[0]] if len(present) == 1 else epochs[present]


def _central_channels(ctx: SubjectContext, epochs: mne.Epochs) -> list[str]:
    channels = ctx.cfg.channels
    sensor_cfg = ctx.cfg.motor.sensor
    wanted = set(sensor_cfg.sensors)
    bads = set(epochs.info["bads"])
    types = dict(zip(epochs.ch_names, epochs.get_channel_types()))
    names = [
        ch for ch in epochs.ch_names
        if types[ch] == "mag" and ch not in bads
        and sensor_id(ch) in wanted and axis_of(ch, channels) == channels.radial_axis.upper()
    ]
    if len(names) < sensor_cfg.min_sensors:
        raise RuntimeError(
            f"{ctx.rec.key}: only {len(names)} of the central sensors {sorted(wanted)} "
            f"survive preprocessing (motor.sensor.min_sensors = {sensor_cfg.min_sensors})"
        )
    return names


def _save_result(path: Path, arrays: dict, metrics: dict) -> None:
    """Write arrays plus the scalar metrics as one JSON string (no pickling)."""
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["metrics"] = np.asarray(json.dumps(metrics, sort_keys=True))
    np.savez(path, **payload)


def load_motor_result(path: Path) -> dict:
    """Read a ``*_motor-sensor.npz`` or ``*_motor-source.npz`` file into a dict."""
    with np.load(path, allow_pickle=False) as stored:
        result = {key: stored[key] for key in stored.files}
    result["metrics"] = json.loads(str(result["metrics"]))
    for key, value in list(result.items()):
        if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind == "U":
            result[key] = str(value)
    return result


# --------------------------------------------------------------------------- #
# 13. Sensor level: central RMS ERF and motor beta
# --------------------------------------------------------------------------- #


def compute_motor_sensor(ctx: SubjectContext, epochs: mne.Epochs | None = None) -> dict:
    """Central-RMS response field and central 15-30 Hz power for one recording.

    *ERF.*  Per-trial RMS across the central radial sensors, averaged over trials.
    The RMS is taken before averaging, so the measure is polarity-invariant and
    insensitive to the sign conventions of individual sensors.  The mean over
    trials is shown with a trial-wise SEM, in femtotesla and in SD of the
    baseline window.

    *Beta.*  Multitaper power (``n_cycles = f / divisor``, 0.5 s windows) averaged
    over the same sensors and expressed as percent change from the baseline
    window.  Negative values around movement indicate beta desynchronisation and
    later positive values the rebound.
    """
    out_file = ctx.paths.motor_sensor
    if not ctx.needs(out_file):
        ctx.logger.info("motor sensor: reusing %s", out_file.name)
        result = load_motor_result(out_file)
        ctx.record(**result["metrics"])
        return result

    if epochs is None:
        from .preprocess import make_epochs
        epochs = make_epochs(ctx)

    cfg = ctx.cfg.motor.sensor
    selected = _pool_conditions(ctx, epochs)
    names = _central_channels(ctx, selected)
    times = selected.times

    # -- response-locked field ------------------------------------------- #
    x = selected.get_data(picks=names) * 1e15  # fT, (epochs, channels, times)
    trial_rms = np.sqrt(np.mean(x**2, axis=1))
    n_epochs = len(trial_rms)
    rms = trial_rms.mean(axis=0)
    sem = (trial_rms.std(axis=0, ddof=1) / np.sqrt(n_epochs)) if n_epochs > 1 else np.zeros_like(rms)
    baseline = _window_mask(times, cfg.erf_baseline, "motor.sensor.erf_baseline")
    active = _window_mask(times, cfg.erf_active, "motor.sensor.erf_active")
    scale = max(float(rms[baseline].std()), 1e-12)
    z = (rms - rms[baseline].mean()) / scale

    # -- motor beta ------------------------------------------------------- #
    nyquist = selected.info["sfreq"] / 2
    freqs = np.arange(cfg.beta_fmin, cfg.beta_fmax + cfg.beta_fstep / 2, cfg.beta_fstep)
    freqs = freqs[freqs < nyquist - 1]
    if freqs.size == 0:
        raise RuntimeError(
            f"{ctx.rec.key}: no beta frequency lies below Nyquist ({nyquist:g} Hz)"
        )
    with timed("motor beta TFR", ctx.logger):
        tfr = selected.compute_tfr(
            method="multitaper", freqs=freqs, n_cycles=freqs / cfg.beta_n_cycles_divisor,
            time_bandwidth=cfg.beta_time_bandwidth, picks=names, average=True,
            return_itc=False, decim=cfg.beta_decim, n_jobs=1, verbose="ERROR",
        )
    power = tfr.data.mean(axis=0)  # (freqs, times), averaged over channels
    base = _window_mask(tfr.times, cfg.beta_baseline, "motor.sensor.beta_baseline")
    reference = power[:, base].mean(axis=1, keepdims=True)
    percent = 100 * (power - reference) / np.maximum(reference, _TINY)
    beta = percent.mean(axis=0)
    move = _window_mask(tfr.times, cfg.movement_window, "motor.sensor.movement_window")
    rebound = _window_mask(tfr.times, cfg.rebound_window, "motor.sensor.rebound_window")

    metrics = {
        "motor_n_epochs": int(n_epochs),
        "motor_n_sensors": len(names),
        "motor_baseline_rms_fT": float(rms[baseline].mean()),
        "motor_peak_erf_rms_fT": float(rms[active].max()),
        "motor_peak_erf_snr": float(z[active].max()),
        "motor_movement_beta_percent": float(percent[:, move].mean()),
        "motor_rebound_beta_percent": float(percent[:, rebound].mean()),
    }
    arrays = {
        "erf_times": times, "erf_rms_ft": rms, "erf_sem_ft": sem, "erf_z": z,
        "erf_z_sem": sem / scale,
        "tfr_freqs": tfr.freqs, "tfr_times": tfr.times, "tfr_percent": percent,
        "beta_percent": beta, "channels": np.asarray(names),
    }
    metrics.update(fingerprints(ctx.cfg))  # which settings made this result
    _save_result(out_file, arrays, metrics)
    ctx.record(**metrics)
    ctx.logger.info(
        "motor sensor: %d epochs, %d central sensors; peak ERF SNR %.1f, beta rebound %+.1f%%",
        n_epochs, len(names), metrics["motor_peak_erf_snr"],
        metrics["motor_rebound_beta_percent"],
    )
    result = {**arrays, "metrics": metrics}
    _plot_sensor(ctx, result)
    return result


def _plot_sensor(ctx: SubjectContext, result: dict) -> None:
    import matplotlib.pyplot as plt

    if not ctx.cfg.output.figures:
        return
    cfg = ctx.cfg.motor.sensor
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)

    ax = axes[0]
    t, y, sem = result["erf_times"], result["erf_rms_ft"], result["erf_sem_ft"]
    ax.plot(t, y, color="k", lw=1.6)
    ax.fill_between(t, y - sem, y + sem, color="k", alpha=0.15, lw=0)
    ax.set(title=f"Central response-locked field (n={result['metrics']['motor_n_epochs']})",
           xlabel="Time from response (s)", ylabel="Central RMS (fT)")

    ax = axes[1]
    ax.plot(result["tfr_times"], result["beta_percent"], color="k", lw=1.6)
    for window, shade in ((cfg.movement_window, "#2a78d6"), (cfg.rebound_window, "#eb6834")):
        ax.axvspan(*window, color=shade, alpha=0.10, lw=0)
    ax.axhline(0, color="0.55", lw=0.8, ls=":")
    ax.set(title=f"Central {cfg.beta_fmin:g}-{cfg.beta_fmax:g} Hz power",
           xlabel="Time from response (s)", ylabel="Change from baseline (%)")

    ax = axes[2]
    percent = result["tfr_percent"]
    limit = float(np.nanpercentile(np.abs(percent), 98)) or 1.0
    mesh = ax.pcolormesh(result["tfr_times"], result["tfr_freqs"], percent, shading="auto",
                         cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(mesh, ax=ax, label="Power change (%)")
    ax.set(title="Time-frequency", xlabel="Time from response (s)", ylabel="Frequency (Hz)")

    for ax in axes:
        ax.axvline(0, color="0.35", lw=0.9, ls="--")
    path = save_figure(fig, ctx.figure_path("11_motor_sensor"), ctx.cfg)
    ctx.add_figure("Motor response", "Central ERF and motor beta", path)


# --------------------------------------------------------------------------- #
# 14. Source level: left primary motor cortex
# --------------------------------------------------------------------------- #


def compute_motor_source(ctx: SubjectContext, epochs: mne.Epochs | None = None,
                         forwards: dict[str, mne.Forward] | None = None) -> dict:
    """LCMV and sliding-window DICS time courses in one cortical parcel.

    *LCMV.*  One covariance over ``cov_tmin..cov_tmax`` and a common filter; the
    parcel's ``mean_flip`` time course of the response-locked average is expressed
    in SD of the baseline window.

    *DICS.*  One filter from the movement and rebound CSDs, then 15-30 Hz CSDs in
    overlapping windows.  Parcel power (``pca_flip``) is expressed in dB relative to
    the baseline window centres (the magnitude of the ``pca_flip`` value, whose sign
    is arbitrary for non-negative power).  The 400 ms spectral window makes the curve
    smooth and truncated at the edges, so it must not be read at 100 ms resolution.
    """
    out_file = ctx.paths.motor_source
    if not ctx.needs(out_file):
        ctx.logger.info("motor source: reusing %s", out_file.name)
        result = load_motor_result(out_file)
        ctx.record(**result["metrics"])
        return result

    from .source import _average_csds, build_forward

    if epochs is None:
        from .preprocess import make_epochs
        epochs = make_epochs(ctx)
    if forwards is None:
        forwards = build_forward(ctx)

    cfg = ctx.cfg
    src_cfg = cfg.motor.source
    sensor_cfg = cfg.motor.sensor
    if src_cfg.space not in forwards:
        raise RuntimeError(
            f"{ctx.rec.key}: no {src_cfg.space} forward model is available "
            f"(built: {sorted(forwards)}); check forward.{src_cfg.space}.enabled"
        )

    selected = _pool_conditions(ctx, epochs)
    picks = resolve_picks(selected, src_cfg.picks, cfg.channels)
    work = selected.copy().pick(picks)
    work.info.normalize_proj()
    fwd = mne.pick_channels_forward(
        forwards[src_cfg.space], include=work.ch_names, ordered=True, copy=True, verbose="ERROR"
    )
    if fwd["nchan"] != len(work.ch_names):
        raise RuntimeError(
            f"{ctx.rec.key}: the {src_cfg.space} forward model covers {fwd['nchan']} of the "
            f"{len(work.ch_names)} analysis channels; rebuild it with output.overwrite=true"
        )

    rank = _rank(work, src_cfg.rank)
    if rank is not None:
        ctx.record(motor_source_rank=dict(rank))
    label = _find_label(ctx)
    src = fwd["src"]

    # -- LCMV ------------------------------------------------------------- #
    with timed("motor LCMV", ctx.logger):
        data_cov = mne.compute_covariance(
            work, tmin=src_cfg.cov_tmin, tmax=src_cfg.cov_tmax, method=src_cfg.cov_method,
            rank=rank, verbose="ERROR",
        )
        lcmv = make_lcmv(
            work.info, fwd, data_cov, reg=src_cfg.reg, noise_cov=None,
            pick_ori=src_cfg.pick_ori, weight_norm=src_cfg.weight_norm,
            reduce_rank=src_cfg.reduce_rank, depth=src_cfg.depth, rank=rank, verbose="ERROR",
        )
        stc = apply_lcmv(work.average(), lcmv, verbose="ERROR")
    lcmv_times = stc.times
    course = _as_real(
        mne.extract_label_time_course(stc, [label], src, mode="mean_flip", verbose="ERROR")[0],
        "the LCMV label time course",
    )
    base = _window_mask(lcmv_times, src_cfg.lcmv_baseline, "motor.source.lcmv_baseline")
    lcmv_z = (course - course[base].mean()) / max(float(course[base].std()), np.finfo(float).eps)
    response = np.flatnonzero(_window_mask(lcmv_times, src_cfg.lcmv_response,
                                           "motor.source.lcmv_response"))
    peak = response[int(np.argmax(np.abs(lcmv_z[response])))]

    # -- DICS ------------------------------------------------------------- #
    with timed("motor DICS", ctx.logger):
        def csd_for(lo: float, hi: float):
            return csd_multitaper(
                work, fmin=src_cfg.dics_fmin, fmax=src_cfg.dics_fmax, tmin=lo, tmax=hi,
                bandwidth=src_cfg.dics_bandwidth, low_bias=True, n_jobs=1, verbose="ERROR",
            ).mean()

        csd_move = csd_for(*sensor_cfg.movement_window)
        csd_rebound = csd_for(*sensor_cfg.rebound_window)
        dics = make_dics(
            work.info, fwd, _average_csds([csd_move, csd_rebound]), reg=src_cfg.reg,
            noise_csd=None, pick_ori=src_cfg.pick_ori, weight_norm=src_cfg.weight_norm,
            reduce_rank=src_cfg.reduce_rank, real_filter=src_cfg.real_filter,
            depth=src_cfg.depth, rank=rank, verbose="ERROR",
        )

        # Window centres from dics_tmin in steps, never past dics_tmax (the notebook's
        # np.arange(-0.75, 1.001, 0.1) ends at 0.95, not 1.05). Flooring, not rounding.
        n_centres = int(np.floor((src_cfg.dics_tmax - src_cfg.dics_tmin) / src_cfg.dics_step
                                 + 1e-9)) + 1
        centres = src_cfg.dics_tmin + src_cfg.dics_step * np.arange(n_centres)
        half, tol = src_cfg.dics_window / 2, 1e-6
        centres_used, power = [], []
        for centre in centres:
            lo, hi = centre - half, centre + half
            if lo < work.tmin - tol or hi > work.tmax + tol:
                ctx.logger.warning(
                    "motor DICS: the window at %.2f s (%.2f to %.2f s) leaves the epochs "
                    "(%.2f to %.2f s) and is skipped", centre, lo, hi, work.tmin, work.tmax,
                )
                continue
            stc_window, _ = apply_dics_csd(
                csd_for(max(lo, work.tmin), min(hi, work.tmax)), dics, verbose="ERROR"
            )
            # Keep the value as returned (DICS power can be complex-typed with a zero
            # imaginary part); ``float()`` would silently drop an imaginary part.
            power.append(mne.extract_label_time_course(
                stc_window, [label], src, mode="pca_flip", verbose="ERROR")[0, 0])
            centres_used.append(float(centre))
    centres_used, signed_power = np.asarray(centres_used), np.asarray(power)
    if centres_used.size == 0:
        raise RuntimeError(f"{ctx.rec.key}: no DICS window fits inside the epochs")
    # ``pca_flip`` returns a signed scalar, but power is non-negative: its sign is
    # an arbitrary orientation convention and carries no information.  The magnitude
    # is the parcel power (unchanged wherever the sign happens to come out positive).
    n_sign_flipped = int((np.real(signed_power) < 0).sum())
    power = np.abs(signed_power)
    if n_sign_flipped:
        ctx.logger.info(
            "motor DICS: pca_flip returned a negative sign for %d of %d windows; "
            "magnitudes are used", n_sign_flipped, len(power),
        )
    dics_base = _window_mask(centres_used, src_cfg.dics_baseline, "motor.source.dics_baseline")
    reference = max(float(power[dics_base].mean()), _TINY)
    dics_db = 10 * np.log10(np.maximum(power, _TINY) / reference)

    metrics = {
        "motor_source_n_epochs": int(len(work)),
        "motor_source_n_channels": len(work.ch_names),
        "motor_lcmv_peak_abs_z": float(abs(lcmv_z[peak])),
        "motor_lcmv_peak_time_s": float(lcmv_times[peak]),
        "motor_dics_min_db": float(dics_db.min()),
        "motor_dics_max_db": float(dics_db.max()),
        "motor_dics_sign_flipped_windows": n_sign_flipped,
    }
    arrays = {
        "lcmv_times": lcmv_times, "lcmv_z": lcmv_z,
        "dics_times": centres_used, "dics_db": dics_db, "dics_power": power,
        "label": np.asarray(label.name),
    }
    metrics.update(fingerprints(ctx.cfg))  # which settings made this result
    _save_result(out_file, arrays, metrics)
    ctx.record(**metrics)
    ctx.logger.info(
        "motor source (%s): LCMV peak |z| %.2f at %.3f s; DICS range %.1f to %+.1f dB",
        label.name, metrics["motor_lcmv_peak_abs_z"], metrics["motor_lcmv_peak_time_s"],
        metrics["motor_dics_min_db"], metrics["motor_dics_max_db"],
    )
    result = {**arrays, "label": label.name, "metrics": metrics}
    _plot_source(ctx, result)
    return result


def _rank(epochs: mne.Epochs, setting: str):
    """Data rank, which HFC and ICA have reduced below the channel count."""
    if setting == "none":
        return None
    if setting == "relative":
        return mne.compute_rank(epochs, tol=1e-5, tol_kind="relative", proj=True, verbose="ERROR")
    return mne.compute_rank(epochs, rank="info", verbose="ERROR")


def _find_label(ctx: SubjectContext) -> mne.Label:
    src_cfg = ctx.cfg.motor.source
    hemi = src_cfg.label.rsplit("-", 1)[1]
    labels = mne.read_labels_from_annot(
        ctx.fs_subject, parc=src_cfg.parc, hemi=hemi,
        subjects_dir=str(ctx.cfg.fs_subjects_dir), verbose="ERROR",
    )
    matches = [label for label in labels if label.name == src_cfg.label]
    if len(matches) != 1:
        raise RuntimeError(
            f"{ctx.rec.key}: expected one {src_cfg.label!r} label in the {src_cfg.parc!r} "
            f"parcellation of {ctx.fs_subject!r}; found {len(matches)}"
        )
    return matches[0]


def _plot_source(ctx: SubjectContext, result: dict) -> None:
    import matplotlib.pyplot as plt

    if not ctx.cfg.output.figures:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    axes[0].plot(result["lcmv_times"], result["lcmv_z"], color="k", lw=1.6)
    axes[0].axvspan(*ctx.cfg.motor.source.lcmv_response, color="0.5", alpha=0.08, lw=0)
    axes[0].set(title=f"{result['label']} LCMV response", ylabel="Parcel response (baseline SD)")
    axes[1].plot(result["dics_times"], result["dics_db"], color="k", lw=1.6, marker="o", ms=3.5)
    axes[1].set(title=f"{result['label']} DICS beta power",
                ylabel="Power change from baseline (dB)")
    for ax in axes:
        ax.axvline(0, color="0.35", lw=0.9, ls="--")
        ax.axhline(0, color="0.55", lw=0.8, ls=":")
        ax.set_xlabel("Time from response (s)")
    path = save_figure(fig, ctx.figure_path("12_motor_source"), ctx.cfg)
    ctx.add_figure("Motor response", f"{result['label']} source time courses", path)
