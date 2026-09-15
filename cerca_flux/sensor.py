"""Sensor-level analyses: ERFs, time-frequency power, and decoding.

    7.  :func:`compute_erf`     - event-related fields (EventRelatedFields)
    8.  :func:`compute_tfr`     - multitaper power (TimeFrequencyPower)
    9.  :func:`run_decoding`    - sliding-window MVPA (Classification)

All three read the epochs written by :func:`cerca_flux.preprocess.make_epochs`
and default to the radial (Z-axis) channels, the OPM analogue of conventional
magnetometers.
"""

from __future__ import annotations

import numpy as np
import mne
from scipy.signal import detrend as _detrend

from .context import SubjectContext
from .paths import sanitize
from .utils import resolve_picks, save_figure, timed


def _conditions(ctx: SubjectContext, epochs: mne.Epochs) -> list[str]:
    """Configured condition names that actually have surviving epochs."""
    names = []
    for name in ctx.cfg.epochs.conditions:
        try:
            if len(epochs[name]) > 0:
                names.append(name)
        except KeyError:
            continue
    return names


def _picked(epochs: mne.Epochs, spec: str, ctx: SubjectContext) -> mne.Epochs:
    picks = resolve_picks(epochs, spec, ctx.cfg.channels)
    out = epochs.copy().pick(picks)
    out.info.normalize_proj()
    return out


# --------------------------------------------------------------------------- #
# 7. Event-related fields
# --------------------------------------------------------------------------- #


def compute_erf(ctx: SubjectContext, epochs: mne.Epochs | None = None) -> list[mne.Evoked]:
    """Average each condition and clean up the average for display.

    Order matters: the average is low-pass filtered, cropped to the analysis
    window, linearly detrended across that window and only then baseline
    corrected, so the pre-event interval is centred on zero.  The detrend is
    fitted over a short window that contains the response itself, so keep the
    window from being dominated by the evoked deflection.
    """
    cfg = ctx.cfg
    out_file = ctx.paths.evoked
    if not ctx.needs(out_file):
        ctx.logger.info("ERF: reusing %s", out_file.name)
        evokeds = mne.read_evokeds(out_file, verbose="ERROR")
        # Re-derive the summary numbers: a cached stage must still contribute
        # its row to the group table.
        _record_erf_metrics(ctx, evokeds)
        return evokeds

    if epochs is None:
        from .preprocess import make_epochs
        epochs = make_epochs(ctx)

    picked = _picked(epochs, cfg.erf.picks, ctx)
    evokeds: list[mne.Evoked] = []

    for name in _conditions(ctx, picked):
        evoked = picked[name].average(method="mean")
        evoked.comment = name
        if cfg.erf.l_freq is not None or cfg.erf.h_freq is not None:
            evoked.filter(cfg.erf.l_freq, cfg.erf.h_freq, verbose="ERROR")
        if cfg.erf.crop:
            evoked.crop(*cfg.erf.crop)
        if cfg.erf.detrend:
            evoked.data = _detrend(evoked.data, axis=-1, type="linear")
        if cfg.erf.baseline:
            evoked.apply_baseline(tuple(cfg.erf.baseline), verbose="ERROR")
        evokeds.append(evoked)

    if not evokeds:
        raise RuntimeError(f"{ctx.rec.key}: no condition had epochs to average")

    mne.write_evokeds(out_file, evokeds, overwrite=True, verbose="ERROR")
    _record_erf_metrics(ctx, evokeds)
    ctx.logger.info("ERF: averaged %d conditions; peaks %s", len(evokeds),
                    {k: round(v["peak_latency_s"], 3) for k, v in ctx.metrics["erf_peaks"].items()})
    _plot_erf(ctx, evokeds)
    return evokeds


def _record_erf_metrics(ctx: SubjectContext, evokeds: list[mne.Evoked]) -> None:
    """Peak latency and global field power of each condition's average."""
    peaks = {}
    for evoked in evokeds:
        gfp = np.sqrt(np.mean(evoked.data**2, axis=0))
        index = int(np.argmax(gfp))
        peaks[evoked.comment] = {
            "peak_latency_s": float(evoked.times[index]),
            "peak_gfp_fT": float(gfp[index] * 1e15),
            "n_trials": int(evoked.nave),
        }
    ctx.record(erf_peaks=peaks, erf_channels=len(evokeds[0].ch_names) if evokeds else 0)


def _plot_erf(ctx: SubjectContext, evokeds: list[mne.Evoked]) -> None:
    import matplotlib.pyplot as plt

    if not ctx.cfg.output.figures:
        return
    for evoked in evokeds:
        label = sanitize(evoked.comment)
        try:
            fig = evoked.plot(gfp=True, show=False, spatial_colors=True)
            path = save_figure(fig, ctx.figure_path(f"07_erf_{label}"), ctx.cfg)
            ctx.add_figure("ERF", f"Butterfly plot: {evoked.comment}", path)
        except (RuntimeError, ValueError) as exc:
            ctx.logger.warning("ERF: butterfly plot failed for %s (%s)", evoked.comment, exc)
        times = [t for t in ctx.cfg.erf.topomap_times
                 if evoked.times[0] <= t <= evoked.times[-1]]
        if not times:
            continue
        try:
            fig = evoked.plot_topomap(
                times=times, ch_type="mag", contours=0, extrapolate="local",
                outlines="head", show=False,
            )
            path = save_figure(fig, ctx.figure_path(f"07_erf_topomap_{label}"), ctx.cfg)
            ctx.add_figure("ERF", f"Topography: {evoked.comment}", path)
        except (RuntimeError, ValueError) as exc:
            ctx.logger.warning("ERF: topomap failed for %s (%s)", evoked.comment, exc)

    if len(evokeds) > 1:
        try:
            fig = mne.viz.plot_compare_evokeds(
                {e.comment: e for e in evokeds}, combine="gfp", show=False,
                title=f"{ctx.rec.key}: GFP by condition",
            )
            fig = fig[0] if isinstance(fig, list) else fig
            path = save_figure(fig, ctx.figure_path("07_erf_gfp_comparison"), ctx.cfg)
            ctx.add_figure("ERF", "GFP by condition", path)
        except (RuntimeError, ValueError) as exc:
            ctx.logger.warning("ERF: GFP comparison failed (%s)", exc)
            plt.close("all")


# --------------------------------------------------------------------------- #
# 8. Time-frequency representations
# --------------------------------------------------------------------------- #


def compute_tfr(ctx: SubjectContext, epochs: mne.Epochs | None = None) -> dict:
    """Multitaper power per configured band and condition.

    ``n_cycles = freqs / n_cycles_divisor`` keeps the sliding window at a fixed
    duration across frequencies, and ``time_bandwidth`` sets the taper count
    (N = TBW - 1) and therefore the spectral smoothing.  The FLUX defaults use a
    narrow 0.5 s / 1-taper window below 30 Hz, where rhythms are narrow-band,
    and a shorter 0.25 s / 3-taper window above it, where gamma is broad-band.
    """
    cfg = ctx.cfg
    outputs = [ctx.paths.tfr(band.name) for band in cfg.tfr.bands]
    if not ctx.needs(*outputs):
        ctx.logger.info("TFR: reusing %d cached band file(s)", len(outputs))
        results = {band.name: _read_band(ctx.paths.tfr(band.name)) for band in cfg.tfr.bands}
        contrast_names = {c.name for c in cfg.tfr.contrasts}
        summary = {
            band.name: {
                k: v for name, tfr in results.get(band.name, {}).items()
                if name not in contrast_names
                for k, v in _band_summary(tfr, band, name).items()
            }
            for band in cfg.tfr.bands
        }
        ctx.record(tfr_summary=summary)
        return results

    if epochs is None:
        from .preprocess import make_epochs
        epochs = make_epochs(ctx)

    picked = _picked(epochs, cfg.tfr.picks, ctx)
    conditions = _conditions(ctx, picked)
    results: dict[str, dict[str, mne.time_frequency.AverageTFR]] = {}
    summary: dict[str, dict[str, float]] = {}

    for band in cfg.tfr.bands:
        freqs = np.arange(band.fmin, band.fmax + band.fstep / 2, band.fstep)
        nyquist = picked.info["sfreq"] / 2
        if freqs.max() >= nyquist:
            ctx.logger.warning(
                "TFR band %r reaches %.0f Hz but Nyquist is %.0f Hz; truncating",
                band.name, freqs.max(), nyquist,
            )
            freqs = freqs[freqs < nyquist - 1]
        if freqs.size == 0:
            ctx.logger.warning("TFR band %r has no usable frequencies; skipping", band.name)
            continue

        per_condition: dict[str, mne.time_frequency.AverageTFR] = {}
        with timed(f"TFR band {band.name} ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz)", ctx.logger):
            for name in conditions:
                tfr = picked[name].compute_tfr(
                    method="multitaper", freqs=freqs, n_cycles=freqs / band.n_cycles_divisor,
                    time_bandwidth=band.time_bandwidth, picks="mag", use_fft=True,
                    return_itc=False, average=True, decim=band.decim, n_jobs=1, verbose="ERROR",
                )
                tfr.comment = name
                per_condition[name] = tfr
                summary.setdefault(band.name, {}).update(
                    _band_summary(tfr, band, name)
                )

        for contrast in cfg.tfr.contrasts:
            if contrast.band != band.name:
                continue
            if contrast.a not in per_condition or contrast.b not in per_condition:
                ctx.logger.warning(
                    "TFR contrast %r skipped: condition missing from this recording", contrast.name
                )
                continue
            per_condition[contrast.name] = _tfr_contrast(
                per_condition[contrast.a], per_condition[contrast.b], contrast
            )

        results[band.name] = per_condition
        mne.time_frequency.write_tfrs(
            ctx.paths.tfr(band.name), list(per_condition.values()), overwrite=True, verbose="ERROR"
        )
        _plot_tfr(ctx, band, per_condition)

    ctx.record(tfr_summary=summary)
    return results


def _read_band(path) -> dict:
    tfrs = mne.time_frequency.read_tfrs(path, verbose="ERROR")
    tfrs = tfrs if isinstance(tfrs, list) else [tfrs]
    return {tfr.comment: tfr for tfr in tfrs}


def _band_summary(tfr, band, condition: str) -> dict[str, float]:
    """Mean baseline-normalised power after the locking event, as a QC number."""
    if not band.baseline:
        return {}
    data = tfr.data.mean(axis=0)  # average over channels
    mask = (tfr.times >= band.baseline[0]) & (tfr.times <= band.baseline[1])
    if not mask.any():
        return {}
    baseline = data[:, mask].mean(axis=1, keepdims=True)
    relative = 100 * (data - baseline) / np.maximum(np.abs(baseline), np.finfo(float).tiny)
    post = tfr.times > band.baseline[1]
    if not post.any():
        return {}
    key = sanitize(condition)
    return {
        f"{key}_mean_percent_change": float(np.nanmean(relative[:, post])),
        f"{key}_peak_percent_change": float(np.nanmax(np.abs(relative[:, post]))),
    }


def _tfr_contrast(a, b, contrast):
    """Combine two TFRs into the contrast named in the config."""
    out = a.copy()
    tiny = np.finfo(float).tiny
    if contrast.kind == "normalised_difference":
        out.data = (a.data - b.data) / np.maximum(a.data + b.data, tiny)
    elif contrast.kind == "relative_change":
        out.data = (a.data - b.data) / np.maximum(np.abs(b.data), tiny)
    elif contrast.kind == "difference":
        out.data = a.data - b.data
    else:
        raise ValueError(f"unknown TFR contrast kind {contrast.kind!r}")
    out.comment = contrast.name
    return out


def _plot_tfr(ctx: SubjectContext, band, per_condition: dict) -> None:
    import matplotlib.pyplot as plt

    if not ctx.cfg.output.figures:
        return
    contrast_names = {c.name for c in ctx.cfg.tfr.contrasts}
    for name, tfr in per_condition.items():
        is_contrast = name in contrast_names
        try:
            data = tfr.data.mean(axis=0)
            if band.baseline and not is_contrast:
                mask = (tfr.times >= band.baseline[0]) & (tfr.times <= band.baseline[1])
                base = data[:, mask].mean(axis=1, keepdims=True)
                data = 100 * (data - base) / np.maximum(np.abs(base), np.finfo(float).tiny)
                label = "Power change from baseline (%)"
            else:
                label = "Contrast" if is_contrast else "Power"
            limit = float(np.nanpercentile(np.abs(data), 98)) or 1.0
            fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
            mesh = ax.pcolormesh(tfr.times, tfr.freqs, data, shading="auto",
                                 cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.axvline(0, color="k", lw=1, ls="--")
            ax.set(title=f"{ctx.rec.key}: {name} ({band.name} band)",
                   xlabel="Time (s)", ylabel="Frequency (Hz)")
            fig.colorbar(mesh, ax=ax, label=label)
            path = save_figure(fig, ctx.figure_path(f"08_tfr_{band.name}_{sanitize(name)}"), ctx.cfg)
            ctx.add_figure("Time-frequency", f"{name} ({band.name})", path)
        except (RuntimeError, ValueError) as exc:
            ctx.logger.warning("TFR plot failed for %s (%s)", name, exc)
            plt.close("all")

        if band.topomap and len(band.topomap) == 4:
            tmin, tmax, fmin, fmax = band.topomap
            try:
                fig = tfr.plot_topomap(
                    tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                    baseline=tuple(band.baseline) if band.baseline and not is_contrast else None,
                    mode=band.baseline_mode, show=False,
                )
                path = save_figure(
                    fig, ctx.figure_path(f"08_tfr_topomap_{band.name}_{sanitize(name)}"), ctx.cfg
                )
                ctx.add_figure("Time-frequency", f"{name} topography ({band.name})", path)
            except (RuntimeError, ValueError) as exc:
                ctx.logger.warning("TFR topomap failed for %s (%s)", name, exc)
                plt.close("all")


# --------------------------------------------------------------------------- #
# 9. Multivariate pattern analysis
# --------------------------------------------------------------------------- #


def run_decoding(ctx: SubjectContext, epochs: mne.Epochs | None = None) -> dict | None:
    """Sliding-window SVM decoding of two conditions.

    At each window position every sensor x sample inside the window forms one
    feature vector, so the decoder uses temporal as well as spatial structure
    and the resulting time course is smoother than single-sample decoding.
    Classes are balanced first, so the AUC cannot be inflated by unequal trial
    counts - which matters far more across a whole study than in one tutorial
    recording.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from mne.decoding import Vectorizer, cross_val_multiscore

    cfg = ctx.cfg.mvpa
    out_file = ctx.paths.decoding
    if not ctx.needs(out_file):
        ctx.logger.info("MVPA: reusing %s", out_file.name)
        with np.load(out_file, allow_pickle=True) as stored:
            cached = {k: stored[k] for k in stored.files}
        _record_mvpa_metrics(ctx, cached["times"], cached["mean_scores"],
                             int(cached["n_per_class"]))
        return cached

    if epochs is None:
        from .preprocess import make_epochs
        epochs = make_epochs(ctx)

    picked = _picked(epochs, cfg.picks, ctx)
    available = _conditions(ctx, picked)
    missing = [name for name in cfg.conditions if name not in available]
    if missing:
        ctx.logger.warning("MVPA skipped: condition(s) %s have no epochs", missing)
        return None

    work = picked.copy()
    if cfg.h_freq is not None:
        # Filter the whole epoch before cropping so the edge effects land outside
        # the retained window.
        work.filter(None, cfg.h_freq, verbose="ERROR")
    if cfg.sfreq and cfg.sfreq < work.info["sfreq"]:
        work.resample(cfg.sfreq, verbose="ERROR")
    if cfg.crop:
        work.crop(*cfg.crop)

    rng = np.random.default_rng(cfg.random_state)
    groups = [work[name] for name in cfg.conditions]
    counts = [len(g) for g in groups]
    n_per_class = min(counts)
    if n_per_class < cfg.cv:
        ctx.logger.warning(
            "MVPA skipped: only %d trials in the smaller class, fewer than cv=%d",
            n_per_class, cfg.cv,
        )
        return None

    arrays, labels = [], []
    for index, group in enumerate(groups):
        if cfg.balance_classes and len(group) > n_per_class:
            keep = np.sort(rng.choice(len(group), size=n_per_class, replace=False))
            group = group[keep]
        arrays.append(group.get_data(picks="mag"))
        labels.extend([index] * len(group))
    X = np.concatenate(arrays)
    y = np.asarray(labels)
    # Remove a linear trend per trial and sensor, in place of a high-pass filter.
    X = _detrend(X, axis=-1, type="linear")

    estimator = (
        SVC(kernel="linear") if cfg.estimator == "linear_svm"
        else SVC(kernel="rbf", C=1.0, gamma="scale")
    )
    clf = make_pipeline(Vectorizer(), StandardScaler(), estimator)
    cv = StratifiedKFold(cfg.cv, shuffle=True, random_state=cfg.random_state)

    sfreq = work.info["sfreq"]
    window = max(1, int(round(cfg.window_s * sfreq)))
    starts = np.arange(0, X.shape[-1] - window + 1, max(1, cfg.step_samples))
    if starts.size == 0:
        ctx.logger.warning("MVPA skipped: the epoch is shorter than one decoding window")
        return None

    scores, times = [], []
    with timed(f"MVPA over {len(starts)} windows", ctx.logger):
        for start in starts:
            fold = cross_val_multiscore(
                clf, X[:, :, start:start + window], y, cv=cv,
                scoring=cfg.scoring, n_jobs=cfg.n_jobs,
            )
            scores.append(fold)
            times.append(float(work.times[start:start + window].mean()))

    scores = np.asarray(scores)          # (n_windows, n_folds)
    times = np.asarray(times)
    mean_scores = scores.mean(axis=1)
    peak = int(np.argmax(mean_scores))

    np.savez(
        out_file, times=times, scores=scores, mean_scores=mean_scores,
        classes=np.array(cfg.conditions, dtype=object), n_per_class=n_per_class,
        scoring=cfg.scoring,
    )
    _record_mvpa_metrics(ctx, times, mean_scores, n_per_class)
    ctx.logger.info(
        "MVPA: peak %s %.3f at %.3f s (%d trials/class)",
        cfg.scoring, mean_scores[peak], times[peak], n_per_class,
    )
    _plot_decoding(ctx, times, scores, mean_scores)
    return {"times": times, "scores": scores, "mean_scores": mean_scores}


def _plot_decoding(ctx: SubjectContext, times, scores, mean_scores) -> None:
    import matplotlib.pyplot as plt

    cfg = ctx.cfg.mvpa
    chance = 0.5 if cfg.scoring == "roc_auc" else None
    sem = scores.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(times, mean_scores, label=" vs ".join(cfg.conditions))
    ax.fill_between(times, mean_scores - sem, mean_scores + sem, alpha=0.2)
    if chance is not None:
        ax.axhline(chance, color="k", ls="--", lw=1, label="chance")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set(title=f"{ctx.rec.key}: sliding-window decoding",
           xlabel="Time (s)", ylabel=cfg.scoring.upper())
    ax.legend(fontsize=8)
    path = save_figure(fig, ctx.figure_path("09_decoding"), ctx.cfg)
    ctx.add_figure("Decoding", "Sliding-window decoding", path)


def _record_mvpa_metrics(ctx: SubjectContext, times, mean_scores, n_per_class: int) -> None:
    """Peak and mean decoding performance, for the group table."""
    times = np.asarray(times)
    mean_scores = np.asarray(mean_scores)
    peak = int(np.argmax(mean_scores))
    ctx.record(
        mvpa_peak_score=float(mean_scores[peak]),
        mvpa_peak_time_s=float(times[peak]),
        mvpa_mean_score=float(mean_scores.mean()),
        mvpa_n_per_class=int(n_per_class),
        mvpa_classes=list(ctx.cfg.mvpa.conditions),
    )
