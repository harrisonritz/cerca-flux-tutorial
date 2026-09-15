"""Group-level aggregation across subjects.

Produces the study's shared outputs: a quality-metrics table, grand-average
evoked responses, time-frequency representations and decoding curves, and the
average of the template-morphed source estimates.

No single metric defines OPM-MEG data quality, so the table keeps acquisition
noise, HFC benefit, artefact burden, epoch retention and effect size as separate
columns rather than collapsing them into one score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import mne

from .config import Config
from .paths import Recording, SubjectPaths
from .utils import get_logger, save_figure


def group_dir(cfg: Config) -> Path:
    path = cfg.deriv_root / "group"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_group_outputs(cfg: Config, rows: list[dict],
                        recordings: list[Recording]) -> dict:
    """Write every group-level product the config asks for."""
    logger = get_logger()
    out = group_dir(cfg)
    outputs: dict[str, Path] = {}

    table = write_quality_table(cfg, rows)
    outputs["quality_metrics"] = table
    logger.info("group: wrote %s", table.name)

    if not cfg.group.enabled:
        return outputs

    usable = [rec for rec, row in zip(recordings, rows) if row["status"] in {"ok", "partial"}]
    if not usable:
        logger.warning("group: no recording completed, skipping the averages")
        return outputs

    if cfg.group.grand_average:
        outputs.update(_grand_average_evoked(cfg, usable, out))
        outputs.update(_grand_average_tfr(cfg, usable, out))
        outputs.update(_average_decoding(cfg, usable, out))
    if cfg.group.average_sources and cfg.morph.enabled:
        outputs.update(_average_sources(cfg, usable, out))
    return outputs


# --------------------------------------------------------------------------- #
# Quality table
# --------------------------------------------------------------------------- #

#: Columns promoted to the front of the table, in reading order.
_PRIMARY = [
    "recording", "subject", "session", "task", "run", "status", "error",
    "sfreq_raw", "duration_s", "n_mag",
    "n_bad_channels", "n_auto_bad_channels", "n_manual_bad_channels",
    "n_hfc_projections", "psd_after_hfc_db", "hfc_reduction_1_40_db",
    "line_excess_after_hfc_db", "first_look_dynamic_rms_pT",
    "blinks_per_min", "muscle_percent",
    "n_ica_components", "n_ica_excluded",
    "epochs_requested", "epochs_retained", "epochs_rejected_percent",
    "mvpa_peak_score", "mvpa_peak_time_s", "mvpa_n_per_class",
]


def write_quality_table(cfg: Config, rows: list[dict]) -> Path:
    """One row per recording, flattening the nested per-stage metrics."""
    flat = [_flatten(row) for row in rows]
    frame = pd.DataFrame(flat)
    ordered = [c for c in _PRIMARY if c in frame.columns]
    rest = sorted(c for c in frame.columns if c not in ordered)
    frame = frame[ordered + rest]
    path = group_dir(cfg) / "quality_metrics.tsv"
    frame.to_csv(path, sep="\t", index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    _plot_quality(cfg, frame)
    return path


def _flatten(row: dict, prefix: str = "") -> dict:
    """Flatten nested metric dicts into ``parent_child`` columns."""
    out: dict[str, object] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, prefix=f"{name}_"))
        elif isinstance(value, (list, tuple)):
            # Lists of channel names stay useful as a count plus the values.
            out[f"{name}_n"] = len(value)
            if len(value) <= 12:
                out[name] = ", ".join(str(v) for v in value)
        else:
            out[name] = value
    return out


def _plot_quality(cfg: Config, frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if not cfg.output.figures or frame.empty:
        return
    panels = [
        ("HFC reduction\n1-40 Hz (dB)", "hfc_reduction_1_40_db"),
        ("Bad channels", "n_bad_channels"),
        ("Muscle artefact (%)", "muscle_percent"),
        ("Epochs rejected (%)", "epochs_rejected_percent"),
        ("Epochs retained", "epochs_retained"),
        ("Peak decoding score", "mvpa_peak_score"),
    ]
    panels = [(label, col) for label, col in panels if col in frame.columns
              and frame[col].notna().any()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.8),
                             constrained_layout=True, squeeze=False)
    labels = frame["recording"].astype(str).str.replace("_", "\n", regex=False)
    for ax, (title, column) in zip(axes[0], panels):
        values = pd.to_numeric(frame[column], errors="coerce")
        ax.bar(range(len(values)), values.fillna(0))
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_title(title, fontsize=9)
    save_figure(fig, group_dir(cfg) / "quality_metrics.png", cfg)


# --------------------------------------------------------------------------- #
# Grand averages
# --------------------------------------------------------------------------- #


def _grand_average_evoked(cfg: Config, recordings, out: Path) -> dict:
    logger = get_logger()
    per_condition: dict[str, list[mne.Evoked]] = {}
    for rec in recordings:
        path = SubjectPaths(cfg, rec).evoked
        if not path.exists():
            continue
        for evoked in mne.read_evokeds(path, verbose="ERROR"):
            per_condition.setdefault(evoked.comment, []).append(evoked)

    averages = []
    for condition, evokeds in per_condition.items():
        # Recordings differ in which channels survived QC, so average over the
        # channels every contributing recording has.
        shared = set(evokeds[0].ch_names).intersection(*(set(e.ch_names) for e in evokeds))
        if len(shared) < 2:
            logger.warning(
                "group: only %d channel(s) shared across recordings for %r; "
                "skipping its grand average", len(shared), condition,
            )
            continue
        aligned = [e.copy().pick(sorted(shared)) for e in evokeds]
        grand = mne.grand_average(aligned, interpolate_bads=False)
        grand.comment = condition
        averages.append(grand)
        logger.info("group: grand-averaged %r over %d recordings (%d channels)",
                    condition, len(aligned), len(shared))

    if not averages:
        return {}
    path = out / "grand_average_ave.fif"
    mne.write_evokeds(path, averages, overwrite=True, verbose="ERROR")
    _plot_grand_evoked(cfg, averages, out)
    return {"grand_average_evoked": path}


def _plot_grand_evoked(cfg: Config, averages: list[mne.Evoked], out: Path) -> None:
    import matplotlib.pyplot as plt

    if not cfg.output.figures:
        return
    try:
        fig = mne.viz.plot_compare_evokeds(
            {e.comment: e for e in averages}, combine="gfp", show=False,
            title="Grand average (GFP)",
        )
        fig = fig[0] if isinstance(fig, list) else fig
        save_figure(fig, out / "grand_average_gfp.png", cfg)
    except (RuntimeError, ValueError):
        plt.close("all")


def _grand_average_tfr(cfg: Config, recordings, out: Path) -> dict:
    logger = get_logger()
    outputs = {}
    for band in cfg.tfr.bands:
        collected: dict[str, list] = {}
        for rec in recordings:
            path = SubjectPaths(cfg, rec).tfr(band.name)
            if not path.exists():
                continue
            tfrs = mne.time_frequency.read_tfrs(path, verbose="ERROR")
            for tfr in (tfrs if isinstance(tfrs, list) else [tfrs]):
                collected.setdefault(tfr.comment, []).append(tfr)

        averages = []
        for condition, tfrs in collected.items():
            shared = set(tfrs[0].ch_names).intersection(*(set(t.ch_names) for t in tfrs))
            if len(shared) < 2 or len({t.data.shape[1:] for t in tfrs}) != 1:
                logger.warning("group: %r/%s not comparable across recordings; skipping",
                               condition, band.name)
                continue
            aligned = [t.copy().pick(sorted(shared)) for t in tfrs]
            grand = aligned[0].copy()
            grand.data = np.mean([t.data for t in aligned], axis=0)
            grand.comment = condition
            averages.append(grand)
        if averages:
            path = out / f"grand_average_band-{band.name}_tfr.h5"
            mne.time_frequency.write_tfrs(path, averages, overwrite=True, verbose="ERROR")
            outputs[f"grand_average_tfr_{band.name}"] = path
            logger.info("group: grand-averaged %d TFR condition(s) for band %r",
                        len(averages), band.name)
    return outputs


def _average_decoding(cfg: Config, recordings, out: Path) -> dict:
    import matplotlib.pyplot as plt

    curves, times = [], None
    for rec in recordings:
        path = SubjectPaths(cfg, rec).decoding
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as stored:
            if times is None:
                times = stored["times"]
            elif len(stored["times"]) != len(times):
                continue  # a differently cropped recording cannot be pooled
            curves.append(stored["mean_scores"])
    if not curves or times is None:
        return {}

    curves = np.asarray(curves)
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean)
    path = out / "group_decoding.npz"
    np.savez(path, times=times, curves=curves, mean=mean, sem=sem)

    if cfg.output.figures:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for curve in curves:
            ax.plot(times, curve, color="0.75", lw=0.7)
        ax.plot(times, mean, lw=2, label=f"mean of {len(curves)}")
        ax.fill_between(times, mean - sem, mean + sem, alpha=0.25)
        if cfg.mvpa.scoring == "roc_auc":
            ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set(title="Group decoding", xlabel="Time (s)", ylabel=cfg.mvpa.scoring.upper())
        ax.legend(fontsize=8)
        save_figure(fig, out / "group_decoding.png", cfg)
    get_logger().info("group: averaged decoding over %d recordings", len(curves))
    return {"group_decoding": path}


def _average_sources(cfg: Config, recordings, out: Path) -> dict:
    """Average the fsaverage-morphed estimates, which share one grid."""
    logger = get_logger()
    outputs = {}
    for space in cfg.source.spaces:
        collected: dict[str, list] = {}
        for rec in recordings:
            paths = SubjectPaths(cfg, rec)
            for path in sorted(paths.analysis_dir.glob(f"{rec.key}_*_{space}_fsaverage-*.h5")):
                name = path.name[len(rec.key) + 1:].rsplit(f"_{space}_fsaverage", 1)[0]
                collected.setdefault(name, []).append(
                    mne.read_source_estimate(path, subject=cfg.morph.subject_to)
                )
        for name, stcs in collected.items():
            shapes = {stc.data.shape for stc in stcs}
            if len(shapes) != 1:
                logger.warning("group: %s/%s has mismatched shapes %s; skipping",
                               name, space, shapes)
                continue
            grand = stcs[0].copy()
            grand._data = np.mean([np.real(stc.data) for stc in stcs], axis=0)
            target = out / f"group_{name}_{space}_{cfg.morph.subject_to}"
            grand.save(target, ftype="h5", overwrite=True, verbose="ERROR")
            outputs[f"group_source_{space}_{name}"] = target
            logger.info("group: averaged %s (%s) over %d recordings", name, space, len(stcs))
    return outputs
