"""Overlay a reference recording on a cohort's motor responses.

The ``FLUX_Oxford_Princeton_Comparison`` notebook sets one Oxford participant next
to one Princeton participant.  This module does the same for a whole cohort: it
reads the per-recording ``motor-sensor`` / ``motor-source`` files that the ``motor``
and ``motor_source`` stages cache, draws the reference as one **thick black** line
and every cohort participant as a **thin coloured** line, and writes the summary
metrics next to the figures.

Colour follows the participant, never their rank: a participant keeps one colour
across every figure and every preprocessing variant, and a participant who is
missing leaves a gap in the colour order instead of repainting the others.  Thirty
or more categorical hues cannot be told apart, so colour here encodes the
participant's *position* in the cohort (a sequential ramp with a labelled colour
bar), and identity is carried by the per-participant small multiples and the
metrics table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .motor import load_motor_result
from .paths import SubjectPaths, discover_recordings
from .utils import get_logger, read_json, use_headless_backend

REFERENCE_COLOUR = "black"
REFERENCE_LINEWIDTH = 3.0
COHORT_LINEWIDTH = 1.0
#: Ramp for the cohort.  Clipped away from both ends: the dark end would be mistaken
#: for the black reference line and the pale end disappears on a white background.
RAMP = "viridis"
RAMP_RANGE = (0.12, 0.88)
INK = "0.15"
INK_MUTED = "0.42"


# --------------------------------------------------------------------------- #
# What gets drawn
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Measure:
    """One curve per participant: where it lives, which arrays, and how to label it."""

    key: str
    file: str  # "sensor" or "source"
    x: str
    y: str
    title: str
    ylabel: str
    zero_line: bool = False
    n_key: str = "motor_n_epochs"


MEASURES: tuple[Measure, ...] = (
    Measure("erf_rms", "sensor", "erf_times", "erf_rms_ft",
            "Central response-locked field", "Central RMS (fT)"),
    Measure("erf_snr", "sensor", "erf_times", "erf_z",
            "Baseline-normalised response", "Baseline SD", zero_line=True),
    Measure("beta", "sensor", "tfr_times", "beta_percent",
            "Central motor beta power", "Change from baseline (%)", zero_line=True),
    Measure("lcmv", "source", "lcmv_times", "lcmv_z",
            "{label}: LCMV response", "Parcel response (baseline SD)", zero_line=True,
            n_key="motor_source_n_epochs"),
    Measure("dics", "source", "dics_times", "dics_db",
            "{label}: DICS beta power", "Power change from baseline (dB)", zero_line=True,
            n_key="motor_source_n_epochs"),
)


@dataclass
class Entry:
    """One participant (or the reference): its cached results, or why they are absent."""

    group: str  # "reference" or "cohort"
    label: str
    key: str
    sensor: dict | None = None
    source: dict | None = None
    status: str = "ok"
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.sensor is not None or self.source is not None


def _series(entry: Entry, measure: Measure):
    data = entry.sensor if measure.file == "sensor" else entry.source
    if data is None or measure.x not in data or measure.y not in data:
        return None
    return np.asarray(data[measure.x]), np.asarray(data[measure.y])


def _measure_title(measure: Measure, entries: list[Entry]) -> str:
    if "{label}" not in measure.title:
        return measure.title
    for entry in entries:
        if entry.source is not None and "label" in entry.source:
            return measure.title.format(label=entry.source["label"])
    return measure.title.format(label="ROI")


# --------------------------------------------------------------------------- #
# Collecting results from the derivatives tree
# --------------------------------------------------------------------------- #


def collect(cfg: Config, group: str) -> list[Entry]:
    """Load every selected recording's cached motor results.

    Requested subjects that have no BIDS recording at all, and recordings whose
    motor stage failed or never ran, are kept as entries with a reason, so the
    figures and the metrics table say who is missing rather than silently
    shrinking.
    """
    recordings = {rec.subject: rec for rec in discover_recordings(cfg)}
    requested = list(cfg.study.subjects or sorted(recordings))
    ordered = requested + [s for s in sorted(recordings) if s not in requested]

    entries: list[Entry] = []
    for label in ordered:
        rec = recordings.get(label)
        if rec is None:
            entries.append(Entry(group, label, f"sub-{label}", status="no BIDS recording",
                                 note=f"sub-{label} has no {cfg.study.datatype} recording "
                                      "matching the study filters"))
            continue
        paths = SubjectPaths(cfg, rec)
        entry = Entry(group, label, rec.key)
        if paths.motor_sensor.exists():
            entry.sensor = load_motor_result(paths.motor_sensor)
        if paths.motor_source.exists():
            entry.source = load_motor_result(paths.motor_source)
        if not entry.usable:
            entry.status, entry.note = "no motor output", _why_missing(paths)
        elif entry.sensor is None:
            entry.status, entry.note = "no sensor output", _why_missing(paths, "motor")
        elif entry.source is None:
            entry.status, entry.note = "no source output", _why_missing(paths, "motor_source")
        entries.append(entry)
    return entries


def _why_missing(paths: SubjectPaths, stage: str | None = None) -> str:
    """The recorded outcome of the stage that should have produced the file."""
    if not paths.metrics.exists():
        return "the pipeline has not been run for this recording"
    stages = read_json(paths.metrics).get("stages", {})
    names = [stage] if stage else ["motor", "motor_source", "epochs", "ica", "annotate",
                                   "hfc", "qc"]
    for name in names:
        state = str(stages.get(name, ""))
        if state.startswith("failed"):
            return f"{name} {state}"
    return f"stage {stage or 'motor'} did not produce output ({stages.get(stage or 'motor', 'not run')})"


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #


def cohort_colours(labels: list[str]) -> dict[str, tuple]:
    """One stable colour per participant label, in cohort order."""
    import matplotlib

    cmap = matplotlib.colormaps[RAMP]
    positions = np.linspace(*RAMP_RANGE, len(labels)) if len(labels) > 1 else [np.mean(RAMP_RANGE)]
    return {label: cmap(float(p)) for label, p in zip(labels, positions)}


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def draw_measure(ax, measure: Measure, reference: list[Entry], cohort: list[Entry],
                 colours: dict[str, tuple], title: str | None = None) -> tuple[list, list]:
    """Cohort lines first, thin and coloured; the reference last, thick and black, on top.

    Returns ``(cohort_lines, reference_lines)`` so callers (and tests) can inspect
    exactly what was drawn.
    """
    cohort_lines, reference_lines = [], []
    for entry in cohort:
        series = _series(entry, measure)
        if series is None:
            continue
        (line,) = ax.plot(*series, color=colours[entry.label], lw=COHORT_LINEWIDTH,
                          alpha=0.9, zorder=3)
        cohort_lines.append(line)
    for entry in reference:
        series = _series(entry, measure)
        if series is None:
            continue
        (line,) = ax.plot(*series, color=REFERENCE_COLOUR, lw=REFERENCE_LINEWIDTH,
                          zorder=10, solid_capstyle="round")
        reference_lines.append(line)
    ax.axvline(0, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
    if measure.zero_line:
        ax.axhline(0, color=INK_MUTED, lw=0.8, ls=":", zorder=1)
    ax.set(title=title or _measure_title(measure, reference + cohort),
           xlabel="Time from response (s)", ylabel=measure.ylabel)
    return cohort_lines, reference_lines


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.title.set_fontsize(10.5)
    ax.title.set_color(INK)


def plot_overlay(reference: list[Entry], cohort: list[Entry], colours: dict[str, tuple],
                 names: tuple[str, str], variant: str, path: Path, dpi: int) -> Path:
    """The composite figure: every measure on one axis per measure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    everyone = reference + cohort
    have_source = any(e.source is not None for e in everyone)
    sensor_measures = [m for m in MEASURES if m.file == "sensor"]
    source_measures = [m for m in MEASURES if m.file == "source"]

    if have_source:
        fig = plt.figure(figsize=(17, 9.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 3)
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        axes += [fig.add_subplot(grid[1, i]) for i in range(2)]
        legend_ax = fig.add_subplot(grid[1, 2])
        measures = sensor_measures + source_measures
    else:
        fig = plt.figure(figsize=(17, 4.8), constrained_layout=True)
        grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.8])
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        legend_ax = fig.add_subplot(grid[0, 3])
        measures = sensor_measures

    for ax, measure in zip(axes, measures):
        draw_measure(ax, measure, reference, cohort, colours)
        _style_axes(ax)

    plotted = [e for e in cohort if e.usable]
    ref_name, cohort_name = names
    legend_ax.set_axis_off()
    # The cohort swatch is a short run of ramp colours, so it reads as "many thin
    # coloured lines" and not as one participant's colour.
    ramp = [Line2D([0], [0], color=cohort_colours(list("abcd"))[k], lw=COHORT_LINEWIDTH * 1.4)
            for k in "abcd"]
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color=REFERENCE_COLOUR, lw=REFERENCE_LINEWIDTH),
            tuple(ramp),
        ],
        labels=[
            f"{ref_name}: " + ", ".join(f"sub-{e.label}" for e in reference if e.usable),
            f"{cohort_name}: {len(plotted)} of {len(cohort)} participants",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="upper left", frameon=False, fontsize=10, labelcolor=INK,
        bbox_to_anchor=(0.0, 1.0), handlelength=3.2,
    )

    if cohort:
        n = len(cohort)
        cmap = ListedColormap([colours[e.label] for e in cohort])
        sm = ScalarMappable(norm=BoundaryNorm(np.arange(-0.5, n + 0.5), n), cmap=cmap)
        cax = legend_ax.inset_axes([0.02, 0.60, 0.94, 0.07])
        bar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        ticks = sorted(set(np.linspace(0, n - 1, min(n, 8)).round().astype(int)))
        bar.set_ticks(ticks)
        bar.set_ticklabels([cohort[i].label for i in ticks], fontsize=8)
        bar.ax.tick_params(length=2, colors=INK_MUTED)
        bar.outline.set_visible(False)
        cax.set_title(f"{cohort_name} participant (colour = position in cohort)",
                      fontsize=9, color=INK_MUTED, loc="left")

    missing = [e for e in cohort if not e.usable]
    lines = []
    if missing:
        lines.append(f"Not plotted ({len(missing)}):")
        lines += [f"  {e.label}: {e.status}" for e in missing[:8]]
        if len(missing) > 8:
            lines.append(f"  ... and {len(missing) - 8} more (see the metrics table)")
    legend_ax.text(0.0, 0.46, "\n".join(lines), transform=legend_ax.transAxes, va="top",
                   fontsize=8.5, color=INK_MUTED, family="monospace")

    fig.suptitle(f"Motor response: {ref_name} (thick black) vs {cohort_name} (thin lines)"
                 f"    [variant: {variant}]", fontsize=12.5, color=INK, x=0.01, ha="left")
    return _save(fig, path, dpi)


def plot_small_multiples(measure: Measure, reference: list[Entry], cohort: list[Entry],
                         colours: dict[str, tuple], names: tuple[str, str], variant: str,
                         path: Path, dpi: int, ncols: int = 6) -> Path | None:
    """One panel per participant, each against the reference, on shared axes."""
    import matplotlib.pyplot as plt

    shown = [e for e in cohort if _series(e, measure) is not None]
    if not shown:
        return None
    nrows = math.ceil(len(shown) / ncols)
    ncols = min(ncols, len(shown))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.15 * nrows + 0.9),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)

    lo, hi, off_scale = _shared_limits(measure, reference, shown)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)

    for ax, entry in zip(axes.flat, shown):
        for ref in reference:
            series = _series(ref, measure)
            if series is not None:
                ax.plot(*series, color=REFERENCE_COLOUR, lw=1.6, zorder=2, alpha=0.9)
        ax.plot(*_series(entry, measure), color=colours[entry.label], lw=1.5, zorder=3)
        ax.axvline(0, color=INK_MUTED, lw=0.7, ls="--")
        if measure.zero_line:
            ax.axhline(0, color=INK_MUTED, lw=0.6, ls=":")
        data = entry.sensor if measure.file == "sensor" else entry.source
        n = data["metrics"].get(measure.n_key)
        ax.set_title(f"{entry.label}" + (f"  (n={n})" if n is not None else "")
                     + ("  off scale" if entry.label in off_scale else ""),
                     fontsize=9, color=INK, loc="left")
        ax.tick_params(labelsize=7.5)
        _style_axes(ax)
        ax.title.set_fontsize(9)
    for ax in list(axes.flat)[len(shown):]:
        ax.set_visible(False)
    axes.flat[0].set_ylim(lo - pad, hi + pad)

    title = _measure_title(measure, reference + cohort)
    fig.suptitle(f"{title}: each {names[1]} participant (colour) vs {names[0]} (black)"
                 f"    [variant: {variant}]", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.supxlabel("Time from response (s)", fontsize=9.5, color=INK_MUTED)
    fig.supylabel(measure.ylabel, fontsize=9.5, color=INK_MUTED)
    return _save(fig, path, dpi)


def _shared_limits(measure: Measure, reference: list[Entry], shown: list[Entry],
                   fence: float = 3.0) -> tuple[float, float, set[str]]:
    """y-limits for shared small-multiple axes.

    The reference's whole curve always fits.  A cohort participant whose extreme
    lies more than ``fence`` interquartile ranges beyond the other participants'
    extremes is an outlier: it is left off the limit calculation (and its panel is
    marked "off scale") so that one noisy recording cannot flatten everybody else.
    Percentiles of the pooled samples would be wrong here, because the peak is a
    handful of samples among many baseline ones and would be clipped.
    """
    ref = [_series(e, measure)[1] for e in reference if _series(e, measure) is not None]
    lows = np.array([np.nanmin(_series(e, measure)[1]) for e in shown])
    highs = np.array([np.nanmax(_series(e, measure)[1]) for e in shown])
    keep = np.ones(len(shown), bool)
    if len(shown) >= 4:
        for values, sign in ((lows, -1), (highs, +1)):
            q1, q3 = np.percentile(values, [25, 75])
            spread = fence * (q3 - q1)
            keep &= (values >= q1 - spread) if sign < 0 else (values <= q3 + spread)
    lo = min([lows[keep].min(), *[np.nanmin(r) for r in ref]]) if keep.any() else lows.min()
    hi = max([highs[keep].max(), *[np.nanmax(r) for r in ref]]) if keep.any() else highs.max()
    return float(lo), float(hi), {e.label for e, k in zip(shown, keep) if not k}


def _save(fig, path: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Metrics table
# --------------------------------------------------------------------------- #


def metrics_table(names: tuple[str, str], entries: list[Entry]) -> pd.DataFrame:
    rows = []
    for entry in entries:
        row = {"dataset": names[0] if entry.group == "reference" else names[1],
               "subject": entry.label, "recording": entry.key,
               "status": entry.status, "note": entry.note}
        for data in (entry.sensor, entry.source):
            if data is not None:
                row.update(data["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def default_output_dir(cfg: Config) -> Path:
    """``<repo>/outputs/motor_response/<variant>``, next to the notebook's outputs."""
    repo = Path(__file__).resolve().parents[1]
    return repo / "outputs" / "motor_response" / (cfg.study.variant or "default")


def compare_motor(reference_cfg: Config, cohort_cfg: Config, out_dir: Path | None = None,
                  small_multiples: bool = True) -> dict[str, Path]:
    """Draw the overlay figures and write the metrics table; returns what was written."""
    use_headless_backend()
    logger = get_logger()
    if reference_cfg.study.variant != cohort_cfg.study.variant:
        logger.warning(
            "the reference uses variant %r but the cohort uses %r; they are compared anyway",
            reference_cfg.study.variant, cohort_cfg.study.variant,
        )
    names = (reference_cfg.study.site or "Reference", cohort_cfg.study.site or "Cohort")
    variant = cohort_cfg.study.variant or "default"
    out_dir = Path(out_dir) if out_dir else default_output_dir(cohort_cfg)
    dpi = cohort_cfg.output.figure_dpi

    reference = collect(reference_cfg, "reference")
    cohort = collect(cohort_cfg, "cohort")
    if not any(e.usable for e in reference):
        raise RuntimeError(
            f"no motor results for the reference under {reference_cfg.deriv_root}. Run "
            "`cerca-flux run --preset motor` for the reference configuration first. "
            f"Reasons: {[(e.label, e.status, e.note) for e in reference]}"
        )
    if not any(e.usable for e in cohort):
        raise RuntimeError(
            f"no motor results for any cohort participant under {cohort_cfg.deriv_root}. "
            f"Reasons: {[(e.label, e.status) for e in cohort][:6]}"
        )

    colours = cohort_colours([e.label for e in cohort])
    outputs: dict[str, Path] = {}
    outputs["overlay"] = plot_overlay(reference, cohort, colours, names, variant,
                                      out_dir / "motor_response_overlay.png", dpi)
    if small_multiples:
        for measure in MEASURES:
            path = plot_small_multiples(measure, reference, cohort, colours, names, variant,
                                        out_dir / f"motor_response_small_multiples_{measure.key}.png",
                                        dpi)
            if path is not None:
                outputs[f"small_multiples_{measure.key}"] = path

    table = metrics_table(names, reference + cohort)
    outputs["metrics"] = out_dir / "motor_response_metrics.csv"
    table.to_csv(outputs["metrics"], index=False)

    n_ok = sum(e.usable for e in cohort)
    logger.info("compare: %s (reference) vs %d/%d %s participants -> %s", names[0], n_ok,
                len(cohort), names[1], out_dir)
    for entry in cohort:
        if not entry.usable:
            logger.warning("compare: %s %s not plotted: %s (%s)", names[1], entry.label,
                           entry.status, entry.note)
    return outputs
