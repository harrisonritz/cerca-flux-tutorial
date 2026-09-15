"""Per-subject HTML report collecting every stage's diagnostic figure."""

from __future__ import annotations

from pathlib import Path

import mne

from .context import SubjectContext
from .utils import write_json


def build_report(ctx: SubjectContext) -> None:
    """Assemble one HTML file per recording, plus the machine-readable metrics.

    The report is the artefact a human actually looks at when deciding whether a
    subject's preprocessing succeeded, so it is written even when later stages
    failed: a partial report showing where things stopped is more useful than none.
    """
    write_json(ctx.paths.metrics, {"recording": ctx.rec.key, "metrics": ctx.metrics,
                                   "stages": ctx.stage_status})
    if not ctx.cfg.output.report:
        return

    report = mne.Report(title=f"Cerca OPM-FLUX: {ctx.rec.key}", verbose="ERROR")

    summary = _summary_html(ctx)
    report.add_html(summary, title="Summary", section="Overview")

    for section, title, path in _collect_figures(ctx):
        try:
            report.add_image(image=path, title=title, section=section)
        except (RuntimeError, ValueError) as exc:
            ctx.logger.warning("report: could not embed %s (%s)", path.name, exc)

    report.save(ctx.paths.report_file, overwrite=True, open_browser=False, verbose="ERROR")
    ctx.logger.info("report: wrote %s", ctx.paths.report_file.name)


def _summary_html(ctx: SubjectContext) -> str:
    rows = []

    def add(label: str, key: str, fmt: str = "{}") -> None:
        if key in ctx.metrics and ctx.metrics[key] is not None:
            try:
                value = fmt.format(ctx.metrics[key])
            except (ValueError, TypeError):
                value = str(ctx.metrics[key])
            rows.append((label, value))

    add("Sampling rate (Hz)", "sfreq_raw", "{:.0f}")
    add("Duration (s)", "duration_s", "{:.1f}")
    add("Magnetometers", "n_mag")
    add("Bad channels", "n_bad_channels")
    add("&nbsp;&nbsp;of which automatic", "n_auto_bad_channels")
    add("HFC projections", "n_hfc_projections")
    add("HFC reduction 1-40 Hz (dB)", "hfc_reduction_1_40_db", "{:.2f}")
    add("Mains excess after HFC (dB)", "line_excess_after_hfc_db", "{:.2f}")
    add("Blinks per minute", "blinks_per_min", "{:.2f}")
    add("Muscle artefact (%)", "muscle_percent", "{:.2f}")
    add("ICA components", "n_ica_components")
    add("ICA components removed", "n_ica_excluded")
    add("Epochs retained", "epochs_retained")
    add("Epochs rejected (%)", "epochs_rejected_percent", "{:.1f}")
    add("Peak decoding score", "mvpa_peak_score", "{:.3f}")

    body = "".join(
        f"<tr><td style='padding:2px 12px 2px 0'>{label}</td>"
        f"<td style='padding:2px 0'><b>{value}</b></td></tr>"
        for label, value in rows
    )
    status = "".join(
        f"<li>{name}: {state}</li>" for name, state in ctx.stage_status.items()
    )
    return (
        f"<table>{body}</table>"
        f"<p><b>Stages</b></p><ul>{status}</ul>"
        f"<p style='color:#666'>Configuration: "
        f"{ctx.cfg.deriv_root / 'config.yaml'}</p>"
    )


#: Filename prefix -> report section, so figures written by an earlier
#: invocation land in the right place even when that stage is not re-run now.
_SECTION_BY_PREFIX = {
    "01": "Sensor quality", "02": "Sensor quality", "03": "HFC",
    "04": "Artefacts", "05": "ICA", "06": "Epochs", "07": "ERF",
    "08": "Time-frequency", "09": "Decoding", "10": "Source",
}


def _collect_figures(ctx: SubjectContext) -> list[tuple[str, str, Path]]:
    """Every figure available for this recording, in pipeline order.

    Figures produced in *this* process carry a hand-written section and title.
    Anything else already on disk - from a run that only re-did some stages -
    is still included, with its section and title derived from the filename, so
    that re-running one stage never silently shrinks the report.
    """
    known = {Path(path).resolve(): (section, title) for section, title, path in ctx.figures}
    prefix = f"{ctx.rec.key}_"
    out: list[tuple[str, str, Path]] = []
    for path in sorted(ctx.paths.figures_dir.glob(f"{prefix}*.png")):
        resolved = path.resolve()
        if resolved in known:
            section, title = known[resolved]
        else:
            stem = path.name[len(prefix):].removesuffix(".png")
            number, _, rest = stem.partition("_")
            section = _SECTION_BY_PREFIX.get(number, "Other")
            title = rest.replace("_", " ").strip() or stem
        out.append((section, title, path))
    return out
