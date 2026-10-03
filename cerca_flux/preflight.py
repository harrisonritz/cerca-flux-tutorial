"""Check, before a long run, that every recording's inputs are in place.

Nothing is processed and no folder is created.  A multi-hour background job over a
whole cohort should learn that participant 020 has no coregistration now, not when the
job reaches them.
"""

from __future__ import annotations

from .config import Config
from .paths import Recording, check_freesurfer, discover_recordings, find_trans, freesurfer_subject


def _label_problem(cfg: Config, fs_subject: str) -> str | None:
    import mne

    motor = cfg.motor.source
    hemi = motor.label.rsplit("-", 1)[1]
    try:
        labels = mne.read_labels_from_annot(
            fs_subject, parc=motor.parc, hemi=hemi, subjects_dir=str(cfg.fs_subjects_dir),
            verbose="ERROR",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return f"cannot read the {motor.parc!r} parcellation ({exc})"
    return None if any(label.name == motor.label for label in labels) else (
        f"no {motor.label!r} label in the {motor.parc!r} parcellation")


def preflight(cfg: Config) -> list[dict]:
    """One row per requested participant: ``{"subject", "ok", "problems"}``."""
    try:
        found = {rec.subject: rec for rec in discover_recordings(cfg)}
    except FileNotFoundError as exc:
        return [{"subject": "(study)", "ok": False, "problems": [str(exc)]}]

    rows = []
    reference = cfg.reference_recording
    if reference is not None and not reference.exists():
        rows.append({"subject": "(duration reference)", "ok": False,
                     "problems": [f"study.match_duration_to does not exist: {reference}"]})

    for subject in list(cfg.study.subjects or sorted(found)):
        rec: Recording | None = found.get(subject)
        if rec is None:
            rows.append({"subject": subject, "ok": False,
                         "problems": ["no raw BIDS recording matches the study filters"]})
            continue
        problems: list[str] = []
        fs_subject = freesurfer_subject(cfg, rec)
        try:
            check_freesurfer(cfg, fs_subject)
        except FileNotFoundError as exc:
            problems.append(str(exc).split(". ")[0])
        else:
            if cfg.motor.enabled and cfg.motor.source.enabled:
                problem = _label_problem(cfg, fs_subject)
                if problem:
                    problems.append(problem)
        if cfg.forward.enabled and find_trans(cfg, rec, fs_subject, None) is None:
            problems.append(
                f"no coregistration transform (*-trans.fif) for {fs_subject!r}; expected in "
                f"{cfg.fs_subjects_dir / fs_subject / 'bem'} (see `cerca-flux export-trans`)")
        rows.append({"subject": subject, "ok": not problems, "problems": problems})
    return rows


def format_preflight(name: str, rows: list[dict]) -> str:
    bad = [r for r in rows if not r["ok"]]
    lines = [f"{name}: {len(rows) - len(bad)} of {len(rows)} ready"]
    lines += [f"  {r['subject']}: {problem}" for r in bad for problem in r["problems"]]
    return "\n".join(lines)
