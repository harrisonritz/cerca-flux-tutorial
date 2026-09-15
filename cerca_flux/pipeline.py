"""Stage registry and orchestration.

The stage order below *is* the OPM-FLUX recommendation, and it is the order the
``FLUX_Oxford_Princeton_Comparison`` notebook follows::

    qc -> hfc -> annotate -> ica -> epochs -> erf -> tfr -> mvpa
       -> forward -> source -> morph -> report

Each stage reads its input from the previous stage's cached derivative, so any
suffix of the pipeline can be re-run on its own once the earlier outputs exist.
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from dataclasses import dataclass
from typing import Callable

import mne

from .config import Config, dump_config
from .context import SubjectContext
from .paths import Recording, SubjectPaths, discover_recordings
from .utils import get_logger, setup_logging, use_headless_backend


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    run: Callable[[SubjectContext], None]
    #: A failing required stage stops that subject, because everything after it
    #: depends on its output.  An optional stage only records the failure.
    required: bool = True
    #: Config section whose ``enabled`` flag switches this stage off.
    switch: str | None = None


# --------------------------------------------------------------------------- #
# Stage implementations: thin wrappers that pass objects through ctx.cache
# --------------------------------------------------------------------------- #


def _stage_qc(ctx: SubjectContext) -> None:
    from .preprocess import check_sensors, load_raw

    ctx.cache["raw"] = check_sensors(ctx, load_raw(ctx))


def _stage_hfc(ctx: SubjectContext) -> None:
    from .preprocess import apply_hfc

    ctx.cache["hfc"] = apply_hfc(ctx, ctx.cache.pop("raw", None))


def _stage_annotate(ctx: SubjectContext) -> None:
    from .preprocess import annotate_artefacts

    ctx.cache["annotated"] = annotate_artefacts(ctx, ctx.cache.pop("hfc", None))


def _stage_ica(ctx: SubjectContext) -> None:
    from .preprocess import run_ica

    ctx.cache["clean"] = run_ica(ctx, ctx.cache.pop("annotated", None))


def _stage_epochs(ctx: SubjectContext) -> None:
    from .preprocess import make_epochs

    ctx.cache["epochs"] = make_epochs(ctx, ctx.cache.pop("clean", None))


def _epochs(ctx: SubjectContext) -> mne.Epochs:
    """The epochs, from this process's cache or from disk."""
    if "epochs" not in ctx.cache:
        from .preprocess import make_epochs

        ctx.cache["epochs"] = make_epochs(ctx)
    return ctx.cache["epochs"]


def _stage_erf(ctx: SubjectContext) -> None:
    from .sensor import compute_erf

    compute_erf(ctx, _epochs(ctx))


def _stage_tfr(ctx: SubjectContext) -> None:
    from .sensor import compute_tfr

    compute_tfr(ctx, _epochs(ctx))


def _stage_mvpa(ctx: SubjectContext) -> None:
    from .sensor import run_decoding

    run_decoding(ctx, _epochs(ctx))


def _stage_forward(ctx: SubjectContext) -> None:
    from .source import build_forward

    ctx.cache["forwards"] = build_forward(ctx)


def _stage_source(ctx: SubjectContext) -> None:
    from .source import reconstruct_sources

    ctx.cache["stcs"] = reconstruct_sources(ctx, _epochs(ctx), ctx.cache.get("forwards"))


def _stage_morph(ctx: SubjectContext) -> None:
    from .source import morph_sources

    morph_sources(ctx, ctx.cache.pop("stcs", None))


def _stage_report(ctx: SubjectContext) -> None:
    from .report import build_report

    build_report(ctx)


STAGES: tuple[Stage, ...] = (
    Stage("qc", "Sensor quality check", _stage_qc, switch=None),
    Stage("hfc", "Homogeneous field correction", _stage_hfc, switch=None),
    Stage("annotate", "Artefact annotation", _stage_annotate, switch=None),
    Stage("ica", "ICA artefact removal", _stage_ica, switch=None),
    Stage("epochs", "Condition-specific epochs", _stage_epochs, switch="epochs"),
    Stage("erf", "Event-related fields", _stage_erf, required=False, switch="erf"),
    Stage("tfr", "Time-frequency power", _stage_tfr, required=False, switch="tfr"),
    Stage("mvpa", "Decoding", _stage_mvpa, required=False, switch="mvpa"),
    Stage("forward", "Forward model", _stage_forward, required=False, switch="forward"),
    Stage("source", "Beamformer source estimates", _stage_source, required=False, switch="source"),
    Stage("morph", "Morph to template", _stage_morph, required=False, switch="morph"),
    Stage("report", "Subject report", _stage_report, required=False, switch=None),
)

STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in STAGES)

#: Named subsets, so a common run does not need an explicit stage list.
PRESETS: dict[str, tuple[str, ...]] = {
    "full": STAGE_NAMES,
    "full_no_mvpa": tuple(n for n in STAGE_NAMES if n != "mvpa"),
    "preproc": ("qc", "hfc", "annotate", "ica", "epochs", "report"),
    "sensor": ("erf", "tfr", "mvpa", "report"),
    "source": ("forward", "source", "morph", "report"),
}


def resolve_stages(preset: str | None, names: list[str] | None) -> list[Stage]:
    """Turn a preset name and/or an explicit stage list into ordered stages."""
    if names:
        unknown = [n for n in names if n not in STAGE_NAMES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; choose from {list(STAGE_NAMES)}")
        wanted = set(names)
    elif preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        wanted = set(PRESETS[preset])
    else:
        wanted = set(STAGE_NAMES)
    # Always in pipeline order, whatever order the user listed them in.
    return [stage for stage in STAGES if stage.name in wanted]


def _stage_enabled(cfg: Config, stage: Stage) -> bool:
    if stage.switch is None:
        return True
    section = getattr(cfg, stage.switch, None)
    return bool(getattr(section, "enabled", True))


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def run_subject(cfg: Config, rec: Recording, stages: list[Stage],
                verbose: bool = False) -> dict:
    """Run the requested stages for one recording and return its summary row.

    Failures are contained: a subject that dies in ICA does not stop the study,
    and the row it returns says which stage failed and why.
    """
    use_headless_backend()
    mne.set_log_level("ERROR")

    paths = SubjectPaths(cfg, rec)
    logger = setup_logging(logging.DEBUG if verbose else logging.INFO, paths.log_file)
    ctx = SubjectContext(cfg=cfg, rec=rec, paths=paths, logger=logger)
    ctx.load_state()

    logger.info("=" * 72)
    logger.info("%s: running %s", rec.key, ", ".join(s.name for s in stages))

    status = "ok"
    error = ""
    for stage in stages:
        if not _stage_enabled(cfg, stage):
            ctx.stage_status[stage.name] = "disabled"
            logger.info("[%s] disabled in config", stage.name)
            continue
        try:
            logger.info("[%s] %s", stage.name, stage.title)
            stage.run(ctx)
            ctx.stage_status[stage.name] = "ok"
        except Exception as exc:  # one bad subject must not stop the study
            ctx.stage_status[stage.name] = f"failed: {exc}"
            logger.error("[%s] failed: %s", stage.name, exc)
            logger.debug("%s", traceback.format_exc())
            if stage.required:
                status, error = "failed", f"{stage.name}: {exc}"
                # Still try to leave a report behind, so the failure is visible.
                if any(s.name == "report" for s in stages) and stage.name != "report":
                    with contextlib.suppress(Exception):
                        _stage_report(ctx)
                break
            if status == "ok":
                status, error = "partial", f"{stage.name}: {exc}"
        finally:
            ctx.save_state()

    ctx.save_state()
    row = {
        "recording": rec.key,
        "subject": rec.subject,
        "session": rec.session,
        "task": rec.task,
        "run": rec.run,
        "status": status,
        "error": error,
    }
    row.update({k: v for k, v in ctx.metrics.items()})
    logger.info("%s: %s", rec.key, status)
    return row


def run_study(cfg: Config, stages: list[Stage], n_jobs: int = 1,
              verbose: bool = False, recordings: list[Recording] | None = None) -> list[dict]:
    """Run the pipeline over every selected recording, then build the group outputs."""
    logger = get_logger()
    recordings = recordings or discover_recordings(cfg)
    cfg.deriv_root.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, cfg.deriv_root / "config.yaml")

    logger.info(
        "Cerca OPM-FLUX: %d recording(s), stages %s, n_jobs=%d",
        len(recordings), [s.name for s in stages], n_jobs,
    )

    if n_jobs == 1:
        rows = [run_subject(cfg, rec, stages, verbose) for rec in recordings]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_subject)(cfg, rec, stages, verbose) for rec in recordings
        )

    ok = sum(row["status"] == "ok" for row in rows)
    partial = sum(row["status"] == "partial" for row in rows)
    failed = [row["recording"] for row in rows if row["status"] == "failed"]
    logger.info("finished: %d ok, %d partial, %d failed", ok, partial, len(failed))
    if failed:
        logger.warning("failed recordings: %s", failed)

    from .group import build_group_outputs

    build_group_outputs(cfg, rows, recordings)
    return rows
