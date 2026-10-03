"""Command-line entry point.

    cerca-flux run      --config study.yaml [--subjects 01 02] [--stages ica epochs]
    cerca-flux list     --config study.yaml
    cerca-flux group    --config study.yaml
    cerca-flux slurm    --config study.yaml --out jobs/
    cerca-flux compare  --reference oxford.yaml --cohort princeton.yaml --variant hfc2
    cerca-flux settings --config princeton.yaml [--changed]
    cerca-flux export-trans --fwd sub-01_fwd.fif --out T1s/bem/T1s-trans.fif
    cerca-flux check    --config princeton.yaml     # are the inputs in place?
    cerca-flux template > study.yaml

Any command that reads a configuration also accepts ``--variant NAME`` (outputs go
to their own ``derivatives/<name>_<variant>`` folder) and repeatable
``--set section.key=value`` overrides, so a preprocessing option can be tried
without editing the YAML::

    cerca-flux run --config princeton.yaml --preset motor --variant hfc3 --set hfc.order=3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigError, dump_config, load_config, parse_overrides
from .pipeline import PRESETS, STAGE_NAMES, resolve_stages, run_study, run_subject
from .paths import discover_recordings
from .utils import setup_logging

TEMPLATE = Path(__file__).parent / "templates" / "study.yaml"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", required=True, type=Path,
                        help="study YAML (see `cerca-flux template`)")
    parser.add_argument("--subjects", nargs="+", metavar="LABEL",
                        help="BIDS subject labels; default is every subject found")
    parser.add_argument("--sessions", nargs="+", metavar="LABEL")
    parser.add_argument("--tasks", nargs="+", metavar="LABEL")
    parser.add_argument("--runs", nargs="+", metavar="LABEL")
    _add_variant(parser)
    parser.add_argument("--verbose", "-v", action="store_true")


def _add_variant(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", metavar="NAME",
                        help="name this set of analysis choices; outputs go to "
                             "derivatives/<derivatives_name>_<NAME>")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        metavar="SECTION.KEY=VALUE",
                        help="override one configuration value (YAML syntax); repeatable, "
                             "e.g. --set hfc.order=3 --set ica.detect_ecg=false")


def _apply_selection(cfg, args) -> None:
    for attr in ("subjects", "sessions", "tasks", "runs"):
        value = getattr(args, attr, None)
        if value:
            setattr(cfg.study, attr, list(value))


def _read_config(path: Path, args):
    try:
        return load_config(path, overrides=parse_overrides(getattr(args, "overrides", None)),
                           variant=getattr(args, "variant", None))
    except ConfigError as exc:
        raise SystemExit(f"configuration error in {path}: {exc}") from None


def _load(args):
    cfg = _read_config(args.config, args)
    _apply_selection(cfg, args)
    if getattr(args, "overwrite", False):
        cfg.output.overwrite = True
    if getattr(args, "no_figures", False):
        cfg.output.figures = False
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cerca-flux",
        description="Batch OPM-MEG analysis following the Cerca OPM-FLUX pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline over the selected recordings")
    _add_common(run)
    run.add_argument("--stages", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
                     help=f"explicit stages, in any order: {', '.join(STAGE_NAMES)}")
    run.add_argument("--preset", choices=sorted(PRESETS), default=None,
                     help="named stage subset (default: every stage)")
    run.add_argument("--n-jobs", type=int, default=1,
                     help="recordings to process in parallel")
    run.add_argument("--overwrite", action="store_true",
                     help="recompute stages whose outputs already exist")
    run.add_argument("--no-figures", action="store_true")
    run.add_argument("--index", type=int, default=None,
                     help="process only the Nth recording (0-based); used by job arrays")
    run.add_argument("--dry-run", action="store_true",
                     help="list what would run, then exit")

    listing = sub.add_parser("list", help="list the recordings the config selects")
    _add_common(listing)

    group = sub.add_parser("group", help="rebuild the group outputs from existing derivatives")
    _add_common(group)

    slurm = sub.add_parser("slurm", help="write a SLURM array script for this study")
    _add_common(slurm)
    slurm.add_argument("--out", type=Path, default=Path("slurm"),
                       help="directory for the generated script and logs")
    slurm.add_argument("--preset", choices=sorted(PRESETS), default=None)
    slurm.add_argument("--stages", nargs="+", choices=STAGE_NAMES, metavar="STAGE")
    slurm.add_argument("--time", default="04:00:00", help="wall clock per task")
    slurm.add_argument("--mem", default="32G", help="memory per task")
    slurm.add_argument("--cpus", type=int, default=4, help="CPUs per task")
    slurm.add_argument("--partition", default=None)
    slurm.add_argument("--account", default=None)
    slurm.add_argument("--overwrite", action="store_true")

    compare = sub.add_parser(
        "compare",
        help="overlay a reference recording (thick black) on a cohort's motor responses",
    )
    compare.add_argument("--reference", required=True, type=Path,
                         help="study YAML of the reference, e.g. the Oxford participant")
    compare.add_argument("--cohort", required=True, type=Path,
                         help="study YAML of the cohort, e.g. every Princeton participant")
    _add_variant(compare)
    compare.add_argument("--out", type=Path, default=None,
                         help="output directory (default: outputs/motor_response/<variant>)")
    compare.add_argument("--no-small-multiples", action="store_true",
                         help="skip the one-panel-per-participant figures")
    compare.add_argument("--allow-mismatch", action="store_true",
                         help="draw the figure even if the two sites' shared preprocessing "
                              "settings differ or a result is stale (the figure says so)")
    compare.add_argument("--verbose", "-v", action="store_true")

    settings = sub.add_parser(
        "settings", help="show the preprocessing settings, flagged where they differ from Cerca")
    settings.add_argument("--config", "-c", required=True, type=Path)
    _add_variant(settings)
    settings.add_argument("--changed", action="store_true",
                          help="list only the settings that differ from the Cerca defaults")

    check = sub.add_parser(
        "check", help="check that every recording's inputs are in place (nothing is processed)")
    _add_common(check)

    export = sub.add_parser(
        "export-trans",
        help="write the MRI/head transform stored in a forward solution to its own file",
        description="Some datasets ship a forward solution but no -trans.fif. The pipeline "
                    "never reads a forward model it did not build, so the coregistration is "
                    "extracted once, here. Only the 4x4 transform is read.",
    )
    export.add_argument("--fwd", required=True, type=Path, help="forward solution (*-fwd.fif)")
    export.add_argument("--out", required=True, type=Path,
                        help="transform to write, e.g. <fs_subjects_dir>/<fs_subject>/bem/"
                             "<fs_subject>-trans.fif (where the pipeline looks for it)")
    export.add_argument("--overwrite", action="store_true")

    sub.add_parser("template", help="print an annotated starter configuration")

    args = parser.parse_args(argv)

    if args.command == "template":
        sys.stdout.write(TEMPLATE.read_text(encoding="utf-8"))
        return 0

    setup_logging(logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)

    if args.command == "export-trans":
        from .source import export_trans

        try:
            out = export_trans(args.fwd, args.out, overwrite=args.overwrite)
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from None
        print(f"wrote {out}\n(only the coregistration transform was read from {args.fwd})")
        return 0

    if args.command == "check":
        from .preflight import format_preflight, preflight

        cfg = _load(args)
        rows = preflight(cfg)
        print(format_preflight(str(args.config.name), rows))
        return 0 if all(r["ok"] for r in rows) else 1

    if args.command == "settings":
        from .provenance import fingerprints, format_settings

        cfg = _read_config(args.config, args)
        print(format_settings(cfg, changed_only=args.changed))
        print(f"\nshared-settings fingerprint: {fingerprints(cfg)['fingerprint_shared']}"
              f"   (results from two sites are comparable only if this matches)")
        return 0

    if args.command == "compare":
        from .compare import compare_motor

        try:
            outputs = compare_motor(
                _read_config(args.reference, args), _read_config(args.cohort, args),
                out_dir=args.out, small_multiples=not args.no_small_multiples,
                check_provenance=not args.allow_mismatch,
            )
        except (RuntimeError, FileNotFoundError) as exc:  # ProvenanceError is a RuntimeError
            raise SystemExit(f"compare: {exc}") from None
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    cfg = _load(args)

    if args.command == "list":
        for i, rec in enumerate(discover_recordings(cfg)):
            print(f"{i:4d}  {rec.key}")
        return 0

    if args.command == "group":
        from .group import build_group_outputs
        from .utils import read_json

        recordings = discover_recordings(cfg)
        rows = []
        for rec in recordings:
            from .paths import SubjectPaths

            metrics_file = SubjectPaths(cfg, rec).metrics
            metrics = read_json(metrics_file).get("metrics", {}) if metrics_file.exists() else {}
            rows.append({
                "recording": rec.key, "subject": rec.subject, "session": rec.session,
                "task": rec.task, "run": rec.run,
                "status": "ok" if metrics else "failed", "error": "",
                **metrics,
            })
        outputs = build_group_outputs(cfg, rows, recordings)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "slurm":
        from .slurm import write_array_script

        stages = resolve_stages(args.preset, args.stages)
        script = write_array_script(cfg, args, stages)
        print(f"wrote {script}")
        print(f"submit with:  sbatch {script}")
        return 0

    # --- run ------------------------------------------------------------- #
    try:
        stages = resolve_stages(args.preset, args.stages)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    recordings = discover_recordings(cfg)
    if args.index is not None:
        if not 0 <= args.index < len(recordings):
            raise SystemExit(
                f"--index {args.index} is out of range: {len(recordings)} recording(s) selected"
            )
        recordings = [recordings[args.index]]

    if args.dry_run:
        print(f"stages:   {', '.join(s.name for s in stages)}")
        print(f"n_jobs:   {args.n_jobs}")
        print(f"outputs:  {cfg.deriv_root}")
        print(f"recordings ({len(recordings)}):")
        for rec in recordings:
            print(f"  {rec.key}")
        return 0

    if len(recordings) == 1:
        # run_study records the resolved configuration; a single-recording run (one
        # SLURM array task) does not go through it, so record it per recording.
        # Separate files keep concurrent array tasks from overwriting each other.
        cfg.deriv_root.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, cfg.deriv_root / f"config_{recordings[0].key}.yaml")
        row = run_subject(cfg, recordings[0], stages, args.verbose)
        return 0 if row["status"] != "failed" else 1

    rows = run_study(cfg, stages, n_jobs=args.n_jobs, verbose=args.verbose,
                     recordings=recordings)
    return 0 if all(row["status"] != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
