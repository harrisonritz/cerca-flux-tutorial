"""Emit a SLURM array script: one array task per recording."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .config import Config, dump_config
from .paths import discover_recordings

_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=cerca-flux
#SBATCH --array=0-{last}%{concurrent}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logs}/%x_%A_%a.out
#SBATCH --error={logs}/%x_%A_%a.err
{extra}
set -euo pipefail

# One array task per recording. The pipeline caches every stage in the BIDS
# derivatives tree, so a failed task can be resubmitted on its own.
export MPLBACKEND=Agg
export OMP_NUM_THREADS={cpus}
export MNE_USE_NUMBA=true

{python} -m cerca_flux.cli run \\
    --config {config} \\
    --index "${{SLURM_ARRAY_TASK_ID}}" \\
    {stages}{overwrite}

# After the array finishes, collect the group outputs with:
#   {python} -m cerca_flux.cli group --config {config}
"""


def write_array_script(cfg: Config, args, stages) -> Path:
    """Write ``<out>/cerca_flux_array.sh`` covering every selected recording."""
    recordings = discover_recordings(cfg)
    out = Path(args.out).expanduser().resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # Freeze the resolved configuration next to the script, so the submitted
    # jobs cannot drift from what was reviewed at submission time.
    frozen = out / "study.resolved.yaml"
    dump_config(cfg, frozen)

    extra = []
    if getattr(args, "partition", None):
        extra.append(f"#SBATCH --partition={args.partition}")
    if getattr(args, "account", None):
        extra.append(f"#SBATCH --account={args.account}")

    stage_names = [s.name for s in stages]
    stage_arg = "--stages " + " ".join(stage_names) if stage_names else ""

    script = out / "cerca_flux_array.sh"
    script.write_text(
        _TEMPLATE.format(
            last=max(len(recordings) - 1, 0),
            concurrent=min(len(recordings), 20),
            cpus=args.cpus,
            mem=args.mem,
            time=args.time,
            logs=shlex.quote(str(logs)),
            extra="\n".join(extra),
            python=shlex.quote(sys.executable),
            config=shlex.quote(str(frozen)),
            stages=stage_arg,
            overwrite=" \\\n    --overwrite" if getattr(args, "overwrite", False) else "",
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)

    manifest = out / "recordings.tsv"
    manifest.write_text(
        "index\trecording\tsubject\tsession\ttask\trun\n"
        + "".join(
            f"{i}\t{r.key}\t{r.subject}\t{r.session or ''}\t{r.task or ''}\t{r.run or ''}\n"
            for i, r in enumerate(recordings)
        ),
        encoding="utf-8",
    )
    return script
