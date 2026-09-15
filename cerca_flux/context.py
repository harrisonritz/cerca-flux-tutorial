"""Per-recording state carried between pipeline stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .paths import Recording, SubjectPaths, freesurfer_subject
from .utils import read_json, write_json


@dataclass
class SubjectContext:
    """Everything a stage needs, plus the state that lets stages resume.

    ``cache`` holds large in-memory objects for the current process only.
    ``state`` holds the small facts a later stage needs but cannot recompute
    (the chosen EOG channel, the event-label map, the excluded ICA components);
    it is persisted so that a stage can be re-run days later without repeating
    the ones before it.  ``metrics`` accumulates the quality numbers that end up
    in the per-subject JSON and the group table.
    """

    cfg: Config
    rec: Recording
    paths: SubjectPaths
    logger: logging.Logger
    cache: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: list[tuple[str, str, Path]] = field(default_factory=list)
    stage_status: dict[str, str] = field(default_factory=dict)

    # -- persisted state -------------------------------------------------- #

    @property
    def state_file(self) -> Path:
        return self.paths.preprocessing("state", ".json")

    def load_state(self) -> None:
        if self.state_file.exists():
            try:
                stored = read_json(self.state_file)
            except (ValueError, OSError) as exc:
                self.logger.warning("ignoring unreadable state file %s (%s)", self.state_file, exc)
                return
            self.state.update(stored.get("state", {}))
            self.metrics.update(stored.get("metrics", {}))
            # Stage outcomes accumulate across invocations, so a run of one
            # stage does not erase what earlier runs recorded about the others.
            for name, status in stored.get("stages", {}).items():
                self.stage_status.setdefault(name, status)

    def save_state(self) -> None:
        write_json(
            self.state_file,
            {"state": self.state, "metrics": self.metrics, "stages": self.stage_status},
        )

    # -- helpers ---------------------------------------------------------- #

    @property
    def fs_subject(self) -> str:
        return freesurfer_subject(self.cfg, self.rec)

    def needs(self, *outputs: Path) -> bool:
        """True when a stage must run: forced, or an output is missing."""
        if self.cfg.output.overwrite:
            return True
        return not all(Path(p).exists() for p in outputs)

    def record(self, **metrics: Any) -> None:
        self.metrics.update(metrics)

    def add_figure(self, section: str, title: str, path: Path | None) -> None:
        if path is not None:
            self.figures.append((section, title, Path(path)))

    def figure_path(self, name: str) -> Path:
        return self.paths.figure(name)
