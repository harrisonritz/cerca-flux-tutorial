"""BIDS discovery and derivative-path construction.

The pipeline writes everything under a single derivatives namespace
(``<bids_root>/derivatives/<derivatives_name>``) but preserves the split the
FLUX notebooks use internally::

    derivatives/cerca-flux/
        preprocessing/sub-01/ses-01/meg/sub-01_..._hfc_raw.fif
                                        sub-01_..._ann_raw.fif
                                        sub-01_..._ica_raw.fif
        analysis/sub-01/ses-01/meg/      sub-01_..._epo.fif
                                        sub-01_..._fwd-vol.fif
        figures/sub-01/...
        reports/sub-01_..._report.html
        logs/  group/
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path

import mne_bids

from .config import Config

#: Stem used when a recording has no session/task/run entity.
_ENTITY_ORDER = (("sub", "subject"), ("ses", "session"), ("task", "task"), ("run", "run"))


@dataclass(frozen=True)
class Recording:
    """One BIDS recording: the unit of work for the pipeline."""

    subject: str
    session: str | None = None
    task: str | None = None
    run: str | None = None

    @property
    def key(self) -> str:
        parts = []
        for prefix, attr in _ENTITY_ORDER:
            value = getattr(self, attr)
            if value is not None:
                parts.append(f"{prefix}-{value}")
        return "_".join(parts)

    @property
    def subject_dir_parts(self) -> tuple[str, ...]:
        parts = [f"sub-{self.subject}"]
        if self.session is not None:
            parts.append(f"ses-{self.session}")
        return tuple(parts)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.key


def discover_recordings(cfg: Config) -> list[Recording]:
    """List the recordings selected by ``cfg.study``.

    Uses ``mne_bids.find_matching_paths`` so that split files, sidecars and the
    ``derivatives/`` tree are handled by the BIDS layer rather than by globbing.
    """
    root = cfg.bids_root
    if not root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {root}")

    matches = mne_bids.find_matching_paths(
        root=root,
        subjects=cfg.study.subjects,
        sessions=cfg.study.sessions,
        tasks=cfg.study.tasks,
        runs=cfg.study.runs,
        datatypes=[cfg.study.datatype],
        suffixes=[cfg.study.datatype],
        extensions=[".fif", ".fif.gz", ".ds", ".con", ".sqd", ".bin", ".vhdr", ".edf"],
        check=False,
    )

    seen: dict[tuple, Recording] = {}
    for bids_path in matches:
        # A split recording (``_split-01_meg.fif``) is one recording; MNE follows
        # the FIFF links from the first part, so later parts are skipped.
        if bids_path.split not in (None, "01", 1):
            continue
        rec = Recording(
            subject=bids_path.subject,
            session=bids_path.session,
            task=bids_path.task,
            run=bids_path.run,
        )
        seen.setdefault((rec.subject, rec.session, rec.task, rec.run), rec)

    recordings = sorted(seen.values(), key=lambda r: (r.subject, r.session or "", r.task or "", r.run or ""))
    if not recordings:
        raise FileNotFoundError(
            f"no {cfg.study.datatype} recordings matched under {root}. "
            "Check study.subjects / study.tasks / study.datatype."
        )
    return recordings


def raw_bids_path(cfg: Config, rec: Recording) -> mne_bids.BIDSPath:
    """The BIDSPath of the *source* recording for one unit of work."""
    return mne_bids.BIDSPath(
        subject=rec.subject,
        session=rec.session,
        task=rec.task,
        run=rec.run,
        datatype=cfg.study.datatype,
        suffix=cfg.study.datatype,
        extension=".fif",
        root=cfg.bids_root,
        check=False,
    )


class SubjectPaths:
    """Derivative file locations for one recording."""

    def __init__(self, cfg: Config, rec: Recording):
        self.cfg = cfg
        self.rec = rec
        self.deriv_root = cfg.deriv_root

    # -- directories ------------------------------------------------------ #

    def _dir(self, kind: str) -> Path:
        path = self.deriv_root.joinpath(kind, *self.rec.subject_dir_parts, self.cfg.study.datatype)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def preprocessing_dir(self) -> Path:
        return self._dir("preprocessing")

    @property
    def analysis_dir(self) -> Path:
        return self._dir("analysis")

    @property
    def figures_dir(self) -> Path:
        path = self.deriv_root.joinpath("figures", *self.rec.subject_dir_parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        path = self.deriv_root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- files ------------------------------------------------------------ #

    def preprocessing(self, suffix: str, extension: str = ".fif") -> Path:
        return self.preprocessing_dir / f"{self.rec.key}_{suffix}{extension}"

    def analysis(self, suffix: str, extension: str = ".fif") -> Path:
        return self.analysis_dir / f"{self.rec.key}_{suffix}{extension}"

    def figure(self, name: str) -> Path:
        return self.figures_dir / f"{self.rec.key}_{name}.png"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / f"{self.rec.key}.log"

    @property
    def report_file(self) -> Path:
        path = self.deriv_root / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{self.rec.key}_report.html"

    # Named outputs, in pipeline order.
    @property
    def hfc_raw(self) -> Path:
        return self.preprocessing("hfc_raw")

    @property
    def ann_raw(self) -> Path:
        return self.preprocessing("ann_raw")

    @property
    def ann_csv(self) -> Path:
        return self.preprocessing("ann_raw", ".csv")

    @property
    def ica_solution(self) -> Path:
        return self.preprocessing("ica")

    @property
    def ica_raw(self) -> Path:
        return self.preprocessing("ica_raw")

    @property
    def epochs(self) -> Path:
        return self.analysis("epo")

    @property
    def evoked(self) -> Path:
        return self.analysis("ave")

    @property
    def decoding(self) -> Path:
        return self.analysis("decoding", ".npz")

    @property
    def bem(self) -> Path:
        return self.analysis("bem-sol")

    @property
    def trans(self) -> Path:
        return self.analysis("trans")

    @property
    def metrics(self) -> Path:
        return self.analysis("qc", ".json")

    def tfr(self, band: str) -> Path:
        return self.analysis(f"band-{band}_tfr", ".h5")

    def src(self, space: str) -> Path:
        return self.analysis(f"{space}-src")

    def fwd(self, space: str) -> Path:
        return self.analysis(f"{space}-fwd")

    def stc(self, name: str, space: str, morphed: bool = False) -> Path:
        tag = "fsaverage" if morphed else "native"
        return self.analysis_dir / f"{self.rec.key}_{name}_{space}_{tag}"


# --------------------------------------------------------------------------- #
# FreeSurfer resolution
# --------------------------------------------------------------------------- #


def freesurfer_subject(cfg: Config, rec: Recording) -> str:
    """The FreeSurfer subject directory name for a BIDS subject label."""
    return cfg.study.fs_subject_template.format(
        subject=rec.subject, session=rec.session or "", task=rec.task or ""
    )


def _expand(template: str, cfg: Config, rec: Recording, fs_subject: str) -> str:
    return template.format(
        bids_root=cfg.bids_root,
        deriv_root=cfg.deriv_root,
        fs_subjects_dir=cfg.fs_subjects_dir,
        fs_subject=fs_subject,
        subject=rec.subject,
        session=rec.session or "",
        task=rec.task or "",
        run=rec.run or "",
    )


def resolve_one(template: str | None, cfg: Config, rec: Recording, fs_subject: str,
                fallbacks: list[str], description: str) -> Path | None:
    """Resolve a path template (or a list of fallback globs) to a single file.

    Returns ``None`` when nothing matches, so callers can decide whether the
    file is required or can be recomputed.
    """
    patterns = [template] if template else list(fallbacks)
    for pattern in patterns:
        if pattern is None:
            continue
        expanded = _expand(str(pattern), cfg, rec, fs_subject)
        hits = sorted(glob.glob(expanded))
        if len(hits) == 1:
            return Path(hits[0])
        if len(hits) > 1:
            raise FileNotFoundError(
                f"{description} is ambiguous for {rec.key}: {len(hits)} files match "
                f"{expanded!r}: {hits[:5]}. Narrow it with the matching config template."
            )
        direct = Path(expanded)
        if direct.exists():
            return direct
    return None


def find_trans(cfg: Config, rec: Recording, fs_subject: str, paths: SubjectPaths) -> Path | None:
    """Locate the MRI/head transform produced by coregistration."""
    fallbacks = [
        str(paths.trans),
        "{fs_subjects_dir}/{fs_subject}/bem/*-trans.fif",
        "{fs_subjects_dir}/{fs_subject}/bem/*_trans.fif",
        "{deriv_root}/analysis/sub-{subject}/**/*_trans.fif",
        "{bids_root}/derivatives/**/sub-{subject}/**/*_trans.fif",
    ]
    return resolve_one(cfg.forward.trans, cfg, rec, fs_subject, fallbacks, "MRI/head transform")


def find_bem(cfg: Config, rec: Recording, fs_subject: str, paths: SubjectPaths) -> Path | None:
    """Locate a precomputed BEM solution, if the study already has one."""
    fallbacks = [
        str(paths.bem),
        "{fs_subjects_dir}/{fs_subject}/bem/*-bem-sol.fif",
        "{fs_subjects_dir}/{fs_subject}/bem/*_bem-sol.fif",
    ]
    return resolve_one(cfg.forward.bem, cfg, rec, fs_subject, fallbacks, "BEM solution")


def check_freesurfer(cfg: Config, fs_subject: str) -> Path:
    """Verify that a usable FreeSurfer reconstruction exists."""
    subjects_dir = cfg.fs_subjects_dir
    subject_dir = subjects_dir / fs_subject
    if not subject_dir.is_dir():
        available = sorted(p.name for p in subjects_dir.glob("*") if (p / "mri").is_dir())
        raise FileNotFoundError(
            f"FreeSurfer subject {fs_subject!r} not found in {subjects_dir}. "
            f"Reconstructions present: {available[:10] or 'none'}. "
            "Adjust study.fs_subjects_dir / study.fs_subject_template."
        )
    required = ["mri/T1.mgz", "surf/lh.white", "surf/rh.white"]
    missing = [r for r in required if not (subject_dir / r).exists()]
    if missing:
        raise FileNotFoundError(
            f"FreeSurfer reconstruction for {fs_subject!r} is incomplete; missing {missing}. "
            "Run `recon-all -all`, then `mne watershed_bem` and `mne make_scalp_surfaces`."
        )
    return subject_dir


def sanitize(name: str) -> str:
    """Make an arbitrary label safe for use inside a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "unnamed"
