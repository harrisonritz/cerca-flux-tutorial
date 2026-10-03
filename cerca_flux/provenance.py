"""Provenance: what a recording may read, and what settings its results came from.

A comparison between sites only means something if every site was processed the
same way, from the same kind of input.  This module makes both properties
checkable rather than a matter of convention:

* **Input audit.**  While a recording is processed, every file the Python process
  opens for reading is classified.  Inside ``provenance.strict`` anything other
  than the recording's raw BIDS data, the shared FreeSurfer reconstruction, the
  declared coregistration transform and the optional duration reference is
  *blocked*, so e.g. another pipeline's processed data or its forward model cannot
  be read even by accident.  The reads are written to ``*_inputs.json`` either way.
  The audit hooks Python's ``open`` event, so it sees reads made by MNE, MNE-BIDS
  or any other library alike.
* **Fingerprints.**  The settings shared by all sites are hashed and stamped into
  every result, so a comparison can verify that both sites' results were made with
  identical settings, and that cached results are not stale.
* **Defaults.**  Settings that differ from the Cerca defaults can be listed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import sysconfig
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config


class ProvenanceError(RuntimeError):
    """A recording tried to read something outside the allowed inputs."""


# --------------------------------------------------------------------------- #
# Which settings are shared, which are per-site
# --------------------------------------------------------------------------- #

#: Settings that decide a result.  ``motor`` is the analysis; the rest is the
#: preprocessing and forward modelling that precede it.
#: ``provenance`` is included so that results computed *without* input enforcement are not
#: mistaken for results computed with it (cached stages are reused by file existence).
RESULT_SECTIONS = ("channels", "qc", "hfc", "annotate", "ica", "epochs", "forward", "motor",
                   "provenance")

#: The steps a user is expected to tune, in pipeline order.
PREPROCESSING_SECTIONS = ("channels", "qc", "hfc", "annotate", "ica", "epochs")

#: Per-site settings: the data, the event labels, the sensors known to be faulty
#: and the span analysed legitimately differ between sites.  Everything else in
#: ``RESULT_SECTIONS`` must be identical.
SITE_SPECIFIC = {
    "channels": ("manual_bads", "bad_sensor_match"),
    "epochs": ("conditions",),
    "forward": ("trans", "bem"),
}
#: ``study`` keys that change what data a site contributes (the rest are paths,
#: names and discovery filters).
SITE_STUDY_KEYS = ("crop_start", "crop_duration")


def _flatten(value, prefix: str = "") -> dict:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}``; lists and scalars are leaves."""
    if isinstance(value, dict):
        out: dict = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: value}


def flat_settings(cfg: Config, sections=RESULT_SECTIONS, shared_only: bool = True) -> dict:
    """``{"hfc.order": 2, ...}`` for the given sections, optionally without per-site keys."""
    data = cfg.to_dict()
    out: dict = {}
    for section in sections:
        skip = SITE_SPECIFIC.get(section, ()) if shared_only else ()
        for key, value in _flatten(data[section], f"{section}.").items():
            if key.split(".")[1] not in skip:
                out[key] = value
    return out


def _digest(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def fingerprints(cfg: Config) -> dict[str, str]:
    """Hashes of the settings behind a result.

    ``fingerprint_shared`` covers what every site must share; two sites' results
    are comparable only if it is equal.  ``fingerprint_site`` covers what may
    differ per site (event labels, bad sensors, crop).  Paths are excluded from
    both, so the same study run on a laptop and on a cluster hashes the same.
    """
    site = {
        key: value
        for section, keys in SITE_SPECIFIC.items() if section != "forward"
        for key, value in _flatten(cfg.to_dict()[section], f"{section}.").items()
        if key.split(".")[1] in keys
    }
    site.update({f"study.{k}": getattr(cfg.study, k) for k in SITE_STUDY_KEYS})
    return {
        "fingerprint_shared": _digest(flat_settings(cfg)),
        "fingerprint_site": _digest(site),
    }


def settings_differences(a: Config, b: Config) -> list[tuple[str, object, object]]:
    """Shared settings on which two configurations disagree: ``(key, a, b)``."""
    fa, fb = flat_settings(a), flat_settings(b)
    return [(key, fa.get(key), fb.get(key)) for key in sorted(set(fa) | set(fb))
            if fa.get(key) != fb.get(key)]


def settings_vs_defaults(cfg: Config, sections=PREPROCESSING_SECTIONS,
                         include_site: bool = False) -> list[tuple[str, object, object, bool]]:
    """Every setting in ``sections`` as ``(key, value, Cerca default, differs)``.

    The Cerca defaults are the dataclass defaults, which reproduce the FLUX/Cerca
    tutorial notebooks.
    """
    mine = flat_settings(cfg, sections, shared_only=not include_site)
    reference = flat_settings(Config(), sections, shared_only=not include_site)
    return [(key, mine[key], reference.get(key), mine[key] != reference.get(key))
            for key in mine]


_STEP_TITLES = {
    "channels": "Bad sensors", "qc": "1. Sensor quality check", "hfc": "2. Homogeneous field correction",
    "annotate": "3. Artefact annotation", "ica": "4. ICA", "epochs": "5. Epoching",
}


def format_settings(cfg: Config, changed_only: bool = False) -> str:
    """The preprocessing settings, one step at a time, flagged where they differ from Cerca.

    ``*`` marks a setting that differs from the Cerca default (the dataclass default,
    which reproduces the FLUX/Cerca tutorials), and the default is shown beside it.
    Settings marked ``(site)`` legitimately differ between sites.
    """
    rows = settings_vs_defaults(cfg, include_site=True)
    site_keys = {f"{sec}.{k}" for sec, keys in SITE_SPECIFIC.items() for k in keys}
    lines = [f"Preprocessing settings   (* = differs from the Cerca default)"]
    for section in PREPROCESSING_SECTIONS:
        block = [r for r in rows if r[0].split(".")[0] == section and (r[3] or not changed_only)]
        if not block:
            continue
        lines += ["", _STEP_TITLES[section]]
        for key, value, default, differs in block:
            site = "(site)" if any(key == sk or key.startswith(sk + ".") for sk in site_keys) else ""
            note = f"   Cerca default: {default!r}" if differs else ""
            lines.append(f"  {'*' if differs else ' '} {key:<34} {value!r:<22}{note}  {site}".rstrip())
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Input audit
# --------------------------------------------------------------------------- #

#: Files a processing pipeline writes (as opposed to FreeSurfer's own formats).
_PRODUCT_SUFFIXES = (".fif", ".fif.gz", ".h5", ".hdf5", ".npz", ".stc", ".pkl", ".pickle")
_THREAD = threading.local()
_INSTALLED = False


def _is_read(mode, flags) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return False
    if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
                                           | os.O_TRUNC | os.O_APPEND):
        return False
    return True


def _hook(event, args):
    if event != "open":
        return
    audit = getattr(_THREAD, "audit", None)
    if audit is None or audit.busy:
        return
    path, mode, flags = (list(args) + [None, None])[:3]
    if isinstance(path, (bytes, bytearray)):
        path = os.fsdecode(path)
    if not isinstance(path, (str, os.PathLike)) or not _is_read(mode, flags):
        return
    audit.observe(os.fspath(path))


def _real(path) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


@dataclass
class InputAudit:
    """Classifies, records and (in strict mode) blocks the files a recording reads."""

    cfg: Config
    recording_key: str
    strict: bool
    allowed: set[str] = field(default_factory=set)
    allowed_prefixes: list[str] = field(default_factory=list)
    reads: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    violations: list[str] = field(default_factory=list)
    busy: bool = False

    def __post_init__(self) -> None:
        self.own = _real(self.cfg.deriv_root)
        self.bids = _real(self.cfg.bids_root)
        self.fs = _real(self.cfg.fs_subjects_dir)
        prefixes = {_real(sys.prefix), _real(sys.base_prefix), _real(sys.exec_prefix)}
        prefixes |= {_real(p) for p in sysconfig.get_paths().values() if p}
        try:
            import mne
            prefixes.add(_real(Path(mne.__file__).parent))
        except ImportError:  # pragma: no cover
            pass
        self.system = sorted(prefixes)
        reference = self.cfg.reference_recording
        if reference is not None:
            # The duration reference is a raw recording; a split one has several parts.
            stem = _real(reference)
            self.allowed_prefixes.append(stem.split("_split-")[0].removesuffix("_meg.fif"))

    # -- declaring and observing ---------------------------------------- #

    def allow(self, path) -> None:
        self.allowed.add(_real(path))

    def observe(self, path: str) -> None:
        self.busy = True
        try:
            real = _real(path)
            category, problem = self.classify(real, os.path.basename(path))
            self.reads[category].add(real)
            if problem:
                message = f"{problem}: {real}"
                self.violations.append(message)
                if self.strict:
                    raise ProvenanceError(
                        f"{self.recording_key}: blocked a read outside the allowed inputs "
                        f"(provenance.strict) - {message}"
                    )
        finally:
            self.busy = False

    # -- the rules -------------------------------------------------------- #

    def classify(self, real: str, name: str) -> tuple[str, str | None]:
        lowered = name.lower()
        is_product = lowered.endswith(_PRODUCT_SUFFIXES)
        if _under(real, self.own):
            return "own", None
        if real in self.allowed:
            return "declared", None
        if is_product and any(real.startswith(p) for p in self.allowed_prefixes):
            return "reference", None
        if _under(real, self.fs):  # before bids: a FreeSurfer tree may sit inside derivatives/
            if is_product:
                return "freesurfer", ("a processed product (BEM solution, forward, source space, "
                                      "...) inside the FreeSurfer tree; only the reconstruction "
                                      "itself and the declared coregistration may be read")
            return "freesurfer", None
        if _under(real, self.bids):
            relative = os.path.relpath(real, self.bids).split(os.sep)
            if relative[0] == "derivatives":
                return "other_derivatives", ("another pipeline's derivatives (processed data, "
                                             "epochs, forward models, ...)")
            if is_product and not self._is_this_raw(name):
                return "bids", "a MEG file that is not this recording's raw data"
            return ("raw_bids" if is_product else "sidecar"), None
        if any(_under(real, root) for root in self.system):
            return "system", None
        if is_product:
            return "outside", "a processed product outside the declared data trees"
        return "other", None

    def _is_this_raw(self, name: str) -> bool:
        return (name.startswith(self.recording_key + "_") and name.endswith("_meg.fif")
                and "_proc-" not in name and "_desc-" not in name)

    # -- reporting -------------------------------------------------------- #

    def manifest(self) -> dict:
        """The external files read, by category (FreeSurfer files are only counted)."""
        listed = ("raw_bids", "sidecar", "declared", "reference", "bids", "other_derivatives",
                  "outside")
        return {
            "recording": self.recording_key,
            "strict": self.strict,
            "violations": list(self.violations),
            "files": {cat: sorted(self.reads[cat]) for cat in listed if self.reads.get(cat)},
            "freesurfer_files_read": len(self.reads.get("freesurfer", ())),
            "own_files_read": len(self.reads.get("own", ())),
        }


@contextmanager
def audit_inputs(cfg: Config, recording_key: str):
    """Audit (and, with ``provenance.strict``, police) the reads inside the block."""
    global _INSTALLED
    if not _INSTALLED:
        sys.addaudithook(_hook)  # audit hooks cannot be removed; ours is inert when unset
        _INSTALLED = True
    audit = InputAudit(cfg, recording_key, strict=cfg.provenance.strict)
    previous = getattr(_THREAD, "audit", None)
    _THREAD.audit = audit
    try:
        yield audit
    finally:
        _THREAD.audit = previous
