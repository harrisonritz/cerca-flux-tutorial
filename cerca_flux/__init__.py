"""A batch implementation of the Cerca/QuSpin OPM-FLUX analysis pipeline.

The FLUX tutorial notebooks describe one recording at a time, with paths and
parameters edited in cells.  This package keeps their analysis choices and their
order but makes each step a function over a BIDS study, so a whole dataset can
be analysed reproducibly from one configuration file.

Typical use::

    cerca-flux template > study.yaml   # edit it
    cerca-flux run --config study.yaml --n-jobs 4

or from Python::

    from cerca_flux import load_config, resolve_stages, run_study

    cfg = load_config("study.yaml")
    run_study(cfg, resolve_stages("full", None), n_jobs=4)
"""

from .config import Config, ConfigError, dump_config, load_config
from .context import SubjectContext
from .paths import Recording, SubjectPaths, discover_recordings
from .pipeline import PRESETS, STAGE_NAMES, STAGES, resolve_stages, run_study, run_subject

__all__ = [
    "Config", "ConfigError", "load_config", "dump_config",
    "SubjectContext", "Recording", "SubjectPaths", "discover_recordings",
    "STAGES", "STAGE_NAMES", "PRESETS", "resolve_stages", "run_subject", "run_study",
]
__version__ = "0.1.0"
