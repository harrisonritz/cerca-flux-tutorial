"""Shared helpers: logging, figure handling, and OPM channel conventions."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import ChannelConfig, Config

LOGGER_NAME = "cerca_flux"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Configure the package logger; optionally tee to a per-subject file."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Replace any handler pointing at a different subject's log.
        for handler in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
            logger.removeHandler(handler)
            handler.close()
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


@contextmanager
def timed(message: str, logger: logging.Logger | None = None):
    logger = logger or get_logger()
    logger.info("%s ...", message)
    start = time.time()
    try:
        yield
    finally:
        logger.info("%s done in %.1f s", message, time.time() - start)


# --------------------------------------------------------------------------- #
# JSON / tabular IO
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
    return path


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def save_figure(fig, path: Path, cfg: Config) -> Path | None:
    """Save a Matplotlib figure and close it, honouring ``output.figures``."""
    import matplotlib.pyplot as plt

    if fig is None:
        return None
    if not cfg.output.figures:
        plt.close(fig)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=cfg.output.figure_dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    return path


def use_headless_backend() -> None:
    """Force a non-interactive Matplotlib backend for batch/cluster runs."""
    import matplotlib

    matplotlib.use("Agg", force=True)


# --------------------------------------------------------------------------- #
# OPM channel conventions
# --------------------------------------------------------------------------- #


def sensor_id(ch_name: str) -> str:
    """First token of a Cerca channel name: the physical sensor (``'F9 C8 Z'`` -> ``'F9'``)."""
    return ch_name.split()[0] if ch_name.split() else ch_name


def axis_of(ch_name: str, channels: ChannelConfig) -> str | None:
    """Measurement axis of a triaxial OPM channel, or ``None`` if it is not one."""
    tokens = ch_name.split()
    if not tokens:
        return None
    last = tokens[-1].upper()
    return last if last in {a.upper() for a in channels.axes} else None


def is_opm(ch_name: str, channels: ChannelConfig) -> bool:
    if any(ch_name.startswith(prefix) for prefix in channels.exclude_name_prefixes):
        return False
    return axis_of(ch_name, channels) is not None


def channels_by_axis(inst, channels: ChannelConfig, axis: str,
                     exclude_bads: bool = True) -> list[str]:
    """Names of the OPM channels measuring along one axis."""
    bads = set(inst.info["bads"]) if exclude_bads else set()
    types = dict(zip(inst.ch_names, inst.get_channel_types()))
    return [
        ch for ch in inst.ch_names
        if types.get(ch) == "mag"
        and is_opm(ch, channels)
        and axis_of(ch, channels) == axis.upper()
        and ch not in bads
    ]


def radial_channels(inst, channels: ChannelConfig, exclude_bads: bool = True) -> list[str]:
    """Radial (``Z``) OPM channels: the analogue of conventional magnetometers."""
    return channels_by_axis(inst, channels, channels.radial_axis, exclude_bads=exclude_bads)


def resolve_picks(inst, spec: str, channels: ChannelConfig,
                  exclude_bads: bool = True) -> list[str]:
    """Turn a config ``picks`` string into an explicit channel-name list.

    ``radial`` -> Z-axis OPMs, ``all``/``mag`` -> every magnetometer,
    ``X``/``Y``/``Z`` -> that axis.
    """
    spec = (spec or "radial").strip()
    if spec.lower() in {"radial", "z-axis"}:
        names = radial_channels(inst, channels, exclude_bads=exclude_bads)
    elif spec.upper() in {a.upper() for a in channels.axes}:
        names = channels_by_axis(inst, channels, spec, exclude_bads=exclude_bads)
    elif spec.lower() in {"all", "mag", "meg"}:
        bads = set(inst.info["bads"]) if exclude_bads else set()
        types = dict(zip(inst.ch_names, inst.get_channel_types()))
        names = [ch for ch in inst.ch_names if types.get(ch) == "mag" and ch not in bads]
    else:
        raise ValueError(f"picks={spec!r} is not recognised (use radial / all / X / Y / Z)")
    if not names:
        raise RuntimeError(
            f"picks={spec!r} selected no channels. Check channels.radial_axis and "
            "channels.exclude_name_prefixes against your channel naming."
        )
    return names


def expand_bad_sensors(ch_names: Sequence[str], sensors: Iterable[str],
                       channels: ChannelConfig) -> list[str]:
    """Expand sensor ids (``'B4'``) into every channel of that sensor (X, Y and Z)."""
    sensors = [s for s in sensors if s]
    if not sensors:
        return []
    if channels.bad_sensor_match == "substring":
        return [ch for ch in ch_names if any(s in ch for s in sensors)]
    wanted = set(sensors)
    return [ch for ch in ch_names if sensor_id(ch) in wanted or ch in wanted]


def manual_bads_for(subject: str, channels: ChannelConfig) -> list[str]:
    """Faulty sensors declared in the config for this subject, plus the ``'*'`` entry."""
    out: list[str] = []
    out.extend(channels.manual_bads.get("*", []))
    out.extend(channels.manual_bads.get(subject, []))
    out.extend(channels.manual_bads.get(f"sub-{subject}", []))
    return sorted(set(out))


def psd_db(values: np.ndarray) -> np.ndarray:
    """Convert PSD in T^2/Hz to dB relative to 1 fT^2/Hz, as the tutorials report it."""
    return 10 * np.log10(np.maximum(values, np.finfo(float).tiny) * 1e30)


def find_channel_by_sensor(inst, sensor: str, channels: ChannelConfig) -> str | None:
    """The radial channel belonging to a named sensor, e.g. ``'F9'`` -> ``'F9 C8 Z'``."""
    axis = channels.radial_axis.upper()
    for ch in inst.ch_names:
        if sensor_id(ch) == sensor and axis_of(ch, channels) == axis:
            return ch
    return None
