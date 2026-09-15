"""Typed configuration for the Cerca OPM-FLUX batch pipeline.

Every analysis constant that the FLUX tutorial notebooks hard-code in a cell is
exposed here, so that a study is described by one YAML file rather than by
edited notebooks.  Defaults reproduce the values used in the Cerca/QuSpin
tutorial notebooks; where the ``FLUX_Oxford_Princeton_Comparison`` notebook
deliberately diverged (it matched two sites on a response-locked contrast) the
difference is noted in the field comment.
"""

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file cannot be interpreted."""


# --------------------------------------------------------------------------- #
# Study-level description
# --------------------------------------------------------------------------- #


@dataclass
class StudyConfig:
    """Where the data live and which recordings to process."""

    bids_root: str = ""
    #: Name of the pipeline's folder under ``<bids_root>/derivatives``.
    derivatives_name: str = "cerca-flux"
    #: ``None`` discovers every subject in ``participants.tsv`` / the BIDS tree.
    subjects: list[str] | None = None
    sessions: list[str] | None = None
    tasks: list[str] | None = None
    runs: list[str] | None = None
    #: Datatype folder holding the recordings (``meg`` for OPM data).
    datatype: str = "meg"
    #: FreeSurfer ``SUBJECTS_DIR``.  Relative paths resolve against ``bids_root``.
    fs_subjects_dir: str = "derivatives/freesurfer"
    #: How a BIDS subject label maps onto a FreeSurfer subject directory.
    fs_subject_template: str = "sub-{subject}"
    #: Mains frequency; written into ``raw.info`` when the sidecar omits it.
    line_freq: float | None = None


# --------------------------------------------------------------------------- #
# Channel conventions
# --------------------------------------------------------------------------- #


@dataclass
class ChannelConfig:
    """Cerca/QuSpin triaxial naming conventions.

    Channels are named ``"<sensor> <slot> <axis>"`` (e.g. ``"F9 C8 Z"``), so the
    sensor identity is the first whitespace-separated token and the measurement
    axis is the last one.  ``Z`` is the radial axis, the OPM analogue of a
    conventional SQUID magnetometer.
    """

    radial_axis: str = "Z"
    axes: list[str] = field(default_factory=lambda: ["X", "Y", "Z"])
    #: Channels whose name starts with any of these are never treated as OPMs.
    exclude_name_prefixes: list[str] = field(default_factory=lambda: ["BNC"])
    #: Sensors known to be faulty, per subject label, e.g. ``{"01": ["B4", "H6"]}``.
    #: The key ``"*"`` applies to every subject.
    manual_bads: dict[str, list[str]] = field(default_factory=dict)
    #: ``token`` matches the sensor id exactly (safer); ``substring`` reproduces
    #: the tutorial's ``any(tag in ch_name)`` behaviour.
    bad_sensor_match: str = "token"


# --------------------------------------------------------------------------- #
# Stage 1 - sensor quality check  (A First Look / A2 Sensor Quality checking)
# --------------------------------------------------------------------------- #


@dataclass
class QCConfig:
    enabled: bool = True
    psd_fmin: float = 1.0
    psd_fmax: float = 120.0
    #: Welch window length in seconds; 2 s gives the tutorial's 0.5 Hz resolution.
    psd_n_fft_seconds: float = 2.0
    #: Segment of the recording used for the quality PSD.
    psd_tmin: float = 30.0
    psd_span: float = 120.0
    #: ``robust_mad`` flags channels at median + k.MAD *within each axis*
    #: (the comparison notebook's data-adaptive rule); ``fixed_db`` reproduces
    #: the tutorial's absolute 39.1 dB cut; ``none`` keeps only manual bads.
    method: str = "robust_mad"
    mad_threshold: float = 6.0
    fixed_db_threshold: float = 39.1
    #: Also flag abnormally *quiet* sensors. HFC is a projection across the
    #: array, so flat/dead sensors must be excluded just as noisy ones are.
    flag_flat: bool = True
    #: A channel is flat when its mean PSD falls this far below the axis median,
    #: in the same robust MAD units as ``mad_threshold``.
    flat_mad_threshold: float = 6.0
    #: 10 s window used for the channel-centred "first look" RMS metric.
    first_look_tmin: float = 40.0
    first_look_span: float = 10.0


# --------------------------------------------------------------------------- #
# Stage 2 - homogeneous field correction  (ArtefactSuppressionHFC)
# --------------------------------------------------------------------------- #


@dataclass
class HFCConfig:
    enabled: bool = True
    #: Spherical-harmonic order of the homogeneous-field model.
    order: int = 2
    #: Resample before HFC to bound memory.  ``None`` keeps the BIDS rate.
    #: Must stay above 2x the muscle band (>= 300 Hz for 110-130 Hz).
    resample_sfreq: float | None = None
    #: ``all`` keeps X/Y/Z through HFC (the triaxial array constrains the
    #: homogeneous model better); ``radial`` drops to Z immediately afterwards.
    keep_axes: str = "all"


# --------------------------------------------------------------------------- #
# Stage 3 - artefact annotation  (ArtefactAnnotation)
# --------------------------------------------------------------------------- #


@dataclass
class EOGConfig:
    enabled: bool = True
    #: Use a recorded EOG channel when one exists, otherwise build the frontal
    #: OPM surrogate below.
    prefer_native: bool = True
    #: Bipolar frontal OPM surrogate: radial(left) - radial(right).
    surrogate_left: str = "F9"
    surrogate_right: str = "F10"
    surrogate_name: str = "OCC"
    l_freq: float = 1.0
    h_freq: float = 10.0
    #: Fixed detection threshold in tesla (tutorial: 0.3e-11 T = 3000 fT).
    #: ``None`` lets MNE pick one adaptively, which travels better across subjects.
    threshold: float | None = None
    #: Annotated window centred on each blink, in seconds.
    window: float = 0.5
    description: str = "BAD_blink"


@dataclass
class MuscleConfig:
    enabled: bool = True
    threshold: float = 3.0
    #: The comparison notebook narrowed this to (110, 130).
    filter_freq: list[float] = field(default_factory=lambda: [110.0, 140.0])
    min_length_good: float = 0.1


@dataclass
class AnnotateConfig:
    enabled: bool = True
    eog: EOGConfig = field(default_factory=EOGConfig)
    muscle: MuscleConfig = field(default_factory=MuscleConfig)


# --------------------------------------------------------------------------- #
# Stage 4 - ICA  (ICA)
# --------------------------------------------------------------------------- #


@dataclass
class ICAConfig:
    enabled: bool = True
    method: str = "fastica"
    #: ``int`` = PCA dimensionality (tutorial: 30); ``float`` < 1 = variance kept.
    #: Clamped to the data rank, which HFC has already reduced.
    n_components: float = 30
    random_state: int = 96
    max_iter: int = 500
    #: ICA is estimated on a band-passed, downsampled copy and applied to the
    #: full-rate data.  The high-pass matters most for decomposition quality.
    fit_l_freq: float = 3.0
    fit_h_freq: float = 30.0
    fit_sfreq: float = 250.0
    detect_eog: bool = True
    eog_threshold: float = 3.0
    detect_ecg: bool = True
    #: ``"auto"`` or a float z-score.
    ecg_threshold: str = "auto"
    #: Cap on removed components; the tutorial reports 2-5 per participant.
    max_exclude: int = 5


# --------------------------------------------------------------------------- #
# Stage 5 - epoching  (ConditionSpecificTrials)
# --------------------------------------------------------------------------- #


@dataclass
class EpochsConfig:
    enabled: bool = True
    #: Continuous filter applied once before epoching, to keep filter edge
    #: effects out of the epochs.  ``continuous_h_freq: null`` preserves the
    #: gamma band the FLUX TFR analysis needs; the comparison notebook used 45 Hz
    #: because it only looked at response-locked beta.
    continuous_l_freq: float | None = 0.1
    continuous_h_freq: float | None = None
    #: Condition name -> BIDS ``trial_type`` labels pooled into it.
    conditions: dict[str, list[str]] = field(default_factory=dict)
    tmin: float = -0.75
    tmax: float = 2.0
    #: ``None`` defers baseline correction to each analysis.
    baseline: list[float] | None = None
    #: 0 = constant, 1 = linear, ``None`` = no per-epoch detrend.
    detrend: int | None = 1
    #: Peak-to-peak rejection in tesla.  Tutorial 1e-11 (10 pT) for cue-locked
    #: epochs; the comparison notebook needed 50e-12 for button responses.
    reject: dict[str, float] = field(default_factory=lambda: {"mag": 1e-11})
    reject_by_annotation: bool = True
    #: Channels retained in the saved epochs (``all`` keeps X/Y/Z for later use).
    picks: str = "all"


# --------------------------------------------------------------------------- #
# Stage 6 - event-related fields  (EventRelatedFields)
# --------------------------------------------------------------------------- #


@dataclass
class ERFConfig:
    enabled: bool = True
    #: ``radial`` = Z-axis channels only, the OPM analogue of magnetometers.
    picks: str = "radial"
    l_freq: float | None = None
    h_freq: float | None = 30.0
    crop: list[float] | None = field(default_factory=lambda: [-0.1, 0.4])
    #: Linear detrend of the averaged, cropped data, applied before the baseline.
    detrend: bool = True
    baseline: list[float] | None = field(default_factory=lambda: [-0.1, 0.0])
    topomap_times: list[float] = field(default_factory=lambda: [0.16])


# --------------------------------------------------------------------------- #
# Stage 7 - time-frequency power  (TimeFrequencyPower)
# --------------------------------------------------------------------------- #


@dataclass
class TFRBand:
    """One multitaper band.

    ``n_cycles = freqs / n_cycles_divisor`` fixes the sliding window at
    dT = ``n_cycles_divisor``/2 seconds for every frequency.  ``time_bandwidth``
    then sets both the taper count (N = TBW - 1) and the spectral smoothing
    (half-bandwidth W = TBW / (2.dT)).
    """

    name: str = "slow"
    fmin: float = 3.0
    fmax: float = 30.0
    fstep: float = 1.0
    n_cycles_divisor: float = 2.0
    time_bandwidth: float = 2.0
    decim: int = 1
    #: Baseline chosen not to overlap t > 0 given the dT-long window.
    baseline: list[float] | None = field(default_factory=lambda: [-0.5, -0.25])
    baseline_mode: str = "percent"
    #: Optional band-limited topomap, ``[tmin, tmax, fmin, fmax]``.
    topomap: list[float] | None = None


@dataclass
class TFRContrast:
    name: str = ""
    band: str = "slow"
    a: str = ""
    b: str = ""
    #: ``normalised_difference`` = (a-b)/(a+b); ``difference`` = a-b;
    #: ``relative_change`` = (a-b)/b.
    kind: str = "normalised_difference"


@dataclass
class TFRConfig:
    enabled: bool = True
    picks: str = "radial"
    bands: list[TFRBand] = field(
        default_factory=lambda: [
            TFRBand(),
            TFRBand(
                name="fast",
                fmin=31.0,
                fmax=100.0,
                fstep=2.0,
                n_cycles_divisor=4.0,
                time_bandwidth=4.0,
                baseline=None,
            ),
        ]
    )
    contrasts: list[TFRContrast] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stage 8 - MVPA  (Classification)
# --------------------------------------------------------------------------- #


@dataclass
class MVPAConfig:
    enabled: bool = True
    #: Exactly two condition names from ``epochs.conditions``.
    conditions: list[str] = field(default_factory=list)
    picks: str = "radial"
    h_freq: float | None = 45.0
    sfreq: float | None = 125.0
    crop: list[float] | None = field(default_factory=lambda: [-0.1, 1.5])
    #: Sliding window length (s) and step (samples) of the decoder.
    window_s: float = 0.1
    step_samples: int = 5
    cv: int = 10
    scoring: str = "roc_auc"
    #: ``linear_svm`` (tutorial) or ``rbf_svm`` (comparison notebook).
    estimator: str = "linear_svm"
    #: Subsample to the smaller class so AUC is not driven by prevalence.
    balance_classes: bool = True
    random_state: int = 99
    n_jobs: int = 1


# --------------------------------------------------------------------------- #
# Stage 9 - forward model  (ForwardModel)
# --------------------------------------------------------------------------- #


@dataclass
class VolumeSourceConfig:
    enabled: bool = True
    #: Grid spacing in mm.
    pos: float = 5.0
    mindist: float = 5.0
    add_interpolator: bool = True


@dataclass
class SurfaceSourceConfig:
    enabled: bool = True
    spacing: str = "oct6"
    surface: str = "white"
    mindist: float = 5.0
    add_dist: bool = False


@dataclass
class ForwardConfig:
    enabled: bool = True
    #: Template for the MRI/head transform.  ``{fs_subjects_dir}``,
    #: ``{fs_subject}``, ``{subject}``, ``{session}``, ``{bids_root}`` and
    #: ``{deriv_root}`` are substituted; globs are allowed.  ``None`` searches
    #: the FreeSurfer ``bem/`` folder for ``*-trans.fif``.
    trans: str | None = None
    #: Template for a precomputed BEM solution.  ``None`` builds one from the
    #: FreeSurfer surfaces and caches it in derivatives.
    bem: str | None = None
    bem_ico: int = 4
    #: Single-shell model; adequate for MEG, unlike EEG.
    bem_conductivity: list[float] = field(default_factory=lambda: [0.3])
    volume: VolumeSourceConfig = field(default_factory=VolumeSourceConfig)
    surface: SurfaceSourceConfig = field(default_factory=SurfaceSourceConfig)


# --------------------------------------------------------------------------- #
# Stage 10 - source reconstruction  (DICSbeamforming + comparison LCMV)
# --------------------------------------------------------------------------- #


@dataclass
class LCMVConfig:
    enabled: bool = True
    #: One common covariance window spanning both baseline and active intervals.
    cov_tmin: float = -0.8
    cov_tmax: float = 0.8
    cov_method: str = "shrunk"
    #: Per-voxel dB power change is computed between these two windows.
    baseline_window: list[float] = field(default_factory=lambda: [-0.8, -0.5])
    active_window: list[float] = field(default_factory=lambda: [-0.2, 0.3])
    pick_ori: str = "max-power"
    weight_norm: str = "unit-noise-gain-invariant"
    #: Conditions to reconstruct; empty = every condition, pooled.
    conditions: list[str] = field(default_factory=list)


@dataclass
class DICSContrast:
    name: str = ""
    a: str = ""
    b: str = ""
    #: ``relative_change`` = (a-b)/b (tutorial); ``normalised_difference`` = (a-b)/(a+b).
    kind: str = "relative_change"


@dataclass
class DICSBand:
    name: str = "alpha"
    fmin: float = 8.0
    fmax: float = 12.0
    #: Multitaper bandwidth: 3 Hz on a 0.5 s window gives 1 taper (alpha),
    #: 8 Hz gives 3 tapers (gamma).
    bandwidth: float = 3.0
    adaptive: bool = False
    low_bias: bool = True
    #: Named time windows, ``{"pre": [-0.7, -0.2], "post": [0.1, 0.6]}``.
    windows: dict[str, list[float]] = field(default_factory=dict)
    #: Conditions pooled into each window's CSD; empty = all conditions.
    conditions: list[str] = field(default_factory=list)
    contrasts: list[DICSContrast] = field(default_factory=list)


@dataclass
class DICSConfig:
    enabled: bool = True
    bands: list[DICSBand] = field(default_factory=list)


@dataclass
class SourceConfig:
    enabled: bool = True
    picks: str = "radial"
    #: Diagonal loading as a fraction of sensor power.
    reg: float = 0.05
    #: ``info`` | ``relative`` (tol 1e-5, as in DICSbeamforming) | ``none``.
    rank: str = "info"
    reduce_rank: bool = True
    real_filter: bool = True
    #: ``None`` disables depth weighting; the tutorial's ``depth=0`` is equivalent.
    depth: float | None = None
    #: Which source spaces to reconstruct on.
    spaces: list[str] = field(default_factory=lambda: ["volume", "surface"])
    lcmv: LCMVConfig = field(default_factory=LCMVConfig)
    dics: DICSConfig = field(default_factory=DICSConfig)


# --------------------------------------------------------------------------- #
# Stage 11 - morphing to a template
# --------------------------------------------------------------------------- #


@dataclass
class MorphConfig:
    enabled: bool = True
    subject_to: str = "fsaverage"
    #: Isotropic resolution (mm) of the morphed volume grid.
    volume_zooms: float = 5.0
    #: ``ico`` subdivision of the template surface (5 ~= 10242 vertices/hemi).
    surface_spacing: int = 5
    #: Download fsaverage if it is missing from ``fs_subjects_dir``.
    fetch_fsaverage: bool = True


# --------------------------------------------------------------------------- #
# Output / execution
# --------------------------------------------------------------------------- #


@dataclass
class OutputConfig:
    #: Write a per-subject MNE ``Report`` HTML summarising every stage.
    report: bool = True
    #: Save diagnostic PNGs alongside the derivatives.
    figures: bool = True
    figure_dpi: int = 150
    #: Recompute a stage even when its output already exists.
    overwrite: bool = False


@dataclass
class GroupConfig:
    enabled: bool = True
    #: Grand-average evoked/TFR/decoding across subjects.
    grand_average: bool = True
    #: Average the fsaverage-morphed source estimates.
    average_sources: bool = True


@dataclass
class Config:
    study: StudyConfig = field(default_factory=StudyConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    hfc: HFCConfig = field(default_factory=HFCConfig)
    annotate: AnnotateConfig = field(default_factory=AnnotateConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)
    epochs: EpochsConfig = field(default_factory=EpochsConfig)
    erf: ERFConfig = field(default_factory=ERFConfig)
    tfr: TFRConfig = field(default_factory=TFRConfig)
    mvpa: MVPAConfig = field(default_factory=MVPAConfig)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    morph: MorphConfig = field(default_factory=MorphConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    group: GroupConfig = field(default_factory=GroupConfig)

    # -- validation ------------------------------------------------------- #

    def validate(self) -> None:
        if not self.study.bids_root:
            raise ConfigError("study.bids_root is required")
        if self.epochs.enabled and not self.epochs.conditions:
            raise ConfigError(
                "epochs.conditions is required: map each condition name onto the "
                "BIDS trial_type label(s) pooled into it, e.g. "
                "{cue_left: [cue_Left], cue_right: [cue_Right]}"
            )
        if self.epochs.tmin >= self.epochs.tmax:
            raise ConfigError("epochs.tmin must be smaller than epochs.tmax")
        if self.qc.method not in {"robust_mad", "fixed_db", "none"}:
            raise ConfigError(f"qc.method {self.qc.method!r} is not recognised")
        if self.channels.bad_sensor_match not in {"token", "substring"}:
            raise ConfigError("channels.bad_sensor_match must be 'token' or 'substring'")
        if self.hfc.keep_axes not in {"all", "radial"}:
            raise ConfigError("hfc.keep_axes must be 'all' or 'radial'")
        if self.source.rank not in {"info", "relative", "none"}:
            raise ConfigError("source.rank must be 'info', 'relative' or 'none'")
        for space in self.source.spaces:
            if space not in {"volume", "surface"}:
                raise ConfigError(f"source.spaces entry {space!r} is not recognised")
        if self.mvpa.enabled and len(self.mvpa.conditions) != 2:
            raise ConfigError("mvpa.conditions must name exactly two conditions")
        known = set(self.epochs.conditions)
        for name in self.mvpa.conditions:
            if name not in known:
                raise ConfigError(f"mvpa condition {name!r} is not in epochs.conditions")
        band_names = {b.name for b in self.tfr.bands}
        for contrast in self.tfr.contrasts:
            if contrast.band not in band_names:
                raise ConfigError(
                    f"tfr contrast {contrast.name!r} refers to unknown band {contrast.band!r}"
                )
            for side in (contrast.a, contrast.b):
                if side not in known:
                    raise ConfigError(
                        f"tfr contrast {contrast.name!r} refers to unknown condition {side!r}"
                    )
        for band in self.source.dics.bands:
            for contrast in band.contrasts:
                for side in (contrast.a, contrast.b):
                    if side not in band.windows:
                        raise ConfigError(
                            f"DICS contrast {contrast.name!r} refers to window "
                            f"{side!r}, which band {band.name!r} does not define"
                        )

    # -- convenience ------------------------------------------------------ #

    @property
    def bids_root(self) -> Path:
        return Path(self.study.bids_root).expanduser().resolve()

    @property
    def deriv_root(self) -> Path:
        return self.bids_root / "derivatives" / self.study.derivatives_name

    @property
    def fs_subjects_dir(self) -> Path:
        path = Path(self.study.fs_subjects_dir).expanduser()
        return path if path.is_absolute() else (self.bids_root / path).resolve()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _coerce(tp, value, path: str):
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        for arg in args:
            if dataclasses.is_dataclass(arg):
                return _build(arg, value, path)
        return value
    if origin in (list, tuple):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        args = typing.get_args(tp)
        inner = args[0] if args else None
        if inner is not None and dataclasses.is_dataclass(inner):
            return [_build(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return list(value)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    return value


def _build(cls, data, path: str = ""):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path or '<root>'} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        where = path or "<root>"
        raise ConfigError(
            f"unknown configuration key(s) {unknown} under {where}; "
            f"valid keys are {sorted(known)}"
        )
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}.{f.name}".lstrip("."))
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Read a YAML study description into a validated :class:`Config`."""
    path = Path(path).expanduser()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg = _build(Config, raw)
    # A relative bids_root is interpreted relative to the config file, so a
    # study directory can be moved without editing the YAML.
    root = Path(cfg.study.bids_root).expanduser()
    if not root.is_absolute():
        cfg.study.bids_root = str((path.parent / root).resolve())
    cfg.validate()
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def dump_config(cfg: Config, path: str | Path) -> Path:
    """Write the fully resolved configuration next to the results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, default_flow_style=False)
    return path
