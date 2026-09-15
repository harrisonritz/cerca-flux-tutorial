# `cerca_flux`: the Cerca OPM-FLUX pipeline as a batch script

The FLUX tutorial notebooks in this repository analyse **one recording at a
time**, with paths and parameters edited in cells.  `cerca_flux` keeps their
analysis choices and their order, but turns each step into a function over a
BIDS study, so a whole dataset can be analysed from a single configuration file.

```bash
uv sync
uv run cerca-flux template > study.yaml    # then edit study.yaml
                                           # (source: cerca_flux/templates/study.yaml)
uv run cerca-flux list  --config study.yaml
uv run cerca-flux run   --config study.yaml --n-jobs 4
```

Assumed inputs:

* a **BIDS** dataset whose `events.tsv` carries meaningful `trial_type` labels
  (i.e. the `DownsamplingBIDSconversion` notebook has already been run, or the
  data were converted equivalently);
* an organised **FreeSurfer** `SUBJECTS_DIR` with `recon-all` complete, plus
  `mne watershed_bem` and `mne make_scalp_surfaces`;
* one **MRI/head transform** per subject from coregistration.

---

## Stage order

This is the order the OPM-FLUX pipeline recommends, and the order the
`FLUX_Oxford_Princeton_Comparison` notebook follows.  It is fixed; only *which*
stages run is configurable.

| # | Stage | Notebook | Output |
|---|-------|----------|--------|
| 1 | `qc` | *A First Look*, *A2 Sensor Quality checking* | bad-channel list, QC figures |
| 2 | `hfc` | *ArtefactSuppressionHFC* | `*_hfc_raw.fif` |
| 3 | `annotate` | *ArtefactAnnotation* | `*_ann_raw.fif`, `*_ann_raw.csv` |
| 4 | `ica` | *ICA* | `*_ica.fif`, `*_ica_raw.fif` |
| 5 | `epochs` | *ConditionSpecificTrials* | `*_epo.fif` |
| 6 | `erf` | *EventRelatedFields* | `*_ave.fif` |
| 7 | `tfr` | *TimeFrequencyPower* | `*_band-<name>_tfr.h5` |
| 8 | `mvpa` | *Classification* | `*_decoding.npz` |
| 9 | `forward` | *ForwardModel* | `*_bem-sol.fif`, `*_{volume,surface}-src.fif`, `*_{volume,surface}-fwd.fif` |
| 10 | `source` | *DICSbeamforming* (+ comparison-notebook LCMV) | `*_<estimate>_<space>_native-stc.h5` |
| 11 | `morph` | comparison notebook §14 | `*_<estimate>_<space>_fsaverage-stc.h5` |
| 12 | `report` | — | `reports/*_report.html`, `*_qc.json` |

Named subsets are available as presets: `full`, `full_no_mvpa`, `preproc`
(1–5), `sensor` (6–8), `source` (9–11).

```bash
cerca-flux run --config study.yaml --preset preproc
cerca-flux run --config study.yaml --stages ica epochs erf   # any subset
```

Every stage caches its output in the derivatives tree and is skipped when that
output already exists, so a suffix of the pipeline can be re-run on its own.
`--overwrite` forces recomputation.

---

## Stage-by-stage

### 1. `qc` — sensor quality check

Bad sensors are identified **before** HFC, because HFC is a linear projection
across the array: a noisy or flat sensor left in would spread its noise
everywhere.  Two sources are combined: sensors noted during acquisition
(`channels.manual_bads`, expanded across all three axes of a sensor) and a
power-spectral outlier rule.

| `qc.method` | Rule |
|---|---|
| `robust_mad` *(default)* | median + k·MAD **within each measurement axis** |
| `fixed_db` | the tutorial's absolute 39.1 dB cut |
| `none` | manual bads only |

> **Deviation from the tutorial.** The tutorial uses a fixed 39.1 dB threshold
> and calls it "admittedly somewhat arbitrary".  An absolute dB value is only
> meaningful for one system and one recording session, so the default here is
> the data-adaptive per-axis rule that the comparison notebook argues for.  Set
> `qc.method: fixed_db` to reproduce the tutorial exactly.

> **Addition.** The tutorial's manually flagged `B4` is a *flat* sensor, and the
> HFC notebook states that flat sensors must be excluded too — but a
> high-power threshold cannot catch them.  `qc.flag_flat` therefore also flags
> sensors that fall abnormally far *below* the axis median.

### 2. `hfc` — homogeneous field correction

`compute_proj_hfc(order=2)` after excluding bad channels.  The full triaxial
array is kept through the projection (`hfc.keep_axes: all`) because the X and Y
channels help constrain the homogeneous model; the reduction to radial channels
happens per analysis.  `hfc.resample_sfreq` optionally resamples first to bound
memory — it is validated against the muscle band, since resampling to at or
below 2× the band's upper edge would make stage 3 meaningless.

### 3. `annotate` — ocular and muscle artefacts

Nothing is rejected here; segments are only marked, so later stages can decide.

* **Blinks.** A recorded EOG channel is used when the system has one; otherwise
  a bipolar frontal OPM surrogate (`F9` − `F10`, radial axis) is built, as in
  the tutorial. When both exist, the recorded channel best correlated with the
  surrogate is chosen. Detection runs on 1–10 Hz.
* **Muscle.** `annotate_muscle_zscore` on 110–140 Hz, z > 3, `min_length_good`
  0.1 s, with bad channels dropped first.

> **Deviation.** `annotate.eog.threshold` defaults to `null` (MNE's adaptive
> threshold) rather than the tutorial's fixed `0.3e-11` T, which was set by
> visual inspection of one recording and will not transfer across subjects. Set
> it explicitly to reproduce the tutorial.

### 4. `ica` — ocular and cardiac components

FastICA on a 3–30 Hz copy resampled to 250 Hz; the unmixing is applied to the
full-rate data.  `n_components` is clamped to the rank HFC has left behind.

> **Deviation — the one that matters most.** The tutorial selects components by
> eye (`ica.exclude = [6, 11]`) and warns that indices are not deterministic.
> That cannot be done for a whole study, so components are selected
> automatically: correlation with the ocular channel (`find_bads_eog`) and CTPS
> against a cardiac signal synthesised from the magnetometers
> (`find_bads_ecg`).  Candidates are ranked by evidence and capped at
> `ica.max_exclude` (default 5, matching the tutorial's reported 2–5 per
> participant).  **Component topographies and scores are written out for every
> subject — inspect them.**  Set `ica.enabled: false` to run the ICA step
> manually instead.

### 5. `epochs` — condition-specific trials

`epochs.conditions` maps each condition name onto the BIDS `trial_type`
label(s) pooled into it:

```yaml
conditions:
  cue_left:  [cue_Left]
  cue_right: [cue_Right]
  response:  [resp_T, resp_L]   # pooled, as Oxford is in the comparison notebook
```

Pooled conditions become MNE hierarchical names (`response/resp_T`), so
`epochs['response']` selects the union and `epochs['response/resp_T']` one arm.
Epochs are saved broadband and un-baselined with only a per-epoch linear
detrend, so each downstream analysis picks its own baseline and filter.

> **Deviation.** `epochs.continuous_h_freq` defaults to `null`.  The comparison
> notebook filtered the continuous data to 0.1–45 Hz, which is right for its
> response-locked beta analysis but would destroy the 60–90 Hz gamma band the
> FLUX TFR analysis targets.

Task events are decoded from a **frozen label→code map** recorded at load time,
before any `BAD_*` annotation is added.  This avoids the trap documented in the
*ConditionSpecificTrials* notebook, where `events_from_annotations(...,
'auto')` renumbers labels alphabetically, and it keeps codes identical across
subjects.

### 6–8. `erf`, `tfr`, `mvpa`

* **ERF.** Average → low-pass 30 Hz → crop → linear detrend → baseline, in that
  order, so the pre-event interval ends up centred on zero.
* **TFR.** Multitaper, one entry per band.  `n_cycles = freqs /
  n_cycles_divisor` fixes the window duration across frequencies;
  `time_bandwidth` sets taper count (TBW − 1) and smoothing (W = TBW / 2ΔT).
  The FLUX defaults are a 0.5 s / 1-taper window below 30 Hz and a 0.25 s /
  3-taper window above it.  Contrasts such as the alpha lateralisation index
  `(right − left)/(right + left)` are computed without a baseline.
* **MVPA.** Sliding 100 ms window; all sensors × all samples in the window form
  one feature vector.  Linear SVM, stratified k-fold, ROC AUC.

> **Addition.** Classes are balanced by subsampling before decoding
> (`mvpa.balance_classes`).  Across a study, trial counts differ between
> conditions and between subjects, and unequal counts inflate AUC.

### 9. `forward` — BEM, source spaces, lead fields

Single-shell BEM (conductivity 0.3 S/m, `ico=4`) — adequate for MEG, unlike
EEG.  Both source spaces are built when configured:

* **volume**: 5 mm grid bounded by the inner-skull surface, `mindist` 5 mm;
* **surface**: `oct6` on the white surface (~4098 vertices/hemisphere).

Coregistration is **not** re-fitted.  The transform is located via
`forward.trans` (templates support `{fs_subject}`, `{subject}`, … and globs) or
by searching the FreeSurfer `bem/` folder.  A missing transform is a clear
error, not a silent fallback — verify each coregistration visually before
trusting a source result.

The forward model is built for every channel that survived preprocessing and
narrowed to the analysis picks at stage 10, so one solution serves both the
radial-only and all-axis analyses.

### 10. `source` — LCMV and DICS

Both beamformers use a **common** spatial filter built from data pooled across
the conditions and intervals later contrasted; a per-condition filter would
differ between the things being compared.  Regularisation is 5 % diagonal
loading, orientation is `max-power`, and the data **rank** is passed explicitly
because HFC and ICA leave the data rank-deficient (in the tutorial, 52 of 60
channels).

* **LCMV** — one covariance window spanning baseline and active intervals;
  output is the per-source dB power change between them.
* **DICS** — per band, a CSD per named time window, one common filter, then the
  configured contrasts.

All results are **relative contrasts**, because beamformer output carries a
depth bias: absolute power grows towards the centre of the head regardless of
the data.  (You can see this in the saved absolute maps, which peak near the
origin.)

Peaks are labelled automatically: volume peaks against `aparc+aseg.mgz` — with
a fall-back to the nearest cortical parcel and its distance when a 5 mm grid
point lands just outside the ribbon — plus MNI coordinates via the FreeSurfer
Talairach transform; surface peaks against `aparc`.

Display thresholds in the figures are the pooled 95th percentile of the
absolute statistic, with the 99th as the colour limit.  **These are exploratory
display thresholds, not statistical ones.**  Confirmatory inference needs an
appropriate permutation/null distribution.

### 11. `morph` — onto a template

Individual source spaces have different grids, so subjects can only be averaged
after a common template is imposed.  Volume and surface estimates are morphed
to `fsaverage` and cached; requires `dipy` (a declared dependency) and
`fsaverage` in `SUBJECTS_DIR`.

### 12. `report`

One HTML report per recording, assembled from **every figure on disk** for that
recording — not only from the stages run in this invocation — so re-running one
stage never silently shrinks the report.  Machine-readable metrics go to
`*_qc.json`.

---

## Outputs

```
<bids_root>/derivatives/cerca-flux/
├── config.yaml                  # the fully resolved configuration actually used
├── preprocessing/sub-XX/ses-YY/meg/   *_hfc_raw.fif  *_ann_raw.fif  *_ica_raw.fif
│                                      *_ica.fif      *_state.json
├── analysis/sub-XX/ses-YY/meg/        *_epo.fif  *_ave.fif  *_band-*_tfr.h5
│                                      *_decoding.npz  *_qc.json
│                                      *_bem-sol.fif  *_{volume,surface}-{src,fwd}.fif
│                                      *_<estimate>_<space>_{native,fsaverage}-stc.h5
├── figures/sub-XX/ses-YY/             numbered PNGs, one set per stage
├── reports/                           sub-XX_..._report.html
├── logs/                              one log per recording
└── group/
    ├── quality_metrics.tsv / .csv / .png
    ├── grand_average_ave.fif, grand_average_band-*_tfr.h5
    ├── group_decoding.npz / .png
    └── group_<estimate>_<space>_fsaverage-stc.h5
```

`quality_metrics.tsv` has one row per recording and keeps acquisition noise,
HFC benefit, artefact burden, epoch retention and effect size as **separate
columns**.  No single number defines OPM-MEG data quality, and collapsing them
would hide which of them a bad subject actually failed on.

---

## Running at scale

```bash
# In-process, N recordings at a time
cerca-flux run --config study.yaml --n-jobs 8

# SLURM: one array task per recording
cerca-flux slurm --config study.yaml --out slurm/ \
    --time 04:00:00 --mem 32G --cpus 8 --partition normal
sbatch slurm/cerca_flux_array.sh
cerca-flux group --config study.yaml     # collect group outputs afterwards
```

`cerca-flux slurm` freezes the resolved configuration next to the script, so
submitted jobs cannot drift from what was reviewed at submission time, and
writes `recordings.tsv` mapping each array index to a recording.

A failing recording never stops the study: required stages abort that subject
only, optional stages mark it `partial`, and the reason lands in the `status`
and `error` columns of the group table.  Resubmit a single array index with
`--index N`.

## From Python

```python
from cerca_flux import load_config, resolve_stages, run_study, run_subject
from cerca_flux.paths import discover_recordings

cfg = load_config("study.yaml")
cfg.output.overwrite = True

run_study(cfg, resolve_stages("full", None), n_jobs=4)

# or drive individual stages against your own objects
from cerca_flux.preprocess import load_raw, check_sensors, apply_hfc
from cerca_flux.context import SubjectContext
from cerca_flux.paths import SubjectPaths
from cerca_flux.utils import setup_logging

rec = discover_recordings(cfg)[0]
ctx = SubjectContext(cfg=cfg, rec=rec, paths=SubjectPaths(cfg, rec), logger=setup_logging())
raw = apply_hfc(ctx, check_sensors(ctx, load_raw(ctx)))
```

---

## Summary of deviations from the notebooks

Each exists because a value tuned by eye on one recording does not transfer to a
study.  All are configurable back to the tutorial behaviour.

| Choice | Tutorial | Default here | Why |
|---|---|---|---|
| Bad-channel rule | fixed 39.1 dB | per-axis median + 6·MAD | an absolute dB cut is site-specific |
| Flat sensors | manual only | also auto-flagged | HFC requires their removal; a high cut cannot find them |
| Blink threshold | fixed 0.3e-11 T | MNE adaptive | set by visual inspection of one trace |
| ICA components | selected by eye | `find_bads_eog` + `find_bads_ecg`, capped | not possible per subject at scale |
| Continuous low-pass | n/a (45 Hz in the comparison notebook) | `null` | 45 Hz would remove the gamma band |
| MVPA classes | as recorded | balanced | unequal trial counts inflate AUC |
| Epoch rejection | 10 pT | 10 pT (unchanged) | response-locked designs typically need ~50 pT |

## Verifying the install

The test suite builds a synthetic triaxial-OPM BIDS dataset and a synthetic
FreeSurfer subject, then runs the whole pipeline over them — so you can check
that the environment works before pointing it at real data, and without a
network download.

```bash
uv sync --all-groups
uv run pytest -q          # ~2 minutes
```

The synthetic recording carries a spatially homogeneous interference field, a
flat sensor, a noisy sensor, blinks on the frontal pair, a cardiac component and
two conditions, so the assertions check real behaviour: that HFC attenuates the
uniform field, that both bad sensors are caught, that blink annotations land on
the blinks (this is what guards the `first_samp` handling), that the beamformer
rank equals channels minus HFC projections, and that re-running a stage reuses
its cache without shrinking the report.

The generators are importable if you want a throwaway dataset to experiment on:

```python
from tests._synthetic_data import make_synthetic_bids
from tests._synthetic_fs import make_synthetic_freesurfer

bids_root = make_synthetic_bids("/tmp/demo/bids")
make_synthetic_freesurfer("/tmp/demo/fs", "sub-01")
```

## Pre-registration

The resolved configuration is written to
`derivatives/cerca-flux/config.yaml` on every run and contains every analysis
constant. It is the artefact to attach to a pre-registration, and to diff when
you want to know what changed between two analyses.
