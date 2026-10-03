# Motor response: Oxford vs every Princeton participant

`FLUX_Oxford_Princeton_Comparison.ipynb` compares **one** Oxford participant with
**one** Princeton participant. This guide describes the batch version: run the
`cerca_flux` pipeline over the whole Princeton TSX sample, and overlay the Oxford
participant's motor response (**thick black line**) on each Princeton participant
(**thinner coloured lines**), so that you can change a preprocessing option and see
what it does to every participant at once.

```bash
# 1. Oxford, then every Princeton participant (locally ...)
uv run cerca-flux run --config configs/motor/oxford.yaml    --preset motor --variant default
uv run cerca-flux run --config configs/motor/princeton.yaml --preset motor --variant default --n-jobs 4

# 2. ... draw the overlay
uv run cerca-flux compare --reference configs/motor/oxford.yaml \
                          --cohort    configs/motor/princeton.yaml --variant default
```

On Della, `run/della_cerca_flux.sh` is an array job (index 0 is Oxford, 7–44 are the
Princeton subject numbers, as in `della_mne_batch.sh`) and
`run/della_cerca_flux_compare.sh` draws the figures afterwards.

## Where things are assumed to live

This repository sits in the TSX folder next to `data/`, `mne-opm/` and `TSX_OPM/`, and the
Princeton paths are the `TSX_OPM` batch-script paths:

```
$TSX_DIR/
├── cerca-flux-tutorial/        # this repository
├── mne-opm/   TSX_OPM/
└── data/
    ├── TSX/{raw,bids,freesurfer}              # Princeton, from mne-opm
    └── oxford_sub-01/Data/Cerca_Spatt_BIDS    # Oxford, laid out as in the notebook
```

`TSX_DIR` defaults to the folder above this repository; export it to override. Configuration
values may use `${VAR}` and `${VAR:-default}`; `OXFORD_BIDS` relocates the Oxford dataset on
its own. **The Oxford location is an assumption** taken from the notebook's paths — adjust
`OXFORD_BIDS` if it lives elsewhere.

## Trying a preprocessing option

`configs/motor/base.yaml` holds every analysis choice shared by both sites;
`oxford.yaml` and `princeton.yaml` extend it and state only what differs (data paths,
event labels, bad sensors, crop). A test asserts that nothing else differs, so a change
to a shared option always reaches both sites.

Name each set of options with `--variant`; its outputs go to their own
`derivatives/cerca-flux-motor_<variant>` folder, so variants never overwrite (or silently
reuse) each other's cached stages. Override any value with `--set section.key=value`:

```bash
for site in oxford princeton; do
  uv run cerca-flux run --config configs/motor/$site.yaml --preset motor \
      --variant hfc3 --set hfc.order=3
done
uv run cerca-flux compare --reference configs/motor/oxford.yaml \
      --cohort configs/motor/princeton.yaml --variant hfc3 --set hfc.order=3
```

The same `--variant` and `--set` must be given to `compare`, because it reads the
configuration too. Quote subject labels in overrides (`'--set study.subjects=["007"]'`):
YAML reads a bare `007` as the number 7.

## What is plotted

For each participant, response-locked epochs (−1.0 to 1.2 s, 0.1–45 Hz, 50 pT rejection)
are pooled over the configured `motor.conditions`.

**Central RMS field.** With $C$ central radial sensors (C1–C6, axis Z) and trial $k$,

$$
\mathrm{RMS}_k(t)=\sqrt{\frac{1}{C}\sum_{c=1}^{C} x_{kc}(t)^{2}},\qquad
\bar r(t)=\frac{1}{K}\sum_{k=1}^{K}\mathrm{RMS}_k(t),\qquad
\mathrm{SEM}(t)=\frac{s_k\!\left[\mathrm{RMS}_k(t)\right]}{\sqrt{K}} .
$$

The RMS is taken *before* averaging over trials. That makes it polarity-invariant, which matters
because corresponding sensors do not share location or sign across participants, but it is
not the RMS of the evoked average: single-trial noise adds in quadrature to the response, so
$\bar r$ has a noise floor that does not shrink with $K$. Baseline-normalised,

$$
z(t)=\frac{\bar r(t)-\mu_b}{\sigma_b},
$$

where $\mu_b$ and $\sigma_b$ are the mean and standard deviation of $\bar r(t)$ over
$t\in[-0.8,-0.6]$ s.

**Central motor beta.** Multitaper power $P(f,t)$ at $f\in\{15,17,\dots,29\}$ Hz with
$n_\mathrm{cycles}=f/2$ (0.5 s windows), averaged over the same sensors, in percent change
from the baseline $B=[-0.9,-0.6]$ s:

$$
\Delta(f,t)=100\,\frac{P(f,t)-\bar P_B(f)}{\bar P_B(f)},\qquad
\bar P_B(f)=\frac{1}{|B|}\sum_{t\in B}P(f,t).
$$

The plotted curve is $\Delta(f,t)$ averaged over $f$. Negative values around movement are beta
desynchronisation; the later positive values are the rebound.

**Left primary motor cortex (`precentral-lh`).** *LCMV:* one covariance over −0.8…0.8 s, 5 %
regularisation, `nai` weight normalisation; the label's `mean_flip` time course of the
response-locked average, in SD of the −0.8…−0.5 s baseline. *DICS:* one filter from the
movement (−0.4…0.1 s) and rebound (0.4…0.9 s) 15–30 Hz CSDs, then CSDs in 400 ms windows
stepped every 100 ms; parcel power $p(t_c)$ (the magnitude of the `pca_flip` value, since the sign
of a non-negative quantity is an arbitrary orientation convention) in dB,

$$
10\log_{10}\frac{p(t_c)}{\bar p_b},
$$

with $\bar p_b$ the mean over window centres $t_c\in[-0.75,-0.5]$ s. The 400 ms window makes this
curve smooth and truncated at the edges; do not read it at 100 ms resolution.

### The figures

| File (in `outputs/motor_response/<variant>/`) | Contents |
|---|---|
| `motor_response_overlay.png` | Five panels: central RMS (fT), baseline-normalised RMS, beta, left-M1 LCMV, left-M1 DICS. Oxford is one thick black line drawn on top; each Princeton participant is a thin line. |
| `motor_response_small_multiples_<measure>.png` | One panel per participant, each against Oxford, on shared axes. |
| `motor_response_metrics.csv` | One row per participant: epoch counts, peak ERF, beta rebound, LCMV/DICS summaries, and a `status`/`note` for anyone not plotted. |

Colour encodes a participant's **position in the cohort** (a sequential ramp with a labelled
colour bar), and it is fixed by the configured subject list — a participant who is missing
leaves a gap rather than repainting everyone else, so colours are comparable across
variants. 35 categorical hues cannot be told apart, so identification is the job of the
small multiples and the table. Small-multiple axes always contain the whole Oxford curve; a
participant whose extremes lie more than three interquartile ranges beyond the others is
left out of the axis limits and marked `off scale`, so one noisy recording cannot flatten
everybody else.

## How this relates to the notebook

All analysis choices are the notebook's (see `configs/motor/base.yaml`). Differences you
should know about:

| Topic | Notebook | Here | Why |
|---|---|---|---|
| Duration matching | Both sites cropped to `min(Oxford, Princeton available)` | Princeton cropped to `min(Oxford, available)`; Oxford uses its full recording | One Oxford run serves all 35 participants |
| Training exclusion | First **400 s** (code), "450 s" (markdown) | Fixed `crop_start: 400` | The notebook's code value; override with `--set study.crop_start=450`. Training length may differ per participant; TSX practice blocks are not read from the behavioural metadata |
| Princeton trials | `response/right`, every such event | same | First-response-per-trial selection (`select_trial_response`) is not applied |
| Beamformer channels | Text says radial; the code (`radial_only = False`, `picks="mag"`) uses X, Y and Z | `motor.source.picks: all` (as run) | Use `--set motor.source.picks=radial` for Z only |
| DICS parcel power | `pca_flip` value, floored at the smallest float | its magnitude | the sign is arbitrary; a negative one would floor to ≈ −3000 dB. Identical wherever the notebook's value was positive |
| Flat sensors | not flagged | `qc.flag_flat: false` | matches the notebook; the `cerca_flux` default is `true` |
| ICA | ocular only, ≤ 3 components | `detect_ecg: false`, `max_exclude: 3` | matches the notebook |
| Blink detector | frontal OPM surrogate at both sites | `annotate.eog.prefer_native: false` | matches `SHARED_BLINK_DETECTOR = True` |

Oxford ships a forward solution and a BEM but no separate `*-trans.fif`, so (as in the notebook)
its MRI/head transform is read out of the existing `*_fwd.fif`; the surface source space and
forward model are then built by the pipeline for both sites.

## Failures are reported, not hidden

A recording that fails a required stage is recorded as `failed` and skipped; an optional stage
marks it `partial`. `compare` then lists the participant under *Not plotted*, writes the reason
into the metrics table, and still draws everyone else. Likely causes in the TSX sample are
missing `response/right` events (the response channels `BNC 1/5 Z`), an incomplete FreeSurfer
reconstruction, a missing coregistration `*-trans.fif`, or fewer than three central sensors
surviving quality control. Rerun a single participant with
`sbatch --array=24 run/della_cerca_flux.sh`.

## Verification status

The code is tested end to end on the synthetic dataset (`tests/test_motor.py`,
`tests/test_compare.py`, `tests/test_motor_configs.py`): crop and duration matching, the
`motor` preset through to the source time courses, cache reuse, the overlay's line styling,
and the CLI. It has **not** been run on real Oxford or Princeton recordings. Before trusting a
comparison, run `--variant default` and check the Oxford row and the Princeton participant used in
the notebook against the notebook's own `quality_metrics.csv` (central ERF SNR, beta rebound).
They should be close but need not match exactly: the notebook also crops Oxford to the matched
duration, and ICA and quality-control details differ slightly.
