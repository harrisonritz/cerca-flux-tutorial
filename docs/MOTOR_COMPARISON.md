# Motor response: Oxford vs every Princeton participant

`FLUX_Oxford_Princeton_Comparison.ipynb` compares **one** Oxford participant with
**one** Princeton participant. This guide describes the batch version: run the
`cerca_flux` pipeline over the whole Princeton TSX sample, and overlay the Oxford
participant's motor response (**thick black line**) on each Princeton participant
(**thinner coloured lines**), so that you can change a preprocessing option and see
what it does to every participant at once.

```bash
run/run_motor_compare.sh start                       # check the setup, then run in the background
run/run_motor_compare.sh status                      # is it running? how far has it got?
run/run_motor_compare.sh log -f                      # follow the log
run/run_motor_compare.sh stop                        # stop it (finished steps stay cached)

run/run_motor_compare.sh start --variant hfc3 --set hfc.order=3     # try another option
```

`start` first runs `cerca-flux check` on both sites (raw recording, FreeSurfer reconstruction,
coregistration transform and ROI label present for every participant) and refuses to start if
the Oxford reference is not ready; participants that are not ready are listed and skipped
(`--require-all` makes that an error). It then runs Oxford, then every Princeton participant
(`--jobs N` at a time), then draws the comparison, detached from your terminal. A participant
failing does not stop the others, and the figure lists who is missing and why. The script is
bash-3.2 compatible (macOS) and keeps a laptop awake with `caffeinate` where available.
`--data DIR` / `--tsx DIR` point it at a data folder that is not under the TSX folder (e.g. a
mounted volume). On Della, `run/della_cerca_flux.sh` is the equivalent array job (index 0 is
Oxford, 7–44 are Princeton subject numbers) and `run/della_cerca_flux_compare.sh` draws the figures.

By hand, the same thing is:

```bash
uv run cerca-flux run --config configs/motor/oxford.yaml    --preset motor --variant default
uv run cerca-flux run --config configs/motor/princeton.yaml --preset motor --variant default --n-jobs 2
uv run cerca-flux compare --reference configs/motor/oxford.yaml \
                          --cohort    configs/motor/princeton.yaml --variant default
```

## Where things are assumed to live

This repository sits in the TSX folder next to `data/`, `mne-opm/` and `TSX_OPM/`, and the
Princeton paths are the `TSX_OPM` batch-script paths:

```
$TSX_DIR/
├── cerca-flux-tutorial/        # this repository
├── mne-opm/   TSX_OPM/
└── data/                       # = $TSX_DATA
    ├── TSX/{raw,bids,freesurfer}              # Princeton, from mne-opm
    └── oxford_sub-01/Data/Cerca_Spatt_BIDS    # Oxford, laid out as in the notebook
```

`TSX_DIR` defaults to the folder above this repository; export it to override, and export
`TSX_DATA` if the data folder lives elsewhere. Configuration values may use `${VAR}` and
`${VAR:-default}`; `OXFORD_BIDS` relocates the Oxford dataset on its own. **The Oxford location is
an assumption** taken from the notebook's paths.

## Both sites go through the same preprocessing, from raw data

The comparison is only meaningful if differences between the curves are differences between
participants, not between pipelines. Three things make that so, and each is enforced rather than
left to convention.

**1. One preprocessing definition.** `configs/motor/preprocessing.yaml` is the single place
where every preprocessing step is defined; `oxford.yaml` and `princeton.yaml` both extend it and
hold only what legitimately differs between sites (data paths, event labels, sensors known to be
faulty, the training crop). A test fails if either site file restates a preprocessing setting.

**2. Only raw inputs.** A recording may read exactly four kinds of pre-existing file:

| Read | What it is |
|---|---|
| the raw BIDS recording and its sidecars | `sub-*_meg.fif`, `events.tsv`, `channels.tsv`, … (mne-opm's BIDS conversion only concatenates runs, renames triggers and records a bad-sensor list; it does no filtering or resampling) |
| the FreeSurfer reconstruction | surfaces, MRI, labels — shared by both sites |
| one coregistration transform | `<subject>/bem/<subject>-trans.fif`; it comes from aligning the head scan to the MRI and cannot be recomputed here |
| the duration reference | Princeton only: Oxford's raw recording, read for its length |

Nothing else is read. Not Princeton's `derivatives/` (processed data, epochs), and not any BEM
solution, source space or forward model made elsewhere: **the BEM, source space and lead field
are rebuilt from the FreeSurfer surfaces, identically for both sites.** With
`provenance.strict: true` (set in `motor.yaml`) this is enforced three ways: the configuration
refuses `forward.bem` and any `forward.trans` that is not a transform; the forward stage no
longer searches other pipelines' derivatives or picks up a `*-bem-sol.fif` lying in the
FreeSurfer folder; and a runtime audit hooks Python's `open`, so *any* library reading *any*
other file is blocked and the recording fails with the offending path. Every recording writes
`<derivatives>/preprocessing/.../*_inputs.json` listing exactly what it read, and a test plants
booby-trapped copies of everything another pipeline would have produced and checks that none
is ever opened.

*Oxford's transform.* Oxford ships its coregistration only inside its forward solution. Extract
it **once, explicitly** (only the 4×4 transform is read from that file; its BEM, source space and
lead field are not used):

```bash
uv run cerca-flux export-trans \
    --fwd <OXFORD_BIDS>/derivatives/analysis/sub-01/ses-01/meg/sub-01_ses-01_task-SpAtt_run-01_fwd.fif \
    --out <OXFORD_BIDS>/derivatives/Freesurfer/T1s/bem/T1s-trans.fif
```

If you have Oxford's original `-trans.fif`, use that instead. Princeton's transform is already
where mne-opm put it, in the FreeSurfer subject's `bem/` folder.

*Bad sensors.* Sensors flagged in the BIDS `channels.tsv` are treated as bad at both sites
(Princeton's list was written when the data were converted; Oxford's two faulty modules are in
`oxford.yaml`). Set `channels.use_metadata_bads: false` to decide every sensor from the raw data
alone.

**3. Proof in every result.** Each motor result is stamped with a hash of the settings that made
it. `compare` draws a comparison only between results that provably share the preprocessing:

* the two site configurations must agree on every shared setting (otherwise it stops and lists the
  disagreements; `--allow-mismatch` overrides, and the figure says so);
* a result **made with settings other than its site's current configuration is stale** and left out.
  Stages are cached by file existence, so changing an option without `--overwrite` or a new
  `--variant` would otherwise silently reuse old results;
* a Princeton result whose shared settings differ from Oxford's, or that read a file outside the
  allowed set, is left out.

Anything left out is listed on the figure and in the metrics table, with the reason. The figure
footer states the settings fingerprint, that inputs were audited, and every setting changed from
the Cerca defaults.

## Changing the preprocessing

Every preprocessing step is in `configs/motor/preprocessing.yaml`, in pipeline order (bad
sensors, 1. sensor quality, 2. HFC, 3. artefact annotation, 4. ICA, 5. epoching). **Every**
setting is listed, and each line says whether it is the Cerca default or was changed — and what
the default is:

```yaml
hfc:
  order: 2               # [= Cerca default] spherical-harmonic order ...
  resample_sfreq: 300.0  # [CHANGED from Cerca default: null] both sites at 300 Hz ...
```

The "Cerca defaults" are the pipeline's built-in defaults, which reproduce the FLUX/Cerca tutorial
notebooks. See what this comparison changes, at any time:

```bash
uv run cerca-flux settings --config configs/motor/princeton.yaml --changed
```

Three ways to change a step — the Oxford and Princeton runs always change together:

1. **One-off, no file edited:** `--variant NAME --set section.key=value` (repeatable). The variant
   keeps its results in their own `derivatives/cerca-flux-motor_<variant>` folder.
   `run/run_motor_compare.sh start --variant no-ecg --set ica.detect_ecg=false`
2. **Permanently:** edit the value in `preprocessing.yaml` and keep its marker honest. Tests
   check that every setting is listed and that every `[= Cerca default]` / `[CHANGED …]` marker is true.
3. **Back to the Cerca default:** set the value to the default shown on its line.

Steps can be switched off (`hfc.enabled`, `annotate.enabled`, `ica.enabled`, `qc.enabled`,
`annotate.eog.enabled`, `annotate.muscle.enabled`) or tuned (HFC order, the artefact thresholds,
ICA components and how many are removed, the continuous filter, the rejection threshold, …);
`preprocessing.yaml` lists them all with what each does. The metrics table reports what each step
did to every recording (bad sensors, HFC reduction, blinks per minute, ICA components removed,
epochs kept), so a change can be read off the data as well as the curves.

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

All analysis choices are the notebook's (see `configs/motor/preprocessing.yaml` and `motor.yaml`). Differences you
should know about:

| Topic | Notebook | Here | Why |
|---|---|---|---|
| Duration matching | Both sites cropped to `min(Oxford, Princeton available)` | Princeton cropped to `min(Oxford, available)`; Oxford uses its full recording | One Oxford run serves all 35 participants |
| Training exclusion | First **400 s** (code), "450 s" (markdown) | Fixed `crop_start: 400` | The notebook's code value; override with `--set study.crop_start=450`. Training length may differ per participant; TSX practice blocks are not read from the behavioural metadata |
| Princeton trials | `response/right`, every such event | same | First-response-per-trial selection (`select_trial_response`) is not applied |
| Forward model | Princeton built in the notebook; Oxford's precomputed forward and BEM were also used for the volume analysis | both rebuilt by the pipeline from the FreeSurfer reconstruction; only the coregistration is taken as given | identical treatment; no other pipeline's products |
| Beamformer channels | Text says radial; the code (`radial_only = False`, `picks="mag"`) uses X, Y and Z | `motor.source.picks: all` (as run) | Use `--set motor.source.picks=radial` for Z only |
| DICS parcel power | `pca_flip` value, floored at the smallest float | its magnitude | the sign is arbitrary; a negative one would floor to ≈ −3000 dB. Identical wherever the notebook's value was positive |
| Flat sensors | not flagged | `qc.flag_flat: false` | matches the notebook; the `cerca_flux` default is `true` |
| ICA | ocular only, ≤ 3 components | `detect_ecg: false`, `max_exclude: 3` | matches the notebook |
| Blink detector | frontal OPM surrogate at both sites | `annotate.eog.prefer_native: false` | matches `SHARED_BLINK_DETECTOR = True` |

## Failures are reported, not hidden

A recording that fails a required stage is recorded as `failed` and skipped; an optional stage
marks it `partial`. `compare` then lists the participant under *Not plotted*, writes the reason
into the metrics table, and still draws everyone else. Likely causes in the TSX sample are
missing `response/right` events (the response channels `BNC 1/5 Z`), an incomplete FreeSurfer
reconstruction, a missing coregistration `*-trans.fif`, or fewer than three central sensors
surviving quality control. Rerun a single participant with
`sbatch --array=24 run/della_cerca_flux.sh`.

## Verification status

The code is tested end to end on the synthetic dataset (`tests/`): crop and duration matching,
the `motor` preset through to the source time courses, cache reuse, the overlay's line styling,
the input audit (rules, blocking of real `open` and MNE reads, and a full run among booby-trapped
decoys), the settings fingerprints and stale detection, that `preprocessing.yaml` lists every
setting with true markers, the background script's mechanics (detaching, ordering, argument
pass-through, `stop` killing child processes, failures), and the whole thing through the real
script on a fake TSX folder with decoys of everything another pipeline would produce. It has
**not** been run on real Oxford or Princeton recordings. Before trusting a comparison, run
`--variant default` and check the Oxford row and the Princeton participant used in the notebook
against the notebook's own `quality_metrics.csv` (central ERF SNR, beta rebound). They should be
close but need not match exactly: the notebook also crops Oxford to the matched duration, and ICA
and quality-control details differ slightly.
