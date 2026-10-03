"""The whole thing on a fake TSX folder, through the real script and the shipped configs.

    $TSX_DIR/data/TSX/{bids,freesurfer}                          Princeton (task TSX, response/right)
    $TSX_DIR/data/oxford_sub-01/Data/Cerca_Spatt_BIDS            Oxford (task SpAtt, resp_T / resp_L)

Both datasets are surrounded by booby-trapped copies of what another pipeline would have left
behind - processed data, epochs, forward models, BEM solutions.  Reading any of them crashes the
run, so a passing test shows they were never touched, and the manifests show what was.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cerca_flux.utils import read_json

from ._motor_support import RESPONSE_LABELS
from ._synthetic_data import make_synthetic_bids
from ._synthetic_fs import make_synthetic_freesurfer

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "run" / "run_motor_compare.sh"
TRAP = b"BOOBY TRAP - another pipeline's product; reading this file is a bug"


def _trap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TRAP)
    return path


@pytest.fixture(scope="module")
def tsx(tmp_path_factory):
    root = tmp_path_factory.mktemp("TSX")
    data = root / "data"
    # Princeton: subject 007, session 01, task TSX, right-hand responses
    princeton = make_synthetic_bids(data / "TSX" / "bids", subjects=("007",),
                                    labels=["response/right"], task="TSX")
    make_synthetic_freesurfer(data / "TSX" / "freesurfer", "sub-007_ses-01")
    # Oxford: sub-01 with its FreeSurfer subject T1s inside derivatives/, as in the notebook
    oxford = make_synthetic_bids(data / "oxford_sub-01" / "Data" / "Cerca_Spatt_BIDS",
                                 subjects=("01",), labels=RESPONSE_LABELS, task="SpAtt")
    make_synthetic_freesurfer(oxford / "derivatives" / "Freesurfer", "T1s")

    traps = [
        # what the Princeton (mne-opm / mne-bids-pipeline) run leaves behind
        princeton / "derivatives/mne-bids-pipeline/sub-007/ses-01/meg/sub-007_ses-01_task-TSX_run-01_proc-filt_raw.fif",
        princeton / "derivatives/mne-bids-pipeline/sub-007/ses-01/meg/sub-007_ses-01_task-TSX_run-01_epo.fif",
        princeton / "derivatives/mne-bids-pipeline/sub-007/ses-01/meg/sub-007_ses-01_task-TSX_run-01_fwd.fif",
        princeton / "derivatives/mne-bids-pipeline/sub-007/ses-01/meg/sub-007_ses-01_task-TSX_run-01_trans.fif",
        data / "TSX/freesurfer/sub-007_ses-01/bem/sub-007_ses-01-bem-sol.fif",
        data / "TSX/freesurfer/sub-007_ses-01/bem/sub-007_ses-01-fwd.fif",
        # Oxford's own precomputed forward model and BEM
        oxford / "derivatives/analysis/sub-01/ses-01/meg/sub-01_ses-01_task-SpAtt_run-01_fwd.fif",
        oxford / "derivatives/analysis/sub-01/ses-01/meg/sub-01_ses-01_task-SpAtt_run-01_bem-sol.fif",
        oxford / "derivatives/Freesurfer/T1s/bem/T1s-bem-sol.fif",
    ]
    for path in traps:
        _trap(path)
    return {"root": root, "princeton": princeton, "oxford": oxford, "traps": traps}


def _env(tsx, tmp_path_factory=None):
    import os
    return {**os.environ, "TSX_DIR": str(tsx["root"]),
            "CERCA_FLUX": f"{sys.executable} -m cerca_flux.cli",
            "MOTOR_COMPARE_LOGS": str(tsx["root"] / "logs")}


@pytest.fixture(scope="module")
def finished(tsx):
    out = tsx["root"] / "figures"
    result = subprocess.run(
        ["bash", str(SCRIPT), "run", "--variant", "e2e", "--subjects", "007", "--jobs", "1",
         "--set", "study.crop_start=0", "--out", str(out)],      # the synthetic recording is short
        env=_env(tsx), capture_output=True, text=True, timeout=900, cwd=REPO,
    )
    return result, out


def test_the_script_ran_every_step_to_completion(finished):
    result, _ = finished
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    for step in ("oxford: done", "princeton: done", "compare: done", "FINISHED"):
        assert step in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_the_script_showed_what_differs_from_the_cerca_defaults(finished):
    result, _ = finished
    assert "settings that differ from the Cerca defaults" in result.stdout
    assert "hfc.resample_sfreq" in result.stdout and "Cerca default: None" in result.stdout


def test_figures_and_table_were_written(finished):
    _, out = finished
    assert (out / "motor_response_overlay.png").stat().st_size > 10_000
    assert (out / "motor_response_small_multiples_erf_rms.png").exists()
    assert (out / "motor_response_metrics.csv").exists()


def _table(out):
    return pd.read_csv(out / "motor_response_metrics.csv", dtype={"subject": str})


def test_both_sites_were_processed_identically_and_the_table_proves_it(finished):
    _, out = finished
    table = _table(out)
    done = table[table["status"] == "ok"].set_index("dataset")
    assert sorted(done.index) == ["Oxford", "Princeton"]
    assert table[table["status"] == "ok"]["subject"].tolist() == ["01", "007"]
    assert done["fingerprint_shared"].nunique() == 1             # same shared settings, both sites
    assert done["fingerprint_site"].nunique() == 2               # but different per-site choices
    assert done["inputs_strict"].all() and (done["inputs_violations"] == 0).all()
    assert done.loc["Oxford", "inputs_transform"] == "T1s-trans.fif"
    assert done.loc["Princeton", "inputs_transform"] == "sub-007_ses-01-trans.fif"


def test_the_comparison_covers_the_full_sample_and_names_who_is_missing(finished):
    """Only 007 was processed, but the other 34 of the 35 are listed rather than dropped."""
    _, out = finished
    missing = _table(out).query("dataset == 'Princeton' and status != 'ok'")
    assert len(missing) == 34
    assert set(missing["status"]) == {"no BIDS recording"}
    assert "020" in set(missing["subject"]) and "007" not in set(missing["subject"])


def test_princeton_was_cropped_to_the_oxford_duration(finished):
    _, out = finished
    table = _table(out).set_index("subject")
    assert table.loc["007", "reference_duration_s"] > 200
    assert table.loc["007", "duration_s"] <= table.loc["007", "reference_duration_s"] + 0.01
    assert pd.isna(table.loc["01", "reference_duration_s"])        # the reference is not cropped


def test_the_table_also_reports_what_preprocessing_did_to_each_recording(finished):
    _, out = finished
    done = _table(out).query("status == 'ok'").set_index("subject")
    for column in ("n_bad_channels", "n_ica_excluded", "epochs_retained", "epochs_rejected_percent",
                   "muscle_percent", "n_hfc_projections"):
        assert column in done.columns and done[column].notna().all(), column


@pytest.mark.parametrize("site", ["oxford", "princeton"])
def test_nothing_another_pipeline_produced_was_read(tsx, finished, site):
    root = tsx[site]
    deriv = next(root.glob("derivatives/cerca-flux-motor_e2e"))
    manifests = list(deriv.glob("preprocessing/**/*_inputs.json"))
    assert len(manifests) == 1
    manifest = read_json(manifests[0])
    assert manifest["strict"] is True and manifest["violations"] == []
    read = {Path(f) for files in manifest["files"].values() for f in files}
    assert not read & {t.resolve() for t in tsx["traps"]}
    allowed = {"raw_bids", "sidecar", "declared"} | ({"reference"} if site == "princeton" else set())
    assert set(manifest["files"]) <= allowed
    assert len(manifest["files"]["raw_bids"]) == 1               # exactly the raw recording
    if site == "princeton":                                      # read only for its duration
        assert [Path(f).name for f in manifest["files"]["reference"]] == [
            "sub-01_ses-01_task-SpAtt_run-01_meg.fif"]
    assert "derivatives" not in manifest["files"]["raw_bids"][0].replace(str(deriv), "")


@pytest.mark.parametrize("site", ["oxford", "princeton"])
def test_the_bem_source_space_and_forward_model_were_rebuilt_not_reused(tsx, finished, site):
    deriv = next(tsx[site].glob("derivatives/cerca-flux-motor_e2e"))
    for kind in ("bem-sol", "surface-src", "surface-fwd"):
        built = list(deriv.glob(f"analysis/**/*_{kind}.fif"))
        assert len(built) == 1, (kind, built)
        assert built[0].read_bytes() != TRAP and built[0].stat().st_size > 1000


def test_the_resolved_configuration_of_each_recording_was_kept(tsx, finished):
    for site in ("oxford", "princeton"):
        deriv = next(tsx[site].glob("derivatives/cerca-flux-motor_e2e"))
        kept = list(deriv.glob("config_*.yaml"))
        assert len(kept) == 1 and "variant: e2e" in kept[0].read_text()


def test_the_preflight_check_passes_for_this_layout(tsx):
    for config in ("oxford", "princeton"):
        extra = ["--subjects", "007"] if config == "princeton" else []
        result = subprocess.run([sys.executable, "-m", "cerca_flux.cli", "check", "--config",
                                 f"configs/motor/{config}.yaml", *extra],
                                env=_env(tsx), capture_output=True, text=True, cwd=REPO, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 of 1 ready" in result.stdout


def test_the_preflight_check_names_what_is_missing(tsx):
    """Subject 020 is in the full sample but has no data in this fake layout."""
    result = subprocess.run([sys.executable, "-m", "cerca_flux.cli", "check", "--config",
                             "configs/motor/princeton.yaml", "--subjects", "007", "020"],
                            env=_env(tsx), capture_output=True, text=True, cwd=REPO, timeout=120)
    assert result.returncode == 1
    assert "1 of 2 ready" in result.stdout and "020: no raw BIDS recording" in result.stdout
