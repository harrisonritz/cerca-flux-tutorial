"""Both sites are processed from raw data by this pipeline alone - and that is enforced.

Nothing another pipeline computed may be read: not its processed data, not its
BEM / source space / forward model.  The rules are tested directly, with real ``open``
and MNE calls, and end to end with booby-trapped decoy files that crash the run if read.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mne
import numpy as np
import pytest

from cerca_flux.config import Config, ConfigError, _build
from cerca_flux.paths import SubjectPaths, discover_recordings, find_bem, find_trans
from cerca_flux.pipeline import resolve_stages, run_subject
from cerca_flux.provenance import (
    InputAudit, ProvenanceError, audit_inputs, fingerprints, format_settings,
    settings_differences, settings_vs_defaults,
)
from cerca_flux.source import export_trans
from cerca_flux.utils import read_json

from ._motor_support import motor_config

KEY = "sub-01_ses-01_task-SpAtt_run-01"


@pytest.fixture
def cfg(response_study) -> Config:
    return motor_config(response_study, study={"variant": "prov"})


def _decoy(path: Path, content: bytes = b"NOT A REAL FIF - reading this file is a bug") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# -- the rules ------------------------------------------------------------------ #


@pytest.fixture
def audit(cfg):
    return InputAudit(cfg, KEY, strict=True)


@pytest.mark.parametrize("relative, category, allowed", [
    ("bids/sub-01/ses-01/meg/sub-01_ses-01_task-SpAtt_run-01_meg.fif", "raw_bids", True),
    ("bids/sub-01/ses-01/meg/sub-01_ses-01_task-SpAtt_run-01_split-02_meg.fif", "raw_bids", True),
    ("bids/sub-01/ses-01/meg/sub-01_ses-01_task-SpAtt_run-01_events.tsv", "sidecar", True),
    ("bids/participants.tsv", "sidecar", True),
    # someone else's recording, an empty-room file, or a processed copy beside the raw data
    ("bids/sub-02/ses-01/meg/sub-02_ses-01_task-SpAtt_run-01_meg.fif", "bids", False),
    ("bids/sub-01/ses-01/meg/sub-01_ses-01_task-noise_meg.fif", "bids", False),
    ("bids/sub-01/ses-01/meg/sub-01_ses-01_task-SpAtt_run-01_proc-filt_meg.fif", "bids", False),
    # another pipeline's derivatives, whatever they are called
    ("bids/derivatives/mne-bids-pipeline/sub-01/ses-01/meg/sub-01_proc-filt_raw.fif",
     "other_derivatives", False),
    ("bids/derivatives/analysis/sub-01/ses-01/meg/sub-01_fwd.fif", "other_derivatives", False),
    ("bids/derivatives/analysis/sub-01/ses-01/meg/sub-01_bem-sol.fif", "other_derivatives", False),
    # inside the FreeSurfer tree, FreeSurfer's own files are fine and pipeline products are not
    ("freesurfer/sub-01/surf/lh.white", "freesurfer", True),
    ("freesurfer/sub-01/mri/T1.mgz", "freesurfer", True),
    ("freesurfer/sub-01/label/lh.aparc.annot", "freesurfer", True),
    ("freesurfer/sub-01/bem/inner_skull.surf", "freesurfer", True),
    ("freesurfer/sub-01/bem/sub-01-bem-sol.fif", "freesurfer", False),
    ("freesurfer/sub-01/bem/sub-01-fwd.fif", "freesurfer", False),
    ("freesurfer/sub-01/bem/sub-01-src.fif", "freesurfer", False),
    # a product anywhere else
    ("elsewhere/TSX_OPM/sub-01_epo.fif", "outside", False),
])
def test_classification_rules(audit, cfg, response_study, relative, category, allowed):
    bids_root, fs_dir = response_study
    base = fs_dir.parent if relative.startswith("freesurfer/") else bids_root.parent
    path = base / relative
    got_category, problem = audit.classify(str(path.resolve()), path.name)
    assert got_category == category
    assert (problem is None) is allowed, problem


def test_the_pipelines_own_outputs_are_always_readable(audit, cfg):
    own = cfg.deriv_root / "analysis/sub-01/ses-01/meg" / f"{KEY}_epo.fif"
    assert audit.classify(str(own.resolve()), own.name) == ("own", None)


def test_the_declared_coregistration_is_the_one_fif_allowed_in_the_freesurfer_tree(audit, cfg):
    trans = cfg.fs_subjects_dir / "sub-01/bem/sub-01-trans.fif"
    assert audit.classify(str(trans.resolve()), trans.name)[1] is not None     # not declared yet
    audit.allow(trans)
    assert audit.classify(str(trans.resolve()), trans.name) == ("declared", None)


def test_a_freesurfer_tree_inside_derivatives_is_still_the_reconstruction(response_study, tmp_path):
    """Oxford keeps FreeSurfer under bids/derivatives/Freesurfer; that is not 'another pipeline'."""
    bids_root, _ = response_study
    fs = bids_root / "derivatives" / "Freesurfer"
    cfg = motor_config(response_study, study={"variant": "p2", "fs_subjects_dir": str(fs)})
    audit = InputAudit(cfg, KEY, strict=True)
    surf = fs / "T1s/surf/lh.white"
    assert audit.classify(str(surf.resolve()), surf.name) == ("freesurfer", None)
    foreign = bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_fwd.fif"
    assert audit.classify(str(foreign.resolve()), foreign.name)[0] == "other_derivatives"


# -- enforcement: real reads, blocked before anything is parsed ------------------ #


def test_strict_mode_blocks_reading_another_pipelines_data(cfg, response_study):
    bids_root, _ = response_study
    raw = next(bids_root.glob("sub-01/ses-01/meg/*_meg.fif"))
    processed = bids_root / "derivatives/mne-bids-pipeline/sub-01/ses-01/meg/sub-01_proc-filt_raw.fif"
    processed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(raw, processed)               # a perfectly VALID FIF: blocked on provenance alone
    with audit_inputs(cfg, KEY):
        mne.io.read_raw_fif(raw, preload=False, verbose="ERROR")                 # allowed
        with pytest.raises(ProvenanceError, match="another pipeline's derivatives"):
            mne.io.read_raw_fif(processed, preload=False, verbose="ERROR")


def test_strict_mode_blocks_a_foreign_forward_model_and_bem(cfg, response_study):
    bids_root, fs_dir = response_study
    forward = _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_fwd.fif")
    bem = _decoy(fs_dir / "sub-01/bem/sub-01-bem-sol.fif")
    with audit_inputs(cfg, KEY):
        with pytest.raises(ProvenanceError):
            mne.read_forward_solution(forward, verbose="ERROR")
        with pytest.raises(ProvenanceError, match="processed product"):
            mne.read_bem_solution(bem, verbose="ERROR")


def test_strict_mode_blocks_other_recordings_and_non_raw_files_beside_the_data(cfg, response_study):
    bids_root, _ = response_study
    other = _decoy(bids_root / "sub-02/ses-01/meg/sub-02_ses-01_task-SpAtt_run-01_meg.fif")
    try:
        with audit_inputs(cfg, KEY):
            with pytest.raises(ProvenanceError, match="not this recording's raw data"):
                open(other, "rb").close()
    finally:
        # The BIDS folder is shared by the whole session, and a second raw file would make
        # "sub-02" a discoverable recording for every later test.
        shutil.rmtree(bids_root / "sub-02")


def test_plain_open_and_os_open_are_both_policed(cfg, response_study):
    import os
    bids_root, _ = response_study
    forward = _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_fwd.fif")
    with audit_inputs(cfg, KEY):
        with pytest.raises(ProvenanceError):
            open(forward, "rb")
        with pytest.raises(ProvenanceError):
            os.close(os.open(forward, os.O_RDONLY))


def test_writing_is_not_policed_only_reading(cfg, response_study):
    bids_root, _ = response_study
    target = bids_root / "derivatives/some-other-pipeline/notes.fif"
    target.parent.mkdir(parents=True, exist_ok=True)
    with audit_inputs(cfg, KEY):
        target.write_bytes(b"x")                       # not a read: no error


def test_non_strict_mode_records_but_does_not_block(response_study):
    cfg = motor_config(response_study, study={"variant": "obs"}, provenance={"strict": False})
    bids_root, _ = response_study
    forward = _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_fwd.fif")
    with audit_inputs(cfg, KEY) as audit:
        open(forward, "rb").close()
    assert audit.violations and not audit.strict
    assert "other_derivatives" in audit.manifest()["files"]


def test_the_audit_is_inert_outside_the_block(cfg, response_study):
    bids_root, _ = response_study
    forward = _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_fwd.fif")
    with audit_inputs(cfg, KEY):
        pass
    open(forward, "rb").close()                        # no error once the block has ended


def test_allow_declares_a_pre_existing_file(cfg, response_study):
    bids_root, _ = response_study
    trans = _decoy(bids_root / "derivatives/coreg/sub-01-trans.fif")
    with audit_inputs(cfg, KEY) as audit:
        with pytest.raises(ProvenanceError):
            open(trans, "rb")
        audit.allow(trans)
        open(trans, "rb").close()


# -- configuration refuses the foreign inputs up front ---------------------------- #


@pytest.mark.parametrize("forward, message", [
    ({"bem": "/data/other/sub-01-bem-sol.fif"}, "forbids forward.bem"),
    ({"trans": "/data/derivs/sub-01_fwd.fif"}, "not a coregistration transform"),
    ({"trans": "/data/derivs/sub-01_task-x_proc-filt_raw.fif"}, "not a coregistration transform"),
])
def test_strict_config_rejects_foreign_forward_inputs(forward, message):
    payload = {"study": {"bids_root": "/x"}, "epochs": {"conditions": {"a": ["x"]}},
               "mvpa": {"enabled": False}, "provenance": {"strict": True}, "forward": forward}
    with pytest.raises(ConfigError, match=message):
        _build(Config, payload).validate()


def test_a_real_coregistration_transform_is_accepted():
    payload = {"study": {"bids_root": "/x"}, "epochs": {"conditions": {"a": ["x"]}},
               "mvpa": {"enabled": False}, "provenance": {"strict": True},
               "forward": {"trans": "/fs/T1s/bem/T1s-trans.fif"}}
    _build(Config, payload).validate()


# -- discovery of the forward-model inputs ---------------------------------------- #


def test_other_pipelines_bem_and_transforms_are_never_picked_up(cfg, response_study):
    """The old search patterns found these; they belong to someone else's pipeline."""
    bids_root, fs_dir = response_study
    _decoy(fs_dir / "sub-01/bem/sub-01-bem-sol.fif")                      # in the FreeSurfer folder
    _decoy(bids_root / "derivatives/mne-bids-pipeline/sub-01/ses-01/meg/sub-01_trans.fif")
    rec = discover_recordings(cfg)[0]
    paths = SubjectPaths(cfg, rec)
    assert find_bem(cfg, rec, "sub-01", paths) is None                    # built fresh instead
    found = find_trans(cfg, rec, "sub-01", paths)
    assert found == fs_dir / "sub-01/bem/sub-01-trans.fif"                # the coregistration only


def test_the_coregistration_is_exported_once_explicitly(motor_run, tmp_path):
    cfg, rec, _ = motor_run
    fwd = SubjectPaths(cfg, rec).fwd("surface")
    out = export_trans(fwd, tmp_path / "bem" / "T1s-trans.fif")
    expected = mne.read_forward_solution(fwd, verbose="ERROR")["mri_head_t"]
    assert np.allclose(mne.read_trans(out)["trans"], expected["trans"])
    with pytest.raises(FileExistsError):
        export_trans(fwd, out)
    export_trans(fwd, out, overwrite=True)


def test_export_trans_command(motor_run, tmp_path, capsys):
    from cerca_flux.cli import main
    cfg, rec, _ = motor_run
    out = tmp_path / "T1s-trans.fif"
    assert main(["export-trans", "--fwd", str(SubjectPaths(cfg, rec).fwd("surface")),
                 "--out", str(out)]) == 0
    assert out.exists() and "only the coregistration transform was read" in capsys.readouterr().out


# -- end to end: decoys everywhere, none of them may be touched --------------------- #


def test_a_full_run_ignores_every_decoy_and_builds_its_own_forward_model(response_study):
    """Booby-trapped copies of everything another pipeline would have produced."""
    bids_root, fs_dir = response_study
    cfg = motor_config(response_study, study={"variant": "decoys"})
    raw = next(bids_root.glob("sub-01/ses-01/meg/*_meg.fif"))
    decoys = [
        _decoy(bids_root / "derivatives/mne-bids-pipeline/sub-01/ses-01/meg/"
               "sub-01_ses-01_task-SpAtt_run-01_proc-filt_raw.fif"),
        _decoy(bids_root / "derivatives/mne-bids-pipeline/sub-01/ses-01/meg/sub-01_ses-01_epo.fif"),
        _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_ses-01_fwd.fif"),
        _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_ses-01_bem-sol.fif"),
        _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_ses-01_trans.fif"),
        _decoy(fs_dir / "sub-01/bem/sub-01-bem-sol.fif"),
        _decoy(fs_dir / "sub-01/bem/sub-01-fwd.fif"),
        _decoy(fs_dir / "sub-01/bem/sub-01-src.fif"),
    ]
    row = run_subject(cfg, discover_recordings(cfg)[0], resolve_stages("motor", None))
    assert row["status"] == "ok", row["error"]

    paths = SubjectPaths(cfg, discover_recordings(cfg)[0])
    manifest = read_json(paths.inputs)
    assert manifest["strict"] is True and manifest["violations"] == []
    read = {Path(f) for files in manifest["files"].values() for f in files}
    assert not read & {d.resolve() for d in decoys}
    assert read >= {raw.resolve()}                                    # the raw recording itself
    assert paths.bem.exists() and paths.src("surface").exists()       # rebuilt, not reused
    assert paths.fwd("surface").exists()
    assert read_json(paths.preprocessing("state", ".json"))["state"]["trans"].endswith("sub-01-trans.fif")


def test_the_manifest_lists_exactly_what_was_read(motor_run):
    cfg, rec, row = motor_run
    manifest = read_json(SubjectPaths(cfg, rec).inputs.with_name("first_run_inputs.json"))
    assert row["inputs_strict"] is True and row["inputs_violations"] == 0
    files = manifest["files"]
    assert [Path(f).name for f in files["raw_bids"]] == [f"{rec.key}_meg.fif"]
    assert [Path(f).name for f in files["declared"]] == ["sub-01-trans.fif"]
    assert set(files) <= {"raw_bids", "sidecar", "declared"}             # nothing else external
    assert manifest["freesurfer_files_read"] > 0


def test_a_blocked_read_fails_the_recording_with_a_clear_reason(response_study):
    """A stage that sneaks in a foreign read fails its recording, loudly and recorded."""
    from cerca_flux.pipeline import Stage

    bids_root, _ = response_study
    foreign = _decoy(bids_root / "derivatives/analysis/sub-01/ses-01/meg/sub-01_fwd.fif")
    cfg = motor_config(response_study, study={"variant": "blocked"})
    rec = discover_recordings(cfg)[0]

    def sneaky(ctx):
        mne.read_forward_solution(foreign, verbose="ERROR")

    row = run_subject(cfg, rec, [Stage("sneaky", "reads another pipeline's forward model", sneaky)])
    assert row["status"] == "failed"
    assert "provenance.strict" in row["error"] and "another pipeline's derivatives" in row["error"]
    assert row["inputs_violations"] == 1
    assert read_json(SubjectPaths(cfg, rec).inputs)["violations"][0].endswith("sub-01_fwd.fif")


def test_an_explicitly_declared_transform_is_honoured_wherever_it_lives(
        motor_run, response_study, tmp_path):
    """The coregistration is the one pre-existing file a user may point the pipeline at."""
    from cerca_flux.context import SubjectContext
    from cerca_flux.source import build_forward
    from cerca_flux.utils import setup_logging

    run_cfg, run_rec, _ = motor_run
    declared = export_trans(SubjectPaths(run_cfg, run_rec).fwd("surface"),
                            response_study[0] / "derivatives" / "coreg" / "sub-01_trans.fif")
    cfg = motor_config(response_study, study={"variant": "declared"},
                       forward={"trans": str(declared)})
    rec = discover_recordings(cfg)[0]
    ctx = SubjectContext(cfg=cfg, rec=rec, paths=SubjectPaths(cfg, rec), logger=setup_logging())
    with audit_inputs(cfg, rec.key) as audit:
        ctx.audit = audit
        forwards = build_forward(ctx)
    manifest = audit.manifest()
    assert "surface" in forwards and manifest["violations"] == []
    assert manifest["files"]["declared"] == [str(declared.resolve())]
    assert ctx.state["trans"] == str(declared)


# -- fingerprints: "the same preprocessing" ----------------------------------------- #


def test_sites_that_differ_only_per_site_share_a_fingerprint(response_study):
    a = motor_config(response_study, epochs={"conditions": {"response": ["resp_T"]}},
                     channels={"manual_bads": {"*": ["B4"]}, "bad_sensor_match": "substring"},
                     study={"crop_start": 0.0})
    b = motor_config(response_study, epochs={"conditions": {"response": ["response/right"]}},
                     channels={"manual_bads": {"*": ["H6"]}, "bad_sensor_match": "token"},
                     study={"crop_start": 400.0, "site": "Other", "subjects": ["07"]})
    assert settings_differences(a, b) == []
    fa, fb = fingerprints(a), fingerprints(b)
    assert fa["fingerprint_shared"] == fb["fingerprint_shared"]
    assert fa["fingerprint_site"] != fb["fingerprint_site"]


@pytest.mark.parametrize("change", [
    {"hfc": {"order": 3}}, {"ica": {"max_exclude": 2}}, {"qc": {"mad_threshold": 5.0}},
    {"epochs": {"reject": {"mag": 4e-11}}}, {"annotate": {"muscle": {"threshold": 4.0}}},
    {"channels": {"use_metadata_bads": False}}, {"motor": {"sensor": {"beta_fmin": 13.0}}},
])
def test_any_shared_setting_changes_the_fingerprint_and_is_reported(response_study, change):
    base = motor_config(response_study)
    changed = motor_config(response_study, **change)
    assert fingerprints(base)["fingerprint_shared"] != fingerprints(changed)["fingerprint_shared"]
    assert len(settings_differences(base, changed)) == 1


def test_presentation_and_path_settings_do_not_change_the_fingerprint(response_study, tmp_path):
    base = motor_config(response_study)
    other = motor_config(response_study, output={"figure_dpi": 50, "report": False},
                         study={"bids_root": str(tmp_path), "variant": "elsewhere"})
    assert fingerprints(base) == fingerprints(other)


def test_cached_results_from_other_settings_are_recognised_as_stale(motor_run, response_study):
    cfg, rec, _ = motor_run
    stamp = {k: v for k, v in
             __import__("cerca_flux.motor", fromlist=["x"]).load_motor_result(
                 SubjectPaths(cfg, rec).motor_sensor)["metrics"].items() if k.startswith("fingerprint")}
    assert stamp == fingerprints(cfg)                                  # stamped when computed
    changed = motor_config(response_study, hfc={"order": 3})
    assert stamp != fingerprints(changed)                              # the same files, new settings


# -- "what changed from the Cerca defaults" ------------------------------------------ #


def test_settings_vs_defaults_flags_exactly_what_differs(response_study):
    rows = {key: (value, default, differs) for key, value, default, differs in
            settings_vs_defaults(motor_config(response_study, hfc={"order": 3}))}
    assert rows["hfc.order"] == (3, 2, True)
    assert rows["hfc.keep_axes"][2] is False


def test_the_settings_listing_shows_changes_and_defaults(response_study):
    text = format_settings(motor_config(response_study, hfc={"order": 3}), changed_only=True)
    assert "* hfc.order" in text and "Cerca default: 2" in text
    assert "hfc.keep_axes" not in text                                  # unchanged, so omitted
    assert "hfc.keep_axes" in format_settings(motor_config(response_study))


def test_the_settings_command(tmp_path, response_study, capsys):
    from cerca_flux.cli import main
    from cerca_flux.config import dump_config
    path = dump_config(motor_config(response_study), tmp_path / "study.yaml")
    assert main(["settings", "--config", str(path), "--changed", "--set", "hfc.order=4"]) == 0
    out = capsys.readouterr().out
    assert "* hfc.order" in out and "4" in out and "fingerprint" in out


def test_a_cached_rerun_rewrites_the_manifest_but_the_transform_is_remembered(motor_run):
    """The manifest describes the latest run; the transform is persisted with the recording."""
    from cerca_flux.compare import _inputs_summary
    cfg, rec, _ = motor_run
    run_subject(cfg, rec, resolve_stages("motor", None))             # everything cached
    paths = SubjectPaths(cfg, rec)
    assert "declared" not in read_json(paths.inputs)["files"]         # nothing needed the transform
    assert _inputs_summary(paths)["inputs_transform"] == "sub-01-trans.fif"


def test_results_made_without_enforcement_are_stale_for_a_strict_comparison(response_study):
    """A non-strict run may have read anything; its cached results must not pass as clean."""
    strict = motor_config(response_study)
    observe = motor_config(response_study, provenance={"strict": False})
    assert fingerprints(strict)["fingerprint_shared"] != fingerprints(observe)["fingerprint_shared"]
    assert len(settings_differences(strict, observe)) == 1
