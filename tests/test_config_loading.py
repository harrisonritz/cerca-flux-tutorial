"""YAML inheritance, ${VAR} expansion, dotted overrides and variants."""

from pathlib import Path

import pytest
import yaml

from cerca_flux.config import (
    Config, ConfigError, _build, default_tsx_root, expand_variables, load_config,
    parse_overrides,
)

BASE = {
    "study": {"bids_root": "/data/bids"},
    "epochs": {"conditions": {"response": ["resp_T"]}},
    "mvpa": {"enabled": False},
}


def _write(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


# -- ${VAR} expansion --------------------------------------------------------- #


def test_variable_from_the_environment_wins_over_the_default():
    env = {"TSX_DIR": "/tsx", "OXFORD": "/elsewhere"}
    assert expand_variables("${OXFORD:-${TSX_DIR}/oxford}/bids", env) == "/elsewhere/bids"
    assert expand_variables("${MISSING:-${TSX_DIR}/oxford}/bids", env) == "/tsx/oxford/bids"


def test_unset_variable_without_default_is_a_clear_error():
    with pytest.raises(ConfigError, match=r"\$\{NOPE\} is not set"):
        expand_variables({"study": {"bids_root": "${NOPE}/x"}}, {})


def test_expansion_leaves_format_placeholders_alone():
    """``{subject}`` is a pipeline placeholder, not a shell variable."""
    template = "sub-{subject}_ses-{session}"
    assert expand_variables(template, {}) == template


def test_expansion_walks_nested_structures():
    out = expand_variables({"a": ["${X}/1", {"b": "${X}/2"}], "n": 3}, {"X": "/r"})
    assert out == {"a": ["/r/1", {"b": "/r/2"}], "n": 3}


def test_tsx_root_defaults_to_the_folder_above_the_repository(monkeypatch, tmp_path):
    """The batch layout keeps this repository next to data/, mne-opm/ and TSX_OPM/."""
    monkeypatch.delenv("TSX_DIR", raising=False)
    path = _write(tmp_path / "study.yaml", {
        **BASE, "study": {"bids_root": "${TSX_DIR}/data/TSX/bids"},
    })
    cfg = load_config(path)
    assert cfg.bids_root == (default_tsx_root() / "data" / "TSX" / "bids").resolve()
    assert default_tsx_root() == Path(__file__).resolve().parents[2]


def test_tsx_dir_in_the_environment_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("TSX_DIR", str(tmp_path / "TSX"))
    path = _write(tmp_path / "study.yaml", {
        **BASE, "study": {"bids_root": "${TSX_DIR}/data/TSX/bids"},
    })
    assert load_config(path).bids_root == tmp_path / "TSX" / "data" / "TSX" / "bids"


# -- extends ------------------------------------------------------------------ #


def test_extends_merges_mappings_and_replaces_lists(tmp_path):
    _write(tmp_path / "base.yaml", {
        "study": {"bids_root": "/base", "line_freq": 50, "subjects": ["01", "02"]},
        "channels": {"manual_bads": {"*": ["B4"]}},
        "mvpa": {"enabled": False},
    })
    _write(tmp_path / "site.yaml", {
        "extends": "base.yaml",
        "study": {"bids_root": "/site", "subjects": ["07"]},
        "channels": {"manual_bads": {"07": ["H6"]}},
        "epochs": {"conditions": {"response": ["resp_T"]}},
    })
    cfg = load_config(tmp_path / "site.yaml")
    assert cfg.study.bids_root == "/site"          # child wins
    assert cfg.study.line_freq == 50               # inherited
    assert cfg.study.subjects == ["07"]            # lists are replaced, not concatenated
    assert cfg.channels.manual_bads == {"*": ["B4"], "07": ["H6"]}   # mappings merge


def test_extends_cycle_is_rejected(tmp_path):
    _write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    _write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


# -- overrides and variants --------------------------------------------------- #


def test_overrides_are_parsed_as_yaml_and_nest():
    assert parse_overrides(["hfc.order=3", "ica.detect_ecg=false", "erf.crop=[-0.1, 0.4]",
                            "study.crop_duration=null"]) == {
        "hfc": {"order": 3}, "ica": {"detect_ecg": False},
        "erf": {"crop": [-0.1, 0.4]}, "study": {"crop_duration": None},
    }


@pytest.mark.parametrize("bad", ["hfc.order", "=3", "  =x"])
def test_malformed_override_is_rejected(bad):
    with pytest.raises(ConfigError, match="section.key=value"):
        parse_overrides([bad])


def test_overrides_beat_the_file_and_the_variant_names_the_output_folder(tmp_path):
    path = _write(tmp_path / "study.yaml", BASE)
    cfg = load_config(path, overrides=parse_overrides(["hfc.order=3"]), variant="hfc3")
    assert cfg.hfc.order == 3
    assert cfg.study.variant == "hfc3"
    assert cfg.deriv_root.name == "cerca-flux_hfc3"
    # Different variants never share a derivatives folder.
    other = load_config(path, variant="hfc2")
    assert other.deriv_root != cfg.deriv_root
    # No variant keeps the historical location.
    assert load_config(path).deriv_root.name == "cerca-flux"


@pytest.mark.parametrize("variant", ["has space", "../escape", "under_score", "-lead"])
def test_unsafe_variant_names_are_rejected(tmp_path, variant):
    path = _write(tmp_path / "study.yaml", BASE)
    with pytest.raises(ConfigError, match="study.variant"):
        load_config(path, variant=variant)


def test_unquoted_subject_labels_are_caught(tmp_path):
    """YAML reads 007 as the integer 7 - a different BIDS subject."""
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(BASE).replace("study:", "study:\n  subjects: [007, 010]", 1))
    with pytest.raises(ConfigError, match="quoted strings"):
        load_config(path)


# -- crop and motor validation ------------------------------------------------ #


def test_crop_settings_are_validated():
    with pytest.raises(ConfigError, match="crop_start"):
        _build(Config, {**BASE, "study": {"bids_root": "/x", "crop_start": -1}}).validate()
    with pytest.raises(ConfigError, match="crop_duration"):
        _build(Config, {**BASE, "study": {"bids_root": "/x", "crop_duration": 0}}).validate()


@pytest.mark.parametrize("motor, message", [
    ({"enabled": True, "conditions": ["nope"]}, "not in epochs.conditions"),
    ({"enabled": True, "source": {"space": "volume"}, "_spaces": ["surface"]},
     "must also be listed in source.spaces"),
    ({"enabled": True, "source": {"label": "precentral"}}, "must end in -lh or -rh"),
    ({"enabled": True, "sensor": {"erf_baseline": [0.1, -0.1]}}, "start < end"),
])
def test_motor_validation_errors(motor, message):
    payload = {**BASE, "study": {"bids_root": "/x"}}
    motor = dict(motor)
    spaces = motor.pop("_spaces", None)
    if spaces:
        payload["source"] = {"spaces": spaces}
    payload["motor"] = motor
    with pytest.raises(ConfigError, match=message):
        _build(Config, payload).validate()


def test_motor_is_off_by_default_so_existing_studies_are_unchanged():
    assert _build(Config, BASE).motor.enabled is False
