"""Configuration parsing and validation."""

import pytest

from cerca_flux.config import Config, ConfigError, _build, dump_config, load_config


def test_template_is_valid():
    """The shipped starter config must parse and validate as-is."""
    from pathlib import Path

    import yaml

    import cerca_flux

    template = Path(cerca_flux.__file__).parent / "templates" / "study.yaml"
    data = yaml.safe_load(template.read_text())
    data["study"]["bids_root"] = "/tmp/example"
    cfg = _build(Config, data)
    cfg.validate()
    assert [b.name for b in cfg.tfr.bands] == ["slow", "fast"]
    assert [b.name for b in cfg.source.dics.bands] == ["alpha", "gamma"]
    # A YAML float in scientific notation must survive as a float, not a string.
    assert cfg.epochs.reject["mag"] == pytest.approx(1e-11)


def test_unknown_key_is_rejected():
    """A typo must fail loudly rather than be silently ignored."""
    with pytest.raises(ConfigError, match="bids_rooot"):
        _build(Config, {"study": {"bids_rooot": "/tmp"}})


@pytest.mark.parametrize("payload, message", [
    ({"study": {"bids_root": "/tmp"}}, "epochs.conditions is required"),
    ({"study": {"bids_root": "/tmp"}, "epochs": {"conditions": {"a": ["x"]}},
      "mvpa": {"conditions": ["a", "nope"]}}, "not in epochs.conditions"),
    ({"study": {"bids_root": "/tmp"}, "epochs": {"conditions": {"a": ["x"]}},
      "mvpa": {"enabled": False},
      "tfr": {"contrasts": [{"name": "c", "band": "nosuch", "a": "a", "b": "a"}]}},
     "unknown band"),
    ({"study": {"bids_root": "/tmp"}, "epochs": {"conditions": {"a": ["x"]}},
      "mvpa": {"enabled": False}, "qc": {"method": "magic"}}, "qc.method"),
])
def test_validation_errors(payload, message):
    with pytest.raises(ConfigError, match=message):
        _build(Config, payload).validate()


def test_roundtrip_through_yaml(tmp_path, config):
    """A dumped configuration must reload to the same settings."""
    path = dump_config(config, tmp_path / "study.yaml")
    reloaded = load_config(path)
    assert reloaded.epochs.conditions == config.epochs.conditions
    assert reloaded.source.spaces == config.source.spaces
    assert reloaded.bids_root == config.bids_root
