"""OPM channel-naming conventions."""

import pytest

from cerca_flux.config import ChannelConfig
from cerca_flux.utils import axis_of, expand_bad_sensors, is_opm, sensor_id


def test_sensor_and_axis_parsing():
    channels = ChannelConfig()
    assert sensor_id("F9 C8 Z") == "F9"
    assert axis_of("F9 C8 Z", channels) == "Z"
    assert axis_of("Trigger 1", channels) is None
    assert is_opm("F9 C8 Z", channels)
    assert not is_opm("BNC 1 Z", channels)


def test_token_matching_does_not_catch_similar_sensor_ids():
    """'B4' must not mark 'B41' bad - the tutorial's substring match would."""
    names = ["B4 A1 X", "B4 A1 Y", "B4 A1 Z", "B41 C2 Z", "H6 D1 Y"]
    token = expand_bad_sensors(names, ["B4"], ChannelConfig(bad_sensor_match="token"))
    assert token == ["B4 A1 X", "B4 A1 Y", "B4 A1 Z"]

    substring = expand_bad_sensors(names, ["B4"], ChannelConfig(bad_sensor_match="substring"))
    assert "B41 C2 Z" in substring


def test_bad_sensor_expands_across_all_three_axes():
    names = [f"{s} A1 {a}" for s in ("B4", "H6", "C1") for a in "XYZ"]
    expanded = expand_bad_sensors(names, ["B4", "H6"], ChannelConfig())
    assert len(expanded) == 6
    assert all(sensor_id(ch) in {"B4", "H6"} for ch in expanded)
