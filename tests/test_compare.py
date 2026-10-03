"""The Oxford-vs-Princeton overlay: styling guarantees, limits, and the full CLI path."""

from __future__ import annotations

import colorsys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cerca_flux.cli import main
from cerca_flux.compare import (
    COHORT_LINEWIDTH, MEASURES, REFERENCE_LINEWIDTH, Entry, _shared_limits, cohort_colours,
    compare_motor, draw_measure, metrics_table, plot_overlay, plot_small_multiples,
)
from cerca_flux.config import dump_config

from ._motor_support import motor_config

TIMES = np.linspace(-1.0, 1.2, 441)
ERF = MEASURES[0]


def _entry(group: str, label: str, peak: float, status: str = "ok") -> Entry:
    """A participant whose ERF is a Gaussian bump of the given height."""
    curve = 90 + peak * np.exp(-0.5 * (TIMES / 0.1) ** 2)
    sensor = {"erf_times": TIMES, "erf_rms_ft": curve, "erf_z": curve - 90,
              "tfr_times": TIMES[::2], "beta_percent": -peak * np.exp(-0.5 * (TIMES[::2] / 0.3) ** 2),
              "metrics": {"motor_n_epochs": 100, "motor_peak_erf_rms_fT": 90 + peak}}
    source = {"lcmv_times": TIMES, "lcmv_z": curve - 90, "dics_times": TIMES[::20],
              "dics_db": -peak / 50 * np.ones_like(TIMES[::20]), "label": "precentral-lh",
              "metrics": {"motor_source_n_epochs": 100}}
    return Entry(group, label, f"sub-{label}", sensor, source, status)


@pytest.fixture
def entries():
    reference = [_entry("reference", "01", 260.0)]
    cohort = [_entry("cohort", f"{i:03d}", 80.0 + 7 * i) for i in range(7, 19)]
    return reference, cohort


def _luminance(rgb) -> float:
    return colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(rgb))[1]


# -- the requested styling ---------------------------------------------------- #


def test_reference_is_thick_black_and_cohort_is_thinner_and_coloured(entries):
    reference, cohort = entries
    colours = cohort_colours([e.label for e in cohort])
    fig, ax = plt.subplots()
    cohort_lines, reference_lines = draw_measure(ax, ERF, reference, cohort, colours)

    assert len(reference_lines) == 1 and len(cohort_lines) == len(cohort)
    (ref,) = reference_lines
    assert matplotlib.colors.to_rgb(ref.get_color()) == (0.0, 0.0, 0.0)
    assert ref.get_linewidth() == REFERENCE_LINEWIDTH
    for line in cohort_lines:
        assert line.get_linewidth() == COHORT_LINEWIDTH < ref.get_linewidth() / 2
        assert matplotlib.colors.to_rgb(line.get_color()) != (0.0, 0.0, 0.0)
        assert ref.get_zorder() > line.get_zorder()            # the reference sits on top
    plt.close(fig)


def test_cohort_colours_are_distinct_and_never_mistaken_for_black(entries):
    _, cohort = entries
    colours = cohort_colours([e.label for e in cohort])
    rgb = [tuple(np.round(c[:3], 4)) for c in colours.values()]
    assert len(set(rgb)) == len(cohort)                        # one colour per participant
    assert min(_luminance(c) for c in colours.values()) > 0.2  # the ramp stays off black
    assert max(_luminance(c) for c in colours.values()) < 0.8  # ... and off the white page


def test_a_participant_keeps_their_colour_when_someone_else_is_missing():
    """Colour follows the participant, not their rank among those who made it."""
    labels = ["007", "008", "009", "010"]
    full = cohort_colours(labels)
    # compare_motor builds colours from *every* requested participant, so dropping 008
    # from the plot leaves the others where they were.
    drawn = {k: v for k, v in full.items() if k != "008"}
    assert drawn["009"] == full["009"] and drawn["010"] == full["010"]
    assert cohort_colours(["007", "009", "010"])["009"] != full["009"]  # the trap this avoids


# -- limits ------------------------------------------------------------------- #


def test_shared_limits_always_contain_the_whole_reference(entries):
    reference, cohort = entries
    lo, hi, off = _shared_limits(ERF, reference, cohort)
    ref = reference[0].sensor["erf_rms_ft"]
    assert lo <= ref.min() and hi >= ref.max()
    assert off == set()


def test_one_runaway_participant_is_flagged_not_allowed_to_flatten_the_rest(entries):
    reference, cohort = entries
    cohort = cohort + [_entry("cohort", "099", 50_000.0)]
    lo, hi, off = _shared_limits(ERF, reference, cohort)
    assert off == {"099"}
    assert hi < 1000                                            # the outlier did not set the scale
    assert hi >= reference[0].sensor["erf_rms_ft"].max()


# -- files -------------------------------------------------------------------- #


def test_overlay_and_small_multiples_are_written(entries, tmp_path):
    reference, cohort = entries
    cohort = cohort + [Entry("cohort", "020", "sub-020", status="no motor output")]
    colours = cohort_colours([e.label for e in cohort])
    names = ("Oxford", "Princeton")
    overlay = plot_overlay(reference, cohort, colours, names, "hfc2", tmp_path / "o.png", 60)
    small = plot_small_multiples(ERF, reference, cohort, colours, names, "hfc2",
                                 tmp_path / "s.png", 60)
    for path in (overlay, small):
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_small_multiples_skip_a_measure_nobody_has(entries, tmp_path):
    reference, cohort = entries
    for entry in cohort:
        entry.source = None
    colours = cohort_colours([e.label for e in cohort])
    assert plot_small_multiples(MEASURES[3], reference, cohort, colours, ("A", "B"), "v",
                                tmp_path / "x.png", 60) is None


def test_metrics_table_names_who_is_missing(entries):
    reference, cohort = entries
    cohort = cohort + [Entry("cohort", "020", "sub-020", status="no motor output",
                             note="the pipeline has not been run for this recording")]
    table = metrics_table(("Oxford", "Princeton"), reference + cohort)
    assert list(table["dataset"].unique()) == ["Oxford", "Princeton"]
    missing = table[table["subject"] == "020"].iloc[0]
    assert missing["status"] == "no motor output" and "not been run" in missing["note"]
    assert table[table["subject"] == "010"].iloc[0]["status"] == "ok"
    assert "motor_peak_erf_rms_fT" in table.columns


# -- end to end on the cached pipeline outputs -------------------------------- #


def test_compare_motor_reads_the_cached_stage_outputs(motor_run, response_study, tmp_path):
    reference_cfg = motor_config(response_study, study={"site": "Oxford"})
    cohort_cfg = motor_config(response_study, study={"site": "Princeton",
                                                     "subjects": ["01", "02"]})
    outputs = compare_motor(reference_cfg, cohort_cfg, out_dir=tmp_path)

    assert outputs["overlay"].exists()
    for key in ("erf_rms", "erf_snr", "beta", "lcmv", "dics"):
        assert outputs[f"small_multiples_{key}"].exists()
    table = pd.read_csv(outputs["metrics"], dtype={"subject": str})
    by_subject = table.set_index(["dataset", "subject"])
    assert by_subject.loc[("Oxford", "01"), "status"] == "ok"
    assert by_subject.loc[("Princeton", "01"), "status"] == "ok"
    # A requested participant with no BIDS recording is reported, not silently dropped.
    assert by_subject.loc[("Princeton", "02"), "status"] == "no BIDS recording"
    assert by_subject.loc[("Oxford", "01"), "motor_peak_erf_snr"] > 0


def test_a_reference_that_has_not_been_run_is_a_clear_error(motor_run, response_study, tmp_path):
    never_run = motor_config(response_study, study={"variant": "never-run"})
    cohort_cfg = motor_config(response_study)
    with pytest.raises(RuntimeError, match=r"cerca-flux run --preset motor"):
        compare_motor(never_run, cohort_cfg, out_dir=tmp_path)


def test_the_compare_command_applies_variant_and_overrides(motor_run, response_study, tmp_path):
    reference = dump_config(motor_config(response_study, study={"site": "Oxford"}),
                            tmp_path / "ref.yaml")
    cohort = dump_config(motor_config(response_study, study={"site": "Princeton"}),
                         tmp_path / "cohort.yaml")
    out = tmp_path / "figs"
    code = main(["compare", "--reference", str(reference), "--cohort", str(cohort),
                 "--variant", "motor-test", "--set", "output.figure_dpi=50",
                 "--out", str(out), "--no-small-multiples"])
    assert code == 0
    assert (out / "motor_response_overlay.png").exists()
    assert (out / "motor_response_metrics.csv").exists()
    assert not list(out.glob("motor_response_small_multiples_*"))


def test_the_compare_command_reports_a_missing_reference_without_a_traceback(
        motor_run, response_study, tmp_path):
    reference = dump_config(motor_config(response_study, study={"variant": "nothing-here"}),
                            tmp_path / "ref.yaml")
    cohort = dump_config(motor_config(response_study), tmp_path / "cohort.yaml")
    with pytest.raises(SystemExit, match="compare: no motor results for the reference"):
        main(["compare", "--reference", str(reference), "--cohort", str(cohort),
              "--out", str(tmp_path / "o")])
