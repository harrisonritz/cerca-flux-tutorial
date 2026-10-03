"""run/run_motor_compare.sh: detaching, ordering, argument pass-through, stopping, failures.

``cerca-flux`` is replaced by a stub that records its arguments, so only the script's own
behaviour is under test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run" / "run_motor_compare.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

STUB = """#!/bin/bash
echo "$*" >> "$FAKE_LOG"
cmd="$1"
case "$cmd" in
  check)    if [ -n "$FAKE_FAIL_CHECK" ] && [[ "$*" == *"$FAKE_FAIL_CHECK"* ]]; then
              echo "  (study): not ready"; exit 1; fi
            echo "ready"; exit 0 ;;
  settings) echo "  * hfc.order  3"; exit 0 ;;
  run)      sleep "${FAKE_SLEEP:-0}"
            if [ -n "$FAKE_FAIL_RUN" ] && [[ "$*" == *"$FAKE_FAIL_RUN"* ]]; then exit 1; fi
            exit 0 ;;
  compare)  exit 0 ;;
esac
exit 0
"""


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "fake_cerca_flux.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    return {
        **os.environ, "CERCA_FLUX": str(stub), "FAKE_LOG": str(tmp_path / "calls.log"),
        "MOTOR_COMPARE_LOGS": str(tmp_path / "logs"), "TSX_DIR": str(tmp_path / "TSX"),
        "STUB": str(stub),
    }


def script(env, *args, check=False):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True,
                          timeout=60, check=check)


def calls(env) -> list[str]:
    path = Path(env["FAKE_LOG"])
    return path.read_text().splitlines() if path.exists() else []


def wait_until_finished(env, variant="default", timeout=40.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "not running" in script(env, "status", "--variant", variant).stdout:
            return
        time.sleep(0.3)
    raise AssertionError("the background run did not finish")


def test_start_returns_immediately_and_runs_oxford_then_princeton_then_compare(env, tmp_path):
    env["FAKE_SLEEP"] = "2"
    started = time.time()
    result = script(env, "start", "--variant", "v1", "--set", "hfc.order=3", "--jobs", "3")
    assert result.returncode == 0 and "Started in the background" in result.stdout
    assert time.time() - started < 6                      # it did not wait for the 4 s of work
    assert "RUNNING" in script(env, "status", "--variant", "v1").stdout
    wait_until_finished(env, "v1")

    runs = [c for c in calls(env) if c.split()[0] in {"run", "compare"}]
    assert [c.split()[0] for c in runs] == ["run", "run", "compare"]
    oxford, princeton, compare = runs
    assert "configs/motor/oxford.yaml" in oxford and "configs/motor/princeton.yaml" in princeton
    for line in runs:                                      # both sites, and the comparison
        assert "--variant v1" in line and "--set hfc.order=3" in line
    assert "--preset motor" in oxford and "--preset motor" in princeton
    assert "--n-jobs 3" in princeton and "--n-jobs" not in oxford
    assert "--reference configs/motor/oxford.yaml --cohort configs/motor/princeton.yaml" in compare

    status = script(env, "status", "--variant", "v1").stdout
    assert "oxford: done" in status and "princeton: done" in status and "FINISHED" in status
    assert (tmp_path / "logs" / "v1" / "latest.log").exists()
    assert "hfc.order" in script(env, "log", "--variant", "v1").stdout   # the settings are logged


def test_it_checks_the_setup_before_detaching(env):
    script(env, "start", "--variant", "v2")
    wait_until_finished(env, "v2")
    kinds = [c.split()[0] for c in calls(env)]
    assert kinds[:2] == ["check", "check"]                 # both sites, before any processing
    assert kinds.index("run") > 1


def test_an_unready_oxford_reference_stops_it_before_anything_runs(env):
    env["FAKE_FAIL_CHECK"] = "oxford"
    result = script(env, "start", "--variant", "v3")
    assert result.returncode != 0 and "nothing to compare against" in result.stderr
    assert not any(c.startswith("run") for c in calls(env))
    assert "no run recorded" in script(env, "status", "--variant", "v3").stdout


def test_unready_princeton_participants_are_skipped_unless_all_are_required(env):
    env["FAKE_FAIL_CHECK"] = "princeton"
    ok = script(env, "start", "--variant", "v4")
    assert ok.returncode == 0 and "will be skipped" in ok.stdout
    wait_until_finished(env, "v4")
    strict = script(env, "start", "--variant", "v5", "--require-all")
    assert strict.returncode != 0 and "Not starting" in strict.stderr


def test_failing_participants_do_not_stop_the_comparison(env):
    env["FAKE_FAIL_RUN"] = "princeton"
    script(env, "start", "--variant", "v6")
    wait_until_finished(env, "v6")
    steps = script(env, "status", "--variant", "v6").stdout
    assert "princeton: finished with failures" in steps and "compare: done" in steps
    assert any(c.startswith("compare") for c in calls(env))


def test_a_failing_reference_is_reported_but_princeton_is_still_processed(env):
    env["FAKE_FAIL_RUN"] = "oxford"
    script(env, "start", "--variant", "v7")
    wait_until_finished(env, "v7")
    steps = script(env, "status", "--variant", "v7").stdout
    assert "oxford: FAILED" in steps and "princeton: done" in steps and "FINISHED with errors" in steps


def test_options_are_passed_through_to_the_right_site(env):
    script(env, "start", "--variant", "v8", "--subjects", "007 008", "--overwrite", "--no-compare")
    wait_until_finished(env, "v8")
    oxford, princeton = [c for c in calls(env) if c.startswith("run")]
    assert "--subjects" not in oxford and "--subjects 007 008" in princeton
    assert "--overwrite" in oxford and "--overwrite" in princeton
    assert not any(c.startswith("compare") for c in calls(env))


def test_stop_ends_the_run_and_every_process_it_started(env):
    env["FAKE_SLEEP"] = "60"
    script(env, "start", "--variant", "v9")
    deadline = time.time() + 10
    while time.time() < deadline and not any(c.startswith("run") for c in calls(env)):
        time.sleep(0.2)
    assert subprocess.run(["pgrep", "-f", env["STUB"]], capture_output=True).returncode == 0
    result = script(env, "stop", "--variant", "v9")
    assert "Stopped" in result.stdout
    time.sleep(1)
    assert subprocess.run(["pgrep", "-f", env["STUB"]], capture_output=True).returncode != 0
    assert "not running" in script(env, "status", "--variant", "v9").stdout
    assert "STOPPED" in script(env, "status", "--variant", "v9").stdout


def test_a_second_start_for_the_same_variant_is_refused_while_it_runs(env):
    env["FAKE_SLEEP"] = "20"
    script(env, "start", "--variant", "v10")
    again = script(env, "start", "--variant", "v10")
    assert again.returncode != 0 and "already running" in again.stdout
    script(env, "stop", "--variant", "v10")


def test_variants_are_independent(env):
    env["FAKE_SLEEP"] = "20"
    script(env, "start", "--variant", "a")
    script(env, "start", "--variant", "b")
    assert "RUNNING" in script(env, "status", "--variant", "a").stdout
    assert "RUNNING" in script(env, "status", "--variant", "b").stdout
    script(env, "stop", "--variant", "a")
    assert "not running" in script(env, "status", "--variant", "a").stdout
    assert "RUNNING" in script(env, "status", "--variant", "b").stdout
    script(env, "stop", "--variant", "b")


@pytest.mark.parametrize("args", [["start", "--variant", "has space"], ["start", "--variant", "a/b"],
                                  ["start", "--jobs", "many"], ["start", "--nonsense"],
                                  ["frobnicate"]])
def test_bad_arguments_are_rejected_up_front(env, args):
    assert script(env, *args).returncode == 2


def test_the_data_folders_are_passed_on_through_the_environment(env, tmp_path):
    probe = tmp_path / "probe_cerca_flux.sh"
    probe.write_text('#!/bin/bash\necho "$1 TSX_DIR=$TSX_DIR TSX_DATA=$TSX_DATA" >> "$FAKE_LOG"\nexit 0\n')
    probe.chmod(0o755)
    env["CERCA_FLUX"] = str(probe)
    script(env, "run", "--variant", "v11", "--data", "/mnt/volume/data", "--tsx", "/proj/TSX", "--no-compare")
    assert any("TSX_DIR=/proj/TSX TSX_DATA=/mnt/volume/data" in c for c in calls(env))


def test_help_describes_every_command(env):
    text = script(env, "--help").stdout
    for word in ("start", "status", "log", "stop", "--variant", "--set", "--jobs", "--data"):
        assert word in text
