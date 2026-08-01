"""Two-tier process governance: command-level reap vs. episode-level roster sweep.

A command that returns inside `timeout` has its own process group reaped, killing
any `cmd &` survivors sharing it. A command still running at `timeout` is DEMOTED,
not killed, and only cleaned up when `sweep()` runs at the end of the episode. An
explicit `setsid` escapes both tiers by design.
"""

import os
import re
import signal
import time

from environment import Environment


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid_from_output(output):
    match = re.search(r"pid (\d+)", output)
    assert match, output
    return int(match.group(1))


def _log_path_from_output(output):
    match = re.search(r"log at (\S+)\.", output)
    assert match, output
    return match.group(1)


def test_demoted_command_reports_pid_log_path_and_is_still_alive(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=1)

    result = env.execute("sleep infinity")

    assert result["returncode"] == -1
    assert "still running" in result["output"]
    pid = _pid_from_output(result["output"])
    log_path = _log_path_from_output(result["output"])
    assert os.path.isabs(log_path)
    assert os.path.exists(log_path)
    assert _alive(pid)

    env.sweep()
    time.sleep(0.2)
    assert not _alive(pid)


def test_demoted_command_is_not_killed_and_later_finishes_on_its_own(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=1)
    marker = tmp_path / "done"

    result = env.execute(f"sleep 2 && touch {marker}")

    assert result["returncode"] == -1
    pid = _pid_from_output(result["output"])
    assert _alive(pid)

    deadline = time.monotonic() + 6
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "demoted command should have kept running to completion"

    env.sweep()  # reaps the already-exited job; a no-op kill, not a kill-while-running
    assert not _alive(pid)


def test_sweep_kills_every_registered_job_and_clears_the_roster(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=1)
    result = env.execute("sleep infinity")
    pid = _pid_from_output(result["output"])
    assert _alive(pid)
    assert len(env.roster) == 1

    env.sweep()
    time.sleep(0.2)

    assert not _alive(pid)
    assert env.roster == []


def test_command_returning_in_budget_reaps_its_backgrounded_survivor(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=5)

    result = env.execute("sleep 100 & echo $!")

    assert result["returncode"] == 0
    pid = int(result["output"].strip())
    time.sleep(0.3)
    assert not _alive(pid)


def test_setsid_escape_hatch_survives_command_end(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=5)

    result = env.execute("setsid nohup sleep 100 > /dev/null 2>&1 & echo $!")

    assert result["returncode"] == 0
    pid = int(result["output"].strip())
    try:
        time.sleep(0.3)
        assert _alive(pid)
    finally:
        os.kill(pid, signal.SIGKILL)


def test_bare_cat_fails_fast_on_stdin_eof(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=60)

    start = time.monotonic()
    result = env.execute("cat")
    elapsed = time.monotonic() - start

    assert result["returncode"] == 0
    assert elapsed < 5


def test_normal_completion_returns_dict_shape_and_log_content(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=10)

    result = env.execute("echo hello")

    assert result == {"output": "hello\n", "returncode": 0}


def test_cleanup_log_dir_removes_the_run_directory(tmp_path):
    env = Environment(cwd=str(tmp_path), timeout=10)
    env.execute("echo hi")
    assert os.path.isdir(env.log_dir)

    env.cleanup_log_dir()

    assert not os.path.exists(env.log_dir)
