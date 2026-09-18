"""Foreground execution with progress notices and a termination cap.

A command runs until it exits. At each notice tick the executor emits a
command_progress event; a command still running at the cap is terminated
(SIGTERM, then SIGKILL after a grace period) and its observation ends with the
termination note. A command that returns on its own has its process group
reaped, killing any `cmd &` survivors sharing it. An explicit `setsid` escapes
by design.
"""

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from environment import Environment


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _env(tmp_path, notices=(), cap=10, emit=None):
    return Environment(cwd=str(tmp_path), progress_notices_seconds=list(notices),
                       kill_after_seconds=cap, log_dir=str(tmp_path), emit=emit)


def test_command_past_the_cap_reports_each_tick_then_is_terminated(tmp_path):
    events = []
    env = _env(tmp_path, notices=[0.2, 0.4], cap=0.8, emit=events.append)

    start = time.monotonic()
    result = env.execute("echo started; sleep 5", 3, 1)
    elapsed = time.monotonic() - start

    assert elapsed < 2, "SIGTERM ended the command without waiting for the SIGKILL grace"
    assert [(e["type"], e["id"], e["elapsed_seconds"], e["killed"]) for e in events] == [
        ("command_progress", "s-3-1", 0.2, False),
        ("command_progress", "s-3-1", 0.4, False),
        ("command_progress", "s-3-1", 0.8, True),
    ]
    pid = events[0]["pid"]
    assert all(e["step"] == 3 and e["pid"] == pid and e["log"] == result["log_path"]
               for e in events)
    assert result["returncode"] == -signal.SIGTERM
    assert result["output"].startswith("started\n")
    assert f"[terminated after 0.8 s: still running as pid {pid}; " in result["output"]
    assert f"log: {result['log_path']}." in result["output"]
    assert "wait for it as your role prompt describes.]" in result["output"]
    assert not _alive(pid)
    time.sleep(0.2)
    assert not _group_alive(pid)


def test_terminated_command_that_traps_sigterm_exits_on_its_own_terms(tmp_path):
    events = []
    env = _env(tmp_path, notices=[], cap=0.5, emit=events.append)

    start = time.monotonic()
    result = env.execute("trap 'exit 7' TERM; sleep 5 & wait", 1, 1)
    elapsed = time.monotonic() - start

    assert result["returncode"] == 7, "the SIGTERM path let the command exit itself"
    assert elapsed < 3, "no escalation to SIGKILL for a command that exits on SIGTERM"
    assert [e["killed"] for e in events] == [True]
    assert "[terminated after 0.5 s:" in result["output"]


def test_command_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path, monkeypatch):
    import environment

    monkeypatch.setattr(environment, "_TERM_GRACE_SECONDS", 0.3)
    env = _env(tmp_path, notices=[], cap=0.3)

    start = time.monotonic()
    result = env.execute("trap '' TERM; sleep 5 & wait", 1, 1)
    elapsed = time.monotonic() - start

    assert result["returncode"] == -signal.SIGKILL
    assert 0.6 <= elapsed < 3
    assert "[terminated after 0.3 s:" in result["output"]


def test_kill_running_kills_the_command_being_waited_on(tmp_path):
    events = []
    env = _env(tmp_path, notices=[5], cap=30, emit=events.append)
    threading.Timer(0.3, env.kill_running).start()

    start = time.monotonic()
    result = env.execute("sleep 30", 1, 1)

    assert time.monotonic() - start < 5
    assert result["returncode"] == -signal.SIGKILL
    assert "[terminated" not in result["output"], "a kill below the cap is not a cap kill"
    assert events == []
    env.kill_running()  # nothing running: a no-op


def test_command_within_the_first_tick_emits_nothing(tmp_path):
    events = []
    env = _env(tmp_path, notices=[5], cap=10, emit=events.append)

    result = env.execute("echo hi", 1, 1)

    assert result["returncode"] == 0
    assert events == []


def test_notices_must_increase_and_stay_below_the_cap(tmp_path):
    with pytest.raises(ValueError, match="stay below kill_after_seconds"):
        _env(tmp_path, notices=[60, 300], cap=200)
    with pytest.raises(ValueError, match="must increase"):
        _env(tmp_path, notices=[300, 60], cap=900)


def test_command_returning_in_budget_reaps_its_backgrounded_survivor(tmp_path):
    env = _env(tmp_path, cap=5)

    result = env.execute("sleep 100 & echo $!", 1, 1)

    assert result["returncode"] == 0
    pid = int(result["output"].strip())
    time.sleep(0.3)
    assert not _alive(pid)


def test_setsid_escape_hatch_survives_command_end(tmp_path):
    env = _env(tmp_path, cap=5)

    result = env.execute("setsid nohup sleep 100 > /dev/null 2>&1 & echo $!", 1, 1)

    assert result["returncode"] == 0
    pid = int(result["output"].strip())
    try:
        time.sleep(0.3)
        assert _alive(pid)
    finally:
        os.kill(pid, signal.SIGKILL)


def test_bare_cat_fails_fast_on_stdin_eof(tmp_path):
    env = _env(tmp_path, cap=60)

    start = time.monotonic()
    result = env.execute("cat", 1, 1)
    elapsed = time.monotonic() - start

    assert result["returncode"] == 0
    assert elapsed < 5


def test_normal_completion_returns_dict_shape_and_log_content(tmp_path):
    env = _env(tmp_path)

    result = env.execute("echo hello", 1, 1)

    assert result == {"output": "hello\n", "returncode": 0,
                      "log_path": str(tmp_path / "s-1-1.log")}


def test_command_log_is_written_to_the_given_dir_and_never_deleted(tmp_path):
    log_dir = tmp_path / "run-logs"
    log_dir.mkdir()
    env = Environment(cwd=str(tmp_path), progress_notices_seconds=[], kill_after_seconds=10,
                      log_dir=str(log_dir))

    result = env.execute("echo hello", 3, 2)

    assert result == {"output": "hello\n", "returncode": 0,
                      "log_path": str(log_dir / "s-3-2.log")}
    # Session logs are never removed: the transcript stubs point at them.
    assert (log_dir / "s-3-2.log").read_text() == "hello\n"


def test_start_background_returns_at_once_and_lists_the_running_task(tmp_path):
    env = _env(tmp_path, cap=10)
    try:
        start = time.monotonic()
        task = env.start_background("sleep 5", 1, 1)
        elapsed = time.monotonic() - start

        assert elapsed < 1, "start_background must not wait on the command"
        assert task.id == "s-1-1"
        assert _alive(task.pid)
        assert task.log_path.endswith("s-1-1.log")
        assert [t.id for t in env.running_background()] == ["s-1-1"]
        assert env.finished_pending() == 0
    finally:
        env.kill_running()


def test_background_task_that_exits_is_reaped_and_handed_out_once(tmp_path):
    env = _env(tmp_path, cap=10)
    try:
        task = env.start_background("echo out; exit 3", 1, 2)

        deadline = time.monotonic() + 1
        while env.finished_pending() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not _alive(task.pid), "the supervising thread reaps the instant it exits"
        assert env.finished_pending() == 1
        (finished,) = env.pop_finished()
        assert finished is task
        assert finished.returncode == 3
        assert finished.killed is False
        assert finished.duration is not None
        assert Path(task.log_path).read_text() == "out\n"
        assert env.pop_finished() == []
    finally:
        env.kill_running()


def test_background_task_past_the_cap_is_terminated_and_recorded(tmp_path):
    env = _env(tmp_path, cap=0.5)
    try:
        task = env.start_background("sleep 30", 1, 3)

        finished = []
        deadline = time.monotonic() + 3
        while not finished and time.monotonic() < deadline:
            finished = env.pop_finished()
            if not finished:
                time.sleep(0.05)

        assert [t.id for t in finished] == ["s-1-3"]
        assert finished[0].killed is True
        assert finished[0].returncode == -signal.SIGTERM
        assert not _alive(task.pid)
    finally:
        env.kill_running()


def test_kill_running_kills_background_groups_too(tmp_path):
    env = _env(tmp_path, cap=10)
    try:
        task = env.start_background("sleep 30", 1, 4)
        env.kill_running()

        deadline = time.monotonic() + 1
        while env.finished_pending() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not _alive(task.pid)
        time.sleep(0.1)
        assert not _group_alive(task.pid)
        assert env.running_background() == []
        (finished,) = env.pop_finished()
        assert finished.returncode == -signal.SIGKILL
    finally:
        env.kill_running()


def test_background_shell_exit_reaps_its_backgrounded_survivor(tmp_path):
    env = _env(tmp_path, cap=10)
    try:
        task = env.start_background("sleep 100 & echo $!", 1, 5)

        deadline = time.monotonic() + 1
        while env.finished_pending() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not _alive(task.pid)
        pid = int(Path(task.log_path).read_text().strip())
        time.sleep(0.3)
        assert not _alive(pid)
    finally:
        env.kill_running()
