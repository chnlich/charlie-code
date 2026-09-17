"""Local subprocess executor.

Each bash command runs in its own fresh subprocess, in its own process group
(`start_new_session=True`), with stdout/stderr captured to a per-command log file
instead of a pipe. There is no persistent shell, so shell state (cwd via `cd`,
exported vars) does not carry over between commands.

A command runs in the foreground until it exits. While it runs, the executor
reports progress: at each `progress_notices_seconds` tick it emits a
`command_progress` event (the harness renders it as a chat note), and a command
still running at `kill_after_seconds` is terminated: SIGTERM to its process
group, SIGKILL after a short grace period if it is still there. The observation
then carries the output so far plus a note naming the cap. There is no
background demotion: a command either returns on its own or is terminated, so
nothing outlives the call except what an explicit `setsid` detaches.

A command returning on its own has its process group reaped right away, which
kills any `cmd &` survivors sharing the group. A command that escapes via an
explicit `setsid` (a new session, hence a different pgid) leaves harness
jurisdiction by design -- that is the documented way to start a real background
service meant to outlive the run.

Every command's full output is written to the run's session log directory, named
s-<step>-<call>.log, and never deleted: the transcript's result stubs point at
these files, so reading an old observation back never means re-running the
command. The directory is created by main.py and passed in; nothing is removed at
exit.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

# A backgrounded `setsid ... &` survivor is forked by the shell into the SAME group
# as the command (job control is off for a non-interactive shell), and only leaves
# it once its own setsid() syscall actually runs. Reaping the group the instant the
# shell returns can race that in-flight setsid() and kill the escaping process
# before it detaches. This grace period gives it room to finish detaching first.
_REAP_GRACE_SECONDS = 0.1

# Time a terminated command gets between SIGTERM and SIGKILL: enough for a
# compiler or test runner to flush its buffers, short enough that a hung command
# costs seconds past the cap, not minutes. Same shape as the harness's own stop
# path, which SIGTERMs charlie-code and SIGKILLs it 5 seconds later.
_TERM_GRACE_SECONDS = 5


def _signal_group(pgid, sig):
    """Send `sig` to a process group; a group with no living members is a silent no-op."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _duration_label(seconds):
    """Whole minutes read as minutes ("15 min"); anything else stays in seconds."""
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60} min"
    return f"{seconds:g} s"


class Environment:
    def __init__(self, cwd, progress_notices_seconds, kill_after_seconds, log_dir,
                 emit=None):
        ticks = [*progress_notices_seconds, kill_after_seconds]
        if any(earlier >= later for earlier, later in zip(ticks, ticks[1:])):
            raise ValueError(
                "progress_notices_seconds must increase and stay below "
                f"kill_after_seconds, got {list(progress_notices_seconds)} and "
                f"{kill_after_seconds}"
            )
        self.cwd = cwd
        self.progress_notices_seconds = list(progress_notices_seconds)
        self.kill_after_seconds = kill_after_seconds
        # The run's session log directory, created by main.py. Command logs land
        # here and stay.
        self.log_dir = str(log_dir)
        # Sink for command_progress events; Agent hands its own emit over so the
        # progress of a command lands in the same stream as its command event.
        self.emit = emit
        # The Popen being waited on, so a signal handler can kill it mid-wait.
        self._running = None

    def execute(self, command, step, call):
        """Run one bash command and return its combined output and exit code.

        The full output goes to <log_dir>/s-<step>-<call>.log and stays there
        after the run, so the transcript's stub line can name it as the way to
        read the observation back. A command still running at the cap is
        terminated and its output gets the termination note appended.
        """
        log_path = os.path.join(self.log_dir, f"s-{step}-{call}.log")

        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=self.cwd,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        self._running = proc
        try:
            killed = self._wait_reporting_progress(proc, step, call, log_path)
        finally:
            self._running = None

        if not killed:
            time.sleep(_REAP_GRACE_SECONDS)
        # start_new_session=True makes the command its own process group leader,
        # so its pgid is its pid. Reaping the group collects `cmd &` survivors.
        _signal_group(proc.pid, signal.SIGKILL)

        output = Path(log_path).read_text(errors="replace")
        if killed:
            output += (
                f"\n[terminated after {_duration_label(self.kill_after_seconds)}: "
                f"still running as pid {proc.pid}; output so far above; log: "
                f"{log_path}. Foreground commands are for work expected within "
                f"5 min; start longer work detached with setsid nohup and wait "
                f"for it as your role prompt describes.]"
            )
        return {"output": output, "returncode": proc.returncode, "log_path": log_path}

    def _wait_reporting_progress(self, proc, step, call, log_path):
        """Wait for `proc`, emitting a progress event at each notice tick.

        Returns False when the command exited on its own, True when it was still
        running at the cap and has been terminated.
        """
        event_id = f"s-{step}-{call}"
        started = time.monotonic()
        for notice in self.progress_notices_seconds:
            if _exited_before(proc, started + notice):
                return False
            self._emit_progress(step, event_id, notice, proc.pid, log_path, killed=False)
        if _exited_before(proc, started + self.kill_after_seconds):
            return False
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_group(proc.pid, signal.SIGKILL)
            proc.wait()
        self._emit_progress(step, event_id, self.kill_after_seconds, proc.pid, log_path,
                            killed=True)
        return True

    def _emit_progress(self, step, event_id, elapsed_seconds, pid, log_path, killed):
        if self.emit is None:
            return
        self.emit({"type": "command_progress", "step": step, "id": event_id,
                   "elapsed_seconds": elapsed_seconds, "pid": pid, "log": log_path,
                   "killed": killed})

    def kill_running(self):
        """SIGKILL the process group of the command being waited on, if any.

        Meant for a signal handler: the interrupted wait in `execute` observes
        the exit and returns, so no reaping happens here.
        """
        if self._running is not None:
            _signal_group(self._running.pid, signal.SIGKILL)


def _exited_before(proc, deadline):
    """True when `proc` exits before the monotonic `deadline`."""
    try:
        proc.wait(timeout=max(deadline - time.monotonic(), 0))
    except subprocess.TimeoutExpired:
        return False
    return True
