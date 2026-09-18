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
then carries the output so far plus a note naming the cap. A command may
instead be started in the background, where a supervising thread waits on it,
terminates it at the same cap, reaps it the instant it exits, and records the
exit for the caller to collect. Nothing outlives the run except what an
explicit `setsid` detaches.

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

import dataclasses
import os
import signal
import subprocess
import threading
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


@dataclasses.dataclass
class BackgroundTask:
    """A background command: launch identity, plus its exit record once reaped.

    `returncode` is what Popen reports (negative when a signal killed it),
    `duration` the seconds from start to exit, and `killed` True when the cap
    terminated the command; they stay None/False until the supervising thread
    records the exit.
    """

    id: str
    command: str
    pid: int
    log_path: str
    started: float
    returncode: int | None = None
    duration: float | None = None
    killed: bool = False


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
        # Background tasks still running: task id -> (BackgroundTask, Popen).
        self._background = {}
        # Background tasks that exited, in exit order, not yet handed out.
        self._finished = []
        # Guards _background and _finished against the supervising threads.
        self._lock = threading.Lock()

    def cap_label(self):
        """The kill cap as notes and reminders name it ("15 min" at the default)."""
        return _duration_label(self.kill_after_seconds)

    def _spawn(self, command, step, call):
        """Start `command` as its own session with output to its command log.

        Shared by `execute` and `start_background` so the two launch paths
        cannot drift. Returns (Popen, log_path); the log file is closed here,
        the child keeps its duplicate of the fd.
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
        return proc, log_path

    def execute(self, command, step, call):
        """Run one bash command and return its combined output and exit code.

        The full output goes to <log_dir>/s-<step>-<call>.log and stays there
        after the run, so the transcript's stub line can name it as the way to
        read the observation back. A command still running at the cap is
        terminated and its output gets the termination note appended.
        """
        proc, log_path = self._spawn(command, step, call)

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
                f"a minute; run longer work with background=true, and start work "
                f"longer than the cap detached with setsid nohup as your role "
                f"prompt describes.]"
            )
        return {"output": output, "returncode": proc.returncode, "log_path": log_path}

    def start_background(self, command, step, call):
        """Start `command` in the background and return its task at once.

        The task is registered under its `s-<step>-<call>` id and a daemon
        thread supervises the process: it terminates the command at the same
        `kill_after_seconds` cap, reaps it the instant it exits, and records
        the exit for `pop_finished` to hand out. No progress events are
        emitted for a background task.
        """
        proc, log_path = self._spawn(command, step, call)
        task = BackgroundTask(id=f"s-{step}-{call}", command=command, pid=proc.pid,
                              log_path=log_path, started=time.monotonic())
        with self._lock:
            self._background[task.id] = (task, proc)
        threading.Thread(target=self._supervise, args=(task, proc), daemon=True).start()
        return task

    def _supervise(self, task, proc):
        """Wait on a background task, terminate it at the cap, record its exit.

        Runs on the task's daemon thread: this wait is what reaps the child, so
        its pid is gone the moment it exits. It never emits an event and only
        touches `task`, `proc`, and the task collections under the lock.
        """
        killed = not _exited_before(proc, task.started + self.kill_after_seconds)
        if killed:
            _signal_group(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _signal_group(proc.pid, signal.SIGKILL)
                proc.wait()
        else:
            # Same grace as the foreground path: an in-flight `setsid` escape
            # needs room to detach before the group is reaped.
            time.sleep(_REAP_GRACE_SECONDS)
        # Collect `cmd &` survivors sharing the group, as the foreground does.
        _signal_group(proc.pid, signal.SIGKILL)
        with self._lock:
            task.returncode = proc.returncode
            task.duration = time.monotonic() - task.started
            task.killed = killed
            del self._background[task.id]
            self._finished.append(task)

    def running_background(self):
        """The background tasks still running, in start order."""
        with self._lock:
            return [task for task, _ in self._background.values()]

    def finished_pending(self):
        """How many exited background tasks have not been handed out yet."""
        with self._lock:
            return len(self._finished)

    def pop_finished(self, limit=None):
        """Remove and return the first `limit` exited tasks, in exit order.

        All of them when `limit` is None.
        """
        with self._lock:
            handed_out = self._finished if limit is None else self._finished[:limit]
            self._finished = self._finished[len(handed_out):]
            return handed_out

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

        Background command groups are killed here too; their supervising
        threads observe the exits and do the reaping. Meant for a signal
        handler: the interrupted wait in `execute` observes the exit and
        returns, so no reaping happens here. The background snapshot is taken
        without the lock -- a handler that blocked on a lock the interrupted
        main thread holds would deadlock, and `list(dict.values())` is atomic
        under the GIL.
        """
        if self._running is not None:
            _signal_group(self._running.pid, signal.SIGKILL)
        for _, proc in list(self._background.values()):
            _signal_group(proc.pid, signal.SIGKILL)


def _exited_before(proc, deadline):
    """True when `proc` exits before the monotonic `deadline`."""
    try:
        proc.wait(timeout=max(deadline - time.monotonic(), 0))
    except subprocess.TimeoutExpired:
        return False
    return True
