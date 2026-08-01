"""Local subprocess executor.

Each bash command runs in its own fresh subprocess, in its own process group
(`start_new_session=True`), with stdout/stderr captured to a per-command log file
instead of a pipe. There is no persistent shell, so shell state (cwd via `cd`,
exported vars) does not carry over between commands.

Process governance is two-tier:
- A command that returns inside `timeout` has its own process group reaped right
  away, on every normal return. This is what kills `cmd &` survivors that share
  the command's group.
- A command still running at `timeout` is DEMOTED, not killed: its group is
  registered on `self.roster` and left alone until `sweep()` runs at the end of
  the episode.

A command that escapes both tiers via an explicit `setsid` (a new session, hence a
different pgid) leaves harness jurisdiction by design -- that is the documented way
to start a real background service meant to outlive the run.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

# A backgrounded `setsid ... &` survivor is forked by the shell into the SAME group
# as the command (job control is off for a non-interactive shell), and only leaves
# it once its own setsid() syscall actually runs. Reaping the group the instant the
# shell returns can race that in-flight setsid() and kill the escaping process
# before it detaches. This grace period gives it room to finish detaching first.
_REAP_GRACE_SECONDS = 0.1


def _killpg(pgid):
    """SIGKILL a process group; a group with no living members is a silent no-op."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Environment:
    def __init__(self, cwd, timeout):
        self.cwd = cwd
        self.timeout = timeout
        self.log_dir = tempfile.mkdtemp(prefix="charlie-code-log-")
        self.roster = []
        self._n_commands = 0

    def execute(self, command):
        """Run one bash command and return its combined output and exit code."""
        self._n_commands += 1
        log_path = os.path.join(self.log_dir, f"cmd-{self._n_commands}.log")

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

        try:
            returncode = proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            # start_new_session=True makes the command its own session and process
            # group leader, so its pgid is always its own pid. `proc` itself is kept
            # so sweep() can reap it through subprocess's own bookkeeping instead of
            # a raw os.waitpid, which would otherwise leave it a zombie forever
            # (we, not init, are its parent).
            self.roster.append({
                "pgid": proc.pid, "pid": proc.pid, "logfile": log_path, "proc": proc,
            })
            output = Path(log_path).read_text(errors="replace")
            marker = (
                f"\n[command timed out after {self.timeout}s: still running as "
                f"pid {proc.pid}, log at {log_path}. It has been demoted to the "
                f"background rather than killed. Polling it, doing other work and "
                f"checking back later, and killing it yourself are all equally "
                f"fine next steps.]"
            )
            return {"output": output + marker, "returncode": -1}

        time.sleep(_REAP_GRACE_SECONDS)
        _killpg(proc.pid)
        return {"output": Path(log_path).read_text(errors="replace"), "returncode": returncode}

    def sweep(self):
        """SIGKILL every registered process group. Call on every controlled exit."""
        for job in self.roster:
            _killpg(job["pgid"])
            job["proc"].wait()
        self.roster = []

    def cleanup_log_dir(self):
        """Remove this run's log directory. Only safe to call after a clean exit."""
        shutil.rmtree(self.log_dir)
