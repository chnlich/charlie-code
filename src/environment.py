"""Local subprocess executor.

Each bash command runs in its own fresh subprocess. There is no persistent shell,
so shell state (cwd via `cd`, exported vars) does not carry over between commands.
"""

import subprocess


class Environment:
    def __init__(self, cwd, timeout):
        self.cwd = cwd
        self.timeout = timeout

    def execute(self, command):
        """Run one bash command and return its combined output and exit code."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (exc.stderr or "")
            return {
                "output": partial + f"\n[command timed out after {self.timeout}s]",
                "returncode": -1,
            }
        return {"output": result.stdout + result.stderr, "returncode": result.returncode}
