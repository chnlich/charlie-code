"""Grader for repro_idempotent: running twice must produce identical output.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) pipeline.py and input.txt. Deterministic: two consecutive runs must
produce the same output.txt. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import os
import subprocess
import sys


REFERENCE = """\
processed:alpha
processed:beta
processed:gamma
processed:delta
processed:epsilon
"""


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _run():
    proc = subprocess.run(
        [sys.executable, "pipeline.py"], capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        fail("pipeline.py exited %d:\n%s" % (proc.returncode, proc.stderr))
        return None
    out_path = os.path.join(os.getcwd(), "output.txt")
    if not os.path.exists(out_path):
        fail("output.txt not found after run")
        return None
    with open(out_path) as f:
        return f.read()


def main():
    out_path = os.path.join(os.getcwd(), "output.txt")
    if os.path.exists(out_path):
        os.unlink(out_path)

    out1 = _run()
    if out1 is None:
        return 1
    out2 = _run()
    if out2 is None:
        return 1

    if out1 != out2:
        return fail("non-idempotent: output changed on second run:\n"
                    "--- run 1 ---\n%s\n--- run 2 ---\n%s" % (out1, out2))

    if out1 != REFERENCE:
        return fail("output does not match reference:\n--- got ---\n%s\n--- expected ---\n%s"
                    % (out1, REFERENCE))

    return 0


if __name__ == "__main__":
    sys.exit(main())
