"""Grader for repro_timestamp: output must be byte-identical across runs.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) generate_report.py and data.txt. Deterministic: two consecutive runs
must produce identical report.txt. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import os
import subprocess
import sys


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _run():
    proc = subprocess.run(
        [sys.executable, "generate_report.py"], capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if proc.returncode != 0:
        fail("generate_report.py exited %d:\n%s" % (proc.returncode, proc.stderr))
        return None
    out_path = os.path.join(os.getcwd(), "report.txt")
    if not os.path.exists(out_path):
        fail("report.txt not found after run")
        return None
    with open(out_path) as f:
        return f.read()


def main():
    out1 = _run()
    if out1 is None:
        return 1
    out2 = _run()
    if out2 is None:
        return 1

    if out1 != out2:
        return fail("non-reproducible: output changed on second run:\n"
                    "--- run 1 ---\n%s\n--- run 2 ---\n%s" % (out1, out2))

    required = ["Total records: 3", "  - alpha", "  - beta", "  - gamma"]
    for req in required:
        if req not in out1:
            return fail("output missing expected line %r:\n%s" % (req, out1))

    return 0


if __name__ == "__main__":
    sys.exit(main())
