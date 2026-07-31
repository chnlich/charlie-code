"""Grader for repro_set_order: output must be byte-identical across hash seeds.

Runs with cwd = the episode work dir, which contains the agent's (possibly
fixed) process.py and data.txt. Deterministic: two runs with different
PYTHONHASHSEED values must produce identical output. No external dependencies.

Exit 0 = solved; non-zero = not solved. Diagnostics go to stderr.
"""

import os
import subprocess
import sys


REFERENCE = """\
fruit:apple,banana,cherry,date
grain:barley,oats,rice,wheat
veg:broccoli,carrot,spinach
"""


def fail(msg):
    sys.stderr.write("FAIL: " + msg + "\n")
    return 1


def _run_with_seed(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    proc = subprocess.run(
        [sys.executable, "process.py"], capture_output=True, text=True,
        cwd=os.getcwd(), env=env,
    )
    if proc.returncode != 0:
        fail("process.py exited %d (seed=%s):\n%s" % (proc.returncode, seed, proc.stderr))
        return None
    out_path = os.path.join(os.getcwd(), "output.txt")
    if not os.path.exists(out_path):
        fail("output.txt not found (seed=%s)" % seed)
        return None
    with open(out_path) as f:
        return f.read()


def main():
    out1 = _run_with_seed(1)
    if out1 is None:
        return 1
    out2 = _run_with_seed(2)
    if out2 is None:
        return 1

    if out1 != out2:
        return fail("non-deterministic output across PYTHONHASHSEED values:\n"
                    "--- seed=1 ---\n%s\n--- seed=2 ---\n%s" % (out1, out2))

    if out1 != REFERENCE:
        return fail("output does not match reference:\n--- got ---\n%s\n--- expected ---\n%s"
                    % (out1, REFERENCE))

    return 0


if __name__ == "__main__":
    sys.exit(main())
