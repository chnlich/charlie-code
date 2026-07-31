# Task: repro_idempotent (reproducible data processing)

`pipeline.py` reads `input.txt`, processes each line, and writes the results
to `output.txt`.

The bug: the script opens `output.txt` in **append mode** (`"a"`), so running
it twice on the same input doubles the output instead of producing the same
result. The pipeline is not idempotent.

Fix `pipeline.py` so it is **idempotent**: running it multiple times on the
same input produces the same `output.txt` every time. The most natural fix
is to open in write mode (`"w"`) instead of append mode (`"a"`), or to
truncate the file before writing.

Do not change `input.txt`. The grader runs the script twice, verifies the
output is identical both times (not doubled), and checks it matches the
expected reference.
