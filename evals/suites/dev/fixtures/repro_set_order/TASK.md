# Task: repro_set_order (reproducible data processing)

`process.py` reads `data.txt`, groups records by their key (first column),
collects the values (second column) into a set per group, and writes the
groups to `output.txt`.

The bug: the script joins the set values with `",".join(groups[key])`, but
**set iteration order for strings is randomized by `PYTHONHASHSEED`**. So
the output differs from run to run, making it non-reproducible.

Fix `process.py` so the output is **byte-identical** across runs regardless
of `PYTHONHASHSEED`. The most natural fix is to sort the values before
joining.

Do not change `data.txt`. The grader runs the script twice with different
`PYTHONHASHSEED` values and checks that the outputs are identical and match
the expected reference.
