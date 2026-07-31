def process(infile, outfile):
    groups = {}
    with open(infile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(",", 1)
            if key not in groups:
                groups[key] = set()
            groups[key].add(value)

    with open(outfile, "w") as f:
        for key in sorted(groups.keys()):
            # BUG: set iteration order is non-deterministic (PYTHONHASHSEED),
            # so the values within each group appear in a different order on
            # different runs.
            f.write("%s:%s\n" % (key, ",".join(groups[key])))


process("data.txt", "output.txt")
