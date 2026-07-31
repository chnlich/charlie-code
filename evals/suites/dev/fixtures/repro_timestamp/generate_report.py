import time


def generate():
    with open("data.txt") as f:
        lines = [line.strip() for line in f if line.strip()]

    report = []
    # BUG: time.time() changes on every run, making the output non-reproducible.
    report.append("Report generated at: %s" % time.time())
    report.append("Total records: %d" % len(lines))
    for line in lines:
        report.append("  - %s" % line)

    with open("report.txt", "w") as f:
        f.write("\n".join(report) + "\n")


generate()
