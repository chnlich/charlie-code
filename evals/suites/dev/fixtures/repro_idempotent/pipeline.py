def run():
    with open("input.txt") as f:
        data = [line.strip() for line in f if line.strip()]

    # BUG: append mode ("a") means running the script twice doubles the
    # output. The pipeline is not idempotent.
    with open("output.txt", "a") as f:
        for item in data:
            f.write("processed:%s\n" % item)


run()
