#!/usr/bin/env python3

"""Run each PBBS input five times and save whole-program timings."""

import csv
import subprocess
import sys
import time
from pathlib import Path

# Usage: python3 bench_pbbs.py EXECUTABLE INPUT_DIR RESULTS_DIR
binary = Path(sys.argv[1]).resolve()
inputs = Path(sys.argv[2]).resolve()
results = Path(sys.argv[3]).resolve()
results.mkdir(parents=True, exist_ok=True)

with (results / "timings.csv").open("w", newline="") as file:
    timings = csv.writer(file)
    timings.writerow(["input", "run", "wall_seconds"])

    for points in sorted(inputs.glob("*.2dpoints")):
        triangles = results / f"{points.stem}.tri"

        for run in range(1, 6):
            if triangles.exists():
                triangles.unlink()

            start = time.perf_counter()
            with (results / f"{points.stem}.{run}.log").open("w") as log:
                subprocess.run(
                    [str(binary), str(points), str(triangles)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            seconds = time.perf_counter() - start

            # The executable must actually write a PBBS triangle file.
            if not triangles.exists():
                raise RuntimeError(f"No output file: {triangles}")
            with triangles.open() as output:
                if output.readline().strip() != "pbbs_triangles":
                    raise RuntimeError(f"Invalid PBBS output: {triangles}")

            timings.writerow([points.name, run, f"{seconds:.6f}"])
            file.flush()  # Keep completed results even if a later run fails.
            print(points.name, run, f"{seconds:.3f}s")