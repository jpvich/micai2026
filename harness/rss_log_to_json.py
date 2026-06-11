#!/usr/bin/env python3
"""Convert per-language RSS log files from wrap_rss.sh into <bench>_memory.json.

Each log file contains one line per benchmark run: the peak RSS in bytes.
Files come from wrap_rss.sh, invoked by hyperfine inside run_single.sh.

Usage:
  python3 rss_log_to_json.py <benchmark_id> --rss-dir <dir> --langs lang1 lang2 ...
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("benchmark")
    p.add_argument("--rss-dir", required=True)
    p.add_argument("--langs", nargs="+", required=True)
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    out = {
        "benchmark": args.benchmark,
        "tool": "/usr/bin/time -l (via wrap_rss.sh, capturing every hyperfine run)",
        "results": [],
    }

    print(f"  {'lang':<8} {'mean':>10}  {'σ':>10}  {'min':>10}  {'max':>10}  {'n':>4}")
    for lang in args.langs:
        log_path = os.path.join(args.rss_dir, f"{args.benchmark}_{lang}.log")
        if not os.path.isfile(log_path):
            print(f"  {lang:<8} (no RSS log at {log_path})", file=sys.stderr)
            continue
        try:
            with open(log_path) as f:
                samples = [int(line.strip()) for line in f if line.strip()]
        except ValueError as e:
            print(f"  {lang:<8} (corrupt RSS log: {e})", file=sys.stderr)
            continue
        if not samples:
            print(f"  {lang:<8} (empty RSS log)", file=sys.stderr)
            continue

        mean = statistics.mean(samples)
        stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0

        def mib(b): return b / 1048576

        print(f"  {lang:<8} {mib(mean):>7.2f} MB {mib(stdev):>7.2f} MB "
              f"{mib(min(samples)):>7.2f} MB {mib(max(samples)):>7.2f} MB {len(samples):>4}")

        out["results"].append({
            "command": lang,
            "rss_bytes_mean": mean,
            "rss_bytes_stdev": stdev,
            "rss_bytes_min": min(samples),
            "rss_bytes_max": max(samples),
            "n_samples": len(samples),
            "samples_bytes": samples,
        })

    output_path = os.path.join(RESULTS, f"{args.benchmark}_memory.json")
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
