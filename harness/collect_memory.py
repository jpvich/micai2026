#!/usr/bin/env python3
"""Capture M2 (peak RSS) for one benchmark across all 10 languages.

Uses /usr/bin/time -l on macOS, where ru_maxrss is reported in BYTES
(unlike Linux, which reports it in KiB). Runs each language N times
(default 5) and records mean, stddev, min, max, and the raw samples.

Output: results/<benchmark_id>_memory.json

Usage:
  python3 collect_memory.py <benchmark_id> [--runs N] [--langs lang1 lang2 ...]

Examples:
  python3 collect_memory.py b1_fibonacci
  python3 collect_memory.py b3_matrix_mul --runs 3
  python3 collect_memory.py b1_fibonacci --langs c rust julia
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARKS = os.path.join(ROOT, "benchmarks")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# MICAI suite: 6 AI-justified languages x 5 AI benchmarks.
ALL_LANGUAGES = ["c", "cpp", "rust", "go", "python", "julia"]
ALL_BENCHMARKS = ["kmeans", "knn", "mlp", "ga", "fuzzy"]

JAVA_CLASS = {
    "b1_fibonacci":   "B1Fibonacci",
    "b2_bubble_sort": "B2BubbleSort",
    "b3_matrix_mul":  "B3MatrixMul",
    "b4_monte_carlo": "B4MonteCarlo",
    "b5_regex":       "B5Regex",
    "b6_file_io":     "B6FileIO",
    "b7_concurrent":  "B7Concurrent",
    "b8_json_parse":  "B8JsonParse",
}

# brew keg-only formulae aren't on the default PATH — match run_single.sh.
for p in ["/opt/homebrew/opt/openjdk/bin", "/opt/homebrew/opt/zig@0.14/bin"]:
    if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")


def build_cmd(lang, bench):
    extra = []
    if bench == "b5_regex":
        extra = [os.path.join(DATA, "regex_input.txt")]
    elif bench == "b8_json_parse":
        extra = [os.path.join(DATA, "json_input.json")]
    jcls = JAVA_CLASS[bench]
    table = {
        "c":      [os.path.join(BENCHMARKS, "c", bench)],
        "cpp":    [os.path.join(BENCHMARKS, "cpp", bench)],
        "rust":   [os.path.join(BENCHMARKS, "rust/target/release", bench)],
        "go":     [os.path.join(BENCHMARKS, "go", bench)],
        "java":   ["java", "-cp", os.path.join(BENCHMARKS, "java"), jcls],
        "python": ["python3", os.path.join(BENCHMARKS, "python", f"{bench}.py")],
        "julia":  ["julia", os.path.join(BENCHMARKS, "julia", f"{bench}.jl")],
        "zig":    [os.path.join(BENCHMARKS, "zig", "zig-out", "bin", bench)],
        "swift":  [os.path.join(BENCHMARKS, "swift", bench)],
        "kotlin": ["java", "-jar", os.path.join(BENCHMARKS, "kotlin", f"{bench}.jar")],
    }
    return table[lang] + extra


_RSS_RE = re.compile(r"\s*(\d+)\s+maximum resident set size")


def measure_rss(cmd):
    """Run cmd under /usr/bin/time -l. Returns peak RSS in bytes, or None on failure."""
    try:
        result = subprocess.run(
            ["/usr/bin/time", "-l", *cmd],
            capture_output=True, text=True, timeout=3600
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return None, f"exec error: {e}"
    if result.returncode != 0:
        return None, f"non-zero exit {result.returncode}: {result.stderr.splitlines()[-1] if result.stderr else ''}"
    for line in result.stderr.splitlines():
        m = _RSS_RE.match(line)
        if m:
            return int(m.group(1)), None
    return None, "no 'maximum resident set size' line found"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", help="Benchmark id, e.g. b1_fibonacci")
    p.add_argument("--runs", type=int, default=5, help="Measurement runs per language (default 5)")
    p.add_argument("--langs", nargs="+", default=ALL_LANGUAGES, help="Subset of languages")
    args = p.parse_args()

    if args.benchmark not in ALL_BENCHMARKS:
        print(f"ERROR: unknown benchmark '{args.benchmark}'. Valid: {', '.join(ALL_BENCHMARKS)}", file=sys.stderr)
        sys.exit(1)
    for lang in args.langs:
        if lang not in ALL_LANGUAGES:
            print(f"ERROR: unknown language '{lang}'. Valid: {', '.join(ALL_LANGUAGES)}", file=sys.stderr)
            sys.exit(1)

    if args.benchmark == "b5_regex" and not os.path.isfile(os.path.join(DATA, "regex_input.txt")):
        print("ERROR: data/regex_input.txt missing. Run: python3 harness/generate_data.py", file=sys.stderr)
        sys.exit(1)
    if args.benchmark == "b8_json_parse" and not os.path.isfile(os.path.join(DATA, "json_input.json")):
        print("ERROR: data/json_input.json missing. Run: python3 harness/generate_data.py", file=sys.stderr)
        sys.exit(1)

    os.makedirs(RESULTS, exist_ok=True)

    print(f"=== collect_memory — {args.benchmark} ({args.runs} runs per lang) ===\n")

    out = {"benchmark": args.benchmark, "runs_per_lang": args.runs, "tool": "/usr/bin/time -l", "results": []}

    for lang in args.langs:
        cmd = build_cmd(lang, args.benchmark)
        print(f"  {lang:<8} ", end="", flush=True)
        samples = []
        failure = None
        for i in range(args.runs):
            rss, err = measure_rss(cmd)
            if rss is None:
                failure = err
                print(f" FAIL ({err})")
                break
            samples.append(rss)
            print(".", end="", flush=True)
        if samples:
            mean = statistics.mean(samples)
            stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
            print(f" mean={mean/1048576:.2f} MiB ± {stdev/1048576:.2f} MiB")
            out["results"].append({
                "command": lang,
                "rss_bytes_mean": mean,
                "rss_bytes_stdev": stdev,
                "rss_bytes_min": min(samples),
                "rss_bytes_max": max(samples),
                "samples_bytes": samples,
            })
        elif failure:
            out["results"].append({"command": lang, "error": failure})

    output_path = os.path.join(RESULTS, f"{args.benchmark}_memory.json")
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
