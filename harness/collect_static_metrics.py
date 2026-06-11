#!/usr/bin/env python3
"""Collect the three non-runtime metrics (M3, M4, M5) for the MICAI suite:
6 languages x 5 AI benchmarks.

Outputs three JSON files in results/:
  - m3_binary_size.json   : per (lang, bench) compiled-artifact size in bytes
  - m4_loc.json           : per (lang, bench) implementation lines of code
  - m5_compile_time.json  : per language total clean-build wall time in seconds

All outputs use the same {"results": [{"command": "<lang>", ...}, ...]} shape as
the hyperfine JSONs (and the wrap_rss aggregator), so downstream analysis looks
entries up by language name rather than by position.

Methodology:
  M3: stat() of the executable produced by each toolchain. Python and Julia have
      no compiled artifact (interpreted); reported as null.
  M4: wc -l of the *implementation* source file only (one file per benchmark).
  M5: total wall time of a clean build of the 5 MICAI benchmarks per language,
      measured after wiping the language's build cache. Per-bench timing is not
      reported (Rust shares dependency/codegen work across bins). Python and
      Julia have no build step (interpreted) -> reported as null.
"""
import json
import os
import subprocess
import time
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LANGUAGES = ["c", "cpp", "rust", "go", "python", "julia"]
BENCHES = ["kmeans", "knn", "mlp", "ga", "fuzzy"]


def artifact_path(lang: str, bench: str):
    """Path to the compiled artifact, or None for interpreted languages."""
    if lang == "c":
        return BENCHMARKS / "c" / bench
    if lang == "cpp":
        return BENCHMARKS / "cpp" / bench
    if lang == "rust":
        return BENCHMARKS / "rust" / "target" / "release" / bench
    if lang == "go":
        return BENCHMARKS / "go" / bench
    if lang in ("python", "julia"):
        return None  # interpreted
    raise ValueError(lang)


def source_path(lang: str, bench: str) -> Path:
    """Path to the implementation source file."""
    if lang == "c":
        return BENCHMARKS / "c" / f"{bench}.c"
    if lang == "cpp":
        return BENCHMARKS / "cpp" / f"{bench}.cpp"
    if lang == "rust":
        return BENCHMARKS / "rust" / "src" / f"{bench}.rs"
    if lang == "go":
        return BENCHMARKS / "go" / f"{bench}.go"
    if lang == "python":
        return BENCHMARKS / "python" / f"{bench}.py"
    if lang == "julia":
        return BENCHMARKS / "julia" / f"{bench}.jl"
    raise ValueError(lang)


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# ─── M3: binary size ─────────────────────────────────────────────────────
def collect_m3():
    print("=== M3 binary size ===")
    results = []
    for lang in LANGUAGES:
        entry = {"command": lang, "bytes_per_bench": {}}
        for bench in BENCHES:
            p = artifact_path(lang, bench)
            if p is None:
                entry["bytes_per_bench"][bench] = None
                continue
            if not p.exists():
                print(f"  ⚠ missing {lang}/{bench}: {p} (build it first)")
                entry["bytes_per_bench"][bench] = None
                continue
            entry["bytes_per_bench"][bench] = p.stat().st_size
        sizes = [v for v in entry["bytes_per_bench"].values() if v is not None]
        entry["bytes_mean"] = sum(sizes) / len(sizes) if sizes else None
        entry["bytes_min"] = min(sizes) if sizes else None
        entry["bytes_max"] = max(sizes) if sizes else None
        print(f"  {lang:<7} mean={entry['bytes_mean']!s:>12}  min={entry['bytes_min']!s:>10}  max={entry['bytes_max']!s:>10}")
        results.append(entry)
    out = RESULTS / "m3_binary_size.json"
    with open(out, "w") as f:
        json.dump({"metric": "M3 binary size (bytes)",
                   "tool": "stat -f%z (macOS bytes)",
                   "notes": "Python and Julia have no compiled artifact (interpreted); reported as null.",
                   "results": results}, f, indent=2)
    print(f"  → {out}\n")


# ─── M4: LOC ─────────────────────────────────────────────────────────────
def collect_m4():
    print("=== M4 lines of code ===")
    results = []
    for lang in LANGUAGES:
        entry = {"command": lang, "loc_per_bench": {}}
        for bench in BENCHES:
            p = source_path(lang, bench)
            if not p.exists():
                print(f"  ⚠ missing {lang}/{bench}: {p}")
                entry["loc_per_bench"][bench] = None
                continue
            entry["loc_per_bench"][bench] = count_lines(p)
        locs = [v for v in entry["loc_per_bench"].values() if v is not None]
        entry["loc_mean"] = sum(locs) / len(locs) if locs else None
        entry["loc_total"] = sum(locs) if locs else None
        print(f"  {lang:<7} total={entry['loc_total']!s:>5}  mean={entry['loc_mean']!s:>6}")
        results.append(entry)
    out = RESULTS / "m4_loc.json"
    with open(out, "w") as f:
        json.dump({"metric": "M4 lines of code (implementation only)",
                   "tool": "wc -l",
                   "notes": "Counts the per-benchmark implementation file only (one file per "
                            "benchmark per language). No external ML libraries are used, so each "
                            "file is the full from-scratch implementation. Comments and blank lines "
                            "are counted; relative ranking across languages is what matters.",
                   "results": results}, f, indent=2)
    print(f"  → {out}\n")


# ─── M5: compile time ───────────────────────────────────────────────────
def time_cmd(cmd, cwd: Path) -> float:
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    if r.returncode != 0:
        raise RuntimeError(f"build failed: {cmd}")
    return t1 - t0


def collect_m5():
    """Total cold-build wall time per language for the 5 MICAI benchmarks."""
    print("=== M5 compile time (cold build, 5 MICAI benchmarks per language) ===")
    results = []

    def add(lang: str, seconds, notes: str = ""):
        print(f"  {lang:<7} {seconds!s:>10}s  {notes}")
        results.append({"command": lang, "compile_time_s": seconds, "notes": notes})

    # C — make clean && make <the 5 benches>
    try:
        subprocess.run(["make", "clean"], cwd=BENCHMARKS / "c",
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t = time_cmd(["make", "-j1", *BENCHES], BENCHMARKS / "c")
        add("c", round(t, 3), "make -j1 (clean), 5 benches")
    except Exception as e:
        add("c", None, f"FAILED: {e}")

    # C++
    try:
        subprocess.run(["make", "clean"], cwd=BENCHMARKS / "cpp",
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t = time_cmd(["make", "-j1", *BENCHES], BENCHMARKS / "cpp")
        add("cpp", round(t, 3), "make -j1 (clean), 5 benches")
    except Exception as e:
        add("cpp", None, f"FAILED: {e}")

    # Rust — cargo clean && build only the 5 MICAI bins
    try:
        subprocess.run(["cargo", "clean"], cwd=BENCHMARKS / "rust",
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        bin_args = []
        for b in BENCHES:
            bin_args += ["--bin", b]
        t = time_cmd(["cargo", "build", "--release", *bin_args], BENCHMARKS / "rust")
        add("rust", round(t, 3), "cargo build --release --bin x5 (clean)")
    except Exception as e:
        add("rust", None, f"FAILED: {e}")

    # Go — clear cache, per-bench build
    try:
        go_dir = BENCHMARKS / "go"
        for b in BENCHES:
            (go_dir / b).unlink(missing_ok=True)
        subprocess.run(["go", "clean", "-cache"], cwd=go_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.perf_counter()
        for b in BENCHES:
            r = subprocess.run(["go", "build", "-o", b, f"{b}.go"], cwd=go_dir,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0:
                raise RuntimeError(f"go build failed for {b}")
        add("go", round(time.perf_counter() - t0, 3),
            "go clean -cache + per-bench go build, 5 benches")
    except Exception as e:
        add("go", None, f"FAILED: {e}")

    # Python / Julia — interpreted, no build step
    add("python", None, "interpreted; no AOT compile step")
    add("julia", None, "interpreted; no AOT compile step (JIT cost included in M1 wall time)")

    out = RESULTS / "m5_compile_time.json"
    with open(out, "w") as f:
        json.dump({"metric": "M5 cold-build wall time (seconds, total per language for the 5 MICAI benchmarks)",
                   "tool": "time.perf_counter wrapping subprocess.run on the build command",
                   "notes": "Per-language totals (cargo clean / make clean / go cache wipe between runs). "
                            "Python and Julia are interpreted -> null.",
                   "results": results}, f, indent=2)
    print(f"  → {out}\n")


if __name__ == "__main__":
    collect_m3()
    collect_m4()
    collect_m5()
    print("Done.")
