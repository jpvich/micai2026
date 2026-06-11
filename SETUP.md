# Setup — Reproducibility recipe

*Implementing Intelligence: Comparing Six Programming Languages on the
Algorithms Behind AI* (MICAI submission).

This study is **not containerized, and Docker is not required**. A Docker
build on an Apple M1 runs Linux inside a VM, which would change the OS,
libc, compilers, and add virtualization overhead — invalidating every
M1/macOS wall-time and peak-RSS number reported in the paper. Reproducibility
is instead guaranteed by **deterministic inputs** (a shared 64-bit LCG, seed
42) and **bit-exact cross-language checksums**: every language provably solves
an identical problem on any machine. Absolute timings are hardware-specific;
the portable result is the *relative* ranking across languages.

## Hardware

- Apple MacBook (M1 or later) with 16 GB+ unified memory
- macOS Sonoma (14.x) or later
- A few GB of free SSD for the toolchain caches

x86-64 and Linux are not validated; the paper's numbers are M1/macOS specific.

## Languages (6)

| Language | Install | Notes |
|----------|---------|-------|
| Python   | preinstalled / `brew install python` | interpreted; reference for productivity tier |
| C        | `xcode-select --install` (Apple Clang) | `-O3 -march=native` |
| C++      | Xcode command-line tools | `-O3 -march=native -std=c++17` |
| Rust     | rustup (below) | release: `opt-level=3, lto=true` |
| Go       | `brew install go` | default `go build` |
| Julia    | `brew install julia` | JIT; runtime startup is included in wall time (documented) |

## One-shot install

```bash
# Xcode command-line tools (C, C++)
xcode-select --install

# Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Go and Julia
brew install go julia hyperfine

# Rust via rustup (brew's rust lags several stable releases)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup default stable

# Python venv for analysis/plots only (no benchmark depends on Python packages)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exact versions used in the paper are recorded in `versions.txt`.

## Running the suite

No data-generation step is needed: every benchmark generates its inputs
in-process from the shared LCG.

```bash
# Full sweep — all 5 benchmarks x 6 languages, ~6.5 h (Python dominates)
bash harness/run_all.sh

# A single benchmark across all 6 languages (mlp ~70 min; fuzzy ~2 h)
bash harness/run_single.sh mlp

# A single benchmark on a subset of languages (quick pipeline test, ~1-2 min)
bash harness/run_single.sh mlp c cpp rust go julia

# Static metrics (M3 binary size, M4 LOC, M5 compile time) — minutes, no heavy run
python3 harness/collect_static_metrics.py

# Aggregate to CSV, then generate figures
python3 harness/aggregate_results.py
python3 analysis/plots.py
```

Each timing run writes `results/<bench>.json` (M1 wall time) and
`results/<bench>_memory.json` (M2 peak RSS). Configuration: `hyperfine`
with 10 measured runs after 3 warmup runs.

## Determinism and verification

All six implementations of a benchmark print a checksum to stdout. The
checksums are **bit-identical** across languages (verified during
development), confirming that every language runs the same workload:

| Benchmark | Checksum |
|-----------|----------|
| `kmeans`  | `559268 20.004093` |
| `knn`     | `8603` |
| `mlp`     | `0.085671 7.648975` |
| `ga`      | `24.460672 4.216343` |
| `fuzzy`   | `999999.524374` |

To re-verify, run any benchmark on all six languages and confirm identical
output:

```bash
for L in c cpp rust go python julia; do bash harness/run_single.sh mlp $L; done
```

## Measurement hygiene

For clean numbers: plug into power, close CPU/memory-heavy apps (browsers,
Docker Desktop, sync clients), pause Time Machine, and prevent sleep via
**System Settings → Battery → Options → "Prevent automatic sleeping on power
adapter when the display is off"** (leave the lid open). Most benchmarks are
single-threaded and sensitive to background CPU contention.

## Known limitations

- Single hardware platform (Apple M1, macOS). Absolute timings are not
  portable; relative rankings are.
- Julia's JIT runtime startup (~0.3–0.5 s) and resident set (~230 MB) are
  included in its measurements, as they are a real cost for the short-lived
  processes typical of AI scripting. Documented as a threat to validity.
- Implementations are idiomatic but from scratch: no SIMD intrinsics, no
  external numerical libraries (BLAS, etc.). The study measures the language,
  not hand-tuned or library-backed peak performance.
