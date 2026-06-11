#!/bin/bash
# Quick (~2 min) sanity check for a freshly-cloned repo.
# Verifies all 10 toolchains are installed and the harness can build + run
# at least one benchmark per language.
#
# Run from repo root:
#   bash harness/smoke.sh
#
# Exit codes:
#   0   all checks passed
#   1   one or more toolchains missing
#   2   one or more builds failed
#   3   one or more outputs unexpected
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Tool PATH setup — matches what run_all.sh expects.
[ -d /opt/homebrew/opt/openjdk/bin ] && export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
[ -d /opt/homebrew/opt/zig@0.14/bin ] && export PATH="/opt/homebrew/opt/zig@0.14/bin:$PATH"

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$1"; }
bold()  { printf "\033[1m%s\033[0m\n" "$1"; }

bold "=== Rust Among Giants — smoke test ==="
echo "Repo: $ROOT"
echo ""

# ── Stage 1: toolchain presence ────────────────────────────────────────
bold "[1/3] Toolchain presence"
MISSING=0
check() {
    local name="$1" cmd="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        local v
        v=$($cmd --version 2>&1 | head -1 | tr -d '\n' | cut -c1-60)
        green "  ✓ $name ($v)"
    else
        red   "  ✗ $name — '$cmd' not found in PATH"
        MISSING=$((MISSING + 1))
    fi
}
check "C/clang"   cc
check "C++/clang" c++
check "Rust"      rustc
check "Cargo"     cargo
check "Go"        go
check "Java"      java
check "javac"     javac
check "Zig"       zig
check "Swift"    swiftc
check "Kotlin"    kotlinc
check "Julia"     julia
check "GnuCOBOL"  cobc
check "hyperfine" hyperfine

if [ "$MISSING" -gt 0 ]; then
    echo ""
    red "❌ $MISSING toolchain(s) missing. See SETUP.md for install instructions."
    exit 1
fi
echo ""

# ── Stage 2: build B6 (file I/O) for each language ─────────────────────
bold "[2/3] Build B6 (file_io) per language"
BUILD_FAIL=0
b6_build() {
    local name="$1" cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        green "  ✓ $name"
    else
        red   "  ✗ $name (build failed)"
        BUILD_FAIL=$((BUILD_FAIL + 1))
    fi
}
b6_build "C"      "cd '$ROOT/benchmarks/c'      && make b6_file_io"
b6_build "C++"    "cd '$ROOT/benchmarks/cpp'    && make b6_file_io"
b6_build "Rust"   "cd '$ROOT/benchmarks/rust'   && cargo build --release --bin b6_file_io"
b6_build "Go"     "cd '$ROOT/benchmarks/go'     && go build -o b6_file_io b6_file_io.go"
b6_build "Java"   "cd '$ROOT/benchmarks/java'   && javac B6FileIO.java"
b6_build "Zig"    "cd '$ROOT/benchmarks/zig'    && zig build"
b6_build "Swift"  "cd '$ROOT/benchmarks/swift'  && swiftc -O -o b6_file_io b6_file_io.swift"
b6_build "Kotlin" "cd '$ROOT/benchmarks/kotlin' && kotlinc b6_file_io.kt -include-runtime -d b6_file_io.jar"
b6_build "COBOL"  "cd '$ROOT/benchmarks/cobol'  && make b6_file_io"
# Julia is interpreted, no build.
green "  ✓ Julia (interpreted, no build step)"

if [ "$BUILD_FAIL" -gt 0 ]; then
    echo ""
    red "❌ $BUILD_FAIL build(s) failed. Try a single language with verbose flags to see why."
    exit 2
fi
echo ""

# ── Stage 3: run B1 (fib) for each language — fast (n=20) ──────────────
bold "[3/3] Run B1 fib(20) per language (output should be 6765)"
EXPECTED=6765
RUN_FAIL=0

# B1 is fastest; we override n=20 for the smoke test so even COBOL's
# expensive PROGRAM-ID IS RECURSIVE finishes in well under a second.
# (Each language's b1 binary accepts an optional command-line n.)

# Build B1 for each language so it's ready.
(cd "$ROOT/benchmarks/c"      && make b1_fibonacci   >/dev/null 2>&1)
(cd "$ROOT/benchmarks/cpp"    && make b1_fibonacci   >/dev/null 2>&1)
(cd "$ROOT/benchmarks/rust"   && cargo build --release --bin b1_fibonacci >/dev/null 2>&1)
(cd "$ROOT/benchmarks/go"     && go build -o b1_fibonacci b1_fibonacci.go)
(cd "$ROOT/benchmarks/java"   && javac B1Fibonacci.java)
(cd "$ROOT/benchmarks/swift"  && swiftc -O -o b1_fibonacci b1_fibonacci.swift)
(cd "$ROOT/benchmarks/kotlin" && kotlinc b1_fibonacci.kt -include-runtime -d b1_fibonacci.jar 2>/dev/null)
(cd "$ROOT/benchmarks/cobol"  && make b1_fibonacci   >/dev/null 2>&1)

check_out() {
    local name="$1" cmd="$2"
    local out
    out=$(eval "$cmd" 2>/dev/null | head -1 | tr -d '\n[:space:]')
    if [ "$out" = "$EXPECTED" ]; then
        green "  ✓ $name → $out"
    else
        red   "  ✗ $name → '$out' (expected $EXPECTED)"
        RUN_FAIL=$((RUN_FAIL + 1))
    fi
}

check_out "C"      "$ROOT/benchmarks/c/b1_fibonacci 20"
check_out "C++"    "$ROOT/benchmarks/cpp/b1_fibonacci 20"
check_out "Rust"   "$ROOT/benchmarks/rust/target/release/b1_fibonacci 20"
check_out "Go"     "$ROOT/benchmarks/go/b1_fibonacci 20"
check_out "Java"   "java -cp $ROOT/benchmarks/java B1Fibonacci 20"
check_out "Zig"    "$ROOT/benchmarks/zig/zig-out/bin/b1_fibonacci 20"
check_out "Swift"  "$ROOT/benchmarks/swift/b1_fibonacci 20"
check_out "Kotlin" "java -jar $ROOT/benchmarks/kotlin/b1_fibonacci.jar 20"
check_out "COBOL"  "$ROOT/benchmarks/cobol/b1_fibonacci 20"
check_out "Julia"  "julia $ROOT/benchmarks/julia/b1_fibonacci.jl 20"

echo ""
if [ "$RUN_FAIL" -gt 0 ]; then
    red "❌ $RUN_FAIL run(s) produced unexpected output."
    exit 3
fi

# ── Final ──────────────────────────────────────────────────────────────
bold "=== ✅ ALL CHECKS PASSED ==="
echo ""
echo "Your environment is ready. Next steps:"
echo "  • Full sweep:       bash harness/run_all.sh    (~90 min + COBOL extra)"
echo "  • Single benchmark: bash harness/run_single.sh b6_file_io"
echo "  • Static metrics:   python3 harness/collect_static_metrics.py"
echo "  • Figures:          python3 analysis/plots.py"
