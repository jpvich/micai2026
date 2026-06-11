#!/bin/bash
# Collect non-timing metrics: Peak RSS, binary size, LOC, compilation time
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
BENCHMARKS="$ROOT/benchmarks"

echo "language,metric,value" > "$RESULTS/static_metrics.csv"

# Binary sizes (compiled languages only)
echo "=== Binary Sizes ==="
for lang in c cpp rust go zig swift; do
    case $lang in
        c|cpp) for f in "$BENCHMARKS/$lang"/b[0-9]*; do [ -x "$f" ] && echo "$lang,binary_size_$(basename $f),$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")" >> "$RESULTS/static_metrics.csv"; done ;;
        rust) for f in "$BENCHMARKS/rust/target/release"/b[0-9]*; do [ -x "$f" ] && echo "rust,binary_size_$(basename $f),$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")" >> "$RESULTS/static_metrics.csv"; done ;;
        go) for f in "$BENCHMARKS/go"/b[0-9]_*; do [ -x "$f" ] && [[ ! "$f" == *.go ]] && echo "go,binary_size_$(basename $f),$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")" >> "$RESULTS/static_metrics.csv"; done ;;
        zig) for f in "$BENCHMARKS/zig/zig-out/bin"/b[0-9]*; do [ -x "$f" ] && echo "zig,binary_size_$(basename $f),$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")" >> "$RESULTS/static_metrics.csv"; done ;;
        swift) for f in "$BENCHMARKS/swift"/b[0-9]*; do [ -x "$f" ] && [[ ! "$f" == *.swift ]] && echo "swift,binary_size_$(basename $f),$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")" >> "$RESULTS/static_metrics.csv"; done ;;
    esac
done

# Lines of code
echo "=== Lines of Code ==="
for lang in c cpp python julia swift; do
    ext=""
    case $lang in c) ext="c" ;; cpp) ext="cpp" ;; python) ext="py" ;; julia) ext="jl" ;; swift) ext="swift" ;; esac
    for f in "$BENCHMARKS/$lang"/*.$ext; do
        name=$(basename "${f%.*}")
        loc=$(wc -l < "$f" | tr -d ' ')
        echo "$lang,loc_$name,$loc" >> "$RESULTS/static_metrics.csv"
    done
done
# Rust
for f in "$BENCHMARKS/rust/src"/*.rs; do
    name=$(basename "${f%.rs}")
    loc=$(wc -l < "$f" | tr -d ' ')
    echo "rust,loc_$name,$loc" >> "$RESULTS/static_metrics.csv"
done
# Go
for f in "$BENCHMARKS/go"/*.go; do
    name=$(basename "${f%.go}")
    loc=$(wc -l < "$f" | tr -d ' ')
    echo "go,loc_$name,$loc" >> "$RESULTS/static_metrics.csv"
done
# Java
for f in "$BENCHMARKS/java"/*.java; do
    name=$(basename "${f%.java}")
    loc=$(wc -l < "$f" | tr -d ' ')
    echo "java,loc_$name,$loc" >> "$RESULTS/static_metrics.csv"
done
# Kotlin
for f in "$BENCHMARKS/kotlin"/*.kt; do
    name=$(basename "${f%.kt}")
    loc=$(wc -l < "$f" | tr -d ' ')
    echo "kotlin,loc_$name,$loc" >> "$RESULTS/static_metrics.csv"
done
# Zig
for f in "$BENCHMARKS/zig"/*.zig; do
    [[ "$(basename $f)" == "build.zig" ]] && continue
    name=$(basename "${f%.zig}")
    loc=$(wc -l < "$f" | tr -d ' ')
    echo "zig,loc_$name,$loc" >> "$RESULTS/static_metrics.csv"
done

echo "Done! Results in $RESULTS/static_metrics.csv"
