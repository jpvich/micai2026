#!/bin/bash
# Master orchestrator for the MICAI AI-benchmark suite.
#
# Runs each benchmark across the 6 languages via run_single.sh, which captures
# M1 (wall-clock time) and M2 (peak RSS) in a single hyperfine pass per
# benchmark. Building is lazy (run_single builds only what it runs).
#
# Usage:
#   bash run_all.sh                 # all 5 AI benchmarks, all 6 languages
#
# The legacy 10-language ICSE orchestration was retired here on the MICAI pivot;
# the old per-benchmark command lists (b5/b6/b8 data files, COBOL skips, JVM
# class names) live in git history if ever needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BENCHMARKS_LIST=(kmeans knn mlp ga fuzzy)

echo "=== MICAI AI-benchmark suite ==="
echo "Date: $(date)"
echo "Machine: $(uname -srm)"
echo "Benchmarks: ${BENCHMARKS_LIST[*]}"
echo ""

for bench in "${BENCHMARKS_LIST[@]}"; do
    echo ""
    echo "########################## $bench ##########################"
    bash "$SCRIPT_DIR/run_single.sh" "$bench"
done

echo ""
echo "=== ALL BENCHMARKS COMPLETE ==="
echo "Results in: $(dirname "$SCRIPT_DIR")/results/"
