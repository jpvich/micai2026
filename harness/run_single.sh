#!/bin/bash
# Run a single benchmark across all 10 languages (or a subset).
#
# Usage:
#   bash run_single.sh <benchmark_id> [lang1 lang2 ...]
#
# Examples:
#   bash run_single.sh b1_fibonacci                # all 10 languages
#   bash run_single.sh b1_fibonacci rust           # just rust
#   bash run_single.sh b3_matrix_mul c rust zig    # only these three
#
# Output: results/<benchmark_id>.json (hyperfine JSON, overwrites prior)
#
# Behavior:
#   - Builds only the languages it will run (lazy)
#   - Uses same hyperfine config as run_all.sh (10 runs, 3 warmup)
#   - Skips build for interpreted languages (Python, Julia)
#   - Errors out if test data is missing for b5_regex / b8_json_parse
set -euo pipefail

# Tool path setup — brew keg-only formulae aren't on the default PATH.
# Java/javac live under openjdk; we pin Zig to 0.14 (0.16 broke benchmark APIs).
[ -d /opt/homebrew/opt/openjdk/bin ] && export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
[ -d /opt/homebrew/opt/zig@0.14/bin ] && export PATH="/opt/homebrew/opt/zig@0.14/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS="$ROOT/results"
BENCHMARKS="$ROOT/benchmarks"
DATA="$ROOT/data"

# Canonical hyperfine config (matches the ICSE methodology): 10 measured runs
# after 3 warmup runs. The 3 warmups absorb cold-start effects (page cache, CPU
# frequency) so even the low-wall fuzzy benchmark stays above the noise floor.
RUNS=10
WARMUP=3

# MICAI suite: 6 AI-justified languages. Julia runs last so its ~210 MB JIT
# runtime does not pollute L2/L3 cache for the next language in the sweep.
ALL_LANGUAGES=(c cpp rust go python julia)

# Retained from the ICSE harness; inert for the MICAI suite (no COBOL).
COBOL_SKIP_BENCHES=(b5_regex b7_concurrent b8_json_parse)
ALL_BENCHMARKS=(kmeans knn mlp ga fuzzy)

# bench id (b1_fibonacci) -> Java class name (B1Fibonacci). BSD sed lacks \u, so hardcode.
java_class() {
    case "$1" in
        b1_fibonacci)   echo "B1Fibonacci" ;;
        b2_bubble_sort) echo "B2BubbleSort" ;;
        b3_matrix_mul)  echo "B3MatrixMul" ;;
        b4_monte_carlo) echo "B4MonteCarlo" ;;
        b5_regex)       echo "B5Regex" ;;
        b6_file_io)     echo "B6FileIO" ;;
        b7_concurrent)  echo "B7Concurrent" ;;
        b8_json_parse)  echo "B8JsonParse" ;;
        *) echo "UNKNOWN_CLASS_FOR_$1" ;;
    esac
}

mkdir -p "$RESULTS"

# ── Argument parsing ──────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: bash $(basename "$0") <benchmark_id> [lang1 lang2 ...]"
    echo ""
    echo "Benchmarks: ${ALL_BENCHMARKS[*]}"
    echo "Languages:  ${ALL_LANGUAGES[*]}"
    exit 1
fi

BENCH="$1"
shift

# Validate benchmark
if [[ ! " ${ALL_BENCHMARKS[*]} " == *" $BENCH "* ]]; then
    echo "ERROR: unknown benchmark '$BENCH'"
    echo "Valid: ${ALL_BENCHMARKS[*]}"
    exit 1
fi

# Pick languages: all if no args, else just the listed ones
if [ $# -eq 0 ]; then
    LANGUAGES=("${ALL_LANGUAGES[@]}")
else
    LANGUAGES=("$@")
    for lang in "${LANGUAGES[@]}"; do
        if [[ ! " ${ALL_LANGUAGES[*]} " == *" $lang "* ]]; then
            echo "ERROR: unknown language '$lang'"
            echo "Valid: ${ALL_LANGUAGES[*]}"
            exit 1
        fi
    done
fi

# Drop COBOL when the benchmark is in its skip list. COBOL has no standard
# regex, no threads/atomics, and GnuCOBOL 3.2 has no JSON PARSE.
if [[ " ${COBOL_SKIP_BENCHES[*]} " == *" $BENCH "* ]]; then
    FILTERED=()
    for lang in "${LANGUAGES[@]}"; do
        [ "$lang" = "cobol" ] || FILTERED+=("$lang")
    done
    if [ ${#FILTERED[@]} -lt ${#LANGUAGES[@]} ]; then
        echo "NOTE: cobol skipped for $BENCH (no COBOL implementation; see paper §Methodology)"
    fi
    LANGUAGES=("${FILTERED[@]}")
fi

# ── Data files needed by some benchmarks ──────────────────────────────
case "$BENCH" in
    b5_regex)
        if [ ! -f "$DATA/regex_input.txt" ]; then
            echo "ERROR: $DATA/regex_input.txt missing. Run: python3 harness/generate_data.py"
            exit 1
        fi
        ;;
    b8_json_parse)
        if [ ! -f "$DATA/json_input.json" ]; then
            echo "ERROR: $DATA/json_input.json missing. Run: python3 harness/generate_data.py"
            exit 1
        fi
        ;;
esac

echo "=== run_single — $BENCH on: ${LANGUAGES[*]} ==="
echo "Runs: $RUNS, Warmup: $WARMUP"
echo ""

# ── Build only what we need ───────────────────────────────────────────
echo "=== BUILD ==="
for lang in "${LANGUAGES[@]}"; do
    case "$lang" in
        c)
            echo "[C] building $BENCH..."
            (cd "$BENCHMARKS/c" && make "$BENCH" >/dev/null)
            ;;
        cpp)
            echo "[C++] building $BENCH..."
            (cd "$BENCHMARKS/cpp" && make "$BENCH" >/dev/null)
            ;;
        rust)
            echo "[Rust] building $BENCH..."
            (cd "$BENCHMARKS/rust" && cargo build --release --bin "$BENCH" 2>&1 | tail -1)
            ;;
        go)
            echo "[Go] building $BENCH..."
            (cd "$BENCHMARKS/go" && go build -o "$BENCH" "$BENCH.go")
            ;;
        java)
            CLASS=$(java_class "$BENCH")
            echo "[Java] building $CLASS..."
            (cd "$BENCHMARKS/java" && javac "${CLASS}.java")
            ;;
        zig)
            echo "[Zig] building all (zig build is whole-project)..."
            (cd "$BENCHMARKS/zig" && zig build 2>&1 | tail -1)
            ;;
        swift)
            echo "[Swift] building $BENCH..."
            (cd "$BENCHMARKS/swift" && swiftc -O -o "$BENCH" "${BENCH}.swift")
            ;;
        kotlin)
            echo "[Kotlin] building $BENCH..."
            (cd "$BENCHMARKS/kotlin" && kotlinc "${BENCH}.kt" -include-runtime -d "${BENCH}.jar" 2>/dev/null)
            ;;
        cobol)
            echo "[COBOL] building $BENCH..."
            (cd "$BENCHMARKS/cobol" && make "$BENCH" >/dev/null)
            ;;
        python|julia)
            ;; # interpreted, no build
    esac
done

# ── Build per-language command strings ────────────────────────────────
build_cmd() {
    local lang="$1"
    local bench="$2"
    local java_cls
    java_cls=$(java_class "$bench")

    local extra=""
    case "$bench" in
        b5_regex)      extra=" $DATA/regex_input.txt" ;;
        b8_json_parse) extra=" $DATA/json_input.json" ;;
        b6_file_io)    extra=" $DATA/fileio_test_${lang}.tmp" ;;
    esac

    case "$lang" in
        c)      echo "$BENCHMARKS/c/$bench$extra" ;;
        cpp)    echo "$BENCHMARKS/cpp/$bench$extra" ;;
        rust)   echo "$BENCHMARKS/rust/target/release/$bench$extra" ;;
        go)     echo "$BENCHMARKS/go/$bench$extra" ;;
        java)   echo "java -cp $BENCHMARKS/java $java_cls$extra" ;;
        python) echo "python3 $BENCHMARKS/python/${bench}.py$extra" ;;
        julia)
            # b7 needs real OS threads; -t 8 matches the C version's pthread count
            if [ "$bench" = "b7_concurrent" ]; then
                echo "julia -t 8 $BENCHMARKS/julia/${bench}.jl$extra"
            else
                echo "julia $BENCHMARKS/julia/${bench}.jl$extra"
            fi
            ;;
        zig)    echo "$BENCHMARKS/zig/zig-out/bin/$bench$extra" ;;
        swift)  echo "$BENCHMARKS/swift/$bench$extra" ;;
        kotlin) echo "java -jar $BENCHMARKS/kotlin/${bench}.jar$extra" ;;
        cobol)  echo "$BENCHMARKS/cobol/$bench$extra" ;;
    esac
}

# ── Run hyperfine with RSS-capturing wrapper (M1 timing + M2 memory in one pass)
echo ""
echo "=== RUN (M1 + M2 in one pass) ==="

RESULT_FILE="$RESULTS/${BENCH}.json"
RSS_DIR="$ROOT/.rss-tmp"
mkdir -p "$RSS_DIR"

HYPERFINE_ARGS=(--warmup "$WARMUP" --runs "$RUNS" --export-json "$RESULT_FILE")
for lang in "${LANGUAGES[@]}"; do
    cmd=$(build_cmd "$lang" "$BENCH")
    log_file="$RSS_DIR/${BENCH}_${lang}.log"
    rm -f "$log_file"   # fresh log per (bench, lang) per run
    HYPERFINE_ARGS+=(-n "$lang" "$SCRIPT_DIR/wrap_rss.sh $log_file $cmd")
done

hyperfine "${HYPERFINE_ARGS[@]}"

# ── Aggregate RSS logs into <bench>_memory.json ───────────────────────
echo ""
echo "=== MEMORY (aggregated from inline RSS capture) ==="
python3 "$SCRIPT_DIR/rss_log_to_json.py" "$BENCH" --rss-dir "$RSS_DIR" --langs "${LANGUAGES[@]}"

echo ""
echo "=== DONE ==="
echo "M1 timing: $RESULT_FILE"
echo "M2 memory: $RESULTS/${BENCH}_memory.json"
