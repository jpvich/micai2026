#!/bin/bash
# Wrapper that runs a benchmark under /usr/bin/time -l, captures peak RSS,
# and forwards the program's exit code to hyperfine.
#
# Usage (invoked by run_single.sh, not directly):
#   wrap_rss.sh <rss_log_file> <command> [args...]
#
# Each invocation appends one line to <rss_log_file>: `<rss_bytes>\n`.
# Failed runs (non-zero exit) are NOT logged, keeping the data clean.
#
# Why: hyperfine's macOS memory reading is unreliable. /usr/bin/time -l
# reports ru_maxrss in BYTES (macOS-specific) which is the authoritative value.
# Combining them eliminates a separate measurement pass.

LOG="$1"
shift

TMP_ERR=$(mktemp)
EXIT=0
# stdout → /dev/null (benchmark output is irrelevant for measurement and would
# flood hyperfine). stderr captures BOTH program stderr and /usr/bin/time's
# report; awk grabs only the RSS line.
/usr/bin/time -l "$@" >/dev/null 2>"$TMP_ERR" || EXIT=$?

if [ "$EXIT" -eq 0 ]; then
    RSS=$(awk '/maximum resident set size/{print $1; exit}' "$TMP_ERR")
    [ -n "$RSS" ] && echo "$RSS" >> "$LOG"
fi

rm -f "$TMP_ERR"
exit "$EXIT"
