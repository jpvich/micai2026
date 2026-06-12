#!/usr/bin/env python3
"""Generate paper figures from benchmark results (MICAI: 6 languages x 5 AI benchmarks).

Outputs PDF + PNG of each figure to ../paper/figures/.

Data sources:
  results/<bench>.json          M1 wall-time (hyperfine)
  results/<bench>_memory.json   M2 peak RSS (wrap_rss.sh aggregator)
  results/m3_binary_size.json   M3 distributable artifact size
  results/m4_loc.json           M4 implementation LOC
  results/m5_compile_time.json  M5 cold-build wall time (per language total)

Robust against language ordering: every entry in every JSON carries a
"command" or "loc_per_bench"/"bytes_per_bench" keyed by language name, so we
look up by name rather than index. (Some early benchmark runs had julia in
the middle of the language sweep, post-2026-05-30 runs have julia last — the
JSONs identify their own rows so this doesn't matter.)
"""
import json
import os
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Config ──────────────────────────────────────────────────────────────
LANGUAGES = ["c", "cpp", "rust", "go", "python", "julia"]
LANG_LABELS = {
    "c": "C", "cpp": "C++", "rust": "Rust", "go": "Go",
    "python": "Python", "julia": "Julia",
}
BENCHMARKS = ["kmeans", "knn", "mlp", "ga", "fuzzy"]
BENCH_LABELS = {
    "kmeans": "K-Means", "knn": "k-NN", "mlp": "MLP",
    "ga": "GA", "fuzzy": "Fuzzy",
}

# Color scheme: AOT-native warm reds/oranges/browns, GC tier green, Python blue,
# Julia purple. Chosen to remain distinguishable in greyscale print.
COLORS = {
    "c":      "#8c2d04",  # dark brown
    "cpp":    "#cc4c02",  # burnt orange
    "rust":   "#d62728",  # red
    "go":     "#2ca02c",  # green
    "python": "#1f77b4",  # blue
    "julia":  "#762a83",  # purple
}

plt.rcParams.update({
    "font.size": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "serif",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# ── Data loading ────────────────────────────────────────────────────────
def load_m1(results_dir):
    """Return {bench: {lang: mean_seconds}}."""
    data = {}
    for b in BENCHMARKS:
        fp = os.path.join(results_dir, f"{b}.json")
        if not os.path.exists(fp):
            print(f"  WARN missing {fp}")
            continue
        with open(fp) as f:
            raw = json.load(f)
        data[b] = {r["command"]: r["mean"] for r in raw["results"]}
    return data


def load_m2(results_dir):
    """Return {bench: {lang: peak_rss_bytes_mean}}."""
    data = {}
    for b in BENCHMARKS:
        fp = os.path.join(results_dir, f"{b}_memory.json")
        if not os.path.exists(fp):
            print(f"  WARN missing {fp}")
            continue
        with open(fp) as f:
            raw = json.load(f)
        data[b] = {r["command"]: r["rss_bytes_mean"] for r in raw["results"]}
    return data


def load_m3(results_dir):
    """Return {lang: {bench: bytes}} with julia → all None."""
    with open(os.path.join(results_dir, "m3_binary_size.json")) as f:
        raw = json.load(f)
    return {r["command"]: r["bytes_per_bench"] for r in raw["results"]}


def load_m4(results_dir):
    """Return {lang: {bench: loc}}."""
    with open(os.path.join(results_dir, "m4_loc.json")) as f:
        raw = json.load(f)
    return {r["command"]: r["loc_per_bench"] for r in raw["results"]}


def load_m5(results_dir):
    """Return {lang: total_compile_seconds or None for julia}."""
    with open(os.path.join(results_dir, "m5_compile_time.json")) as f:
        raw = json.load(f)
    return {r["command"]: r["compile_time_s"] for r in raw["results"]}


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ── Figure 1: Wall-time grouped bar chart (log scale) ──────────────────
def fig1_walltime(m1, output_dir):
    """Grouped bar: per-benchmark wall time, one bar per language. Log y."""
    benches = [b for b in BENCHMARKS if b in m1]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(benches))
    width = 0.08
    n = len(LANGUAGES)

    for i, lang in enumerate(LANGUAGES):
        means = [m1[b].get(lang) for b in benches]
        means = [v if v is not None else float("nan") for v in means]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width,
               label=LANG_LABELS[lang], color=COLORS[lang],
               edgecolor="white", linewidth=0.3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABELS[b] for b in benches])
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("Execution time per benchmark across the six languages")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6, fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(output_dir, f"fig1_walltime.{ext}"))
    plt.close()
    print("  fig1_walltime")


# ── Figure 2: Slowdown-relative-to-C heatmap ───────────────────────────
def fig2_vs_c_heatmap(m1, output_dir):
    """Heatmap: how many times slower than C is each (lang, bench)? 1.0 = tied, >1 = slower."""
    benches = [b for b in BENCHMARKS if b in m1 and "c" in m1[b]]
    matrix = np.full((len(LANGUAGES), len(benches)), np.nan)
    for i, lang in enumerate(LANGUAGES):
        for j, b in enumerate(benches):
            cv = m1[b].get("c")
            lv = m1[b].get(lang)
            if cv and lv:
                matrix[i, j] = lv / cv

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    # log-color so 0.1× and 10× are visually symmetric around 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   norm=LogNorm(vmin=0.05, vmax=20))
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels([BENCH_LABELS[b] for b in benches], rotation=20, ha="right")
    ax.set_yticks(range(len(LANGUAGES)))
    ax.set_yticklabels([LANG_LABELS[l] for l in LANGUAGES])
    for i in range(len(LANGUAGES)):
        for j in range(len(benches)):
            v = matrix[i, j]
            if not np.isnan(v):
                txt = f"{v:.2f}×" if v < 10 else f"{v:.0f}×"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if (v > 5 or v < 0.2) else "black")
    cb = plt.colorbar(im, ax=ax, label="Wall-time relative to C (=1.0)")
    cb.set_ticks([0.1, 0.5, 1, 2, 5, 10])
    cb.set_ticklabels(["0.1×", "0.5×", "1×", "2×", "5×", "10×"])
    ax.set_title("Wall-time relative to C (green = faster, red = slower)")
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(output_dir, f"fig2_vs_c_heatmap.{ext}"))
    plt.close()
    print("  fig2_vs_c_heatmap")


# ── Figure 3: Memory clustering (log scale, per benchmark) ─────────────
def fig3_rss(m2, output_dir):
    """Per-benchmark RSS, one bar per language. Log scale so all tiers visible."""
    benches = [b for b in BENCHMARKS if b in m2]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(benches))
    width = 0.08
    n = len(LANGUAGES)

    for i, lang in enumerate(LANGUAGES):
        vals = [(m2[b].get(lang) or float("nan")) / 1048576 for b in benches]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               label=LANG_LABELS[lang], color=COLORS[lang],
               edgecolor="white", linewidth=0.3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABELS[b] for b in benches])
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Peak RSS (MB, log scale)")
    ax.set_title("Peak resident-set size per benchmark across the six languages")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6, fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(output_dir, f"fig3_rss.{ext}"))
    plt.close()
    print("  fig3_rss")


# ── Figure 4: LOC vs time scatter (productivity vs performance) ────────
def fig4_loc_vs_time(m1, m4, output_dir):
    """Productivity vs performance. X = mean LOC, Y = geomean wall-time (log).
    The takeaway: code length barely varies (~1.4x) while speed varies ~300x,
    and Python sits at almost exactly C's code length yet ~300x slower."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    pts = {}
    for lang in LANGUAGES:
        loc_vals = [v for v in m4.get(lang, {}).values() if v is not None]
        times = [m1[b][lang] for b in BENCHMARKS if b in m1 and lang in m1[b]]
        if not loc_vals or not times:
            continue
        pts[lang] = (sum(loc_vals) / len(loc_vals), geomean(times))
    c_time = pts["c"][1]

    # faint horizontal tier bands relative to C (systems / managed / interpreted)
    ax.axhspan(c_time * 0.5, c_time * 1.5,   color="#2ca02c", alpha=0.06)
    ax.axhspan(c_time * 1.5, c_time * 50,    color="#ff7f0e", alpha=0.06)
    ax.axhspan(c_time * 50,  c_time * 1000,  color="#d62728", alpha=0.06)

    # label offsets to avoid the C/C++ overlap at the bottom-left
    dy = {"c": 9, "cpp": -13}
    for lang, (x, y) in pts.items():
        ax.scatter(x, y, s=210, c=COLORS[lang], edgecolor="black", linewidth=0.7, zorder=5)
        slow = y / c_time
        tag = f"{LANG_LABELS[lang]} ({slow:.0f}×)" if slow >= 2 else f"{LANG_LABELS[lang]} ({slow:.2f}×)"
        ax.annotate(tag, (x, y), fontsize=9, fontweight="bold",
                    textcoords="offset points", xytext=(11, dy.get(lang, 0)), va="center")

    # connector: C and Python share almost the same LOC -> the headline contrast
    cx, cy = pts["c"]; px, py = pts["python"]
    ax.annotate("", xy=(px, py), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.4, ls="--"))
    ax.text(cx - 1.5, (cy * py) ** 0.5, "same code length,\n${\\sim}300\\times$ slower",
            fontsize=9, ha="right", va="center", color="0.3", style="italic")

    ax.set_yscale("log")
    ax.set_xlabel("Mean lines of code per implementation")
    ax.set_ylabel("Geometric-mean wall-time over 5 benchmarks (s, log scale)")
    ax.set_title("Code length buys little: speed spans ${\\sim}300\\times$, length barely $1.4\\times$")
    ax.margins(x=0.20)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(output_dir, f"fig4_loc_vs_time.{ext}"), bbox_inches="tight")
    plt.close()
    print("  fig4_loc_vs_time")


# ── Figure 5 (static metrics scatter) removed: redundant with the M3/M4/M5
#    table in the paper, and "compile time vs binary size" is not a real
#    trade-off axis. The table communicates those numbers more clearly.


# ── Figure 6: Radar — multi-dimensional comparison of top languages ────
def fig6_radar(m1, m2, m3, m4, m5, output_dir):
    """Normalized radar for all six languages across 5 dimensions.

    Each axis scored 0..1 where 1 = best observed across the 9-language set.
    Dimensions: Speed (1/M1 geomean), Memory (1/M2 geomean), Compactness (1/M4 mean),
    Compile (1/M5), Binary (1/M3 mean). Julia compile=N/A → use median of others.
    """
    # compute per-lang raw scores (higher = better, so use reciprocal of cost)
    raw = {}
    for lang in LANGUAGES:
        times = [m1[b][lang] for b in BENCHMARKS if b in m1 and lang in m1[b]]
        rsses = [m2[b][lang] for b in BENCHMARKS if b in m2 and lang in m2[b]]
        locs = [v for v in m4.get(lang, {}).values() if v is not None]
        sizes = [v for v in (m3.get(lang) or {}).values() if v is not None]
        ct = m5.get(lang)
        raw[lang] = {
            "Speed": 1.0 / geomean(times) if times else None,
            "Memory":  1.0 / geomean(rsses) if rsses else None,
            "LOC":     1.0 / (sum(locs)/len(locs)) if locs else None,
            "Compile": 1.0 / ct if ct else None,
            "Binary":  1.0 / (sum(sizes)/len(sizes)) if sizes else None,
        }

    dims = ["Speed", "Memory", "LOC", "Compile", "Binary"]
    # normalize each dimension to [0, 1] over languages with data
    normed = {l: {} for l in LANGUAGES}
    for d in dims:
        vals = [raw[l][d] for l in LANGUAGES if raw[l][d] is not None]
        if not vals:
            continue
        mx = max(vals)
        for l in LANGUAGES:
            v = raw[l][d]
            normed[l][d] = (v / mx) if v is not None else 0.0  # 0 = missing

    # Show all six languages (C and C++ nearly overlap, confirming their tie).
    selected = LANGUAGES

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for lang in selected:
        vals = [normed[lang].get(d, 0.0) for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=LANG_LABELS[lang], color=COLORS[lang])
        ax.fill(angles, vals, alpha=0.10, color=COLORS[lang])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Five-dimensional comparison (1.0 = best across the six languages)",
                 pad=20, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(output_dir, f"fig6_radar.{ext}"))
    plt.close()
    print("  fig6_radar")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(here, "..", "results"))
    ap.add_argument("--output-dir", default=os.path.join(here, "..", "paper", "figures"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading from {args.results_dir}")
    m1 = load_m1(args.results_dir)
    m2 = load_m2(args.results_dir)
    m3 = load_m3(args.results_dir)
    m4 = load_m4(args.results_dir)
    m5 = load_m5(args.results_dir)

    print(f"Writing figures to {args.output_dir}")
    fig1_walltime(m1, args.output_dir)
    fig2_vs_c_heatmap(m1, args.output_dir)
    fig3_rss(m2, args.output_dir)
    fig4_loc_vs_time(m1, m4, args.output_dir)
    fig6_radar(m1, m2, m3, m4, m5, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
