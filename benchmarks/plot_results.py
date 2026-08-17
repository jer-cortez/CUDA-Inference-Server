"""Charts for the load-test results. Produces the figures used in the README.

    python benchmarks/plot_results.py [--input latest] [--dark]

Two figures, because throughput and latency answer different questions and
sharing an axis between them would be a lie about scale:

  throughput_vs_concurrency.png -- does batching serve more requests per second?
  latency_percentiles.png       -- what does that cost an individual request?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Non-interactive backend, selected before pyplot is imported: this runs over
# SSH on a headless GPU box where the default backend would fail to open a
# display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Categorical slots 1 and 2 of the reference palette, validated as a pair
# (CVD dE 24.7 protan, normal-vision dE 33.6 -- both far above their floors).
# Colour identifies the mode and nothing else; it never encodes rank.
LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#dedcd5",
    "dynamic": "#2a78d6",
    "serial": "#eb6834",
}
DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#3a3a37",
    "dynamic": "#3987e5",
    "serial": "#d95926",
}

MODE_LABEL = {"dynamic": "Dynamic batching", "serial": "Serial (batch size 1)"}


def load(stem: str) -> dict:
    path = RESULTS_DIR / f"{stem}.json"
    if not path.is_file():
        raise SystemExit(f"no results at {path}; run benchmarks/load_test.py first")
    return json.loads(path.read_text())


def style_axes(ax, theme: dict) -> None:
    """Recessive grid and axes so the data marks carry the chart."""
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9)


def plot_throughput(data: dict, theme: dict, out_path: Path) -> None:
    runs = data["runs"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=theme["surface"])
    style_axes(ax, theme)

    series = {}
    for mode in ("serial", "dynamic"):
        points = sorted(
            ((r["concurrency"], r["throughput_rps"]) for r in runs if r["mode"] == mode)
        )
        if not points:
            continue
        xs, ys = zip(*points)
        series[mode] = dict(zip(xs, ys))
        # 2px lines, >=8px markers per the mark spec.
        ax.plot(xs, ys, color=theme[mode], linewidth=2, marker="o", markersize=8,
                markeredgecolor=theme["surface"], markeredgewidth=2,
                label=MODE_LABEL[mode], zorder=3)

    # Annotate the peak speedup directly on the chart: this is the number the
    # figure exists to communicate, and a reader should not have to compute it.
    if "dynamic" in series and "serial" in series:
        shared = sorted(set(series["dynamic"]) & set(series["serial"]))
        if shared:
            best = max(shared, key=lambda c: series["dynamic"][c] / max(series["serial"][c], 1e-9))
            dynamic_y = series["dynamic"][best]
            factor = dynamic_y / max(series["serial"][best], 1e-9)

            # Keep the label inside the axes. Centred text on the first or last
            # point runs half its width past the edge and gets clipped, so the
            # edge cases anchor inward instead.
            if best == shared[-1]:
                ha, x_offset = "right", -8
            elif best == shared[0]:
                ha, x_offset = "left", 8
            else:
                ha, x_offset = "center", 0

            # Place the label on the far side from the other series, so it does
            # not land on top of that line where the two run close together.
            above = dynamic_y >= series["serial"][best]
            y_offset = 18 if above else -26

            # ...unless that would push it into the title. The annotated point
            # is usually the throughput peak, which sits near the top of the
            # axes, so "above" is exactly where there is no room.
            peak = max(max(series[mode].values()) for mode in series)
            if y_offset > 0 and dynamic_y > 0.85 * peak:
                y_offset = -26

            ax.annotate(
                f"{factor:.1f}x at concurrency {best}",
                xy=(best, dynamic_y),
                xytext=(x_offset, y_offset), textcoords="offset points",
                ha=ha, fontsize=10, fontweight="bold", color=theme["text"],
            )

    ax.set_xlabel("Concurrent clients", color=theme["muted"], fontsize=10)
    ax.set_ylabel("Throughput (requests/sec)", color=theme["muted"], fontsize=10)
    ax.set_title("Throughput vs. concurrency", color=theme["text"], fontsize=13,
                 fontweight="bold", loc="left", pad=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["concurrency"] for r in runs}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)

    legend = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for text in legend.get_texts():
        text.set_color(theme["text"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_latency(data: dict, theme: dict, out_path: Path, concurrency: int | None) -> None:
    runs = data["runs"]
    levels = sorted({r["concurrency"] for r in runs})
    if concurrency is None:
        concurrency = levels[-1]  # highest load: where the modes differ most

    selected = {r["mode"]: r for r in runs if r["concurrency"] == concurrency}
    if not selected:
        raise SystemExit(f"no runs at concurrency {concurrency}")

    metrics = ["p50", "p90", "p99"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=theme["surface"])
    style_axes(ax, theme)
    ax.grid(axis="x", visible=False)

    width = 0.34
    positions = range(len(metrics))
    for offset, mode in ((-width / 2 - 0.01, "serial"), (width / 2 + 0.01, "dynamic")):
        run = selected.get(mode)
        if run is None:
            continue
        values = [run["latency_ms"][m] for m in metrics]
        bars = ax.bar([p + offset for p in positions], values, width,
                      color=theme[mode], label=MODE_LABEL[mode], zorder=3)
        # Direct labels: three bars per series is few enough that every value
        # can be labelled without clutter, and it removes a lookup to the axis.
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.0f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=theme["text"])

    ax.set_xticks(list(positions))
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Latency (ms)", color=theme["muted"], fontsize=10)
    ax.set_title(f"Latency percentiles at concurrency {concurrency}",
                 color=theme["text"], fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.set_ylim(bottom=0)

    legend = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for text in legend.get_texts():
        text.set_color(theme["text"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


def print_table(data: dict) -> None:
    """Text equivalent of the figures, so the numbers are readable without them."""
    print(f"\n{'mode':<9}{'conc':>6}{'rps':>10}{'p50':>9}{'p90':>9}{'p99':>9}{'batch':>8}")
    print("-" * 60)
    for run in sorted(data["runs"], key=lambda r: (r["concurrency"], r["mode"])):
        lat = run["latency_ms"]
        print(f"{run['mode']:<9}{run['concurrency']:>6}{run['throughput_rps']:>10.1f}"
              f"{lat['p50']:>9.1f}{lat['p90']:>9.1f}{lat['p99']:>9.1f}"
              f"{run.get('observed_mean_batch', 0):>8.1f}")

    meta = data.get("metadata", {})
    print(f"\nGPU: {meta.get('gpu', 'unknown')}   model: {meta.get('model_path', 'unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="latest", help="result stem under benchmarks/results/")
    parser.add_argument("--dark", action="store_true", help="render for a dark surface")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="concurrency level for the latency chart (default: highest)")
    args = parser.parse_args()

    data = load(args.input)
    theme = DARK if args.dark else LIGHT
    suffix = "_dark" if args.dark else ""

    plot_throughput(data, theme, RESULTS_DIR / f"throughput_vs_concurrency{suffix}.png")
    plot_latency(data, theme, RESULTS_DIR / f"latency_percentiles{suffix}.png",
                 args.concurrency)
    print_table(data)


if __name__ == "__main__":
    main()
