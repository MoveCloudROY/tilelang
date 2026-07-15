"""Plot the complete uniform1d performance matrix from plr_uniform1d_full.py.

python maint/plr_uniform1d_plot.py --csv plr_u1d_full.csv --out plr_u1d_full.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(csv_path):
    data = defaultdict(dict)  # lmax -> (variant, budget) -> row
    budgets = set()
    for row in csv.DictReader(open(csv_path)):
        budget = int(row["reg_budget"]) if row["reg_budget"] else None
        if budget is not None:
            budgets.add(budget)
        data[int(row["lmax"])][(row["variant"], budget)] = row
    return dict(sorted(data.items())), sorted(budgets)


def bar_group(ax, lmaxes, series, value_fn, fmt, log=False):
    width = 0.8 / len(series)
    for si, (label, key_fn, color) in enumerate(series):
        xs, ys = [], []
        for li, lmax in enumerate(lmaxes):
            row = key_fn(lmax)
            if row is None:
                continue
            xs.append(li + si * width - 0.4 + width / 2)
            ys.append(value_fn(row))
        bars = ax.bar(xs, ys, width=width * 0.92, label=label, color=color)
        for bar, value in zip(bars, ys):
            ax.annotate(
                fmt(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=0,
            )
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(lmaxes)))
    ax.set_xticklabels([f"lmax={lm}" for lm in lmaxes])
    ax.grid(alpha=0.25, axis="y", which="both")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="plr_u1d_full.png")
    parser.add_argument("--title", default="uniform1d on H100: complete PathLocalityReorder matrix")
    args = parser.parse_args()

    data, budgets = load(args.csv)
    lmaxes = list(data)
    b_lo, b_hi = (budgets[0], budgets[-1]) if budgets else (None, None)

    def get(lmax, variant, budget=None):
        return data[lmax].get((variant, budget))

    series_lat = [
        ("baseline (pass off)", lambda lm: get(lm, "const_off"), "tab:gray"),
        ("serial loop (off)", lambda lm: get(lm, "serial_off"), "lightsteelblue"),
        (f"pass default (b={b_hi})", lambda lm: get(lm, "const_default", b_hi), "tab:red"),
        (f"pass relaxed (b={b_lo})", lambda lm: get(lm, "const_relaxed", b_lo), "tab:cyan"),
        (f"pass relaxed (b={b_hi})", lambda lm: get(lm, "const_relaxed", b_hi), "tab:blue"),
        (f"serial+desc relaxed (b={b_hi})", lambda lm: get(lm, "serial_desc", b_hi), "tab:green"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0][0]
    bar_group(ax, lmaxes, series_lat, lambda r: float(r["lat_us"]), lambda v: f"{v:.0f}", log=True)
    ax.set_ylabel("latency (us, log)")
    ax.set_title("Latency per variant")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0][1]
    bar_group(ax, lmaxes, series_lat[1:], lambda r: float(r["speedup_vs_off"]), lambda v: f"{v:.2f}x")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("speedup vs baseline")
    ax.set_title("Speedup vs fully-unrolled baseline")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1][0]
    series_atomics = [
        ("baseline (= P paths)", lambda lm: get(lm, "const_off"), "tab:gray"),
        (f"relaxed (b={b_lo})", lambda lm: get(lm, "const_relaxed", b_lo), "tab:cyan"),
        (f"relaxed (b={b_hi})", lambda lm: get(lm, "const_relaxed", b_hi), "tab:blue"),
    ]
    bar_group(ax, lmaxes, series_atomics, lambda r: int(r["atomics"]), lambda v: f"{v:.0f}", log=True)
    for li, lm in enumerate(lmaxes):
        dim = int(data[lm][("const_off", None)]["dim"])
        ax.hlines(dim, li - 0.4, li + 0.4, color="tab:green", linestyle=":", linewidth=1.5)
    ax.set_ylabel("AtomicAdd per warp (log)")
    ax.set_title("Atomic count (dotted green = per-element minimum)")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    series_regs = [
        ("baseline", lambda lm: get(lm, "const_off"), "tab:gray"),
        (f"default (b={b_hi})", lambda lm: get(lm, "const_default", b_hi), "tab:red"),
        (f"relaxed (b={b_hi})", lambda lm: get(lm, "const_relaxed", b_hi), "tab:blue"),
        ("serial loop (off)", lambda lm: get(lm, "serial_off"), "lightsteelblue"),
    ]
    bar_group(ax, lmaxes, series_regs, lambda r: int(r["regs"]), lambda v: f"{v:.0f}")
    ax.axhline(255, color="tab:red", linestyle="--", linewidth=1)
    ax.text(0.02, 256, "255 register ceiling", fontsize=7, color="tab:red")
    ax.set_ylabel("ptxas registers")
    ax.set_title("Register usage")
    ax.legend(fontsize=7)

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
