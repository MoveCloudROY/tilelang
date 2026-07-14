"""Plot PathLocalityReorder reg_budget sweep results.

Reads the CSV produced by plr_budget_sweep.py and renders one figure with
three panels: latency vs budget (per lmax), speedup vs budget, and fused
atomic count vs budget.

    python maint/plr_budget_plot.py --csv plr_budget_sweep.csv --out sweep.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(csv_path):
    rows = defaultdict(list)
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            rows[int(row["lmax"])].append(row)
    for lmax in rows:
        rows[lmax].sort(key=lambda r: int(r["reg_budget"]))
    return dict(sorted(rows.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="plr_budget_sweep.png")
    parser.add_argument("--title", default="uniform1d on H100: PathLocalityReorder reg_budget sweep")
    args = parser.parse_args()

    data = load(args.csv)
    lmaxes = list(data)
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, len(lmaxes), height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)

    # Row 1: latency vs budget, one panel per lmax.
    for i, lmax in enumerate(lmaxes):
        ax = fig.add_subplot(grid[0, i])
        rows = data[lmax]
        budgets = [int(r["reg_budget"]) for r in rows]
        off = float(rows[0]["off_us"])
        num_paths = rows[0]["num_paths"]
        ax.axhline(off, color="gray", linestyle="--", linewidth=1.5, label=f"pass off ({off:.0f} us)")
        ax.plot(budgets, [float(r["on_default_us"]) for r in rows], marker="s", markersize=4, color="tab:red", label="on (default policy)")
        ax.plot(
            budgets,
            [float(r["on_relaxed_us"]) for r in rows],
            marker="o",
            markersize=4,
            color=colors.get(lmax, "tab:blue"),
            label="on (relaxed + acc fusion)",
        )
        ax.set_title(f"lmax={lmax}  (P={num_paths} paths)")
        ax.set_xlabel("reg_budget")
        ax.set_ylabel("latency (us)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Row 2 left/middle: speedup vs budget (all lmax).
    ax = fig.add_subplot(grid[1, 0:2])
    for lmax in lmaxes:
        rows = data[lmax]
        budgets = [int(r["reg_budget"]) for r in rows]
        ax.plot(
            budgets,
            [float(r["speedup_relaxed"]) for r in rows],
            marker="o",
            markersize=4,
            color=colors.get(lmax),
            label=f"lmax={lmax} relaxed",
        )
        ax.plot(
            budgets,
            [float(r["speedup_default"]) for r in rows],
            marker="s",
            markersize=3,
            color=colors.get(lmax),
            alpha=0.35,
            linestyle=":",
            label=f"lmax={lmax} default",
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("reg_budget")
    ax.set_ylabel("speedup vs pass off")
    ax.set_title("Speedup vs reg_budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Row 2 right: atomic count vs budget (relaxed).
    ax = fig.add_subplot(grid[1, len(lmaxes) - 1])
    for lmax in lmaxes:
        rows = data[lmax]
        budgets = [int(r["reg_budget"]) for r in rows]
        ax.plot(
            budgets,
            [int(r["atomics_relaxed"]) for r in rows],
            marker="o",
            markersize=4,
            color=colors.get(lmax),
            label=f"lmax={lmax} relaxed",
        )
        ax.axhline(int(rows[0]["atomics_off"]), color=colors.get(lmax), linestyle="--", linewidth=1, alpha=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("reg_budget")
    ax.set_ylabel("atomics per warp (log)")
    ax.set_title("Fused atomic count (dashed = pass off)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(args.title, fontsize=14)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
