"""Plot the optimization trace produced by plr_trace.py.

Generates several figures:
  <prefix>_fp32.png / _fp64.png   per-dtype optimization trajectory
  <prefix>_budget.png             latency vs register budget (all dtypes/lmax)
  <prefix>_metrics.png            atomics / SASS / redundant loads / registers

    python maint/plr_trace_plot.py --csv plr_trace.csv --prefix plr_trace
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGE_COLORS = {
    "baseline": "#9e9e9e",
    "original_loop": "#bdd7ee",
    "s1": "#f4b183",
    "s2": "#ffd966",
    "s3": "#a9d18e",
    "s4": "#70ad47",
    "final": "#2e75b6",
}


def load(csv_path):
    rows = defaultdict(list)  # (dtype, lmax) -> [row]
    for row in csv.DictReader(open(csv_path)):
        rows[(row["dtype"], int(row["lmax"]))].append(row)
    return rows


def stage_rows(cell):
    """Ordered (key, label, row) trajectory entries for one (dtype, lmax)."""
    by_stage = defaultdict(dict)
    for row in cell:
        budget = int(row["reg_budget"]) if row["reg_budget"] else None
        by_stage[row["stage"]][budget] = row

    sweep = by_stage.get("sweep_relaxed", {})
    best_budget = min(sweep, key=lambda b: float(sweep[b]["lat_us"])) if sweep else None

    entries = [
        ("baseline", "Baseline: fully unrolled, pass off", by_stage["baseline_unrolled"].get(None)),
        ("original_loop", "Original loop (runtime descriptors)", by_stage["original_loop"].get(None)),
        ("s1", "Step 1: + operand register reuse", by_stage["s1_load_reuse"].get(32)),
        ("s2", "Step 2: + cross-path reordering", by_stage["s2_reorder"].get(32)),
        ("s3", "Step 3: + accumulator fusion (b=16)", sweep.get(16)),
        ("s4", f"Step 4: + tuned budget (b={best_budget})", sweep.get(best_budget)),
        ("final", f"Final: loop API + specialization (b={best_budget})", by_stage["final_serial_api"].get(best_budget)),
    ]
    return [(key, label, row) for key, label, row in entries if row is not None], best_budget


def plot_trajectory(rows, dtype, out_path, title):
    lmaxes = sorted(lm for dt, lm in rows if dt == dtype)
    fig, axes = plt.subplots(1, len(lmaxes), figsize=(6.4 * len(lmaxes), 5.4))
    if len(lmaxes) == 1:
        axes = [axes]
    for ax, lmax in zip(axes, lmaxes):
        entries, _ = stage_rows(rows[(dtype, lmax)])
        labels = [label for _, label, _ in entries][::-1]
        lats = [float(row["lat_us"]) for _, _, row in entries][::-1]
        speedups = [float(row["speedup_vs_baseline"]) for _, _, row in entries][::-1]
        colors = [STAGE_COLORS[key] for key, _, _ in entries][::-1]
        bars = ax.barh(range(len(entries)), lats, color=colors)
        for i, (bar, lat, spd) in enumerate(zip(bars, lats, speedups)):
            ax.annotate(
                f" {lat:.0f} us ({spd:.2f}x)",
                (bar.get_width(), i),
                va="center",
                fontsize=8,
            )
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("latency (us)")
        num_paths = entries[0][2]["num_paths"]
        ax.set_title(f"lmax={lmax} ({num_paths} paths)")
        ax.set_xlim(0, max(lats) * 1.42)
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_budget(rows, out_path, title):
    dtypes = sorted({dt for dt, _ in rows})
    lmaxes = sorted({lm for _, lm in rows})
    fig, axes = plt.subplots(len(dtypes), len(lmaxes), figsize=(4.8 * len(lmaxes), 3.9 * len(dtypes)), squeeze=False)
    for di, dtype in enumerate(dtypes):
        for li, lmax in enumerate(lmaxes):
            ax = axes[di][li]
            cell = rows.get((dtype, lmax), [])
            sweep = sorted(
                ((int(r["reg_budget"]), float(r["lat_us"])) for r in cell if r["stage"] == "sweep_relaxed"),
            )
            base = next((float(r["lat_us"]) for r in cell if r["stage"] == "baseline_unrolled"), None)
            if sweep:
                budgets, lats = zip(*sweep)
                ax.plot(budgets, lats, marker="o", color="tab:blue", label="pass (all optimizations)")
                best_i = min(range(len(lats)), key=lambda i: lats[i])
                ax.annotate(
                    f"best: b={budgets[best_i]}\n{lats[best_i]:.0f} us",
                    (budgets[best_i], lats[best_i]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                    color="tab:blue",
                )
            if base is not None:
                ax.axhline(base, color="gray", linestyle="--", linewidth=1.2, label=f"baseline ({base:.0f} us)")
            ax.set_title(f"{dtype}, lmax={lmax}")
            ax.set_xlabel("register budget")
            ax.set_ylabel("latency (us)")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_metrics(rows, dtype, out_path, title):
    lmaxes = sorted(lm for dt, lm in rows if dt == dtype)
    panels = [
        ("atomics", "AtomicAdd per warp (log)", True, int),
        ("sass_instrs", "SASS instructions", False, int),
        ("redundant_loads", "redundant repeated-address accesses", False, int),
        ("regs", "ptxas registers", False, int),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for ax, (field, ylabel, log, conv) in zip(axes.flat, panels):
        entry_sets = {lmax: stage_rows(rows[(dtype, lmax)])[0] for lmax in lmaxes}
        keys = [key for key, _, _ in entry_sets[lmaxes[0]]]
        labels = {key: label.split(":")[0] for key, label, _ in entry_sets[lmaxes[0]]}
        width = 0.8 / len(keys)
        for si, key in enumerate(keys):
            xs, ys = [], []
            for li, lmax in enumerate(lmaxes):
                row = next((r for k, _, r in entry_sets[lmax] if k == key), None)
                if row is None:
                    continue
                xs.append(li + si * width - 0.4 + width / 2)
                ys.append(conv(row[field]))
            bars = ax.bar(xs, ys, width=width * 0.9, color=STAGE_COLORS[key], label=labels[key])
            for bar, value in zip(bars, ys):
                ax.annotate(f"{value}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=6)
        if log:
            ax.set_yscale("log")
        if field == "regs":
            ax.axhline(255, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(lmaxes)))
        ax.set_xticklabels([f"lmax={lm}" for lm in lmaxes])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y", which="both")
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prefix", default="plr_trace")
    args = parser.parse_args()

    rows = load(args.csv)
    dtypes = sorted({dt for dt, _ in rows})
    for dtype in dtypes:
        plot_trajectory(
            rows, dtype, f"{args.prefix}_{dtype.replace('float', 'fp')}.png", f"uniform1d optimization trajectory on H100 ({dtype})"
        )
        plot_metrics(rows, dtype, f"{args.prefix}_metrics_{dtype.replace('float', 'fp')}.png", f"uniform1d per-stage metrics ({dtype})")
    plot_budget(rows, f"{args.prefix}_budget.png", "uniform1d latency vs register budget (all optimizations on)")


if __name__ == "__main__":
    main()
