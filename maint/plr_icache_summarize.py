import csv
import glob
import re
import sys


def load_cell(path):
    m = re.search(r"ncu_l(\d+)_(\w+)\.csv", path)
    lmax, variant = int(m.group(1)), m.group(2)
    vals = {}
    with open(path) as fh:
        lines = [l for l in fh if l.startswith('"')]
    if not lines:
        return None
    reader = csv.DictReader(lines)
    for row in reader:
        name = row["Metric Name"]
        try:
            v = float(row["Metric Value"].replace(",", ""))
        except ValueError:
            continue
        vals.setdefault(name, []).append(v)
    avg = {k: sum(v) / len(v) for k, v in vals.items()}
    return lmax, variant, avg


SHORT = {
    "gpu__time_duration.sum": "dur_us",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy%",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct": "no_instr%",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "long_sb%",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct": "lg_throttle%",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct": "mio%",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct": "wait%",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct": "math%",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct": "drain%",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct": "not_sel%",
}
order = ["dur_us", "occupancy%", "no_instr%", "long_sb%", "lg_throttle%", "mio%", "wait%", "math%", "drain%", "not_sel%"]

cells = {}
for path in sorted(glob.glob(sys.argv[1] + "/ncu_l*_*.csv")):
    cell = load_cell(path)
    if cell:
        lmax, variant, avg = cell
        cells[(lmax, variant)] = {SHORT.get(k, k): v for k, v in avg.items()}

hdr = f"{'lmax':<5}{'variant':<15}" + "".join(f"{c:>12}" for c in order)
print(hdr)
print("-" * len(hdr))
for (lmax, variant), vals in sorted(cells.items()):
    # gpu__time_duration.sum is ns; convert to us
    if "dur_us" in vals:
        vals = dict(vals)
        vals["dur_us"] = vals["dur_us"] / 1000.0
    print(f"{lmax:<5}{variant:<15}" + "".join(f"{vals.get(c, float('nan')):>12.2f}" for c in order))
