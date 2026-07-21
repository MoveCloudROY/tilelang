"""Dump the generated CUDA of every optimization stage measured by
plr_trace.py.

Reads the trace CSV to pick each (dtype, lmax) cell's latency-optimal
register budget, rebuilds the seven stage kernels and writes their CUDA
sources as

    u1d_<fp32|fp64>_l<lmax>_<stage>[_b<budget>].cu

    TILELANG_DISABLE_CACHE=1 python maint/plr_trace_dump.py \
        --csv plr_trace.csv --out-dir u1d_cuda_stages
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plr_trace import compile_variant, redundant_loads  # noqa: E402
from plr_uniform1d_bench import build_const_kernel, build_serial_kernel, cg_paths  # noqa: E402

from tilelang.transform import annotate_path_descriptors  # noqa: E402


def best_budgets(csv_path):
    sweep = defaultdict(dict)
    for row in csv.DictReader(open(csv_path)):
        if row["stage"] == "sweep_relaxed":
            sweep[(row["dtype"], int(row["lmax"]))][int(row["reg_budget"])] = float(row["lat_us"])
    return {cell: min(lats, key=lats.get) for cell, lats in sweep.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--warps", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    best = best_budgets(args.csv)
    cells = sorted(best)

    for dtype, lmax in cells:
        paths, dim, n_trios = cg_paths(lmax)
        short = dtype.replace("float", "fp")
        tag = f"dp_{short}_{lmax}"
        cfunc = build_const_kernel(paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"c{tag}", dtype)
        sfunc = build_serial_kernel(len(paths), args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"s{tag}", dtype)
        tables = {name: [p[i] for p in paths] for i, name in enumerate(["i_list", "j_list", "k_list", "v_list", "coeff_list"])}
        b_best = best[(dtype, lmax)]

        stages = [
            ("baseline_unrolled", compile_variant(cfunc, enable=False), None),
            ("original_loop", compile_variant(sfunc, enable=False), None),
            ("s1_load_reuse", compile_variant(cfunc, True, 32), 32),
            ("s2_reorder", compile_variant(cfunc, True, 32, relaxed=True, fuse_outputs=False), 32),
            ("s3_acc_fusion", compile_variant(cfunc, True, 16, relaxed=True), 16),
            ("s4_tuned_budget", compile_variant(cfunc, True, b_best, relaxed=True), b_best),
            (
                "final_serial_api",
                compile_variant(annotate_path_descriptors(sfunc, tables), True, b_best, relaxed=True),
                b_best,
            ),
        ]
        for stage, kernel, budget in stages:
            source = kernel.get_kernel_source()
            body = source[source.index("__launch_bounds__") :]
            suffix = "" if budget is None else f"_b{budget}"
            name = f"u1d_{short}_l{lmax}_{stage}{suffix}.cu"
            with open(os.path.join(args.out_dir, name), "w") as fh:
                fh.write(source)
            print(
                f"{name:<48} lines={len(source.splitlines()):<5} atomics={body.count('AtomicAdd'):<4} "
                f"redundant={redundant_loads(body)}",
                flush=True,
            )
    print("DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
