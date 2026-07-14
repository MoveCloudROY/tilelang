"""Sweep PathLocalityReorder reg_budget and record uniform1d latency to CSV.

For each lmax the pass-off baseline is measured once (it does not depend on
reg_budget); each budget point then measures the pass with the default policy
and with allow_atomic_reorder (accumulator fusion). Rows are flushed to the
CSV incrementally so partial results survive interruption.

Run inside the build container:
    TILELANG_DISABLE_CACHE=1 python maint/plr_budget_sweep.py \
        --edges 65536 --nodes 4096 --budgets-from 16 --budgets-to 200 \
        --budgets-step 8 --csv /ws/plr_budget_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import sys

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plr_uniform1d_bench import bench, build_const_kernel, cg_paths, reference  # noqa: E402


def compile_variant(func, enable, reg_budget=None, relaxed=False):
    pass_configs = {"tl.disable_safe_memory_legalize": True}
    if enable:
        pass_configs["tl.enable_path_locality_reorder"] = True
        pass_configs["tl.PathLocalityReorder"] = {
            "max_paths": 1024,
            "reg_budget": int(reg_budget),
            "allow_atomic_reorder": relaxed,
        }
    return tilelang.compile(func, pass_configs=pass_configs)


def atomic_count(kernel):
    source = kernel.get_kernel_source()
    body = source[source.index("__launch_bounds__") :]
    return body.count("AtomicAdd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--lmax", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--budgets-from", type=int, default=16)
    parser.add_argument("--budgets-to", type=int, default=200)
    parser.add_argument("--budgets-step", type=int, default=8)
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    budgets = list(range(args.budgets_from, args.budgets_to + 1, args.budgets_step))
    dev = "cuda"
    print(f"device={torch.cuda.get_device_name(0)} budgets={budgets}", flush=True)

    fields = [
        "lmax",
        "num_paths",
        "reg_budget",
        "off_us",
        "on_default_us",
        "on_relaxed_us",
        "speedup_default",
        "speedup_relaxed",
        "atomics_off",
        "atomics_relaxed",
    ]
    csv_file = open(args.csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    csv_file.flush()

    for lmax in args.lmax:
        paths, dim, n_trios = cg_paths(lmax)
        num_paths = len(paths)
        func = build_const_kernel(paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"sweep_l{lmax}")

        torch.manual_seed(0)
        w = torch.randn(args.edges, dim, args.lanes, device=dev)
        x_all = torch.randn(args.nodes, dim, args.lanes, device=dev)
        y = torch.randn(args.edges, n_trios, device=dev)
        src = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
        dst = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
        b_list = torch.arange(args.edges, dtype=torch.int32, device=dev)
        ref = reference(paths, w, x_all, y, src, dst, args.nodes, dim, args.lanes)
        out = torch.zeros(args.nodes, dim, args.lanes, device=dev)

        def run_variant(kernel, w=w, x_all=x_all, y=y, out=out, src=src, dst=dst, b_list=b_list, ref=ref):
            out.zero_()
            kernel(w, x_all, y, out, src, dst, b_list)
            torch.testing.assert_close(out.double(), ref, rtol=1e-3, atol=1e-3)
            out.zero_()
            return bench(functools.partial(kernel, w, x_all, y, out, src, dst, b_list), args.iters, args.warmup)

        kernel_off = compile_variant(func, enable=False)
        off_ms = run_variant(kernel_off)
        atomics_off = atomic_count(kernel_off)
        print(f"lmax={lmax} P={num_paths} off={off_ms * 1e3:.1f}us atomics={atomics_off}", flush=True)

        for budget in budgets:
            kernel_default = compile_variant(func, enable=True, reg_budget=budget, relaxed=False)
            default_ms = run_variant(kernel_default)
            kernel_relaxed = compile_variant(func, enable=True, reg_budget=budget, relaxed=True)
            relaxed_ms = run_variant(kernel_relaxed)
            row = {
                "lmax": lmax,
                "num_paths": num_paths,
                "reg_budget": budget,
                "off_us": round(off_ms * 1e3, 2),
                "on_default_us": round(default_ms * 1e3, 2),
                "on_relaxed_us": round(relaxed_ms * 1e3, 2),
                "speedup_default": round(off_ms / default_ms, 4),
                "speedup_relaxed": round(off_ms / relaxed_ms, 4),
                "atomics_off": atomics_off,
                "atomics_relaxed": atomic_count(kernel_relaxed),
            }
            writer.writerow(row)
            csv_file.flush()
            print(
                f"lmax={lmax} budget={budget:>3} default={default_ms * 1e3:8.1f}us "
                f"relaxed={relaxed_ms * 1e3:8.1f}us ({row['speedup_relaxed']:.3f}x) "
                f"atomics={row['atomics_relaxed']}",
                flush=True,
            )

    csv_file.close()
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
