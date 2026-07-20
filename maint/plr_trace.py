"""Optimization-trace measurements for tl.PathLocalityReorder on uniform1d.

Reproduces every optimization step as a configuration of the current
compiler, for fp32 and fp64, across lmax path tables and register budgets:

  baseline_unrolled  fully unrolled kernel, pass disabled
  original_loop      runtime-descriptor serial loop, pass disabled
  s1_load_reuse      pass on, order-preserving policy (register reuse of
                     repeated operand loads + scheduling)
  s2_reorder         + relaxed same-output ordering, still one atomic per
                     path (enable_output_accumulation=False)
  sweep_relaxed      + accumulator fusion (one atomic per output element),
                     measured at each register budget; budget 16 is
                     "step 3" and the latency-optimal budget is "step 4"
  final_serial_api   the loop-form kernel + annotate_path_descriptors at
                     the optimal budget (usability endpoint)

Every kernel is verified against a float64 reference. Rows are flushed to
the CSV incrementally and include latency, ptxas registers, SASS size,
atomic count and the number of redundant (repeated-address) memory
accesses in the emitted CUDA.

    TILELANG_DISABLE_CACHE=1 python maint/plr_trace.py --csv /ws/plr_trace.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import re
import sys
from collections import Counter

import torch

import tilelang
from tilelang.transform import annotate_path_descriptors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plr_uniform1d_bench import (  # noqa: E402
    bench,
    build_const_kernel,
    build_serial_kernel,
    cg_paths,
    reference,
)
from plr_uniform1d_full import sass_stats  # noqa: E402

_TENSORS = "w|x_all|y|b_list|src_idx|dst_idx|i_list|j_list|k_list|v_list|coeff_list|out"


def redundant_loads(body):
    """Number of redundant repeated-address memory accesses in the source."""
    accesses = Counter(f"{m.group(1)}[{m.group(2)}]" for m in re.finditer(rf"\b({_TENSORS})\[((?:[^\[\]]|\[[^\[\]]*\])*)\]", body))
    return sum(count - 1 for count in accesses.values() if count >= 2)


def compile_variant(func, enable, reg_budget=16, relaxed=False, fuse_outputs=True):
    pass_configs = {"tl.disable_safe_memory_legalize": True}
    if enable:
        pass_configs["tl.enable_path_locality_reorder"] = True
        pass_configs["tl.PathLocalityReorder"] = {
            "max_paths": 1024,
            "reg_budget": int(reg_budget),
            "allow_atomic_reorder": relaxed,
            "enable_output_accumulation": fuse_outputs,
        }
    return tilelang.compile(func, pass_configs=pass_configs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--lmax", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dtypes", type=str, nargs="+", default=["float32", "float64"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 24, 32, 40, 48, 64, 96])
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    dev = "cuda"
    print(f"device={torch.cuda.get_device_name(0)} dtypes={args.dtypes} budgets={args.budgets}", flush=True)

    fields = [
        "dtype",
        "lmax",
        "num_paths",
        "dim",
        "stage",
        "reg_budget",
        "lat_us",
        "speedup_vs_baseline",
        "regs",
        "sass_instrs",
        "atomics",
        "redundant_loads",
        "max_rel_err",
    ]
    csv_file = open(args.csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    csv_file.flush()

    for dtype in args.dtypes:
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        tol = 1e-3 if dtype == "float32" else 1e-6
        for lmax in args.lmax:
            paths, dim, n_trios = cg_paths(lmax)
            num_paths = len(paths)
            tag = f"tr_{dtype[-2:]}_{lmax}"
            cfunc = build_const_kernel(paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"c{tag}", dtype)
            sfunc = build_serial_kernel(num_paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"s{tag}", dtype)
            tables = {name: [p[i] for p in paths] for i, name in enumerate(["i_list", "j_list", "k_list", "v_list", "coeff_list"])}

            torch.manual_seed(0)
            w = torch.randn(args.edges, dim, args.lanes, dtype=torch_dtype, device=dev)
            x_all = torch.randn(args.nodes, dim, args.lanes, dtype=torch_dtype, device=dev)
            y = torch.randn(args.edges, n_trios, dtype=torch_dtype, device=dev)
            src = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
            dst = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
            b_list = torch.arange(args.edges, dtype=torch.int32, device=dev)
            desc = tuple(
                torch.tensor(tables[name], dtype=torch.int32 if name != "coeff_list" else torch_dtype, device=dev)
                for name in ["i_list", "j_list", "k_list", "v_list", "coeff_list"]
            )
            ref = reference(paths, w, x_all, y, src, dst, args.nodes, dim, args.lanes)
            out = torch.zeros(args.nodes, dim, args.lanes, dtype=torch_dtype, device=dev)

            def measure(kernel, serial, out=out, w=w, x_all=x_all, y=y, src=src, dst=dst, b_list=b_list, desc=desc, ref=ref, tol=tol):
                call = (w, x_all, y, out, src, dst, b_list, *desc) if serial else (w, x_all, y, out, src, dst, b_list)
                out.zero_()
                kernel(*call)
                max_rel = ((out.double() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
                torch.testing.assert_close(out.double(), ref, rtol=tol, atol=tol)
                out.zero_()
                return bench(functools.partial(kernel, *call), args.iters, args.warmup), max_rel

            def record(stage, kernel, budget, serial, base_ms=None, dtype=dtype, lmax=lmax, num_paths=num_paths, dim=dim, measure=measure):
                ms, max_rel = measure(kernel, serial)
                source = kernel.get_kernel_source()
                body = source[source.index("__launch_bounds__") :]
                stats = sass_stats(source, args.arch)
                row = {
                    "dtype": dtype,
                    "lmax": lmax,
                    "num_paths": num_paths,
                    "dim": dim,
                    "stage": stage,
                    "reg_budget": budget if budget is not None else "",
                    "lat_us": round(ms * 1e3, 2),
                    "speedup_vs_baseline": round(base_ms / ms, 4) if base_ms else 1.0,
                    "regs": stats["regs"],
                    "sass_instrs": stats["sass_instrs"],
                    "atomics": body.count("AtomicAdd"),
                    "redundant_loads": redundant_loads(body),
                    "max_rel_err": f"{max_rel:.2e}",
                }
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"{dtype} lmax={lmax} {stage:<18} b={str(budget) if budget is not None else '-':>3} "
                    f"lat={ms * 1e3:9.1f}us ({row['speedup_vs_baseline']:6.3f}x) regs={stats['regs']:<3} "
                    f"atomics={row['atomics']:<4} redundant={row['redundant_loads']}",
                    flush=True,
                )
                return ms

            base_ms = record("baseline_unrolled", compile_variant(cfunc, enable=False), None, False)
            record("original_loop", compile_variant(sfunc, enable=False), None, True, base_ms)
            record("s1_load_reuse", compile_variant(cfunc, True, 32), 32, False, base_ms)
            record("s2_reorder", compile_variant(cfunc, True, 32, relaxed=True, fuse_outputs=False), 32, False, base_ms)
            best = (None, None)
            for budget in args.budgets:
                ms = record("sweep_relaxed", compile_variant(cfunc, True, budget, relaxed=True), budget, False, base_ms)
                if best[0] is None or ms < best[0]:
                    best = (ms, budget)
            record(
                "final_serial_api",
                compile_variant(annotate_path_descriptors(sfunc, tables), True, best[1], relaxed=True),
                best[1],
                True,
                base_ms,
            )

    csv_file.close()
    print("TRACE_DONE", flush=True)


if __name__ == "__main__":
    main()
