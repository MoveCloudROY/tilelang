"""Complete uniform1d performance matrix for tl.PathLocalityReorder.

Runs every kernel variant across path-table sizes and register budgets,
verifies each against a float64 reference, records latency / ptxas
registers / spills / SASS size / atomic counts into a CSV (flushed row by
row), and archives the generated CUDA source of every variant.

Variants:
  const_off        fully unrolled baseline (pass disabled)
  const_default    pass on, association-preserving policy
  const_relaxed    pass on, allow_atomic_reorder + accumulator fusion
  const_nopair     const_relaxed with pair CSE disabled
  serial_off       runtime-descriptor loop (pass is a no-op by design)
  serial_desc      serial kernel + annotate_path_descriptors, relaxed

Run inside the build container:
    TILELANG_DISABLE_CACHE=1 python maint/plr_uniform1d_full.py \
        --csv /ws/plr_u1d_full.csv --dump-dir /ws/u1d_cuda_full
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import re
import subprocess
import sys
import tempfile

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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compile_variant(func, enable, reg_budget, relaxed=False, pair_cse=True):
    pass_configs = {"tl.disable_safe_memory_legalize": True}
    if enable:
        pass_configs["tl.enable_path_locality_reorder"] = True
        pass_configs["tl.PathLocalityReorder"] = {
            "max_paths": 1024,
            "reg_budget": int(reg_budget),
            "allow_atomic_reorder": relaxed,
            "enable_pair_cse": pair_cse,
        }
    return tilelang.compile(func, pass_configs=pass_configs)


def sass_stats(kernel_source, arch):
    """ptxas register/spill counts plus SASS instruction count."""
    with tempfile.TemporaryDirectory() as tmp:
        cu = os.path.join(tmp, "k.cu")
        cubin = os.path.join(tmp, "k.cubin")
        with open(cu, "w") as fh:
            fh.write(kernel_source)
        cmd = [
            "nvcc",
            "--cubin",
            "-O3",
            f"-arch={arch}",
            "-std=c++20",
            "-Xptxas",
            "-v",
            f"-I{_REPO}/src",
            f"-I{_REPO}/3rdparty/cutlass/include",
            "-o",
            cubin,
            cu,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        regs = re.search(r"Used (\d+) registers", proc.stderr)
        spill_st = re.search(r"(\d+) bytes spill stores", proc.stderr)
        spill_ld = re.search(r"(\d+) bytes spill loads", proc.stderr)
        instrs = -1
        if os.path.exists(cubin):
            dump = subprocess.run(["cuobjdump", "--dump-sass", cubin], capture_output=True, text=True)
            instrs = len(re.findall(r"/\*[0-9a-f]{4,}\*/", dump.stdout))
    return {
        "regs": int(regs.group(1)) if regs else -1,
        "spill_st": int(spill_st.group(1)) if spill_st else 0,
        "spill_ld": int(spill_ld.group(1)) if spill_ld else 0,
        "sass_instrs": instrs,
    }


def kernel_body(kernel):
    source = kernel.get_kernel_source()
    return source[source.index("__launch_bounds__") :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--lmax", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--dump-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    dev = "cuda"
    print(f"device={torch.cuda.get_device_name(0)} budgets={args.budgets}", flush=True)

    fields = [
        "lmax",
        "num_paths",
        "dim",
        "variant",
        "reg_budget",
        "lat_us",
        "speedup_vs_off",
        "regs",
        "spill_st",
        "spill_ld",
        "sass_instrs",
        "atomics",
        "max_rel_err",
        "dump_file",
    ]
    csv_file = open(args.csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    csv_file.flush()

    for lmax in args.lmax:
        paths, dim, n_trios = cg_paths(lmax)
        num_paths = len(paths)
        cfunc = build_const_kernel(paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"full_c{lmax}")
        sfunc = build_serial_kernel(num_paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"full_s{lmax}")
        tables = {name: [p[i] for p in paths] for i, name in enumerate(["i_list", "j_list", "k_list", "v_list", "coeff_list"])}

        torch.manual_seed(0)
        w = torch.randn(args.edges, dim, args.lanes, device=dev)
        x_all = torch.randn(args.nodes, dim, args.lanes, device=dev)
        y = torch.randn(args.edges, n_trios, device=dev)
        src = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
        dst = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
        b_list = torch.arange(args.edges, dtype=torch.int32, device=dev)
        desc = [
            torch.tensor(tables["i_list"], dtype=torch.int32, device=dev),
            torch.tensor(tables["j_list"], dtype=torch.int32, device=dev),
            torch.tensor(tables["k_list"], dtype=torch.int32, device=dev),
            torch.tensor(tables["v_list"], dtype=torch.int32, device=dev),
            torch.tensor(tables["coeff_list"], dtype=torch.float32, device=dev),
        ]
        ref = reference(paths, w, x_all, y, src, dst, args.nodes, dim, args.lanes)
        out = torch.zeros(args.nodes, dim, args.lanes, device=dev)

        def measure(kernel, serial, w=w, x_all=x_all, y=y, out=out, src=src, dst=dst, b_list=b_list, desc=tuple(desc), ref=ref):
            call_args = (w, x_all, y, out, src, dst, b_list, *desc) if serial else (w, x_all, y, out, src, dst, b_list)
            out.zero_()
            kernel(*call_args)
            max_rel = ((out.double() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            torch.testing.assert_close(out.double(), ref, rtol=1e-3, atol=1e-3)
            out.zero_()
            return bench(functools.partial(kernel, *call_args), args.iters, args.warmup), max_rel

        def record(variant, kernel, budget, serial, off_ms=None, lmax=lmax, num_paths=num_paths, dim=dim, measure=measure):
            ms, max_rel = measure(kernel, serial)
            body = kernel_body(kernel)
            stats = sass_stats(kernel.get_kernel_source(), args.arch)
            suffix = "" if budget is None else f"_b{budget}"
            dump_file = f"u1d_l{lmax}_{variant}{suffix}.cu"
            with open(os.path.join(args.dump_dir, dump_file), "w") as fh:
                fh.write(kernel.get_kernel_source())
            row = {
                "lmax": lmax,
                "num_paths": num_paths,
                "dim": dim,
                "variant": variant,
                "reg_budget": budget if budget is not None else "",
                "lat_us": round(ms * 1e3, 2),
                "speedup_vs_off": round(off_ms / ms, 4) if off_ms else 1.0,
                "regs": stats["regs"],
                "spill_st": stats["spill_st"],
                "spill_ld": stats["spill_ld"],
                "sass_instrs": stats["sass_instrs"],
                "atomics": body.count("AtomicAdd"),
                "max_rel_err": f"{max_rel:.2e}",
                "dump_file": dump_file,
            }
            writer.writerow(row)
            csv_file.flush()
            print(
                f"lmax={lmax} {variant:<14} budget={str(budget) if budget is not None else '-':>4} "
                f"lat={ms * 1e3:8.1f}us ({row['speedup_vs_off']:6.3f}x) regs={stats['regs']:<3} "
                f"atomics={row['atomics']:<4} sass={stats['sass_instrs']}",
                flush=True,
            )
            return ms

        off_ms = record("const_off", compile_variant(cfunc, enable=False, reg_budget=0), None, serial=False)
        record("serial_off", compile_variant(sfunc, enable=False, reg_budget=0), None, serial=True, off_ms=off_ms)
        for budget in args.budgets:
            record("const_default", compile_variant(cfunc, True, budget), budget, False, off_ms)
            record("const_relaxed", compile_variant(cfunc, True, budget, relaxed=True), budget, False, off_ms)
            record("const_nopair", compile_variant(cfunc, True, budget, relaxed=True, pair_cse=False), budget, False, off_ms)
            record(
                "serial_desc",
                compile_variant(annotate_path_descriptors(sfunc, tables), True, budget, relaxed=True),
                budget,
                True,
                off_ms,
            )

    csv_file.close()
    print("FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
