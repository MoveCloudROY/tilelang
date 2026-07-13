"""Spill comparison under forced register caps for tl.PathLocalityReorder.

Compiles the generated CUDA of each benchmark kernel with
`-maxrregcount=<cap>` (the occupancy-tuning regime the pass targets) and
reports ptxas spill bytes with the pass off vs on.

    python maint/plr_regcap.py --paths 64 128 --caps 32 40 64
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plr_bench import build_prim_func, compile_kernel, grouped_paths  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ptxas_stats_capped(kernel_source, arch, cap):
    # __launch_bounds__ overrides -maxrregcount, so strip it for this
    # offline spill experiment.
    kernel_source = re.sub(r"__launch_bounds__\([^)]*\)", "", kernel_source)
    with tempfile.TemporaryDirectory() as tmp:
        cu = os.path.join(tmp, "k.cu")
        with open(cu, "w") as fh:
            fh.write(kernel_source)
        cmd = [
            "nvcc",
            "--cubin",
            "-O3",
            f"-arch={arch}",
            "-std=c++20",
            f"-maxrregcount={cap}",
            "-Xptxas",
            "-v",
            f"-I{_REPO}/src",
            f"-I{_REPO}/3rdparty/cutlass/include",
            "-o",
            os.path.join(tmp, "k.cubin"),
            cu,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        text = proc.stderr
    regs = re.search(r"Used (\d+) registers", text)
    spill_st = re.search(r"(\d+) bytes spill stores", text)
    spill_ld = re.search(r"(\d+) bytes spill loads", text)
    return (
        int(regs.group(1)) if regs else -1,
        int(spill_st.group(1)) if spill_st else 0,
        int(spill_ld.group(1)) if spill_ld else 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=65536)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--n-y", type=int, default=16)
    parser.add_argument("--paths", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--forms", type=str, nargs="+", default=["atomic", "dense"])
    parser.add_argument("--caps", type=int, nargs="+", default=[32, 40, 64])
    parser.add_argument("--reg-budget", type=int, default=None)
    args = parser.parse_args()

    print(f"{'form':>6} {'P':>4} {'cap':>4} | {'off regs/spill_st/spill_ld':>26} | {'on regs/spill_st/spill_ld':>26}")
    for form in args.forms:
        for num_paths in args.paths:
            paths = grouped_paths(num_paths, args.group, args.n_y)
            tag = f"cap_{form}_p{num_paths}"
            func, _ = build_prim_func(paths, args.blocks, args.lanes, form, tag)
            src_off = compile_kernel(func, False).get_kernel_source()
            src_on = compile_kernel(func, True, args.reg_budget, 256).get_kernel_source()
            assert "plr_" in src_on
            for cap in args.caps:
                off = ptxas_stats_capped(src_off, args.arch, cap)
                on = ptxas_stats_capped(src_on, args.arch, cap)
                print(
                    f"{form:>6} {num_paths:>4} {cap:>4} | {off[0]:>8} /{off[1]:>6} /{off[2]:>6} | {on[0]:>8} /{on[1]:>6} /{on[2]:>6}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
