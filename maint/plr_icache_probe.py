"""Launch one uniform1d kernel variant for Nsight Compute profiling.

Compiles the requested variant (pass off / relaxed+fusion / serial loop) and
launches it repeatedly so `ncu --launch-skip/--launch-count` can sample
steady-state executions.

    ncu --metrics ... python maint/plr_icache_probe.py --lmax 3 --variant const_off
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plr_uniform1d_bench import build_const_kernel, build_serial_kernel, cg_paths  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmax", type=int, required=True)
    parser.add_argument("--variant", choices=["const_off", "const_relaxed", "serial_off"], required=True)
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--reg-budget", type=int, default=32)
    parser.add_argument("--launches", type=int, default=8)
    args = parser.parse_args()

    paths, dim, n_trios = cg_paths(args.lmax)
    pass_configs = {"tl.disable_safe_memory_legalize": True}
    if args.variant == "const_relaxed":
        pass_configs["tl.enable_path_locality_reorder"] = True
        pass_configs["tl.PathLocalityReorder"] = {
            "max_paths": 4096,
            "reg_budget": args.reg_budget,
            "allow_atomic_reorder": True,
        }

    tag = f"icache_{args.variant}_l{args.lmax}"
    if args.variant == "serial_off":
        func = build_serial_kernel(len(paths), args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, tag)
    else:
        func = build_const_kernel(paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, tag)
    kernel = tilelang.compile(func, pass_configs=pass_configs)

    dev = "cuda"
    torch.manual_seed(0)
    w = torch.randn(args.edges, dim, args.lanes, device=dev)
    x_all = torch.randn(args.nodes, dim, args.lanes, device=dev)
    y = torch.randn(args.edges, n_trios, device=dev)
    src = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
    dst = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
    b_list = torch.arange(args.edges, dtype=torch.int32, device=dev)
    out = torch.zeros(args.nodes, dim, args.lanes, device=dev)

    if args.variant == "serial_off":
        desc = [torch.tensor([p[i] for p in paths], dtype=torch.int32, device=dev) for i in range(4)] + [
            torch.tensor([p[4] for p in paths], dtype=torch.float32, device=dev)
        ]
        run = lambda: kernel(w, x_all, y, out, src, dst, b_list, *desc)  # noqa: E731
    else:
        run = lambda: kernel(w, x_all, y, out, src, dst, b_list)  # noqa: E731

    for _ in range(args.launches):
        run()
    torch.cuda.synchronize()
    print(f"PROBE_DONE variant={args.variant} lmax={args.lmax} P={len(paths)}", flush=True)


if __name__ == "__main__":
    main()
