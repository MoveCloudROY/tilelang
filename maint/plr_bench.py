"""Before/after benchmark for tl.PathLocalityReorder on real GPUs.

Generates uniform1d-style path kernels with compile-time constant path
tables (the V1 pattern contract), compiles each with the pass disabled and
enabled, validates numerics against a torch reference, and reports latency
plus ptxas register/spill counts.

Path table structure (FastEq-like): paths t = 0..P-1 grouped by `group`
consecutive paths sharing the same (w[i], x[j]) operand pair, with y[k]
cycling through a small pool and one distinct output element per path:

    out[bx, t, lane] += c_t * w[bx, t//group, lane] * x[bx, t//group, lane]
                            * y[bx, t % n_y]

Run inside the build container:
    python maint/plr_bench.py --blocks 65536 --iters 200
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

import torch

import tilelang

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grouped_paths(num_paths: int, group: int, n_y: int):
    paths = []
    for t in range(num_paths):
        g = t // group
        coeff = 0.5 + 0.03125 * (t % 7)
        paths.append((g, g, t % n_y, t, coeff))
    return paths


def build_prim_func(paths, blocks, lanes, form, tag):
    """Write a temp module with literal path statements and import it."""
    n_w = max(p[0] for p in paths) + 1
    n_x = max(p[1] for p in paths) + 1
    n_y = max(p[2] for p in paths) + 1
    n_v = max(p[3] for p in paths) + 1

    lines = [
        "import tilelang.language as T",
        "",
        "@T.prim_func",
        "def main(",
        f"    w: T.Tensor(({blocks}, {n_w}, {lanes}), 'float32'),",
        f"    x: T.Tensor(({blocks}, {n_x}, {lanes}), 'float32'),",
        f"    y: T.Tensor(({blocks}, {n_y}), 'float32'),",
        f"    out: T.Tensor(({blocks}, {n_v}, {lanes}), 'float32'),",
        "):",
        f"    with T.Kernel({blocks}, threads={lanes}) as bx:",
        "        lane = T.get_thread_binding(0)",
    ]
    for i, j, k, v, c in paths:
        product = f"T.float32({c!r}) * w[bx, {i}, lane] * x[bx, {j}, lane] * y[bx, {k}]"
        if form == "atomic":
            lines.append(f"        T.atomic_add(out[bx, {v}, lane], {product})")
        else:
            lines.append(f"        out[bx, {v}, lane] = out[bx, {v}, lane] + {product}")
    source = "\n".join(lines) + "\n"

    path = os.path.join(tempfile.gettempdir(), f"plr_kernel_{tag}.py")
    with open(path, "w") as fh:
        fh.write(source)
    spec = importlib.util.spec_from_file_location(f"plr_kernel_{tag}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main, (n_w, n_x, n_y, n_v)


def compile_kernel(func, enable, reg_budget=None, max_paths=None):
    pass_configs = {}
    if enable:
        pass_configs["tl.enable_path_locality_reorder"] = True
        sub = {}
        if reg_budget is not None:
            sub["reg_budget"] = reg_budget
        if max_paths is not None:
            sub["max_paths"] = max_paths
        if sub:
            pass_configs["tl.PathLocalityReorder"] = sub
    return tilelang.compile(func, pass_configs=pass_configs)


def ptxas_stats(kernel_source, arch):
    """Compile the generated CUDA standalone to read ptxas register usage."""
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
    return {
        "regs": int(regs.group(1)) if regs else -1,
        "spill_stores": int(spill_st.group(1)) if spill_st else 0,
        "spill_loads": int(spill_ld.group(1)) if spill_ld else 0,
    }


def reference(paths, w, x, y, n_v, lanes):
    out = torch.zeros(w.shape[0], n_v, lanes, dtype=torch.float32, device=w.device)
    for i, j, k, v, c in paths:
        out[:, v, :] += c * w[:, i, :] * x[:, j, :] * y[:, k, None]
    return out


def bench(run, iters, warmup):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_config(form, num_paths, group, n_y, args, variants):
    tag = f"{form}_p{num_paths}"
    paths = grouped_paths(num_paths, group, n_y)
    func, (n_w, n_x, ny, n_v) = build_prim_func(paths, args.blocks, args.lanes, form, tag)

    torch.manual_seed(0)
    dev = "cuda"
    w = torch.randn(args.blocks, n_w, args.lanes, dtype=torch.float32, device=dev)
    x = torch.randn(args.blocks, n_x, args.lanes, dtype=torch.float32, device=dev)
    y = torch.randn(args.blocks, ny, dtype=torch.float32, device=dev)
    ref = reference(paths, w, x, y, n_v, args.lanes)

    rows = []
    for label, enable, reg_budget in variants:
        kernel = compile_kernel(func, enable, reg_budget, args.max_paths)
        source = kernel.get_kernel_source()
        rewritten = "plr_" in source

        out = torch.zeros_like(ref)
        kernel(w, x, y, out)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

        out.zero_()
        ms = bench(functools.partial(kernel, w, x, y, out), args.iters, args.warmup)
        stats = ptxas_stats(source, args.arch)
        rows.append(
            {
                "form": form,
                "paths": num_paths,
                "variant": label,
                "rewritten": rewritten,
                "ms": ms,
                **stats,
            }
        )
        print(
            f"  {form:>6} P={num_paths:<3} {label:<12} rewritten={str(rewritten):<5} "
            f"lat={ms * 1e3:8.1f} us  regs={stats['regs']:<3} "
            f"spill={stats['spill_stores']}/{stats['spill_loads']}",
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=65536)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--n-y", type=int, default=8)
    parser.add_argument("--paths", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--forms", type=str, nargs="+", default=["atomic", "dense"])
    parser.add_argument("--max-paths", type=int, default=None, help="tl.PathLocalityReorder max_paths override for enabled variants")
    args = parser.parse_args()

    print(f"device={torch.cuda.get_device_name(0)} blocks={args.blocks} lanes={args.lanes} group={args.group} n_y={args.n_y}", flush=True)

    all_rows = []
    for form in args.forms:
        for num_paths in args.paths:
            variants = [("off", False, None), ("on", True, None)]
            if num_paths == max(args.paths):
                variants += [("on_rb8", True, 8), ("on_rb32", True, 32)]
            all_rows += run_config(form, num_paths, args.group, args.n_y, args, variants)

    print("\n=== summary (latency us, speedup vs off) ===", flush=True)
    by_key = {}
    for row in all_rows:
        by_key.setdefault((row["form"], row["paths"]), {})[row["variant"]] = row
    header = f"{'form':>6} {'P':>4} {'off_us':>9} {'on_us':>9} {'speedup':>8} {'regs off->on':>13} {'spills off->on':>15}"
    print(header, flush=True)
    for (form, num_paths), group_rows in sorted(by_key.items()):
        off, on = group_rows.get("off"), group_rows.get("on")
        if not off or not on:
            continue
        speedup = off["ms"] / on["ms"] if on["ms"] else float("nan")
        print(
            f"{form:>6} {num_paths:>4} {off['ms'] * 1e3:>9.1f} {on['ms'] * 1e3:>9.1f} "
            f"{speedup:>7.3f}x {off['regs']:>6}->{on['regs']:<5} "
            f"{off['spill_stores']:>7}->{on['spill_stores']:<6}",
            flush=True,
        )


if __name__ == "__main__":
    main()
