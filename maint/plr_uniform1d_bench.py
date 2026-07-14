"""Correctness/performance validation of tl.PathLocalityReorder on the
FastEq uniform1d forward operator.

Reproduces the FastEq edge-parallel kernel shape (one warp per edge with
src/dst indirection and atomic scatter accumulation):

    out[dst(b), v, lane] += c * w[b, i, lane] * x_all[src(b), j, lane] * y[b, k]

Path tables (i, j, k, v, c) are real SO(3) tensor-product tables built from
Clebsch-Gordan coefficients up to `lmax`: one k per (l1, l2, l3) trio and
one path per non-zero CG entry, so output elements v are shared by many
paths (the defining property of real uniform1d tables).

Two kernel forms are compared:
  - const:  the path table baked as literal statements (the V1 pattern
            contract, matching FastEq's generated CUDA);
  - serial: the original `for t in T.serial(P)` kernel with runtime
            descriptor tensors, which V1 intentionally leaves untouched.

Because real tables repeat v, the default pass config must skip the region
(float-addition association); `allow_atomic_reorder=True` opts into the
rewrite. Both are measured.

Run inside the build container:
    python maint/plr_uniform1d_bench.py --edges 65536 --nodes 4096
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


def cg_paths(lmax):
    """Real SO(3) uniform1d path table from Clebsch-Gordan coefficients."""
    from sympy import Rational
    from sympy.physics.quantum.cg import CG

    def slot(l, m):  # noqa: E741
        return l * l + (m + l)

    trios = [(l1, l2, l3) for l1 in range(lmax + 1) for l2 in range(lmax + 1) for l3 in range(abs(l1 - l2), min(lmax, l1 + l2) + 1)]
    paths = []
    for k, (l1, l2, l3) in enumerate(trios):
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m3 = m1 + m2
                if abs(m3) > l3:
                    continue
                coeff = float(CG(Rational(l1), Rational(m1), Rational(l2), Rational(m2), Rational(l3), Rational(m3)).doit().evalf())
                if coeff == 0.0:
                    continue
                paths.append((slot(l1, m1), slot(l2, m2), k, slot(l3, m3), coeff))
    dim = (lmax + 1) ** 2
    return paths, dim, len(trios)


def _import_kernel(source, tag):
    path = os.path.join(tempfile.gettempdir(), f"plr_u1d_{tag}.py")
    with open(path, "w") as fh:
        fh.write(source)
    spec = importlib.util.spec_from_file_location(f"plr_u1d_{tag}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def _kernel_header(edges, nodes, dim, n_trios, lanes, warps, serial_paths):
    blocks = (edges + warps - 1) // warps
    lines = [
        "import tilelang.language as T",
        "",
        "@T.prim_func",
        "def main(",
        f"    w: T.Tensor(({edges}, {dim}, {lanes}), 'float32'),",
        f"    x_all: T.Tensor(({nodes}, {dim}, {lanes}), 'float32'),",
        f"    y: T.Tensor(({edges}, {n_trios}), 'float32'),",
        f"    out: T.Tensor(({nodes}, {dim}, {lanes}), 'float32'),",
        f"    src_idx: T.Tensor(({edges},), 'int32'),",
        f"    dst_idx: T.Tensor(({edges},), 'int32'),",
        f"    b_list: T.Tensor(({edges},), 'int32'),",
    ]
    if serial_paths:
        num_paths = serial_paths
        lines += [
            f"    i_list: T.Tensor(({num_paths},), 'int32'),",
            f"    j_list: T.Tensor(({num_paths},), 'int32'),",
            f"    k_list: T.Tensor(({num_paths},), 'int32'),",
            f"    v_list: T.Tensor(({num_paths},), 'int32'),",
            f"    coeff_list: T.Tensor(({num_paths},), 'float32'),",
        ]
    lines += [
        "):",
        f"    with T.Kernel({blocks}, threads={warps * 32}) as bx:",
        "        tid = T.get_thread_binding(0)",
        "        lane = tid % 32",
        "        warp = tid // 32",
        f"        warp_global = bx * {warps} + warp",
        f"        if warp_global < {edges}:",
        "            b = b_list[warp_global]",
        "            src = src_idx[b]",
        "            dst = dst_idx[b]",
    ]
    return lines


def build_const_kernel(paths, edges, nodes, dim, n_trios, lanes, warps, tag):
    lines = _kernel_header(edges, nodes, dim, n_trios, lanes, warps, None)
    for i, j, k, v, c in paths:
        lines.append(
            f"            T.atomic_add(out[dst, {v}, lane], T.float32({c!r}) * w[b, {i}, lane] * x_all[src, {j}, lane] * y[b, {k}])"
        )
    return _import_kernel("\n".join(lines) + "\n", tag)


def build_serial_kernel(num_paths, edges, nodes, dim, n_trios, lanes, warps, tag):
    lines = _kernel_header(edges, nodes, dim, n_trios, lanes, warps, num_paths)
    lines += [
        f"            for t in T.serial({num_paths}):",
        "                i = i_list[t]",
        "                j = j_list[t]",
        "                k = k_list[t]",
        "                v = v_list[t]",
        "                c = coeff_list[t]",
        "                T.atomic_add(out[dst, v, lane], c * w[b, i, lane] * x_all[src, j, lane] * y[b, k])",
    ]
    return _import_kernel("\n".join(lines) + "\n", tag)


def compile_kernel(func, enable, relaxed=False, pair_cse=True, max_paths=1024):
    # Safe-memory legalization wraps every indirect access in bounds-check
    # predicates, which fence path regions. The operator contract guarantees
    # in-bounds indices (FastEq's reference CUDA performs no bounds checks
    # either), so it is disabled identically for baseline and pass runs.
    pass_configs = {"tl.disable_safe_memory_legalize": True}
    if enable:
        pass_configs["tl.enable_path_locality_reorder"] = True
        pass_configs["tl.PathLocalityReorder"] = {
            "max_paths": max_paths,
            "allow_atomic_reorder": relaxed,
            "enable_pair_cse": pair_cse,
        }
    return tilelang.compile(func, pass_configs=pass_configs)


def ptxas_stats(kernel_source, arch):
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
        "spills": (int(spill_st.group(1)) if spill_st else 0, int(spill_ld.group(1)) if spill_ld else 0),
    }


def reference(paths, w, x_all, y, src, dst, nodes, dim, lanes):
    out = torch.zeros(nodes, dim, lanes, dtype=torch.float64, device=w.device)
    for i, j, k, v, c in paths:
        contrib = (c * w[:, i, :] * x_all[src.long(), j, :] * y[:, k, None]).double()
        acc = torch.zeros(nodes, lanes, dtype=torch.float64, device=w.device)
        acc.index_add_(0, dst.long(), contrib)
        out[:, v, :] += acc
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, default=65536)
    parser.add_argument("--nodes", type=int, default=4096)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--arch", type=str, default="sm_90a")
    parser.add_argument("--lmax", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--skip-serial", action="store_true")
    args = parser.parse_args()

    dev = "cuda"
    print(f"device={torch.cuda.get_device_name(0)} edges={args.edges} nodes={args.nodes} lanes={args.lanes}", flush=True)

    for lmax in args.lmax:
        paths, dim, n_trios = cg_paths(lmax)
        num_paths = len(paths)
        shared_w = num_paths - len({(p[0],) for p in paths})
        print(f"\n--- lmax={lmax}: P={num_paths} paths, dim={dim}, trios={n_trios} ---", flush=True)

        torch.manual_seed(0)
        w = torch.randn(args.edges, dim, args.lanes, device=dev)
        x_all = torch.randn(args.nodes, dim, args.lanes, device=dev)
        y = torch.randn(args.edges, n_trios, device=dev)
        src = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
        dst = torch.randint(0, args.nodes, (args.edges,), dtype=torch.int32, device=dev)
        b_list = torch.arange(args.edges, dtype=torch.int32, device=dev)
        ref = reference(paths, w, x_all, y, src, dst, args.nodes, dim, args.lanes)

        func = build_const_kernel(paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"const_l{lmax}")
        variants = [
            ("off", dict(enable=False)),
            ("on_default", dict(enable=True)),
            ("on_relaxed", dict(enable=True, relaxed=True)),
            ("on_rlx_nopair", dict(enable=True, relaxed=True, pair_cse=False)),
        ]
        base_ms = None
        for label, kwargs in variants:
            kernel = compile_kernel(func, **kwargs)
            source = kernel.get_kernel_source()
            rewritten = "plr_" in source

            out = torch.zeros(args.nodes, dim, args.lanes, device=dev)
            kernel(w, x_all, y, out, src, dst, b_list)
            max_rel = ((out.double() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            torch.testing.assert_close(out.double(), ref, rtol=1e-3, atol=1e-3)

            out.zero_()
            ms = bench(functools.partial(kernel, w, x_all, y, out, src, dst, b_list), args.iters, args.warmup)
            stats = ptxas_stats(source, args.arch)
            if label == "off":
                base_ms = ms
            speedup = base_ms / ms if base_ms else float("nan")
            print(
                f"  const  {label:<14} rewritten={str(rewritten):<5} lat={ms * 1e3:9.1f} us "
                f"({speedup:5.3f}x) regs={stats['regs']:<3} spill={stats['spills'][0]}/{stats['spills'][1]} "
                f"max_rel_err={max_rel:.2e}",
                flush=True,
            )

        if args.skip_serial:
            continue
        sfunc = build_serial_kernel(num_paths, args.edges, args.nodes, dim, n_trios, args.lanes, args.warps, f"serial_l{lmax}")
        i_l = torch.tensor([p[0] for p in paths], dtype=torch.int32, device=dev)
        j_l = torch.tensor([p[1] for p in paths], dtype=torch.int32, device=dev)
        k_l = torch.tensor([p[2] for p in paths], dtype=torch.int32, device=dev)
        v_l = torch.tensor([p[3] for p in paths], dtype=torch.int32, device=dev)
        c_l = torch.tensor([p[4] for p in paths], dtype=torch.float32, device=dev)

        src_off = compile_kernel(sfunc, enable=False)
        src_on = compile_kernel(sfunc, enable=True, relaxed=True)
        identical = src_off.get_kernel_source() == src_on.get_kernel_source()
        out = torch.zeros(args.nodes, dim, args.lanes, device=dev)
        src_on(w, x_all, y, out, src, dst, b_list, i_l, j_l, k_l, v_l, c_l)
        torch.testing.assert_close(out.double(), ref, rtol=1e-3, atol=1e-3)
        ms = bench(
            functools.partial(src_on, w, x_all, y, out, src, dst, b_list, i_l, j_l, k_l, v_l, c_l),
            args.iters,
            args.warmup,
        )
        print(
            f"  serial runtime-desc   pass_noop={identical}   lat={ms * 1e3:9.1f} us (vs const off {base_ms * 1e3:9.1f} us)",
            flush=True,
        )

        # The same serial kernel with descriptor tables annotated: the pass
        # folds the descriptor loads and schedules the specialized paths, so
        # the user writes only the loop form.
        annotated = tilelang.transform.annotate_path_descriptors(
            sfunc,
            {
                "i_list": [p[0] for p in paths],
                "j_list": [p[1] for p in paths],
                "k_list": [p[2] for p in paths],
                "v_list": [p[3] for p in paths],
                "coeff_list": [p[4] for p in paths],
            },
        )
        kernel = compile_kernel(annotated, enable=True, relaxed=True)
        source = kernel.get_kernel_source()
        out = torch.zeros(args.nodes, dim, args.lanes, device=dev)
        kernel(w, x_all, y, out, src, dst, b_list, i_l, j_l, k_l, v_l, c_l)
        max_rel = ((out.double() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
        torch.testing.assert_close(out.double(), ref, rtol=1e-3, atol=1e-3)
        out.zero_()
        ms = bench(
            functools.partial(kernel, w, x_all, y, out, src, dst, b_list, i_l, j_l, k_l, v_l, c_l),
            args.iters,
            args.warmup,
        )
        stats = ptxas_stats(source, args.arch)
        print(
            f"  serial +descriptors   rewritten={str('plr_' in source):<5} lat={ms * 1e3:9.1f} us "
            f"({base_ms / ms:5.3f}x vs const off) regs={stats['regs']:<3} "
            f"spill={stats['spills'][0]}/{stats['spills'][1]} max_rel_err={max_rel:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
