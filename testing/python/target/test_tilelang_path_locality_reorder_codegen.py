"""CUDA codegen and correctness tests for tl.PathLocalityReorder.

The kernels bake a compile-time path table (the V1 pattern contract) as
literal statements: each path contributes

    out[v, lane] += c * w[i, lane] * x[j, lane] * y[k]

Paths sharing the (w, x) operand pair should be scheduled so the shared
loads become scalar temporaries in the generated CUDA.
"""

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

# Path table from the V1 design doc: (i, j, k, v, c) with shared (w, x)
# labels between adjacent paths and pairwise-distinct outputs.
_PATHS = [(0, 0, 0, 0, 2.0), (0, 0, 1, 1, 3.0), (1, 1, 0, 2, 4.0), (1, 1, 1, 3, 5.0)]
_U = 32


def _atomic_kernel():
    @T.prim_func
    def main(
        w: T.Tensor((2, _U), "float32"),
        x: T.Tensor((2, _U), "float32"),
        y: T.Tensor((2,), "float32"),
        out: T.Tensor((4, _U), "float32"),
    ):
        with T.Kernel(1, threads=_U):
            lane = T.get_thread_binding(0)
            T.atomic_add(out[0, lane], T.float32(2.0) * w[0, lane] * x[0, lane] * y[0])
            T.atomic_add(out[1, lane], T.float32(3.0) * w[0, lane] * x[0, lane] * y[1])
            T.atomic_add(out[2, lane], T.float32(4.0) * w[1, lane] * x[1, lane] * y[0])
            T.atomic_add(out[3, lane], T.float32(5.0) * w[1, lane] * x[1, lane] * y[1])

    return main


def _dense_kernel():
    @T.prim_func
    def main(
        w: T.Tensor((2, _U), "float32"),
        x: T.Tensor((2, _U), "float32"),
        y: T.Tensor((2,), "float32"),
        out: T.Tensor((4, _U), "float32"),
    ):
        with T.Kernel(1, threads=_U):
            lane = T.get_thread_binding(0)
            out[0, lane] = out[0, lane] + T.float32(2.0) * w[0, lane] * x[0, lane] * y[0]
            out[1, lane] = out[1, lane] + T.float32(3.0) * w[0, lane] * x[0, lane] * y[1]
            out[2, lane] = out[2, lane] + T.float32(4.0) * w[1, lane] * x[1, lane] * y[0]
            out[3, lane] = out[3, lane] + T.float32(5.0) * w[1, lane] * x[1, lane] * y[1]

    return main


def _compile(func, enable, extra_config=None):
    pass_configs = {}
    if enable:
        pass_configs["tl.enable_path_locality_reorder"] = True
    if extra_config:
        pass_configs["tl.PathLocalityReorder"] = extra_config
    # The kernels accumulate into `out`, so the caller passes a zeroed output
    # tensor explicitly instead of relying on out_idx allocation.
    return tilelang.compile(func, pass_configs=pass_configs)


def _reference(w, x, y):
    import torch

    ref = torch.zeros(4, _U, dtype=torch.float32, device=w.device)
    for i, j, k, v, c in _PATHS:
        ref[v] += c * w[i] * x[j] * y[k]
    return ref


def _random_inputs():
    import torch

    torch.manual_seed(0)
    w = torch.randn(2, _U, dtype=torch.float32, device="cuda")
    x = torch.randn(2, _U, dtype=torch.float32, device="cuda")
    y = torch.randn(2, dtype=torch.float32, device="cuda")
    return w, x, y


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("build_kernel", [_atomic_kernel, _dense_kernel])
def test_path_locality_reorder_cuda_matches_reference(build_kernel):
    import torch

    kernel_off = _compile(build_kernel(), enable=False)
    kernel_on = _compile(build_kernel(), enable=True)

    assert "plr_" not in kernel_off.get_kernel_source()
    assert "plr_" in kernel_on.get_kernel_source(), "shared operand labels should be materialized as scalar temporaries"

    w, x, y = _random_inputs()
    ref = _reference(w, x, y)
    out_off = torch.zeros(4, _U, dtype=torch.float32, device="cuda")
    out_on = torch.zeros(4, _U, dtype=torch.float32, device="cuda")
    kernel_off(w, x, y, out_off)
    kernel_on(w, x, y, out_on)

    torch.testing.assert_close(out_off, ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_on, ref, rtol=1e-4, atol=1e-5)


@tilelang.testing.requires_cuda
def test_path_locality_reorder_cuda_shared_loads_emitted_once():
    source = _compile(_atomic_kernel(), enable=True).get_kernel_source()
    kernel_body = source[source.index("main_kernel") :]
    # w[0, lane] / x[0, lane] / w[1, lane] / x[1, lane] are each used by two
    # paths but must be loaded from memory only once.
    for expr in ["w[((int)threadIdx.x)]", "x[((int)threadIdx.x)]"]:
        assert kernel_body.count(expr) == 1, kernel_body


@tilelang.testing.requires_cuda
def test_path_locality_reorder_cuda_repeated_output_gated():
    import torch

    @T.prim_func
    def main(
        w: T.Tensor((2, _U), "float32"),
        x: T.Tensor((2, _U), "float32"),
        y: T.Tensor((2,), "float32"),
        out: T.Tensor((4, _U), "float32"),
    ):
        with T.Kernel(1, threads=_U):
            lane = T.get_thread_binding(0)
            T.atomic_add(out[0, lane], T.float32(2.0) * w[0, lane] * x[0, lane] * y[0])
            T.atomic_add(out[0, lane], T.float32(3.0) * w[0, lane] * x[0, lane] * y[1])

    # Repeated output element: without explicit opt-in the region must stay
    # in original order (no scalar label temporaries are introduced).
    kernel_default = _compile(main, enable=True)
    assert "plr_" not in kernel_default.get_kernel_source()

    # Opting into relaxed addition association enables the rewrite; the
    # result is correct up to floating-point association.
    kernel_relaxed = _compile(main, enable=True, extra_config={"allow_atomic_reorder": True})
    assert "plr_" in kernel_relaxed.get_kernel_source()

    w, x, y = _random_inputs()
    ref = 2.0 * w[0] * x[0] * y[0] + 3.0 * w[0] * x[0] * y[1]
    out = torch.zeros(4, _U, dtype=torch.float32, device="cuda")
    kernel_relaxed(w, x, y, out)
    torch.testing.assert_close(out[0], ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
