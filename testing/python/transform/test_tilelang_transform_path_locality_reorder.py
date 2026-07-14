"""IR-shape tests for tl.PathLocalityReorder.

The pass groups unrolled path statements such as

    out[v] = out[v] + coeff * w[i] * x[j] * y[k]

into regions, loads shared operand labels once into scalar Binds, and
list-schedules the path computations under a virtual register budget.
"""

from collections import Counter

import tilelang
import tilelang.testing
from tilelang import tvm
from tvm.script import tirx as T
from tvm.tirx.stmt_functor import post_order_visit


def _apply(func, config=None):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    pass_config = {}
    if config is not None:
        pass_config["tl.PathLocalityReorder"] = config
    with tvm.transform.PassContext(config=pass_config):
        return tilelang.transform.PathLocalityReorder()(mod)["main"]


def _assert_unchanged(func, config=None):
    after = _apply(func, config)
    tvm.ir.assert_structural_equal(after, func.with_attr("global_symbol", "main"), map_free_vars=True)


def _collect(stmt, klass):
    nodes = []
    post_order_visit(stmt, lambda n: nodes.append(n) if isinstance(n, klass) else None)
    return nodes


def _label_bind_counts(func):
    """Count Bind statements whose value is a constant-indexed BufferLoad."""
    counts = Counter()
    for bind in _collect(func.body, tvm.tirx.Bind):
        if isinstance(bind.value, tvm.tirx.BufferLoad):
            load = bind.value
            key = (load.buffer.name, tuple(int(idx) for idx in load.indices))
            counts[key] += 1
    return counts


def _pair_binds(func):
    """Bind statements holding a product of two scalar label vars."""
    pairs = []
    for bind in _collect(func.body, tvm.tirx.Bind):
        value = bind.value
        if isinstance(value, tvm.tirx.Mul) and isinstance(value.a, tvm.tirx.Var) and isinstance(value.b, tvm.tirx.Var):
            pairs.append(bind)
    return pairs


def _four_path_func():
    # The V1 design-doc example: paths (w0,x0,y0,out0), (w0,x0,y1,out1),
    # (w1,x1,y0,out2), (w1,x1,y1,out3).
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        y: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        out[0] = out[0] + T.float32(2) * w[0] * x[0] * y[0]
        out[1] = out[1] + T.float32(3) * w[0] * x[0] * y[1]
        out[2] = out[2] + T.float32(4) * w[1] * x[1] * y[0]
        out[3] = out[3] + T.float32(5) * w[1] * x[1] * y[1]

    return before


def test_shared_labels_loaded_once():
    after = _apply(_four_path_func(), {"reg_budget": 4, "enable_pair_cse": False})

    counts = _label_bind_counts(after)
    for key in [("w", (0,)), ("x", (0,)), ("w", (1,)), ("x", (1,))]:
        assert counts[key] == 1, f"label {key} expected one load, got {counts}"
    for key in [("y", (0,)), ("y", (1,))]:
        assert counts[key] >= 1, f"label {key} missing: {counts}"

    stores = _collect(after.body, tvm.tirx.BufferStore)
    assert len(stores) == 4
    for store in stores:
        input_loads = [load for load in _collect(store.value, tvm.tirx.BufferLoad) if load.buffer.name != "out"]
        assert not input_loads, "path compute must consume scalar label vars"


def test_pair_cse_materialized_when_enabled():
    after = _apply(_four_path_func(), {"reg_budget": 16, "enable_pair_cse": True})
    # (w0, x0) and (w1, x1) each have two uses -> exactly two pair temporaries.
    assert len(_pair_binds(after)) == 2


def test_pair_cse_absent_when_disabled():
    after = _apply(_four_path_func(), {"reg_budget": 16, "enable_pair_cse": False})
    assert len(_pair_binds(after)) == 0


def test_enable_false_is_noop():
    _assert_unchanged(_four_path_func(), {"enable": False})


def test_same_output_updates_not_reordered():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        y: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        out[0] = out[0] + T.float32(2) * w[0] * x[0] * y[0]
        out[0] = out[0] + T.float32(3) * w[0] * x[0] * y[1]

    # Two updates to out[0]: reordering would change float association.
    _assert_unchanged(before)

    # Explicitly relaxing association allows the rewrite.
    after = _apply(before, {"allow_atomic_reorder": True})
    assert len(_label_bind_counts(after)) > 0


def test_unknown_output_aliasing_keeps_original_order():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        idx: T.Buffer((2,), "int32"),
        out: T.Buffer((4,), "float32"),
    ):
        out[idx[0]] = out[idx[0]] + T.float32(2) * w[0] * x[0]
        out[idx[1]] = out[idx[1]] + T.float32(3) * w[0] * x[1]

    # idx[0] and idx[1] cannot be proven disjoint at compile time.
    _assert_unchanged(before)


def test_no_shared_reads_skips_region():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        y: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        out[0] = out[0] + T.float32(2) * w[0] * x[0] * y[0]
        out[1] = out[1] + T.float32(3) * w[1] * x[1] * y[1]

    _assert_unchanged(before)


def test_opaque_call_fences_region():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        out[0] = out[0] + T.float32(2) * w[0] * x[0]
        T.call_extern("int32", "opaque_barrier")
        out[1] = out[1] + T.float32(3) * w[0] * x[0]

    # The opaque call splits the two paths into single-path regions, which
    # are never rewritten; original relative order is preserved.
    _assert_unchanged(before)


def test_per_statement_predicates_fence_region():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        cond: T.Buffer((1,), "int32"),
        out: T.Buffer((4,), "float32"),
    ):
        if cond[0] > 0:
            out[0] = out[0] + T.float32(2) * w[0] * x[0]
        if cond[0] > 1:
            out[1] = out[1] + T.float32(3) * w[0] * x[0]

    _assert_unchanged(before)


def test_predicated_region_schedules_inside_guard():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        cond: T.Buffer((1,), "int32"),
        out: T.Buffer((4,), "float32"),
    ):
        if cond[0] > 0:
            out[0] = out[0] + T.float32(2) * w[0] * x[0]
            out[1] = out[1] + T.float32(3) * w[0] * x[0]

    after = _apply(before)
    ifs = _collect(after.body, tvm.tirx.IfThenElse)
    assert len(ifs) == 1, "the guarding predicate must stay attached"
    counts = _label_bind_counts(after)
    assert counts[("w", (0,))] == 1
    assert counts[("x", (0,))] == 1


def test_serial_path_loop_is_unrolled_and_scheduled():
    @T.prim_func
    def before(
        x: T.Buffer((5,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(4):
            out[t] = out[t] + T.float32(0.5) * x[t] * x[t + 1]

    after = _apply(before)
    assert not _collect(after.body, tvm.tirx.For), "path loop must be unrolled"
    counts = _label_bind_counts(after)
    # x[1..3] are shared by adjacent paths; every element is loaded once.
    for i in range(5):
        assert counts[("x", (i,))] == 1, counts
    assert len(_collect(after.body, tvm.tirx.BufferStore)) == 4


def test_serial_loop_without_shared_reads_kept():
    @T.prim_func
    def before(
        x: T.Buffer((4,), "float32"),
        y: T.Buffer((4,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(4):
            out[t] = out[t] + T.float32(0.5) * x[t] * y[t]

    # Elementwise chain: no label is shared between paths, so unrolling has
    # no register-reuse benefit and the loop is preserved.
    _assert_unchanged(before)


def test_recurrence_dependency_rejected():
    @T.prim_func
    def before(
        x: T.Buffer((5,), "float32"),
        y: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(4):
            x[t + 1] = x[t + 1] + T.float32(0.5) * x[t] * y[t]

    # x is both an input label and the output buffer: a prefix/scan-like
    # recurrence whose iterations must not be reordered.
    _assert_unchanged(before)


def test_loop_annotation_fences_pass():
    @T.prim_func
    def before(
        x: T.Buffer((5,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(4, annotations={"tl.path_locality_reorder": 0}):
            out[t] = out[t] + T.float32(0.5) * x[t] * x[t + 1]

    _assert_unchanged(before)


def test_max_paths_rejects_large_regions():
    @T.prim_func
    def before(
        x: T.Buffer((5,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(4):
            out[t] = out[t] + T.float32(0.5) * x[t] * x[t + 1]

    _assert_unchanged(before, {"max_paths": 3})


def test_scalar_binds_are_inlined_and_consumed():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        y: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        wval = w[0]
        xval = x[0]
        out[0] = out[0] + T.float32(2) * wval * xval * y[0]
        out[1] = out[1] + T.float32(3) * wval * xval * y[1]

    after = _apply(before, {"enable_pair_cse": False})
    counts = _label_bind_counts(after)
    assert counts[("w", (0,))] == 1
    assert counts[("x", (0,))] == 1
    stores = _collect(after.body, tvm.tirx.BufferStore)
    assert len(stores) == 2


def test_peeled_bind_used_after_region_is_kept():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
        other: T.Buffer((1,), "float32"),
    ):
        wval = w[0]
        out[0] = out[0] + T.float32(2) * wval * x[0]
        out[1] = out[1] + T.float32(3) * wval * x[0]
        other[0] = wval

    after = _apply(before, {"enable_pair_cse": False})
    other_stores = [s for s in _collect(after.body, tvm.tirx.BufferStore) if s.buffer.name == "other"]
    assert len(other_stores) == 1
    kept_var = other_stores[0].value
    assert isinstance(kept_var, tvm.tirx.Var)
    bind_vars = [b.var for b in _collect(after.body, tvm.tirx.Bind)]
    assert any(bv.same_as(kept_var) for bv in bind_vars), "the peeled bind consumed after the region must be re-emitted"


_ATOMIC_ADD_ELEM = tvm.ir.Op.get("tl.atomic_add_elem_op")


def _atomic_two_path_func(second_offset):
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        y: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        T.evaluate(
            T.call_intrin(
                "float32",
                _ATOMIC_ADD_ELEM,
                T.tvm_access_ptr(T.type_annotation("float32"), out.data, 0, 1, 3),
                T.float32(2) * w[0] * x[0] * y[0],
                0,
            )
        )
        T.evaluate(
            T.call_intrin(
                "float32",
                _ATOMIC_ADD_ELEM,
                T.tvm_access_ptr(T.type_annotation("float32"), out.data, second_offset, 1, 3),
                T.float32(3) * w[0] * x[0] * y[1],
                0,
            )
        )

    return before


def test_atomic_add_paths_scheduled():
    after = _apply(_atomic_two_path_func(second_offset=1), {"enable_pair_cse": False})
    counts = _label_bind_counts(after)
    assert counts[("w", (0,))] == 1
    assert counts[("x", (0,))] == 1


def test_atomic_same_output_not_reordered():
    before = _atomic_two_path_func(second_offset=0)
    _assert_unchanged(before)
    after = _apply(before, {"allow_atomic_reorder": True})
    assert len(_label_bind_counts(after)) > 0


def test_non_relaxed_atomic_fences_region():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        T.evaluate(
            T.call_intrin(
                "float32",
                _ATOMIC_ADD_ELEM,
                T.tvm_access_ptr(T.type_annotation("float32"), out.data, 0, 1, 3),
                T.float32(2) * w[0] * x[0],
                2,  # acquire
            )
        )
        T.evaluate(
            T.call_intrin(
                "float32",
                _ATOMIC_ADD_ELEM,
                T.tvm_access_ptr(T.type_annotation("float32"), out.data, 1, 1, 3),
                T.float32(3) * w[0] * x[1],
                2,  # acquire
            )
        )

    # Acquire/release/seq_cst atomics carry synchronization semantics: the
    # region must not be reordered even though the outputs are disjoint.
    _assert_unchanged(before)


def _count_atomic_calls(func):
    calls = []
    post_order_visit(
        func.body,
        lambda n: calls.append(n) if isinstance(n, tvm.tirx.Call) and n.op.name == "tl.atomic_add_elem_op" else None,
    )
    return len(calls)


def test_output_accumulation_fuses_atomics():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        y: T.Buffer((4,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        T.evaluate(
            T.call_intrin(
                "float32",
                _ATOMIC_ADD_ELEM,
                T.tvm_access_ptr(T.type_annotation("float32"), out.data, 0, 1, 3),
                T.float32(2) * w[0] * x[0] * y[0],
                0,
            )
        )
        T.evaluate(
            T.call_intrin(
                "float32",
                _ATOMIC_ADD_ELEM,
                T.tvm_access_ptr(T.type_annotation("float32"), out.data, 0, 1, 3),
                T.float32(3) * w[0] * x[0] * y[1],
                0,
            )
        )
        T.evaluate(
            T.call_intrin(
                "float32",
                _ATOMIC_ADD_ELEM,
                T.tvm_access_ptr(T.type_annotation("float32"), out.data, 0, 1, 3),
                T.float32(4) * w[1] * x[1] * y[2],
                0,
            )
        )

    # Three contributions to out[0] fuse into one accumulator chain and a
    # single final atomic.
    after = _apply(before, {"allow_atomic_reorder": True})
    assert _count_atomic_calls(after) == 1
    acc_binds = [b for b in _collect(after.body, tvm.tirx.Bind) if isinstance(b.value, tvm.tirx.Add)]
    assert len(acc_binds) == 2, "chain must add the 2nd and 3rd contributions"

    # The escape hatch keeps one atomic per path.
    after = _apply(before, {"allow_atomic_reorder": True, "enable_output_accumulation": False})
    assert _count_atomic_calls(after) == 3

    # Default policy still refuses to touch same-output updates entirely.
    _assert_unchanged(before)


def test_descriptor_tables_specialize_serial_loop():
    @T.prim_func
    def before(
        x: T.Buffer((8,), "float32"),
        idx: T.Buffer((4,), "int32"),
        coeff: T.Buffer((4,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(4):
            out[t] = out[t] + coeff[t] * x[idx[t]] * x[idx[t] + 1]

    annotated = tilelang.transform.annotate_path_descriptors(before, {"idx": [0, 1, 1, 2], "coeff": [0.5, 0.25, 0.125, 2.0]})
    mod = tvm.IRModule.from_expr(annotated.with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config={"tl.PathLocalityReorder": {"enable_pair_cse": False}}):
        after = tilelang.transform.PathLocalityReorder()(mod)["main"]

    assert not _collect(after.body, tvm.tirx.For), "annotated path loop must specialize and unroll"
    counts = _label_bind_counts(after)
    # idx = [0,1,1,2] makes paths x[0]*x[1], x[1]*x[2], x[1]*x[2], x[2]*x[3].
    for i in range(4):
        assert counts[("x", (i,))] == 1, counts
    for store in _collect(after.body, tvm.tirx.BufferStore):
        loads = [ld for ld in _collect(store.value, tvm.tirx.BufferLoad) if ld.buffer.name in ("idx", "coeff")]
        assert not loads, "descriptor loads must fold into constants"


def test_serial_loop_without_descriptor_tables_kept():
    @T.prim_func
    def before(
        x: T.Buffer((8,), "float32"),
        idx: T.Buffer((4,), "int32"),
        coeff: T.Buffer((4,), "float32"),
        out: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(4):
            out[t] = out[t] + coeff[t] * x[idx[t]] * x[idx[t] + 1]

    # Without the descriptor attr the labels stay runtime loads with no
    # sharing, so the loop is preserved.
    _assert_unchanged(before)


def test_loop_annotation_fences_body_regions():
    @T.prim_func
    def before(
        w: T.Buffer((2,), "float32"),
        x: T.Buffer((2,), "float32"),
        out: T.Buffer((4,), "float32"),
        flag: T.Buffer((4,), "float32"),
    ):
        for t in T.serial(2, annotations={"tl.path_locality_reorder": 0}):
            out[0] = out[0] + T.float32(2) * w[0] * x[0]
            out[1] = out[1] + T.float32(3) * w[0] * x[1]
            flag[t] = T.float32(1)

    # The fence annotation must also protect statement runs INSIDE the
    # annotated loop's body, not only the loop-unroll rewrite itself.
    _assert_unchanged(before)


if __name__ == "__main__":
    tilelang.testing.main()
