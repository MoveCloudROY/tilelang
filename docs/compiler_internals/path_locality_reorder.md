# Path Locality Reorder Pass Proposal

This note proposes a TileLang TIR pass for kernels that compute many
small "paths" from a compact set of operands and accumulate one or more results
per path. The motivating reference is the local FastEq checkout:

- `FastEq/fasteq/tilelang/uniform1d.py`: a direct TileLang implementation with
  a `for t in T.serial(P)` path loop.
- `FastEq/fasteq/ops/uniform1d_auto_schedule.py`: the LARS scheduler and CUDA
  emitter used to generate register-budget-aware uniform1d kernels.
- `FastEq/fasteq/cuda/src/uniform1d_fused_fwd.cu` and related CUDA files:
  baseline fused kernels.

The requested `FastEq/fasteq/cuda/src/uniform1d_codegen` directory is not
present in the current checkout. The auto-schedule file nevertheless contains
the optimized code-generation path, including logical `load`, `release`,
`fma_u1d_resident`, `pair_cse`, and backward variants.

## Goal

Add a late TIR transform that:

1. Detects a small constant path loop whose iterations are structurally the
   same computation at adjacent affine offsets.
2. Explicitly unrolls that path loop when needed.
3. Reorders the resulting statements so equivalent instructions from adjacent
   paths are adjacent, while preserving dependencies and side effects.

The intended effect is better register reuse for common path operands and pair
products. In FastEq forward uniform1d, each path computes:

```text
out[v] += coeff * w[i] * x[j] * y[k]
```

The optimized scheduler treats each path as a set of labels such as `("x", j)`,
`("y", k)`, and `("w", i)`, schedules loads under a virtual register budget,
fires any path whose labels are resident, and optionally caches repeated
`w[i] * x[j]` pairs. The backward pass generalizes this to labels
`w`, `x`, `y`, and `grad_out`, with optional pair reuse for `w * grad_out`.

## Pattern Recognition

The pass should match either the pre-unroll loop form or the post-unroll
`SeqStmt` form. The first implementation should target the FastEq TileLang
shape directly, then generalize to structurally equivalent loops.

The canonical pre-unroll pattern is:

```python
for t in T.serial(P):
    i = i_list[t]
    j = j_list[t]
    k = k_list[t]
    v = v_list[t]
    c = coeff_list[t]

    wval = w[b, i, lane]
    xval = x_all[src, j, lane]
    yval = y[b, k]
    acc = c * wval * xval * yval
    T.atomic_add(out[dst, v, lane], acc)
```

Required checks:

- The path loop is `ForKind::kSerial` or `ForKind::kUnrolled`, and its extent
  is either a small constant or a compile-time tensor of path descriptors that
  can be read by the front-end/autotuner before lowering.
- Path descriptors are invariant within the kernel launch. For FastEq this is
  the tuple `(i_list[t], j_list[t], k_list[t], v_list[t], coeff_list[t])`.
- The loop body is a straight-line product/sum tree over path-local loads,
  scalar coefficient loads, and one accumulation store.
- Input labels can be normalized into stable identities. For forward uniform1d,
  use `x[j]`, `y[k]`, and `w[i]`; for backward, use `w[i]`, `x[j]`, `y[k]`,
  and `grad_out[v]`.
- Output accumulators can be kept resident or safely accumulated. Atomic stores
  and scatter stores should be treated as final sinks; the pass should not
  reorder two updates to the same output unless addition associativity policy
  and dtype policy explicitly allow it.
- The loop body contains only reorderable statements:
  `BufferStore` to local/fragment/global output buffers, pure `Evaluate`
  expressions, and `LetStmt`/`AttrStmt` wrappers that can be kept with the
  statement they guard.
- Calls with side effects, barriers, async copies, atomics, volatile loads,
  `tvm_storage_sync`, mbarrier/tma operations, and opaque extern calls terminate
  the reorderable region.
- Writes are single-assignment inside the region, or all write-after-write
  ordering edges are made explicit in the dependency graph.
- Stores for different paths write disjoint affine indices. This should be
  proven with `arith::Analyzer`; if the proof fails, the region is skipped.

For each path, compute a `PathRecord`:

- `path_id`: original loop iteration.
- `input_labels`: normalized operand labels, e.g. `{x[j], y[k], w[i]}`.
- `output_labels`: normalized output/gradient labels, e.g. `{out[v]}` or
  `{grad_w[i], grad_x[j], grad_y[k]}`.
- `pair_keys`: profitable common products, e.g. `(w[i], x[j])` for forward and
  `(w[i], grad_out[v])` for backward.
- `compute_stmt`: the pure arithmetic statement or small statement bundle that
  emits this path's contribution.
- `sink_stmt`: final store/atomic/add statement, if it cannot be represented as
  a local accumulator update.

The pass should only accept a region when at least two adjacent `path_id` values
or two paths in the same region share at least one input label or pair key;
otherwise there is no expected register-reuse benefit.

## Reorder Algorithm

The implementation should be a new C++ pass, for example
`src/transform/path_locality_reorder.cc`, following the local pass style:

```cpp
Pass PathLocalityReorder() {
  auto pass_func = [](PrimFunc func, const IRModule& mod,
                      PassContext ctx) -> PrimFunc {
    arith::Analyzer analyzer;
    return PathLocalityRewriter::Rewrite(std::move(func), &analyzer);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PathLocalityReorder", {});
}
```

Use TVM handle types (`Stmt`, `For`, `Buffer`, `Var`) across helper boundaries,
and use `ObjectPtrHash`/`ObjectPtrEqual` for identity maps. Raw `*Node` pointers
should stay local to visitor callbacks and pattern checks.

The rewrite has four local steps:

1. **Region formation**

   Visit `SeqStmt` bodies and split them into maximal reorderable regions.
   Non-reorderable statements act as fences. A candidate path loop can either be
   sent through the existing `UnrollLoop` logic first or locally unrolled using
   the same substitution plus definition-freshening strategy used by
   `src/transform/unroll_loop.cc`.

2. **Path and label graph**

   Convert the region into a small scheduling problem similar to FastEq LARS:

   - `Label`: a stable operand identity, for example `(buffer=w, index=i)` or
     `(buffer=x_all, row=src, index=j, lane=lane)`.
   - `Path`: one contribution with a set of input labels and output labels.
   - `label_to_paths`: map from each label to paths that still need it.
   - `remaining_uses`: use count per label.
   - `live`: labels currently represented by local scalar temporaries.
   - `pair_key`: optional common product such as `(w[i], x[j])`.

   In TIR this means converting repeated buffer loads to named local scalar
   definitions when the scheduler decides to load a label, then substituting
   those scalars into each fired path's compute statement.

3. **Dependency graph**

   Create one node per statement and add edges for:

   - true dependencies: a read of a local/fragment value must stay after the
     statement that writes it;
   - anti/output dependencies when the same buffer/index may alias;
   - all original-order edges involving opaque calls, barriers, volatile memory,
     atomics, async copy, or output stores whose relative order cannot be proven
     irrelevant;
   - wrapper dependencies so an `AttrStmt`, predicate, or `LetStmt` remains
     attached to the statement it semantically guards.

   Alias checks should first use exact buffer identity plus analyzer-normalized
   indices. Unknown aliasing should conservatively keep original order.

4. **Register-budget-aware list scheduling**

   Schedule with the same policy shape as FastEq's LARS scheduler:

   1. If any path is fireable, meaning all its `input_labels` are live, emit
      its compute bundle immediately.
   2. Otherwise choose a non-live label to load. Prefer labels that:
      create more fireable paths, release more last-use labels, have higher
      affinity with already-live labels, and have more remaining uses.
   3. If the live label count reaches `reg_budget`, choose a spill victim
      jointly with the next label. Avoid spilling labels required by the same
      target path.
   4. If global selection makes no progress, use path-directed fallback: choose
      an unscheduled path closest to fireable, load one missing label from that
      path, and avoid evicting the path's already-live labels.
   5. After a path fires, decrement remaining uses and release labels whose use
      count reaches zero.
   6. Opportunistically materialize pair CSE only when the pair has at least two
      remaining uses and a free virtual register exists. Pair temporaries should
      be dropped before spilling normal operand labels under pressure.

   Logical instruction stream:

   ```text
   load        r0, w[i0]
   load        r1, x[j0]
   load        r2, y[k0]
   pair_cse    r3, r0, r1      # optional
   fma_path    path0, r3, r2
   release     r2
   load        r2, y[k1]
   fma_path    path1, r3, r2   # reuses pair
   release_pair r3
   ```

   The TileLang pass should not literally emit `load` or `release` pseudo-ops.
   They are internal scheduling actions. The final TIR should be a `SeqStmt`
   of scalar `LetStmt`/local `BufferStore` definitions, path compute statements,
   and final stores/atomics in the scheduled order. Run `ConvertSSA` only if
   definitions were duplicated during local unroll.

## Pipeline Placement

For CUDA, place the pass after general loop unrolling and before final cleanup:

```python
mod = tilelang.transform.LoopUnswitching()(mod)
mod = tilelang.transform.UnrollLoop()(mod)
mod = tilelang.transform.PathLocalityReorder()(mod)
mod = s_tir.transform.RenormalizeSplitPattern()(mod)
mod = tirx.transform.Simplify()(mod)
```

This placement sees scalarized/local-access loops after `StorageRewrite` and
`UnrollLoop`, but still leaves simplification and no-op removal to clean up the
result. The first implementation should enable the pass only for CUDA because
the optimization target is CUDA register pressure and instruction order. ROCm
and other SIMT backends can opt in later once codegen effects are measured.

## Configuration And User Interface

No TileLang language syntax needs to change.

Recommended controls are pass-context only:

```python
with tilelang.transform.PassContext(config={
    "tl.enable_path_locality_reorder": True,
    "tl.PathLocalityReorder": {
        "max_paths": 8,
        "reg_budget": 16,
        "enable_pair_cse": True,
        "enable_secondary_affinity": True,
        "min_shared_reads": 1,
        "path_fallback_after": 32,
    },
}):
    kernel = tilelang.compile(fn, target="cuda")
```

Default rollout should be disabled or target-gated at first, then enabled for
recognized safe patterns after correctness and performance coverage is stable.
If users need a manual escape hatch later, add a loop/block annotation such as
`T.annotate("tl.path_locality_reorder", 0)` to fence a problematic region. Avoid
adding a new public scheduling primitive unless autotuning needs to select this
explicitly.

## Test Set

Build the test set around semantic equivalence, pass shape, and performance
signals.

Correctness tests:

- FastEq forward `uniform1d` TileLang kernel:
  `out[dst, v, lane] += c * w[b, i, lane] * x_all[src, j, lane] * y[b, k]`.
- FastEq forward dense output mode without scatter atomics.
- FastEq backward `uniform1d`:
  `grad_w`, `grad_x`, and `grad_y` contributions from
  `grad_out[v] * w[i] * x[j] * y[k]`.
- `uniform1d` three-point stencil: `y[i] = a*x[i-1] + b*x[i] + c*x[i+1]`.
- `uniform1d` five-point stencil with two outputs per thread.
- First-order upwind / linear advection update with adjacent output paths.
- 1D heat equation update with two time steps fused in registers.
- 1D wave equation update reading two input state arrays and writing one output.
- Elementwise polynomial chain over adjacent elements, to confirm the pass
  skips when there are no shared reads.
- Prefix/scan-like recurrence, to confirm dependency edges prevent illegal
  reordering.
- Atomic or barrier-containing body, to confirm the region is fenced.
- Predicated boundary stencil, to confirm predicates stay attached.
- Dynamic shape with constant path extent, to confirm affine proofs do not rely
  on static global length.

IR-shape tests:

- A path loop with constant extent or compile-time path descriptors is
  explicitly unrolled when the path-locality pattern is detected.
- Repeated operand labels are loaded once and used by multiple scheduled path
  compute statements when dependencies permit.
- Repeated pair keys such as `(w[i], x[j])` are materialized only when
  `enable_pair_cse=True`, the pair has at least two uses, and the register
  budget has slack.
- Under tight register budgets, emitted order follows path-directed fallback
  rather than oscillating between loading and spilling labels for the same
  target path.
- Non-reorderable calls preserve original relative order.
- Unknown aliasing falls back to the original order.

Performance/regression tests:

- Compare generated CUDA for `uniform1d` against the baseline with the pass
  disabled.
- Compare against FastEq generated LARS schedules for representative path lists.
- Track `ptxas` register count, local-memory spill count, and instruction count.
- Benchmark path extents 2, 4, 8, and 16 for stencil radius 1, 2, and 4.
- Sweep `reg_budget` values such as 8, 12, 16, 24, and 32 to verify the
  scheduler changes only when extra budget exposes useful liveness or CSE.
- Include small and large `N` to catch instruction-cache and occupancy tradeoffs.

Useful kernel families beyond FastEq uniform1d:

- batched 1D finite-difference stencils;
- fused finite-volume flux computations over adjacent cells;
- sliding-window signal filters such as FIR and Savitzky-Golay;
- local polynomial transforms over contiguous vectors;
- activation/normalization kernels that compute several adjacent elements per
  lane from a shared neighborhood;
- shallow explicit time-stepping kernels where two or three time levels are kept
  in registers.

## V1 Implementation Design

V1 should be deliberately narrow: implement a working pass for forward
uniform1d-style path loops whose path descriptors are compile-time constants.
This gives a measurable pass without requiring runtime inspection of
`i_list[t]`, `j_list[t]`, `k_list[t]`, `v_list[t]`, or `coeff_list[t]`.
Runtime descriptor tensors can be supported later by a front-end helper that
specializes them into PrimFunc attrs before this pass runs.

### Files To Touch

- `src/transform/path_locality_reorder.cc`: new C++ pass, config attrs, pattern
  extractor, scheduler, and rewriter.
- `tilelang/transform/__init__.py`: Python wrapper `PathLocalityReorder()`.
- `tilelang/transform/pass_config.py`: pass-config keys and documentation.
- `tilelang/cuda/pipeline.py`: optional CUDA pipeline call behind
  `tl.enable_path_locality_reorder`.
- `testing/python/transform/test_path_locality_reorder.py`: IR-shape tests.
- `testing/python/target/test_tilelang_path_locality_reorder_codegen.py`:
  CUDA correctness/codegen tests once the IR-shape tests are stable.

The top-level `CMakeLists.txt` already glob-builds `src/transform/*.cc`, so a
new `.cc` file is enough for the C++ object to be compiled.

### Public Pass API

Register the pass like other TileLang transforms:

```cpp
TVM_REGISTER_PASS_CONFIG_OPTION("tl.PathLocalityReorder",
                                PathLocalityReorderConfig);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.PathLocalityReorder",
                        PathLocalityReorder);
}
```

The config object should be an `AttrsNodeReflAdapter`:

```cpp
struct PathLocalityReorderConfigNode
    : public AttrsNodeReflAdapter<PathLocalityReorderConfigNode> {
  bool enable;
  int max_paths;
  int reg_budget;
  bool enable_pair_cse;
  bool enable_secondary_affinity;
  int path_fallback_after;
  bool allow_atomic_reorder;
};
```

Recommended defaults:

- `enable=false`: pipeline-gated while experimental.
- `max_paths=64`: avoids code-size blowups.
- `reg_budget=16`: matches FastEq's default forward scheduler scale.
- `enable_pair_cse=true`: only materializes pairs when there is slack.
- `enable_secondary_affinity=true`: useful for path clusters with partial
  overlap.
- `path_fallback_after=2 * reg_budget`.
- `allow_atomic_reorder=false`: V1 preserves atomic update order for identical
  output labels; users can opt into relaxed floating-point association later.

### V1 Pattern Contract

The C++ pass should match a loop only when all path records are explicit in IR.
Use one of these two encodings:

1. A loop or block annotation containing a serialized path table:

   ```python
   T.annotate("tl.path_locality", {
       "kind": "uniform1d_fwd",
       "paths": [(i0, j0, k0, v0, c0), ...],
   })
   ```

2. A PrimFunc attr containing the same table, plus an annotation identifying
   the loop to rewrite.

The body must be equivalent to:

```text
wval = w[b, i, lane]
xval = x_all[src, j, lane]
yval = y[b, k]
acc = coeff * wval * xval * yval
atomic_add(out[dst, v, lane], acc)
```

Reject the candidate if:

- the path table length is zero or greater than `max_paths`;
- any path index is symbolic in V1;
- the loop contains barriers, async copies, non-uniform control flow, or opaque
  calls other than the final atomic/add sink;
- two paths write the same output label and `allow_atomic_reorder=false`;
- dtype is not a normal floating dtype supported by the target codegen.

### C++ Data Structures

Keep raw `*Node` pointers local to visitors and store IR handles in records:

```cpp
struct LabelKey {
  Buffer buffer;
  ffi::Array<PrimExpr> indices;
  std::string kind;  // "w", "x", "y", "out"
};

struct PairKey {
  LabelId lhs;
  LabelId rhs;
};

struct PathRecord {
  int path_id;
  std::vector<LabelId> input_labels;
  LabelId output_label;
  Optional<PairKey> pair_key;
  PrimExpr coeff;
  Stmt sink_template;
};

struct ScheduledOp {
  enum class Kind { kLoad, kPairCse, kCompute, kRelease, kReleasePair };
  Kind kind;
  int path_id;
  LabelId label;
  PairKey pair_key;
};
```

Use an interner to assign dense `LabelId` values. The interner should key labels
by buffer identity and analyzer-normalized indices using TVM object handles and
structural equality. Unknown structural comparison should make extraction fail
instead of guessing.

### Scheduler

V1 should port the FastEq LARS policy, but emit internal `ScheduledOp` records
rather than text instructions.

Pseudo-code:

```text
while unscheduled_paths is not empty:
    fireable = paths whose input_labels are all live
    if fireable:
        pid = choose_fireable_path(fireable)
        maybe_emit_pair_cse(pid)
        emit_compute(pid)
        consume uses and release dead labels/pairs
        continue

    if no progress for path_fallback_after or live.size == reg_budget:
        label, victim = select_by_target_path()
    else:
        label, victim = select_by_global_score()

    if live.size == reg_budget:
        emit_release(victim)
    emit_load(label)
```

The global score mirrors FastEq:

```text
score(label) =
  fireable_count_after_load,
  last_use_release_count_after_load,
  primary_affinity_with_live_labels,
  secondary_affinity_with_neighbor_labels,
  -non_live_affinity_penalty,
  remaining_uses
```

Pair CSE rule:

- V1 pair key is `(w_label, x_label)`.
- Only create the pair if `enable_pair_cse`, the pair has at least two
  remaining uses, and there is a free virtual register.
- If register pressure appears later, drop pair temporaries before normal
  operand labels.

### IR Emission

The scheduler's `load` and `release` operations are virtual. Emission should
turn them into a straight-line `SeqStmt`:

- `load(label)`: create a fresh scalar/local variable definition for the
  corresponding `BufferLoad`.
- `pair_cse(pair)`: create a fresh scalar/local variable for the product.
- `compute(path)`: emit the original sink with loads substituted by the live
  scalar variables and, if available, the pair scalar.
- `release(...)`: no IR statement; only ends an internal live range.

Prefer `LetStmt` for scalar temporaries if downstream TileLang passes preserve
them well at this stage. If CSE through `LetStmt` is undone too early, use
`local.var` buffer stores instead and let `StorageRewrite`/codegen scalarize
them. V1 placement after `StorageRewrite` favors `LetStmt`, followed by
`Simplify` and `RemoveNoOp`.

Atomic sink policy:

- For scatter atomics, V1 can reorder only if output labels are distinct or
  `allow_atomic_reorder=true`.
- For dense owned outputs, rewrite repeated output updates into one local
  accumulator per `out[v]` and emit a final store after scheduled computes.

### Pipeline Placement

Wire CUDA as:

```python
mod = tilelang.transform.StorageRewrite()(mod)
mod = tilelang.transform.LoopUnswitching()(mod)
mod = tilelang.transform.UnrollLoop()(mod)
if enable_path_locality_reorder(pass_ctx):
    mod = tilelang.transform.PathLocalityReorder()(mod)
mod = s_tir.transform.RenormalizeSplitPattern()(mod)
mod = tirx.transform.Simplify()(mod)
```

Do not enable it for CPU/Metal/WebGPU in V1. ROCm can reuse the common pass
later after CUDA correctness and register metrics are understood.

### Test-Driven Slice

Start with pure IR tests before CUDA:

1. Build a tiny PrimFunc with four compile-time paths:
   `(w0,x0,y0,out0)`, `(w0,x0,y1,out1)`, `(w1,x1,y0,out2)`,
   `(w1,x1,y1,out3)`.
1. Run `PathLocalityReorder` with `reg_budget=4`.
1. Assert the transformed IR contains one load-like scalar binding for each
   repeated `w/x` label before multiple compute sinks.
1. Assert pair CSE appears for `(w0,x0)` and `(w1,x1)` when enabled, and does
   not appear when disabled.
1. Assert same-output atomics are not reordered unless explicitly allowed.
1. Add a barrier/opaque-call body and assert the pass is a no-op.

Then add CUDA correctness with small tensors and compare against the original
TileLang uniform1d loop for deterministic path tables.

## Implementation Milestones

1. Add the pass skeleton, FFI registration, Python wrapper, and pass config keys.
1. Implement the post-unroll `SeqStmt` recognizer first. This is easier to test
   and avoids duplicating `UnrollLoop` logic initially.
1. Add path/label extraction for FastEq-style forward uniform1d.
1. Add dependency graph construction and conservative LARS-style list
   scheduling with label live ranges, victim selection, and path fallback.
1. Add optional pair CSE for repeated products such as `(w, x)`.
1. Add a small optional pre-unroll path-loop recognizer that rewrites only when
   `UnrollLoop` would not otherwise expose the pattern.
1. Wire the pass into the CUDA pipeline behind
   `tl.enable_path_locality_reorder`.
1. Add IR-shape tests and CUDA correctness tests.
1. Add backward uniform1d extraction and `w * grad_out` pair CSE.
1. Add opt-in benchmarks and compare register/spill metrics before enabling by
   default for any pattern.
