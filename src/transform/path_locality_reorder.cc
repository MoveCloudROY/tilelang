/*!
 * \file path_locality_reorder.cc
 * \brief Reorder unrolled "path" statements for register locality.
 *
 * Many small kernels compute a set of paths from a compact operand set and
 * accumulate one contribution per path:
 *
 *   out[v] += coeff * w[i] * x[j] * y[k]
 *
 * After unrolling, adjacent paths frequently share operand loads such as
 * `w[i]` or `x[j]`. This pass groups structurally-equivalent path statements
 * into regions, converts repeated buffer loads into named scalar bindings
 * (virtual "registers"), and list-schedules the path computations under a
 * virtual register budget so equivalent instructions from related paths are
 * adjacent and shared operands are loaded once. Optionally, repeated pair
 * products (e.g. `w[i] * x[j]`) are materialized as a common subexpression.
 *
 * The scheduling policy is a port of the FastEq LARS uniform1d scheduler
 * (see docs/compiler_internals/path_locality_reorder.md):
 *   1. fire any path whose input labels are all live;
 *   2. otherwise load the label with the best global score;
 *   3. under register pressure or lack of progress, fall back to
 *      path-directed selection that avoids evicting labels required by the
 *      chosen target path;
 *   4. release labels when their remaining use count reaches zero;
 *   5. opportunistically cache pair products when a register is free.
 *
 * V1 recognizes two path sink forms:
 *   - dense accumulate:  BufferStore(out, out[idx] + product, idx)
 *   - scatter atomic:    Evaluate(tl.atomic_add_elem(tvm_access_ptr(out, off),
 *                                                    product[, memory_order]))
 * where `product` is a multiplication tree over scalar constants, scalar
 * variables and scalar BufferLoads (the operand labels). Atomic sinks are
 * accepted only with relaxed (or absent) memory order; acquire/release/
 * seq_cst atomics carry synchronization semantics and fence the region.
 *
 * Correctness is guaranteed by region acceptance instead of a general
 * dependency graph: a region is rewritten only when all path statements are
 * pure except for their single output update, no input label aliases any
 * output, and all outputs are provably pairwise-disjoint (or duplicates are
 * explicitly allowed via `allow_atomic_reorder`). Any statement that does not
 * match the pattern (barriers, calls, control flow, ...) fences the region.
 * Scheduled paths reproduce the original value expression bit-exactly except
 * under pair CSE, which reassociates the multiplication chain of paths that
 * consume a cached pair; set `enable_pair_cse=false` (or fence the loop with
 * the `tl.path_locality_reorder = 0` annotation) for bit-exact scheduling.
 */

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/*! \brief Loop/region annotation that fences the pass when set to 0. */
constexpr const char *kPathLocalityReorderAnnotation =
    "tl.path_locality_reorder";

/*! \brief Boolean pass-context key that gates pipeline wiring. */
constexpr const char *kEnablePathLocalityReorder =
    "tl.enable_path_locality_reorder";

/*!
 * \brief PrimFunc attr carrying compile-time path-descriptor tables.
 *
 * Maps a descriptor buffer's data var to the array of values the caller
 * will pass at runtime (see tilelang.transform.annotate_path_descriptors).
 * Loads of these buffers at constant indices fold into the table entries,
 * so runtime-descriptor path loops such as
 *
 *   for t in T.serial(P):
 *       T.atomic_add(out[dst, v_list[t], lane],
 *                    coeff_list[t] * w[b, i_list[t], lane] * ...)
 *
 * specialize into the compile-time path form this pass schedules.
 */
constexpr const char *kPathLocalityDescriptorsAttr =
    "tl.path_locality_descriptors";

struct PathLocalityReorderConfigNode
    : public AttrsNodeReflAdapter<PathLocalityReorderConfigNode> {
  bool enable;
  int max_paths;
  int reg_budget;
  bool enable_pair_cse;
  bool enable_secondary_affinity;
  int min_shared_reads;
  int path_fallback_after;
  bool allow_atomic_reorder;

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<PathLocalityReorderConfigNode>()
        .def_ro("enable", &PathLocalityReorderConfigNode::enable,
                "Whether the pass rewrites anything when invoked. Rollout "
                "gating happens in the pipeline via "
                "tl.enable_path_locality_reorder; this key is an escape hatch "
                "to hard-disable an explicitly wired pass.",
                refl::DefaultValue(true))
        .def_ro("max_paths", &PathLocalityReorderConfigNode::max_paths,
                "Maximum number of path statements per region; larger regions "
                "are left untouched to avoid code-size blowups.",
                refl::DefaultValue(64))
        .def_ro("reg_budget", &PathLocalityReorderConfigNode::reg_budget,
                "Virtual register budget for live operand labels and pair "
                "temporaries. This is a scheduling budget, not the ptxas "
                "register count.",
                refl::DefaultValue(16))
        .def_ro("enable_pair_cse",
                &PathLocalityReorderConfigNode::enable_pair_cse,
                "Materialize repeated pair products (e.g. w[i]*x[j]) as a "
                "scalar temporary when the pair has at least two remaining "
                "uses and a virtual register is free. Paths consuming a "
                "cached pair have their multiplication chain reassociated, "
                "which can change bit-level rounding; disable for bit-exact "
                "scheduling.",
                refl::DefaultValue(true))
        .def_ro("enable_secondary_affinity",
                &PathLocalityReorderConfigNode::enable_secondary_affinity,
                "Include second-order affinity with neighbor labels in the "
                "label selection score.",
                refl::DefaultValue(true))
        .def_ro("min_shared_reads",
                &PathLocalityReorderConfigNode::min_shared_reads,
                "Minimum number of avoided reloads (sum over labels of "
                "uses-1) required before a region is rewritten.",
                refl::DefaultValue(1))
        .def_ro("path_fallback_after",
                &PathLocalityReorderConfigNode::path_fallback_after,
                "Force path-directed label selection after this many "
                "scheduler iterations without firing a path. Non-positive "
                "values mean max(8, 2 * reg_budget).",
                refl::DefaultValue(0))
        .def_ro("allow_atomic_reorder",
                &PathLocalityReorderConfigNode::allow_atomic_reorder,
                "Allow reordering two updates to the same output element. "
                "This relaxes floating-point addition association and is "
                "disabled by default.",
                refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.PathLocalityReorderConfig",
                                    PathLocalityReorderConfigNode,
                                    BaseAttrsNode);
};

class PathLocalityReorderConfig : public Attrs {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PathLocalityReorderConfig,
                                                Attrs,
                                                PathLocalityReorderConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  PathLocalityReorderConfigNode::RegisterReflection();
}

TVM_REGISTER_PASS_CONFIG_OPTION("tl.PathLocalityReorder",
                                PathLocalityReorderConfig);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnablePathLocalityReorder, Bool);

namespace {

/*!
 * \brief A stable operand identity: a buffer-like base plus normalized
 * indices.
 *
 * The base is a Buffer handle for BufferLoad/BufferStore labels, or the
 * buffer data Var for tvm_access_ptr-based atomic destinations. Indices are
 * analyzer-simplified and compared structurally, so `w[v + 0]` and `w[v]`
 * intern to the same label.
 */
struct LabelInfo {
  ObjectRef base;
  Array<PrimExpr> indices;
  /*! \brief Representative load expression, used to emit label bindings. */
  PrimExpr repr_load;
  std::string name_hint;
};

class LabelInterner {
public:
  int Intern(const ObjectRef &base, Array<PrimExpr> indices, PrimExpr repr_load,
             std::string name_hint) {
    int found = Find(base, indices);
    if (found >= 0) {
      return found;
    }
    labels_.push_back(LabelInfo{base, std::move(indices), std::move(repr_load),
                                std::move(name_hint)});
    return static_cast<int>(labels_.size()) - 1;
  }

  /*! \brief Find an existing label without inserting; returns -1 if absent. */
  int Find(const ObjectRef &base, const Array<PrimExpr> &indices) const {
    StructuralEqual eq;
    for (size_t i = 0; i < labels_.size(); ++i) {
      const LabelInfo &info = labels_[i];
      if (!info.base.same_as(base) || info.indices.size() != indices.size()) {
        continue;
      }
      bool all_equal = true;
      for (size_t d = 0; d < indices.size(); ++d) {
        if (!eq(info.indices[d], indices[d])) {
          all_equal = false;
          break;
        }
      }
      if (all_equal) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  const LabelInfo &at(int id) const { return labels_[id]; }
  int size() const { return static_cast<int>(labels_.size()); }

private:
  std::vector<LabelInfo> labels_;
};

/*! \brief One multiplicative leaf of a path product. */
struct Factor {
  PrimExpr expr;
  /*! \brief Input label id, or -1 for constant/variable factors. */
  int label{-1};
};

/*! \brief One recognized path statement. */
struct ParsedPath {
  Stmt original;
  bool is_atomic{false};
  /*! \brief The product expression contributing this path's value. */
  PrimExpr product;
  std::vector<Factor> factors;
  /*! \brief Sorted unique input label ids. */
  std::vector<int> input_labels;
  int output_label{-1};
  /*! \brief Pair-CSE key (input label ids, lhs < rhs), or (-1, -1). */
  std::pair<int, int> pair_key{-1, -1};
};

std::string SanitizeName(const std::string &name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    out.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  }
  return out.empty() ? std::string("v") : out;
}

bool IsFloatingDType(DataType dtype) {
  return dtype.lanes() == 1 && (dtype.is_float() || dtype.is_bfloat16());
}

bool IsPureExpr(const PrimExpr &expr) {
  return SideEffect(expr) <= CallEffectKind::kReadState;
}

/*! \brief Collect the data vars of every buffer read inside an expression. */
void CollectReadDataVars(const PrimExpr &expr,
                         std::unordered_set<const VarNode *> *reads) {
  PostOrderVisit(expr, [&](const ObjectRef &obj) {
    if (const auto *load = obj.as<BufferLoadNode>()) {
      reads->insert(load->buffer->data.get());
    }
  });
}

/*!
 * \brief Fold loads of compile-time descriptor tables into constants.
 *
 * Only loads with a provably constant, in-bounds index fold; everything else
 * is left untouched (and typically rejects the region later). The caller of
 * the annotated kernel is responsible for passing tensors whose contents
 * match the annotated tables.
 */
class DescriptorFolder : public StmtExprMutator {
public:
  DescriptorFolder(
      const std::unordered_map<const VarNode *, Array<PrimExpr>> *tables,
      arith::Analyzer *analyzer)
      : tables_(tables), analyzer_(analyzer) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    PrimExpr visited = StmtExprMutator::VisitExpr_(op);
    const auto *load = visited.as<BufferLoadNode>();
    if (load == nullptr || load->indices.size() != 1 ||
        load->predicate.defined()) {
      return visited;
    }
    auto it = tables_->find(load->buffer->data.get());
    if (it == tables_->end()) {
      return visited;
    }
    PrimExpr index = analyzer_->Simplify(load->indices[0]);
    const auto *imm = index.as<IntImmNode>();
    if (imm == nullptr || imm->value < 0 ||
        imm->value >= static_cast<int64_t>(it->second.size())) {
      return visited;
    }
    PrimExpr value = it->second[imm->value];
    DataType dtype = load->buffer->dtype;
    if (value.dtype() != dtype) {
      if (const auto *int_value = value.as<IntImmNode>()) {
        value = dtype.is_float()
                    ? PrimExpr(FloatImm(dtype,
                                        static_cast<double>(int_value->value)))
                    : PrimExpr(IntImm(dtype, int_value->value));
      } else if (const auto *float_value = value.as<FloatImmNode>()) {
        if (!dtype.is_float() && !dtype.is_bfloat16()) {
          return visited;
        }
        value = FloatImm(dtype, float_value->value);
      } else {
        return visited;
      }
    }
    return value;
  }

private:
  const std::unordered_map<const VarNode *, Array<PrimExpr>> *tables_;
  arith::Analyzer *analyzer_;
};

/*! \brief Whether the fence annotation disables the pass for a scope. */
bool PassFencedByAnnotation(const Map<String, Any> &annotations) {
  auto fence = annotations.Get(kPathLocalityReorderAnnotation);
  if (!fence) {
    return false;
  }
  // Annotation values may be plain ints or IntImm objects depending on the
  // frontend that produced them.
  if (auto as_int = fence->try_cast<int64_t>()) {
    return *as_int == 0;
  }
  if (const auto *as_imm = fence->as<IntImmNode>()) {
    return as_imm->value == 0;
  }
  return false;
}

/*! \brief Flatten a multiplication tree into ordered leaves. */
void FlattenProduct(const PrimExpr &expr, std::vector<PrimExpr> *leaves) {
  if (const auto *mul = expr.as<MulNode>()) {
    FlattenProduct(mul->a, leaves);
    FlattenProduct(mul->b, leaves);
    return;
  }
  leaves->push_back(expr);
}

/*!
 * \brief Extracts ParsedPath records from candidate statements.
 *
 * Input labels and output labels use separate interners because they live in
 * different scheduling spaces: inputs are load candidates for the register
 * budget, outputs only participate in disjointness/ordering policy.
 */
class PathParser {
public:
  PathParser(arith::Analyzer *analyzer, LabelInterner *inputs,
             LabelInterner *outputs)
      : analyzer_(analyzer), inputs_(inputs), outputs_(outputs) {}

  static constexpr int kMaxFactors = 16;

  bool Parse(const Stmt &stmt, ParsedPath *path) {
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      return ParseDenseAccumulate(stmt, store, path);
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      return ParseAtomicAdd(stmt, eval, path);
    }
    return false;
  }

private:
  Array<PrimExpr> NormalizeIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> normalized;
    normalized.reserve(indices.size());
    for (const PrimExpr &index : indices) {
      normalized.push_back(analyzer_->Simplify(index));
    }
    return normalized;
  }

  bool IndicesArePure(const Array<PrimExpr> &indices) {
    for (const PrimExpr &index : indices) {
      if (index.dtype().lanes() != 1 || !IsPureExpr(index)) {
        return false;
      }
    }
    return true;
  }

  // out[idx] = out[idx] + product   /   out[idx] = product + out[idx]
  bool ParseDenseAccumulate(const Stmt &stmt, const BufferStoreNode *store,
                            ParsedPath *path) {
    if (store->predicate.defined() || !IsFloatingDType(store->value.dtype()) ||
        !IndicesArePure(store->indices)) {
      return false;
    }
    const auto *add = store->value.as<AddNode>();
    if (add == nullptr) {
      return false;
    }
    Array<PrimExpr> store_indices = NormalizeIndices(store->indices);
    auto is_self_load = [&](const PrimExpr &expr) {
      const auto *load = expr.as<BufferLoadNode>();
      if (load == nullptr || !load->buffer.same_as(store->buffer) ||
          load->predicate.defined() ||
          load->indices.size() != store_indices.size()) {
        return false;
      }
      StructuralEqual eq;
      Array<PrimExpr> load_indices = NormalizeIndices(load->indices);
      for (size_t d = 0; d < load_indices.size(); ++d) {
        if (!eq(load_indices[d], store_indices[d])) {
          return false;
        }
      }
      return true;
    };

    PrimExpr product;
    if (is_self_load(add->a)) {
      product = add->b;
    } else if (is_self_load(add->b)) {
      product = add->a;
    } else {
      return false;
    }

    path->original = stmt;
    path->is_atomic = false;
    path->product = product;
    path->output_label =
        outputs_->Intern(store->buffer, store_indices, PrimExpr(),
                         SanitizeName(std::string(store->buffer->name)));
    return ParseProduct(product, path);
  }

  // Evaluate(tl.atomic_add_elem(tvm_access_ptr(ty, data, off, 1, mask),
  //                             product[, memory_order]))
  bool ParseAtomicAdd(const Stmt &stmt, const EvaluateNode *eval,
                      ParsedPath *path) {
    const auto *call = eval->value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(atomic_add_elem_op()) ||
        call->args.size() < 2 || call->args.size() > 3) {
      return false;
    }
    if (call->args.size() == 3) {
      // Only relaxed atomics are safe to reorder and to hoist label loads
      // across: acquire/release/seq_cst orders carry synchronization
      // semantics and must fence the region. Memory-order ids follow
      // std::memory_order, so relaxed == 0 (the two-argument form defaults
      // to relaxed in codegen).
      const auto *order = call->args[2].as<IntImmNode>();
      if (order == nullptr || order->value != 0) {
        return false;
      }
    }
    const auto *ptr = call->args[0].as<CallNode>();
    if (ptr == nullptr || !ptr->op.same_as(builtin::tvm_access_ptr()) ||
        ptr->args.size() != 5) {
      return false;
    }
    const auto *data = ptr->args[1].as<VarNode>();
    const auto *extent = ptr->args[3].as<IntImmNode>();
    if (data == nullptr || extent == nullptr || extent->value != 1) {
      return false;
    }
    PrimExpr offset = ptr->args[2];
    if (offset.dtype().lanes() != 1 || !IsPureExpr(offset)) {
      return false;
    }
    PrimExpr product = call->args[1];
    if (!IsFloatingDType(product.dtype())) {
      return false;
    }

    path->original = stmt;
    path->is_atomic = true;
    path->product = product;
    path->output_label = outputs_->Intern(
        GetRef<Var>(data), {analyzer_->Simplify(offset)}, PrimExpr(),
        SanitizeName(std::string(data->name_hint)));
    return ParseProduct(product, path);
  }

  bool ParseProduct(const PrimExpr &product, ParsedPath *path) {
    std::vector<PrimExpr> leaves;
    FlattenProduct(product, &leaves);
    if (leaves.size() > static_cast<size_t>(kMaxFactors)) {
      return false;
    }
    for (const PrimExpr &leaf : leaves) {
      Factor factor;
      factor.expr = leaf;
      if (leaf->IsInstance<IntImmNode>() || leaf->IsInstance<FloatImmNode>() ||
          leaf->IsInstance<VarNode>()) {
        factor.label = -1;
      } else if (const auto *load = leaf.as<BufferLoadNode>()) {
        if (load->predicate.defined() || leaf.dtype().lanes() != 1 ||
            !IndicesArePure(load->indices)) {
          return false;
        }
        // The representative load uses normalized indices so emitted label
        // bindings are canonical (e.g. x[1] instead of x[0 + 1] after a
        // local unroll substituted the loop variable).
        Array<PrimExpr> normalized = NormalizeIndices(load->indices);
        factor.label = inputs_->Intern(
            load->buffer, normalized, BufferLoad(load->buffer, normalized),
            SanitizeName(std::string(load->buffer->name)));
      } else {
        return false;
      }
      path->factors.push_back(factor);
    }

    for (const Factor &factor : path->factors) {
      if (factor.label >= 0) {
        path->input_labels.push_back(factor.label);
      }
    }
    std::sort(path->input_labels.begin(), path->input_labels.end());
    path->input_labels.erase(
        std::unique(path->input_labels.begin(), path->input_labels.end()),
        path->input_labels.end());
    return !path->input_labels.empty();
  }

  arith::Analyzer *analyzer_;
  LabelInterner *inputs_;
  LabelInterner *outputs_;
};

/*! \brief A pure scalar Bind peeled from a candidate region. */
struct PeeledBind {
  Var var;
  /*! \brief Value with all earlier peeled binds substituted away. */
  PrimExpr resolved_value;
};

/*! \brief One candidate region extracted from a SeqStmt run or a path loop. */
struct Region {
  std::vector<ParsedPath> paths;
  std::vector<PeeledBind> binds;
  LabelInterner inputs;
  LabelInterner outputs;
};

/*! \brief Replace interned input-label loads with their live scalar vars. */
class LabelLoadReplacer : public StmtExprMutator {
public:
  LabelLoadReplacer(arith::Analyzer *analyzer, const LabelInterner &inputs,
                    const std::vector<Var> &reg_of)
      : analyzer_(analyzer), inputs_(inputs), reg_of_(reg_of) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Array<PrimExpr> normalized;
    normalized.reserve(op->indices.size());
    for (const PrimExpr &index : op->indices) {
      normalized.push_back(analyzer_->Simplify(index));
    }
    int id = inputs_.Find(op->buffer, normalized);
    if (id >= 0 && reg_of_[id].defined()) {
      return reg_of_[id];
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  arith::Analyzer *analyzer_;
  const LabelInterner &inputs_;
  const std::vector<Var> &reg_of_;
};

/*!
 * \brief Register-budget-aware list scheduler over a validated region.
 *
 * Port of the FastEq LARS/CSE-LARS policy. "load" and "release" are virtual
 * scheduling actions: a load emits a fresh scalar Bind of the label's
 * representative BufferLoad, a release only ends the internal live range
 * (labels are pure reads and can be re-materialized later).
 */
class LarsScheduler {
public:
  LarsScheduler(const Region &region, const PathLocalityReorderConfigNode *cfg,
                arith::Analyzer *analyzer)
      : region_(region), cfg_(cfg), analyzer_(analyzer) {
    int num_paths = static_cast<int>(region.paths.size());
    int num_labels = region.inputs.size();

    label_paths_.resize(num_labels);
    remaining_uses_.assign(num_labels, 0);
    live_.assign(num_labels, false);
    reg_of_.assign(num_labels, Var());
    reload_count_.assign(num_labels, 0);

    size_t max_path_labels = 0;
    for (int pid = 0; pid < num_paths; ++pid) {
      unscheduled_.insert(pid);
      const ParsedPath &path = region.paths[pid];
      max_path_labels = std::max(max_path_labels, path.input_labels.size());
      for (int label : path.input_labels) {
        label_paths_[label].push_back(pid);
        remaining_uses_[label] += 1;
      }
      if (path.pair_key.first >= 0) {
        pair_remaining_[path.pair_key] += 1;
        pair_initial_[path.pair_key] += 1;
      }
    }

    // The budget must admit at least one full path; clamp instead of failing
    // so misconfiguration degrades to a correct (if less shared) schedule.
    reg_budget_ =
        std::max({cfg->reg_budget, static_cast<int>(max_path_labels), 2});
    path_fallback_after_ = cfg->path_fallback_after > 0
                               ? cfg->path_fallback_after
                               : std::max(8, 2 * reg_budget_);
  }

  bool Schedule(Array<Stmt> *result) {
    // Termination: each fallback round loads a missing label of the target
    // path while protecting its live labels, so every path fires after at
    // most |labels| fallback rounds. The cap is a fail-safe: on overflow the
    // region is left untouched rather than aborting compilation.
    int64_t max_iters = 1024 + 64 * static_cast<int64_t>(region_.paths.size()) *
                                   std::max(1, region_.inputs.size());
    int64_t iters = 0;
    size_t done = 0;
    int no_progress = 0;

    while (!unscheduled_.empty()) {
      if (++iters > max_iters) {
        return false;
      }
      size_t now_done = region_.paths.size() - unscheduled_.size();
      if (now_done == done) {
        ++no_progress;
      } else {
        no_progress = 0;
        done = now_done;
      }

      std::vector<int> fireable = FireablePaths();
      if (!fireable.empty()) {
        Fire(ChooseFireablePath(fireable));
        continue;
      }

      int label = -1;
      int victim = -1;
      if (no_progress > path_fallback_after_ || FreeRegs() == 0) {
        SelectByTargetPath(&label, &victim);
        no_progress = 0;
      } else {
        label = SelectGlobal();
      }
      if (label < 0) {
        return false;
      }
      LoadLabel(label, victim);
    }

    *result = stmts_;
    return true;
  }

private:
  using PairKey = std::pair<int, int>;

  int FreeRegs() const {
    return reg_budget_ - live_count_ - static_cast<int>(live_pairs_.size());
  }

  bool IsFireable(int pid) const {
    for (int label : region_.paths[pid].input_labels) {
      if (!live_[label]) {
        return false;
      }
    }
    return true;
  }

  std::vector<int> FireablePaths() const {
    std::vector<int> fireable;
    for (int pid : unscheduled_) {
      if (IsFireable(pid)) {
        fireable.push_back(pid);
      }
    }
    return fireable;
  }

  int ReleaseNowCount(int pid) const {
    int count = 0;
    for (int label : region_.paths[pid].input_labels) {
      count += remaining_uses_[label] == 1 ? 1 : 0;
    }
    return count;
  }

  int ChooseFireablePath(const std::vector<int> &fireable) const {
    int best = -1;
    std::array<int, 3> best_key{};
    for (int pid : fireable) {
      int reuse = 0;
      for (int label : region_.paths[pid].input_labels) {
        reuse += remaining_uses_[label];
      }
      std::array<int, 3> key{ReleaseNowCount(pid), reuse, -pid};
      if (best < 0 || key > best_key) {
        best = pid;
        best_key = key;
      }
    }
    return best;
  }

  // ---- label scoring (FastEq LARS score) ----

  bool FireableWith(int pid, int extra_label) const {
    for (int label : region_.paths[pid].input_labels) {
      if (!live_[label] && label != extra_label) {
        return false;
      }
    }
    return true;
  }

  int FireableCountAfterLoad(int label) const {
    int count = 0;
    for (int pid : unscheduled_) {
      count += FireableWith(pid, label) ? 1 : 0;
    }
    return count;
  }

  int BestFireableReleaseAfterLoad(int label) const {
    int best = 0;
    for (int pid : unscheduled_) {
      if (FireableWith(pid, label)) {
        best = std::max(best, ReleaseNowCount(pid));
      }
    }
    return best;
  }

  int ReleasePotentialIfAdd(int label) const {
    int score = 0;
    for (int pid : unscheduled_) {
      if (!FireableWith(pid, label)) {
        continue;
      }
      for (int l : region_.paths[pid].input_labels) {
        if (remaining_uses_[l] == 1) {
          score += live_[l] ? 2 : 1;
        }
      }
    }
    return score;
  }

  int FirePotentialIfAdd(int label) const {
    int count = 0;
    for (int pid : label_paths_[label]) {
      if (unscheduled_.count(pid) && FireableWith(pid, label)) {
        ++count;
      }
    }
    return count;
  }

  int PrimaryAffinity(int label) const {
    int score = 0;
    for (int pid : label_paths_[label]) {
      if (!unscheduled_.count(pid)) {
        continue;
      }
      for (int l : region_.paths[pid].input_labels) {
        score += live_[l] ? 1 : 0;
      }
    }
    return score;
  }

  int SecondaryAffinity(int label) const {
    if (!cfg_->enable_secondary_affinity) {
      return 0;
    }
    std::set<int> neighbors;
    for (int pid : unscheduled_) {
      bool touches_live = false;
      for (int l : region_.paths[pid].input_labels) {
        if (live_[l]) {
          touches_live = true;
          break;
        }
      }
      if (touches_live) {
        neighbors.insert(region_.paths[pid].input_labels.begin(),
                         region_.paths[pid].input_labels.end());
      }
    }
    int score = 0;
    for (int pid : label_paths_[label]) {
      if (!unscheduled_.count(pid)) {
        continue;
      }
      for (int l : region_.paths[pid].input_labels) {
        if (neighbors.count(l) && !live_[l] && l != label) {
          ++score;
        }
      }
    }
    return score;
  }

  int NonLiveAffinityPenalty(int label) const {
    std::set<int> neighbors;
    for (int pid : label_paths_[label]) {
      if (unscheduled_.count(pid)) {
        neighbors.insert(region_.paths[pid].input_labels.begin(),
                         region_.paths[pid].input_labels.end());
      }
    }
    int penalty = 0;
    for (int l : neighbors) {
      if (!live_[l] && l != label) {
        ++penalty;
      }
    }
    return penalty;
  }

  std::array<int, 6> LabelScore(int label) const {
    return {ReleasePotentialIfAdd(label),   FirePotentialIfAdd(label),
            PrimaryAffinity(label),         SecondaryAffinity(label),
            -NonLiveAffinityPenalty(label), remaining_uses_[label]};
  }

  std::set<int> CandidateLabels() const {
    std::set<int> candidates;
    for (int pid : unscheduled_) {
      for (int label : region_.paths[pid].input_labels) {
        if (!live_[label]) {
          candidates.insert(label);
        }
      }
    }
    return candidates;
  }

  /*! \brief Global selection; only used while a free register exists. */
  int SelectGlobal() const {
    int best = -1;
    std::array<int, 9> best_key{};
    for (int label : CandidateLabels()) {
      std::array<int, 6> score = LabelScore(label);
      std::array<int, 9> key{FireableCountAfterLoad(label),
                             BestFireableReleaseAfterLoad(label),
                             score[0],
                             score[1],
                             score[2],
                             score[3],
                             score[4],
                             score[5],
                             -label};
      if (best < 0 || key > best_key) {
        best = label;
        best_key = key;
      }
    }
    return best;
  }

  /*!
   * \brief Path-directed fallback: pick the unscheduled path closest to
   * fireable, load one of its missing labels, and avoid evicting labels the
   * target path already holds live.
   */
  void SelectByTargetPath(int *out_label, int *out_victim) const {
    int best_pid = -1;
    std::array<int, 4> best_key{};
    for (int pid : unscheduled_) {
      int live_hits = 0;
      int missing = 0;
      for (int label : region_.paths[pid].input_labels) {
        live_[label] ? ++live_hits : ++missing;
      }
      std::array<int, 4> key{live_hits, -missing, ReleaseNowCount(pid), -pid};
      if (best_pid < 0 || key > best_key) {
        best_pid = pid;
        best_key = key;
      }
    }
    ICHECK_GE(best_pid, 0);

    const std::vector<int> &labels = region_.paths[best_pid].input_labels;
    int best_label = -1;
    std::array<int, 7> best_label_key{};
    for (int label : labels) {
      if (live_[label]) {
        continue;
      }
      std::array<int, 6> score = LabelScore(label);
      std::array<int, 7> key{score[0], score[1], score[2], score[3],
                             score[4], score[5], -label};
      if (best_label < 0 || key > best_label_key) {
        best_label = label;
        best_label_key = key;
      }
    }
    *out_label = best_label;

    *out_victim = -1;
    if (FreeRegs() == 0 && live_count_ > 0) {
      std::set<int> avoid(labels.begin(), labels.end());
      *out_victim = ChooseSpillVictimAvoid(avoid);
    }
  }

  int ChooseSpillVictimAvoid(const std::set<int> &avoid) const {
    int best = -1;
    std::pair<int, int> best_key{};
    for (int label = 0; label < static_cast<int>(live_.size()); ++label) {
      if (!live_[label] || avoid.count(label)) {
        continue;
      }
      std::pair<int, int> key{remaining_uses_[label], label};
      if (best < 0 || key < best_key) {
        best = label;
        best_key = key;
      }
    }
    if (best >= 0) {
      return best;
    }
    // All live labels belong to the target path; fall back to any live label.
    for (int label = 0; label < static_cast<int>(live_.size()); ++label) {
      if (live_[label]) {
        std::pair<int, int> key{remaining_uses_[label], label};
        if (best < 0 || key < best_key) {
          best = label;
          best_key = key;
        }
      }
    }
    return best;
  }

  // ---- virtual register file mutation ----

  void ReleasePairsTouchingLabel(int label) {
    std::vector<PairKey> touching;
    for (const auto &kv : live_pairs_) {
      if (kv.first.first == label || kv.first.second == label) {
        touching.push_back(kv.first);
      }
    }
    for (const PairKey &key : touching) {
      live_pairs_.erase(key);
    }
  }

  void ReleaseLabel(int label) {
    if (!live_[label]) {
      return;
    }
    ReleasePairsTouchingLabel(label);
    live_[label] = false;
    reg_of_[label] = Var();
    --live_count_;
  }

  void DropOnePairForPressure() {
    if (live_pairs_.empty()) {
      return;
    }
    PairKey best{};
    std::array<int, 2> best_key{};
    bool first = true;
    for (const auto &kv : live_pairs_) {
      std::array<int, 2> key{pair_remaining_.at(kv.first),
                             pair_initial_.at(kv.first)};
      if (first || key < best_key) {
        best = kv.first;
        best_key = key;
        first = false;
      }
    }
    live_pairs_.erase(best);
  }

  void LoadLabel(int label, int victim) {
    if (live_[label]) {
      return;
    }
    if (FreeRegs() == 0) {
      // Pair temporaries are cheaper to drop than spilling operand labels.
      DropOnePairForPressure();
    }
    if (FreeRegs() == 0) {
      if (victim < 0) {
        victim = ChooseSpillVictimAvoid({});
      }
      ICHECK_GE(victim, 0);
      ReleaseLabel(victim);
    }
    const LabelInfo &info = region_.inputs.at(label);
    std::string name = "plr_" + info.name_hint + "_" + std::to_string(label);
    if (reload_count_[label] > 0) {
      name += "_r" + std::to_string(reload_count_[label]);
    }
    ++reload_count_[label];
    Var reg(name, info.repr_load.dtype());
    stmts_.push_back(Bind(reg, info.repr_load));
    reg_of_[label] = reg;
    live_[label] = true;
    ++live_count_;
  }

  // ---- path firing ----

  void Fire(int pid) {
    const ParsedPath &path = region_.paths[pid];

    Var pair_var;
    bool use_pair = false;
    if (cfg_->enable_pair_cse && path.pair_key.first >= 0) {
      auto it = live_pairs_.find(path.pair_key);
      if (it != live_pairs_.end()) {
        pair_var = it->second;
        use_pair = true;
      } else if (pair_remaining_.at(path.pair_key) >= 2 && FreeRegs() >= 1) {
        pair_var = MaterializePair(pid);
        use_pair = true;
      }
    }

    stmts_.push_back(EmitCompute(path, use_pair, pair_var));
    unscheduled_.erase(pid);

    if (path.pair_key.first >= 0) {
      int &remaining = pair_remaining_.at(path.pair_key);
      remaining -= 1;
      if (remaining <= 0) {
        live_pairs_.erase(path.pair_key);
      }
    }
    for (int label : path.input_labels) {
      remaining_uses_[label] -= 1;
    }
    for (int label : path.input_labels) {
      if (remaining_uses_[label] == 0) {
        ReleaseLabel(label);
      }
    }
  }

  /*! \brief The two pair operands in factor order for a given path. */
  std::pair<int, int> PairOperandOrder(const ParsedPath &path) const {
    for (const Factor &factor : path.factors) {
      if (factor.label == path.pair_key.first) {
        return {path.pair_key.first, path.pair_key.second};
      }
      if (factor.label == path.pair_key.second) {
        return {path.pair_key.second, path.pair_key.first};
      }
    }
    return path.pair_key;
  }

  Var MaterializePair(int pid) {
    const ParsedPath &path = region_.paths[pid];
    std::pair<int, int> ordered = PairOperandOrder(path);
    Var pair_var("plr_pair_" + std::to_string(pair_ordinal_++),
                 region_.inputs.at(ordered.first).repr_load.dtype());
    stmts_.push_back(
        Bind(pair_var, Mul(reg_of_[ordered.first], reg_of_[ordered.second])));
    live_pairs_[path.pair_key] = pair_var;
    return pair_var;
  }

  PrimExpr RebuildProductWithPair(const ParsedPath &path,
                                  const Var &pair_var) const {
    // Replace the first occurrence of each pair operand with the pair
    // temporary (one occurrence contributes the pair itself, the other is
    // dropped). Remaining factors keep their original order.
    int drop_first = -1;
    int drop_second = -1;
    for (size_t i = 0; i < path.factors.size(); ++i) {
      int label = path.factors[i].label;
      if (label == path.pair_key.first && drop_first < 0) {
        drop_first = static_cast<int>(i);
      } else if (label == path.pair_key.second && drop_second < 0) {
        drop_second = static_cast<int>(i);
      }
    }
    ICHECK_GE(drop_first, 0);
    ICHECK_GE(drop_second, 0);
    int pair_pos = std::min(drop_first, drop_second);
    int skip_pos = std::max(drop_first, drop_second);

    PrimExpr result;
    for (size_t i = 0; i < path.factors.size(); ++i) {
      PrimExpr piece;
      if (static_cast<int>(i) == pair_pos) {
        piece = pair_var;
      } else if (static_cast<int>(i) == skip_pos) {
        continue;
      } else if (path.factors[i].label >= 0) {
        piece = reg_of_[path.factors[i].label];
      } else {
        piece = path.factors[i].expr;
      }
      result = result.defined() ? Mul(result, piece) : piece;
    }
    ICHECK(result.defined());
    return result;
  }

  Stmt EmitCompute(const ParsedPath &path, bool use_pair,
                   const Var &pair_var) const {
    PrimExpr new_product;
    if (use_pair) {
      new_product = RebuildProductWithPair(path, pair_var);
    } else {
      // Substitution keeps the original multiplication tree shape, so the
      // scheduled statement is bit-identical to the source path.
      LabelLoadReplacer replacer(analyzer_, region_.inputs, reg_of_);
      new_product = replacer(path.product);
    }

    if (path.is_atomic) {
      const auto *eval = path.original.as<EvaluateNode>();
      Call call = Downcast<Call>(eval->value);
      Array<PrimExpr> args = call->args;
      args.Set(1, new_product);
      call.CopyOnWrite()->args = std::move(args);
      return Evaluate(call);
    }

    BufferStore store = Downcast<BufferStore>(path.original);
    const auto *add = store->value.as<AddNode>();
    PrimExpr new_value = add->b.same_as(path.product)
                             ? Add(add->a, new_product)
                             : Add(new_product, add->b);
    store.CopyOnWrite()->value = std::move(new_value);
    return store;
  }

  const Region &region_;
  const PathLocalityReorderConfigNode *cfg_;
  arith::Analyzer *analyzer_;

  int reg_budget_{0};
  int path_fallback_after_{0};

  std::vector<std::vector<int>> label_paths_;
  std::vector<int> remaining_uses_;
  std::vector<bool> live_;
  std::vector<Var> reg_of_;
  std::vector<int> reload_count_;
  int live_count_{0};
  std::set<int> unscheduled_;

  std::map<PairKey, int> pair_remaining_;
  std::map<PairKey, int> pair_initial_;
  std::map<PairKey, Var> live_pairs_;
  int pair_ordinal_{0};

  Array<Stmt> stmts_;
};

/*! \brief The data var written by an output label. */
const VarNode *OutputDataVar(const LabelInfo &info) {
  if (const auto *buffer = info.base.as<BufferNode>()) {
    return buffer->data.get();
  }
  return info.base.as<VarNode>();
}

/*!
 * \brief An analyzer-simplified index split into a symbolic part and a
 * constant offset.
 *
 * Output offsets of scatter kernels are typically `sym + const` where `sym`
 * contains runtime indirection loads (e.g. dst_idx[e] * stride + lane). The
 * analyzer cannot cancel opaque loads across two expressions, but within a
 * validated region no statement writes the buffers those loads read, so
 * structurally equal symbolic parts are value-equal and disjointness reduces
 * to comparing the constant offsets.
 */
struct IndexParts {
  PrimExpr sym;
  int64_t offset{0};
};

IndexParts DecomposeIndex(const PrimExpr &index) {
  IndexParts parts;
  parts.sym = index;
  if (const auto *add = parts.sym.as<AddNode>()) {
    if (const auto *imm = add->b.as<IntImmNode>()) {
      parts.offset = imm->value;
      parts.sym = add->a;
    }
  }
  if (const auto *imm = parts.sym.as<IntImmNode>()) {
    parts.offset += imm->value;
    parts.sym = PrimExpr();
  }
  return parts;
}

/*! \brief Per-dimension equal/distinct decision for two output indices. */
void CompareIndexDim(const PrimExpr &a, const PrimExpr &b,
                     arith::Analyzer *analyzer, bool *dim_equal,
                     bool *dim_distinct) {
  IndexParts pa = DecomposeIndex(a);
  IndexParts pb = DecomposeIndex(b);
  bool sym_equal = (!pa.sym.defined() && !pb.sym.defined()) ||
                   (pa.sym.defined() && pb.sym.defined() &&
                    StructuralEqual()(pa.sym, pb.sym));
  if (sym_equal) {
    *dim_equal = pa.offset == pb.offset;
    *dim_distinct = pa.offset != pb.offset;
    return;
  }
  // Heterogeneous symbolic parts: fall back to the analyzer. Simplifying the
  // difference lets the canonical simplifier cancel common opaque subterms
  // that CanProve alone would not.
  PrimExpr diff = analyzer->Simplify(a - b);
  if (const auto *imm = diff.as<IntImmNode>()) {
    *dim_equal = imm->value == 0;
    *dim_distinct = imm->value != 0;
    return;
  }
  *dim_equal = analyzer->CanProveEqual(a, b);
  *dim_distinct = analyzer->CanProve(a != b);
}

/*!
 * \brief Validate a candidate region. Rejection keeps the original order, so
 * every check may be conservative.
 */
bool ValidateRegion(const Region &region,
                    const PathLocalityReorderConfigNode *cfg,
                    arith::Analyzer *analyzer) {
  int num_paths = static_cast<int>(region.paths.size());
  if (num_paths < 2 || num_paths > cfg->max_paths) {
    return false;
  }

  // Output policy: duplicated output elements imply a reordering of adds to
  // the same location, which changes floating-point association.
  std::vector<int> output_uses(region.outputs.size(), 0);
  for (const ParsedPath &path : region.paths) {
    output_uses[path.output_label] += 1;
  }
  if (!cfg->allow_atomic_reorder) {
    for (int uses : output_uses) {
      if (uses > 1) {
        return false;
      }
    }
  }

  // Distinct output labels must be provably disjoint (or provably equal with
  // reordering explicitly allowed). Unknown aliasing keeps original order.
  for (int a = 0; a < region.outputs.size(); ++a) {
    for (int b = a + 1; b < region.outputs.size(); ++b) {
      const LabelInfo &la = region.outputs.at(a);
      const LabelInfo &lb = region.outputs.at(b);
      if (!la.base.same_as(lb.base)) {
        const VarNode *va = OutputDataVar(la);
        const VarNode *vb = OutputDataVar(lb);
        // Distinct buffers with distinct data vars are assumed non-aliasing
        // (tir.noalias); the same data var behind different bases cannot be
        // reasoned about here.
        if (va == nullptr || vb == nullptr || va == vb) {
          return false;
        }
        continue;
      }
      if (la.indices.size() != lb.indices.size()) {
        return false;
      }
      bool provably_equal = true;
      bool provably_distinct = false;
      for (size_t d = 0; d < la.indices.size(); ++d) {
        bool dim_equal = false;
        bool dim_distinct = false;
        CompareIndexDim(la.indices[d], lb.indices[d], analyzer, &dim_equal,
                        &dim_distinct);
        if (!dim_equal) {
          provably_equal = false;
        }
        if (dim_distinct) {
          provably_distinct = true;
        }
      }
      if (provably_distinct) {
        continue;
      }
      if (provably_equal && cfg->allow_atomic_reorder) {
        continue;
      }
      return false;
    }
  }

  // No input label (or peeled bind) may read a buffer any path writes.
  std::unordered_set<const VarNode *> written;
  for (int label = 0; label < region.outputs.size(); ++label) {
    const VarNode *data = OutputDataVar(region.outputs.at(label));
    if (data == nullptr) {
      return false;
    }
    written.insert(data);
  }
  std::unordered_set<const VarNode *> reads;
  for (const ParsedPath &path : region.paths) {
    CollectReadDataVars(path.product, &reads);
    for (int label : path.input_labels) {
      for (const PrimExpr &index : region.inputs.at(label).indices) {
        CollectReadDataVars(index, &reads);
      }
    }
    for (const PrimExpr &index : region.outputs.at(path.output_label).indices) {
      CollectReadDataVars(index, &reads);
    }
  }
  for (const PeeledBind &bind : region.binds) {
    CollectReadDataVars(bind.resolved_value, &reads);
  }
  for (const VarNode *read : reads) {
    if (written.count(read)) {
      return false;
    }
  }

  // Register reuse requires shared reads; otherwise reordering has no
  // expected benefit.
  std::map<int, int> uses;
  for (const ParsedPath &path : region.paths) {
    for (int label : path.input_labels) {
      uses[label] += 1;
    }
  }
  int shared_reads = 0;
  for (const auto &kv : uses) {
    shared_reads += std::max(0, kv.second - 1);
  }
  return shared_reads >= std::max(1, cfg->min_shared_reads);
}

/*! \brief Assign each path its most frequent candidate pair key. */
void AssignPairKeys(Region *region) {
  std::map<std::pair<int, int>, int> pair_count;
  for (const ParsedPath &path : region->paths) {
    const std::vector<int> &labels = path.input_labels;
    for (size_t i = 0; i < labels.size(); ++i) {
      for (size_t j = i + 1; j < labels.size(); ++j) {
        pair_count[{labels[i], labels[j]}] += 1;
      }
    }
  }
  for (ParsedPath &path : region->paths) {
    const std::vector<int> &labels = path.input_labels;
    std::pair<int, int> best{-1, -1};
    int best_count = 1;
    for (size_t i = 0; i < labels.size(); ++i) {
      for (size_t j = i + 1; j < labels.size(); ++j) {
        std::pair<int, int> key{labels[i], labels[j]};
        int count = pair_count[key];
        if (count > best_count ||
            (count == best_count && best.first >= 0 && key < best)) {
          best = key;
          best_count = count;
        }
      }
    }
    path.pair_key = best;
  }
}

class PathLocalityRewriter : public StmtExprMutator {
public:
  PathLocalityRewriter(const PathLocalityReorderConfigNode *cfg,
                       arith::Analyzer *analyzer)
      : cfg_(cfg), analyzer_(analyzer) {}

  static PrimFunc Rewrite(PrimFunc func,
                          const PathLocalityReorderConfigNode *cfg,
                          arith::Analyzer *analyzer) {
    PathLocalityRewriter rewriter(cfg, analyzer);
    if (auto tables = func->GetAttr<Map<Var, Array<PrimExpr>>>(
            kPathLocalityDescriptorsAttr)) {
      for (const auto &kv : tables.value()) {
        rewriter.descriptor_tables_[kv.first.get()] = kv.second;
      }
    }
    auto *node = func.CopyOnWrite();
    node->body = rewriter(std::move(node->body));
    return func;
  }

private:
  /*! \brief Fold annotated descriptor loads; identity without tables. */
  Stmt FoldDescriptors(Stmt stmt) {
    if (descriptor_tables_.empty()) {
      return stmt;
    }
    return DescriptorFolder(&descriptor_tables_, analyzer_)(std::move(stmt));
  }
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    const auto *seq = visited.as<SeqStmtNode>();
    if (seq == nullptr) {
      return visited;
    }

    Array<Stmt> result;
    bool changed = false;
    size_t i = 0;
    size_t n = seq->seq.size();
    while (i < n) {
      size_t run_end = i;
      Array<Stmt> scheduled;
      if (TryScheduleRun(seq->seq, i, &run_end, &scheduled)) {
        for (const Stmt &stmt : scheduled) {
          result.push_back(stmt);
        }
        changed = true;
        i = run_end;
        continue;
      }
      result.push_back(seq->seq[i]);
      ++i;
    }

    if (!changed) {
      return visited;
    }
    return SeqStmt::Flatten(result);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // The fence annotation excludes the loop and everything inside it, so it
    // must be honored before visiting children.
    if (PassFencedByAnnotation(op->annotations)) {
      return GetRef<Stmt>(op);
    }
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    const auto *loop = visited.as<ForNode>();
    if (loop == nullptr) {
      return visited;
    }
    Stmt rewritten = TryRewriteLoop(loop);
    return rewritten.defined() ? rewritten : visited;
  }

  /*!
   * \brief Try to extract, validate and schedule a maximal run of peelable
   * Binds and path statements starting at `start`.
   */
  bool TryScheduleRun(const Array<Stmt> &seq, size_t start, size_t *run_end,
                      Array<Stmt> *result) {
    Region region;
    PathParser parser(analyzer_, &region.inputs, &region.outputs);

    Map<Var, PrimExpr> pending;
    size_t end = start;
    while (end < seq.size() &&
           static_cast<int>(region.paths.size()) <= cfg_->max_paths) {
      const Stmt &stmt = seq[end];
      if (const auto *bind = stmt.as<BindNode>()) {
        if (bind->value.dtype().lanes() != 1 || !IsPureExpr(bind->value)) {
          break;
        }
        PrimExpr resolved = Substitute(bind->value, pending);
        pending.Set(bind->var, resolved);
        region.binds.push_back(PeeledBind{bind->var, resolved});
        ++end;
        continue;
      }
      Stmt resolved =
          FoldDescriptors(pending.empty() ? stmt : Substitute(stmt, pending));
      ParsedPath path;
      if (!parser.Parse(resolved, &path)) {
        break;
      }
      region.paths.push_back(std::move(path));
      ++end;
    }

    if (region.paths.size() < 2 || !ValidateRegion(region, cfg_, analyzer_)) {
      return false;
    }

    AssignPairKeys(&region);
    Array<Stmt> scheduled;
    LarsScheduler scheduler(region, cfg_, analyzer_);
    if (!scheduler.Schedule(&scheduled)) {
      return false;
    }

    // Peeled binds whose vars are still used after the run must be kept;
    // their values are pure and fully resolved, so hoisting them to the top
    // of the region is safe.
    for (const PeeledBind &bind : region.binds) {
      bool used_later = false;
      for (size_t j = end; j < seq.size() && !used_later; ++j) {
        used_later = UsesVar(
            seq[j], [&](const VarNode *v) { return v == bind.var.get(); });
      }
      if (used_later) {
        result->push_back(Bind(bind.var, bind.resolved_value));
      }
    }
    for (const Stmt &stmt : scheduled) {
      result->push_back(stmt);
    }
    *run_end = end;
    return true;
  }

  /*!
   * \brief Locally unroll a small constant-extent serial path loop whose body
   * reduces to path statements, then schedule the unrolled paths. Peeled
   * binds are substituted away during extraction, so no definition is
   * duplicated and ConvertSSA is not required.
   */
  Stmt TryRewriteLoop(const ForNode *loop) {
    if (loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled) {
      return Stmt();
    }
    if (loop->thread_binding.defined()) {
      return Stmt();
    }
    PrimExpr extent_expr = analyzer_->Simplify(loop->extent);
    const auto *extent = extent_expr.as<IntImmNode>();
    if (extent == nullptr || extent->value < 2 ||
        extent->value > cfg_->max_paths) {
      return Stmt();
    }

    // Peel the loop body into pure binds plus path sink templates.
    Array<Stmt> body_stmts;
    if (const auto *body_seq = loop->body.as<SeqStmtNode>()) {
      body_stmts = body_seq->seq;
    } else {
      body_stmts.push_back(loop->body);
    }
    Map<Var, PrimExpr> pending;
    std::vector<Stmt> sink_templates;
    for (const Stmt &stmt : body_stmts) {
      if (const auto *bind = stmt.as<BindNode>()) {
        if (bind->value.dtype().lanes() != 1 || !IsPureExpr(bind->value)) {
          return Stmt();
        }
        pending.Set(bind->var, Substitute(bind->value, pending));
        continue;
      }
      Stmt resolved = pending.empty() ? stmt : Substitute(stmt, pending);
      LabelInterner scratch_inputs;
      LabelInterner scratch_outputs;
      PathParser scratch(analyzer_, &scratch_inputs, &scratch_outputs);
      ParsedPath probe;
      if (!scratch.Parse(resolved, &probe)) {
        return Stmt();
      }
      sink_templates.push_back(resolved);
    }
    if (sink_templates.empty() ||
        extent->value * static_cast<int64_t>(sink_templates.size()) >
            cfg_->max_paths) {
      return Stmt();
    }

    Region region;
    PathParser parser(analyzer_, &region.inputs, &region.outputs);
    for (int64_t iter = 0; iter < extent->value; ++iter) {
      Map<Var, PrimExpr> vmap;
      vmap.Set(loop->loop_var,
               loop->min + make_const(loop->loop_var.dtype(), iter));
      for (const Stmt &tmpl : sink_templates) {
        ParsedPath path;
        if (!parser.Parse(FoldDescriptors(Substitute(tmpl, vmap)), &path)) {
          return Stmt();
        }
        region.paths.push_back(std::move(path));
      }
    }

    if (!ValidateRegion(region, cfg_, analyzer_)) {
      return Stmt();
    }
    AssignPairKeys(&region);
    Array<Stmt> scheduled;
    LarsScheduler scheduler(region, cfg_, analyzer_);
    if (!scheduler.Schedule(&scheduled)) {
      return Stmt();
    }
    return SeqStmt::Flatten(scheduled);
  }

  const PathLocalityReorderConfigNode *cfg_;
  arith::Analyzer *analyzer_;
  std::unordered_map<const VarNode *, Array<PrimExpr>> descriptor_tables_;
};

} // namespace

namespace transform {

using namespace tirx::transform;

tvm::transform::Pass PathLocalityReorder() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto cfg =
        ctx->GetConfig<PathLocalityReorderConfig>("tl.PathLocalityReorder");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<PathLocalityReorderConfig>();
    }
    if (!cfg.value()->enable) {
      return f;
    }
    arith::Analyzer analyzer;
    return PathLocalityRewriter::Rewrite(std::move(f), cfg.value().get(),
                                         &analyzer);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PathLocalityReorder", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.PathLocalityReorder",
                        PathLocalityReorder);
}

} // namespace transform

} // namespace tl
} // namespace tvm
