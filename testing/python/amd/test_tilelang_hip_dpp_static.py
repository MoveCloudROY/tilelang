from pathlib import Path


def test_hip_common_exposes_dpp_reduce_fast_path():
    """HIP reduce helpers should contain an AMD DPP fast path."""
    repo_root = Path(__file__).resolve().parents[3]
    common_h = repo_root / "src" / "tl_templates" / "hip" / "common.h"
    reduce_h = repo_root / "src" / "tl_templates" / "hip" / "reduce.h"

    common_source = common_h.read_text()
    reduce_source = reduce_h.read_text()

    assert "dpp" in common_source
    assert "__builtin_amdgcn_update_dpp" in common_source
    assert "dpp_shfl_xor_32" in common_source
    assert "dpp_shfl_xor_32" in reduce_source
    assert "logicalWidth = threads <= 32 ? 32 : warpSize" in reduce_source
    assert "return tl::shfl_xor(val, delta);" in common_source
    assert "__shfl_xor" in common_source
