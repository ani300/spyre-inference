# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused Gemma FP8-cache assessment, including the default-path compiler failure.

Run explicitly with pytest; SPYRE_ATTN_FOR_EACH_TILE=0 tests the unrolled workaround.
The standard attention suite supplies the cache-write and CPU-reference checks.
"""

import runpy
from pathlib import Path

import pytest

_support = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tests/attention/test_spyre_head_major_attn.py")
)
configure_device = _support["configure_device"]
configure_compilation = _support["configure_compilation"]
_run_head_major_attn_test = _support["_run_head_major_attn_test"]


@pytest.mark.parametrize(
    "num_kv_heads, head_size, sliding_window",
    [(8, 256, 1024), (2, 512, None)],
    ids=["gemma_sliding", "gemma_global"],
)
@pytest.mark.parametrize(
    "seq_lens",
    [[(23, 23)], [(23, 23), (24, 24), (22, 22), (19, 19)], [(1, 24)], [(1, 24)] * 4],
    ids=["prefill_one", "prefill_four", "decode_one", "decode_four"],
)
@pytest.mark.parametrize("configure_compilation", ["STOCK_TORCH_COMPILE"], indirect=True)
@pytest.mark.parametrize("configure_device", ["spyre"], indirect=True)
def test_gemma_fp8_cache(
    default_vllm_config,
    seq_lens,
    num_kv_heads,
    head_size,
    sliding_window,
    configure_compilation,
    configure_device,
):
    """Gemma's two head geometries exercise FP8 page conversion and attention."""
    _run_head_major_attn_test(
        seq_lens=seq_lens,
        block_size=128,
        sliding_window=sliding_window,
        configure_compilation=configure_compilation,
        configure_device=configure_device,
        num_query_heads=16,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        kv_cache_dtype="fp8",
    )
