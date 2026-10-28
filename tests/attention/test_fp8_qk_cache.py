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

"""CPU checks of the native writer's logical assignment; hardware probe is an example.

These tests do not establish Spyre packing/scatter support. The example executes
those operations without xfail or exception suppression and retains their errors.
"""

import pytest
import torch

from spyre_inference.v1.attention.ops.fp8_qk_cache import (
    allocate_fp8_qk_cache,
    store_packed_kv_page,
)


@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize("offsets", [[0], [63], [64], [127], [1, 63, 64], list(range(128))])
def test_packed_store_preserves_other_tokens_and_pages(heads, offsets):
    torch.manual_seed(7)
    k, v = allocate_fp8_qk_cache(3, heads, 128, 128, "cpu", path="packed")
    k.copy_(torch.randn(k.shape, dtype=torch.float16).to(k.dtype))
    v.copy_(torch.randn(v.shape, dtype=torch.float16).to(v.dtype))
    expected_k, expected_v = k.float(), v.float()
    key = torch.randn(len(offsets), heads, 128, dtype=torch.float16)
    value = torch.randn_like(key)
    key[0, 0, :2] = torch.tensor([-100.0, 100.0])  # Exercise saturation as well as placement.
    ks, vs = 0.02, 0.015
    store_packed_kv_page(key, value, k, v, 1, torch.tensor(offsets), ks, vs)
    quantized_k = (key / ks).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    quantized_v = (value / vs).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    for row, offset in enumerate(offsets):
        expected_k[1, :, :, offset] = quantized_k[row]
        expected_v[1, :, offset, :] = quantized_v[row]
    torch.testing.assert_close(k.float(), expected_k, atol=0, rtol=0)
    torch.testing.assert_close(v.float(), expected_v, atol=0, rtol=0)
