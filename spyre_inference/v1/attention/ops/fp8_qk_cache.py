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

"""Experimental FP8 QK cache paths, independent of production dispatch.

``staged`` retains the production QFP8CH cache and explicitly expands/re-packs K.
``packed`` stores K as QFP8WT [page, head, D, token] and assumes the compiler can
scatter into and gather from that arrangement. Unsupported operations propagate
their errors; the packed path never switches to the staged path.

Readers handle one page per sequence. They are building blocks for a future
paged online-softmax loop, not a replacement vLLM attention backend.
"""

import math

import torch

from spyre_inference.v1.attention.ops.fp8_qk import fp8_attention
from spyre_inference.v1.attention.ops.layout import head_major_kv_layout
from spyre_inference.v1.attention.ops.reshape_and_cache_head_major import quantize_kv


def allocate_fp8_qk_cache(num_pages, num_kv_heads, block_size, head_size, device, *, path):
    """Allocate separate K/V layouts; use byte fills rather than unsupported FP8 fills."""
    if path not in ("staged", "packed"):
        raise ValueError(f"Unknown FP8 QK path: {path}")
    if min(num_pages, num_kv_heads, block_size, head_size) <= 0:
        raise ValueError("Cache dimensions must be positive")
    if block_size % 64 or head_size % 64:
        raise ValueError("This experiment requires block_size and head_size multiples of 64")
    device = torch.device(device)
    v_shape = (num_pages, num_kv_heads, block_size, head_size)
    k_shape = (num_pages, num_kv_heads, head_size, block_size) if path == "packed" else v_shape
    k_kwargs, v_kwargs = {}, {}
    if device.type == "spyre":
        from torch_spyre._C import ElementArrangement, SpyreTensorLayout

        v_kwargs["device_layout"] = head_major_kv_layout(
            num_pages * num_kv_heads, block_size, head_size, torch.float8_e4m3fn
        )
        k_kwargs.update(v_kwargs)
        if path == "packed":
            strides = [math.prod(k_shape[i + 1 :]) for i in range(len(k_shape))]
            # Match qfp8wt's [..., D, N] producer: D is directly outside the stick.
            k_kwargs["device_layout"] = SpyreTensorLayout(
                list(k_shape), strides, torch.float8_e4m3fn, [2, 0, 1, 3], ElementArrangement.QFP8WT
            )
    k = torch.empty(k_shape, dtype=torch.float8_e4m3fn, device=device, **k_kwargs)
    v = torch.empty(v_shape, dtype=torch.float8_e4m3fn, device=device, **v_kwargs)
    k.view(torch.uint8).zero_()
    v.view(torch.uint8).zero_()
    return k, v


def pack_fp8_key(key_t, k_scale):
    """Quantize an already-transposed [..., D, token] key, preserving a fixed scale."""
    scaled = (key_t / k_scale).clamp(-448.0, 448.0)
    if key_t.device.type == "spyre":
        return torch.ops.spyre.qfp8wt(scaled)
    return scaled.to(torch.float8_e4m3fn)


def store_packed_kv_page(key, value, k_pages, v_pages, page, token_indices, k_scale, v_scale):
    """Native specification: write [T, H, D] into one physical page at distinct token slots.

    K's token-axis index_copy_ requires partial-stick QFP8WT scatter. There is no
    read/expand/repack workaround here. The same assignment runs on CPU as a
    logical reference, where FP8 tensors have ordinary storage.
    """
    key_t = pack_fp8_key(key.permute(1, 2, 0), k_scale)
    value = quantize_kv(value, v_scale, v_pages.dtype)
    if key.device.type == "cpu":
        # CPU index_copy has no FP8 kernel. Byte copies implement the same assignment.
        key_t, value = key_t.view(torch.uint8), value.view(torch.uint8)
        k_pages, v_pages = k_pages.view(torch.uint8), v_pages.view(torch.uint8)
    for h in range(key.shape[1]):
        k_pages[page, h].index_copy_(1, token_indices, key_t[h])
        v_pages[page, h].index_copy_(0, token_indices, value[:, h])


def _expand_pages(pages, page_ids):
    return pages.index_select(0, page_ids).to(torch.float16)


def _repack_expanded_key(key):
    # Values are still k / k_scale. Compensation happens after QK, not here.
    return torch.ops.spyre.qfp8wt(key.transpose(-1, -2))


def _scale_value(value, v_scale):
    return value * v_scale


def gqa_fp8_attention(query, key_t, value, mask, k_scale, attention_scale):
    """Query/output [B, M, Hq, D]; mask [B, Hkv, groups*M, N]."""
    batch, query_len, query_heads, head_size = query.shape
    kv_heads = key_t.shape[1]
    groups = query_heads // kv_heads
    q = query.transpose(1, 2).reshape(batch, kv_heads, groups * query_len, head_size)
    out = fp8_attention(q, key_t, value, mask, k_scale, attention_scale)
    return out.reshape(batch, query_heads, query_len, head_size).transpose(1, 2)


def packed_fp8_attention_kernel(query, k_pages, v_pages, page_ids, mask, k_scale, v_scale, scale):
    """Direct QFP8WT gather and matmul; only V expands to DL16."""
    key_t = k_pages.index_select(0, page_ids)
    value = v_pages.index_select(0, page_ids).to(query.dtype) * v_scale
    return gqa_fp8_attention(query, key_t, value, mask, k_scale, scale)


_expand_compiled = torch.compile(_expand_pages, fullgraph=True)
_repack_compiled = torch.compile(_repack_expanded_key, fullgraph=True)
_scale_value_compiled = torch.compile(_scale_value, fullgraph=True)
_gqa_compiled = torch.compile(gqa_fp8_attention, fullgraph=True)
packed_fp8_attention = torch.compile(packed_fp8_attention_kernel, fullgraph=True)
store_packed_kv_page_compiled = torch.compile(store_packed_kv_page, fullgraph=True)


def staged_fp8_attention(query, k_pages, v_pages, page_ids, mask, k_scale, v_scale, scale):
    """Working workaround. Call from Python to retain the necessary graph boundaries.

    Both caches are QFP8CH. Separate compiled calls force K expansion before
    transpose/QFP8WT repacking. V expansion/scaling and DL16 P×V are retained.
    No tensor is transferred to CPU. Fusing this wrapper into one graph loses
    the layout boundaries that make the workaround work.
    """
    key16 = _expand_compiled(k_pages, page_ids)
    value16 = _expand_compiled(v_pages, page_ids)
    key_t = _repack_compiled(key16)
    value = _scale_value_compiled(value16, v_scale)
    return _gqa_compiled(query, key_t, value, mask, k_scale, scale)
