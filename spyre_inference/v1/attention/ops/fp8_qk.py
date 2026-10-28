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

"""Experimental FP8 QK: dynamic query-row scale, fixed per-tensor key scale.

``query`` is DL16 ``[..., M, D]`` and ``key_t`` is FP8 ``[..., D, N]``.
``key_t`` must already have the weight packing required by Spyre's FP8 matmul.
A transpose of the existing QFP8CH KV cache does not establish that packing.
The caller supplies matching batch dimensions; grouped query heads can be
folded into the query-row dimension to avoid batched-matmul broadcasting.
Each query row/head gets a dynamic scale; ``k_scale`` is a fixed, positive
per-tensor dequantization scale. This prototype is not wired into paged attention.
"""

import torch


def fp8_qk(query, key_t, k_scale, attention_scale):
    query_scale = torch.ops.spyre.quantscalepertokenfp8(query, 448.0)  # ty: ignore[invalid-argument-type]
    query_fp8 = torch.ops.spyre.quantize_fp8_with_scale(query, query_scale)
    raw_scores = torch.ops.spyre.scaled_bmm(
        query_fp8,
        key_t,
        out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
    )
    return raw_scores * (query_scale * (k_scale * attention_scale))


def fp8_attention(query, key_t, value, mask, k_scale, attention_scale):
    """Dense attention control: only QK is FP8; softmax and P×V stay DL16.

    ``value`` is already dequantized. Each mask row must expose at least one key.
    """
    scores = fp8_qk(query, key_t, k_scale, attention_scale) + mask
    probs = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.matmul(probs, value.to(query.dtype))
