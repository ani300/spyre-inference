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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Independent concatenated-projection oracle for merged vLLM XPress at ce9467b757c0."""

import pytest
import torch
from torch.nn import functional as F

from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


def merged_refine(head, base, hidden, anchors, passes):
    summary = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
    cache = torch.cat((head.down_h(hidden), head.down_g(summary)), dim=-1)
    draft = base[:, 1:].argmax(-1)
    candidates = None
    if head.topc:
        _, candidates = base[:, 1:].topk(min(head.topc, base.shape[-1]), dim=-1)
    for _ in range(passes):
        # XPressSpeculator uses the verified anchor as its own predecessor.
        previous = torch.cat((anchors[:, None], anchors[:, None], draft[:, :-1]), dim=1)
        x = head.in_proj(torch.cat((cache, head.w1(previous)), dim=-1))
        x = torch.bmm(head.mix_L.to(x.dtype), x.permute(2, 1, 0)).permute(2, 1, 0)
        latent = x + head.mlp_down(F.silu(head.mlp_gate(x)) * head.mlp_up(x))
        scores = base[:, 1:] + head.w2(latent)[:, 1:]
        if candidates is None:
            draft = scores.argmax(-1)
        else:
            winners = scores.gather(-1, candidates).argmax(-1, keepdim=True)
            draft = candidates.gather(-1, winners).squeeze(-1)
    return draft


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("topc", [0, 128, 512])
@pytest.mark.parametrize("passes", [0, 1, 6])
def test_refinement_matches_merged_vllm(batch, topc, passes):
    torch.manual_seed(20261027)
    head = XPressRefinerHead(1031, 128, 16, rank=64, mlp_hidden=128, topc=topc).double()
    with torch.no_grad():
        head.mix_L.copy_(torch.randn_like(head.mix_L).tril() / 8 + torch.eye(16))
        hidden = torch.randn(batch, 16, 128, dtype=torch.float64)
        base = torch.randn(batch, 16, 1031, dtype=torch.float64)
        anchors = torch.arange(batch) + 1025
        expected = merged_refine(head, base, hidden, anchors, passes)
        actual = head(base, hidden, anchors, anchors, passes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
