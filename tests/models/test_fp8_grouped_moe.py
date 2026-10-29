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

"""Logical checks for device dispatch, padding and routed output reduction."""

import pytest
import torch

from spyre_inference.experimental.fp8_grouped_moe import (
    _activate,
    _combine_device,
    _dispatch_device,
)


@pytest.mark.parametrize("tokens", [1, 4, 128, 512])
def test_dispatch_keeps_each_token_with_its_selected_experts(tokens):
    x = torch.arange(tokens * 128, dtype=torch.float32).reshape(tokens, 128)
    logits = torch.full((tokens, 128), -20.0)
    selected = (torch.arange(8)[None, :] + torch.arange(tokens)[:, None] * 8) % 128
    logits.scatter_(1, selected, torch.arange(8, dtype=torch.float32).expand(tokens, 8))
    grouped, ids, weights = _dispatch_device(x, logits, 8)
    torch.testing.assert_close(ids.reshape(tokens, 8).long(), selected.flip(-1))
    for token in range(tokens):
        for slot in range(8):
            torch.testing.assert_close(grouped[token * 8 + slot, 0], x[token])
    expected_weights = torch.softmax(torch.arange(7, -1, -1, dtype=torch.float32), 0)
    torch.testing.assert_close(weights, expected_weights.expand(tokens, 8))


def test_output_reduction_preserves_distinct_token_and_expert_contributions():
    grouped = torch.tensor([[[1.0, 3.0]], [[4.0, -2.0]], [[-5.0, 7.0]], [[8.0, 1.0]]])
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
    expected = torch.tensor([[3.25, -0.75], [0.2, 4.6]])
    torch.testing.assert_close(_combine_device(grouped, weights), expected)


def test_gemma_padding_contributes_zero_to_down_projection():
    torch.random.default_generator.manual_seed(42)
    gate, up = torch.randn(8, 1, 704), torch.randn(8, 1, 704)
    actual = _activate(gate, up)
    assert actual.shape == (8, 1, 768)
    assert torch.count_nonzero(actual[..., 704:]) == 0
    down = torch.randn(768, 64)
    torch.testing.assert_close(actual @ down, actual[..., :704] @ down[:704], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("chunks", [1, 4])
def test_chunked_padded_equations_match_unchunked_experts(chunks):
    from spyre_inference.experimental.fp8_grouped_moe import (
        _activate_chunks,
        _combine_chunks,
        prepare_experts,
    )

    torch.random.default_generator.manual_seed(71)
    e, h, m = 3, 192, 96
    host = {
        p: (torch.randn(e, n, k) / 16).to(torch.float8_e4m3fn)
        for p, n, k in (("gate", m, h), ("up", m, h), ("down", h, m))
    }
    scales = {p: torch.rand(e, w.shape[1], 1) + 0.1 for p, w in host.items()}
    multiplier = torch.tensor([0.75, 1.25, 0.9])
    pools, sp = prepare_experts(host, scales, multiplier, chunks)
    ids = torch.tensor([2, 0])
    entries = (ids[:, None] * chunks + torch.arange(chunks)).reshape(-1)
    x = torch.randn(1, h)
    hp = pools["gate"].shape[-1] * chunks
    xp = torch.nn.functional.pad(x, (0, hp - h))
    inputs = xp.reshape(1, chunks, -1).expand(2, chunks, -1).reshape(2 * chunks, 1, -1)
    gu = [
        torch.bmm(inputs, pools[p][entries].float().transpose(-1, -2)) * sp[p][entries]
        for p in ("gate", "up")
    ]
    z = _activate_chunks(gu[0], gu[1], chunks)
    parts = torch.bmm(z, pools["down"][entries].float().transpose(-1, -2)) * sp["down"][entries]
    routing = torch.tensor([[0.4, 0.6]])
    actual = _combine_chunks(parts, routing, h)
    expected = torch.zeros_like(x)
    for slot, expert in enumerate(ids):
        g, u, d = [
            (host[p][expert].float() * scales[p][expert]).t() for p in ("gate", "up", "down")
        ]
        value = (torch.nn.functional.gelu(x @ g, approximate="tanh") * (x @ u)) @ d
        expected += value * multiplier[expert] * routing[0, slot]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
