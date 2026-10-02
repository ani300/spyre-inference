# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


@pytest.fixture
def head_and_weights():
    torch.manual_seed(7)
    head = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128).double()
    state = head.state_dict()
    weights = {
        "xpress_head." + source: torch.randn_like(state[dest]) * 0.05
        for source, dest in head._PUBLISHED_KEYS.items()
    }
    head.load_checkpoint_weights(weights)
    return head, weights


def _unfolded_bias(weights, hidden, previous):
    w = {key.removeprefix("xpress_head."): value for key, value in weights.items()}
    summary = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
    features = torch.cat(
        (
            F.linear(hidden, w["down_h.weight"]),
            F.linear(summary, w["down_g.weight"]),
            F.embedding(previous, w["w1.weight"]),
        ),
        dim=-1,
    )
    x = F.linear(features, w["in_proj.weight"])
    mixed = torch.bmm(w["mix.L"].tril(), x.permute(2, 1, 0)).permute(2, 1, 0)
    x = x + mixed
    x = x + F.linear(
        F.silu(F.linear(x, w["mlp.gate_proj.weight"])) * F.linear(x, w["mlp.up_proj.weight"]),
        w["mlp.down_proj.weight"],
    )
    return F.linear(x, w["w2.weight"])


def test_fold_matches_training_sublayer(head_and_weights):
    head, weights = head_and_weights
    hidden = torch.randn(3, 16, 128, dtype=torch.float64)
    previous = torch.randint(73, (3, 16))
    expected = _unfolded_bias(weights, hidden, previous)
    actual = head.refine_bias(previous, head.hidden_cache(hidden))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_later_tokens_cannot_change_earlier_bias(head_and_weights):
    head, _ = head_and_weights
    hidden = torch.randn(2, 16, 128, dtype=torch.float64)
    previous = torch.randint(73, (2, 16))
    changed = previous.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 73
    cache = head.hidden_cache(hidden)
    first = head.refine_bias(previous, cache)
    second = head.refine_bias(changed, cache)
    torch.testing.assert_close(first[:, :8], second[:, :8], atol=0, rtol=0)
    assert not torch.equal(first[:, 8:], second[:, 8:])


@pytest.mark.parametrize("passes", [0, 1, 2, 6])
def test_jacobi_matches_unfolded_reference(head_and_weights, passes):
    head, weights = head_and_weights
    hidden = torch.randn(2, 16, 128, dtype=torch.float64)
    base = torch.randn(2, 16, 73, dtype=torch.float64) * 0.02
    anchor = torch.tensor([3, 17])
    predecessor = torch.tensor([5, 9])
    block = base.argmax(-1)
    block[:, 0] = anchor
    for _ in range(passes):
        previous = torch.cat((predecessor[:, None], block[:, :-1]), dim=1)
        block = (base + _unfolded_bias(weights, hidden, previous)).argmax(-1)
        block[:, 0] = anchor
    actual = head(base, hidden, anchor, predecessor, passes)
    assert torch.equal(actual, block[:, 1:])
    assert torch.equal(actual, head(base, hidden, anchor, predecessor, passes))


def test_training_and_serving_checkpoint_roundtrip(head_and_weights):
    original, published = head_and_weights
    reverse = {dest: source for source, dest in original._PUBLISHED_KEYS.items()}
    training = {
        "refiner_head." + source: published["xpress_head." + reverse[dest]]
        for source, dest in original._TRAINING_KEYS.items()
    }
    loaded = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128).double()
    loaded.load_checkpoint_weights(training)
    restored = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128).double()
    restored.load_state_dict(loaded.state_dict())
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, atol=0, rtol=0)
        torch.testing.assert_close(restored.state_dict()[name], expected, atol=0, rtol=0)


def test_missing_mixer_is_rejected(head_and_weights):
    head, weights = head_and_weights
    del weights["xpress_head.mix.L"]
    with pytest.raises(ValueError, match="mix.L"):
        head.load_checkpoint_weights(weights)


def test_wrong_block_size_is_rejected(head_and_weights):
    head, _ = head_and_weights
    with pytest.raises(ValueError, match="block size"):
        head(torch.zeros(1, 8, 73), torch.zeros(1, 8, 128), torch.tensor([1]), torch.tensor([2]))
