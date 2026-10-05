# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3Model
from spyre_inference.v1.spec_decode.dflash import SpyreDFlashProposer
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.mark.parametrize(
    "output_count,max_tokens,skip",
    [
        (0, 1, False),
        (1, 1, True),
        (23, 24, False),
        (24, 24, True),
        (29, 24, True),
        (24, None, False),
    ],
)
def test_output_limit_uses_tokens_already_appended_by_bookkeeping(output_count, max_tokens, skip):
    calls = []

    def propose(*args, skip_proposal=False):
        calls.append(skip_proposal)
        return [[]] if skip_proposal else [[7]]

    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        requests={
            "request": SimpleNamespace(
                output_token_ids=[3] * output_count,
                sampling_params=SimpleNamespace(max_tokens=max_tokens),
            )
        },
        input_batch=SimpleNamespace(req_ids=["request"]),
        drafter=SimpleNamespace(propose_spyre=propose),
        _get_positions=lambda count: torch.arange(count),
    )
    result = TorchSpyreModelRunner.propose_draft_token_ids(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=4),
        [[3]],
        SimpleNamespace(all_greedy=True),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert calls == [skip]
    assert result == ([[]] if skip else [[7]])


def test_skipped_final_proposal_commits_only_accepted_context():
    writes = []
    features = torch.randn(16, 128)
    drafter = SimpleNamespace(
        spyre_device=torch.device("cpu"),
        model=SimpleNamespace(
            combine_hidden_states=lambda states: states,
            precompute_and_store_context_kv=lambda *args: writes.append(args),
        ),
        context_tokens=0,
        context_seconds=0.0,
        proposal_calls=0,
        proposal_seconds=0.0,
    )
    result = SpyreDFlashProposer.propose_spyre(
        drafter,
        [[7, 9]],
        [features],
        torch.arange(10, 14),
        SimpleNamespace(num_draft_tokens=[3]),
        SimpleNamespace(slot_mapping=torch.arange(128, 132)),
        skip_proposal=True,
    )
    assert result == [[]]
    assert drafter.proposal_calls == 0
    assert drafter.context_tokens == 2
    assert len(writes) == 1
    context, positions, slots = writes[0]
    torch.testing.assert_close(context, features)
    assert positions.tolist() == [10, 11, 12, 13] + [0] * 12
    assert slots.tolist() == [128, 129] + [0] * 14


@pytest.mark.parametrize("rows", [1, 16, 64])
def test_context_projection_packs_live_transposed_weights_without_query_columns(monkeypatch, rows):
    torch.manual_seed(93)
    attentions, references = [], []
    for _ in range(2):
        query = torch.randn(256, 192)
        key, value = torch.randn(2, 128, 192)
        attention = torch.nn.Module()
        attention.qkv_proj = torch.nn.Module()
        attention.qkv_proj.weight = torch.nn.Parameter(
            torch.cat((query, key, value)).t().contiguous()
        )
        attention.q_size, attention.kv_size = 256, 128
        attention.num_kv_heads, attention.head_dim = 2, 64
        attention.k_norm = torch.nn.Identity()
        attention.rotary_emb = lambda positions, query, key: (query, key)
        attentions.append(attention)
        references.append((key, value))
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=attention) for attention in attentions],
        hidden_norm=torch.nn.Identity(),
    )
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    SpyreDFlashQwen3Model.prepare_for_spyre(model)
    context = torch.randn(rows, 192)
    for attention, reference in zip(attentions, references):
        assert attention._context_kv_weight_t.shape == (192, 256)
        assert attention._context_kv_weight_t.is_contiguous()
        assert "_context_kv_weight_t" not in attention.state_dict()
        actual = model._context_project(attention, context, torch.arange(rows))
        for result, weight in zip(actual, reference):
            expected = torch.nn.functional.linear(context, weight).view(rows, 2, 64)
            torch.testing.assert_close(result, expected)
