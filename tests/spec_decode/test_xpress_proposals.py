# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch

from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


@pytest.mark.parametrize("topc", [0, 1, 7, 73])
@pytest.mark.parametrize("passes", [0, 2, 6])
def test_serving_selection_loop_matches_reference_and_transfers_only_candidate_scores(topc, passes):
    torch.manual_seed(81)
    head = XPressRefinerHead(73, 128, 16, rank=64, mlp_hidden=128, topc=topc).eval()
    hidden, base = torch.randn(16, 128), torch.randn(16, 73)
    anchor = torch.tensor([3])
    model = SimpleNamespace(
        xpress_head=head,
        xpress_num_passes=passes,
        xpress_topc=topc,
        compute_device_logits=lambda _: base,
        _hidden_cache=head.project_hidden_cache,
        _refine=head.refine_full,
        _gather_readout=head.gather_readout,
        _refine_candidates=head.refine_candidates,
    )
    for name in (
        "device_to_host_seconds",
        "host_argmax_seconds",
        "host_topk_seconds",
        "host_to_device_seconds",
        "candidate_gather_seconds",
        "candidate_transfer_bytes",
        "refiner_seconds",
        "logits_seconds",
        "selection_calls",
        "logits_transfer_bytes",
    ):
        setattr(model, name, 0)
    for name in ("_copy_logits_to_cpu", "_argmax_on_cpu"):
        setattr(model, name, MethodType(getattr(SpyreDFlashQwen3ForCausalLM, name), model))
    with torch.inference_mode():
        expected = head(base[None], hidden[None], anchor, anchor, passes).tolist()
        actual = SpyreDFlashQwen3ForCausalLM.propose_block(model, hidden, anchor)
    assert actual == expected
    assert model.logits_transfer_bytes == (16 * 73 + passes * 15 * (topc or 73)) * 4
