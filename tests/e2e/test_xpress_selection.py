# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resident full-vocabulary selection through XPress candidate gather and feedback."""

import warnings
from types import MethodType, SimpleNamespace

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present

pytestmark = pytest.mark.model_quality


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("topc", [128, 512])
def test_fp16_selection_drives_candidate_gather_and_refiner_without_logit_copies(batch, topc):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    import torch_spyre

    torch_spyre._autoload()
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.models.qwen3_dflash import (
        SpyreDFlashQwen3ForCausalLM,
        _assemble_feedback,
        _select_candidate_parts,
    )
    from spyre_inference.v1.spec_decode.selection import FP16VocabularySelector
    from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead

    torch.manual_seed(918)
    if not hasattr(torch.ops.spyre, "topk_payload"):
        pytest.skip("Requires the optional torch-spyre FP16 payload top-k implementation")
    vocab, block = 151936, 16
    head = (
        XPressRefinerHead(vocab, 128, block, rank=256, mlp_hidden=512, topc=topc)
        .eval()
        .half()
        .to("spyre")
    )
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        head.prepare_for_spyre()
        hidden = torch.randn(batch * block, 128).half().to("spyre")
        anchors = torch.arange(batch).int() * 30001 + 7
        raw = torch.randn(batch * block, 153600).half()
        # Every position has a different, unambiguous maximum, including anchors.
        maxima = (100003 + torch.arange(batch * block) * 997) % vocab
        raw.scatter_(1, maxima[:, None], 32.0)
        raw[:, vocab:] = 1000.0
        backing = raw.to("spyre")
        base = backing[:, :vocab]
        resident = base.to(device="cpu", dtype=torch.float32).view(batch, block, vocab)
        selector = FP16VocabularySelector(vocab)
        selected_parts = []
        top = selector._top
        payloads_by_layout = {}

        def checked_top(scores, low, high):
            layout = scores.device_tensor_layout()
            assert low.device_tensor_layout() == high.device_tensor_layout() == layout
            key = (tuple(layout.device_size), tuple(layout.stride_map))
            if key in payloads_by_layout:
                old_low, old_high = payloads_by_layout[key]
                assert low is old_low and high is old_high
            else:
                payloads_by_layout[key] = low, high
            return top(scores, low, high)

        selector._top = checked_top

        def checked_selector(scores, k, block_size):
            greedy, shortlist = selector(scores, k, block_size)
            ids = greedy[0].cpu().int() + greedy[1].cpu().int() * 512
            assert torch.equal(ids.long(), resident[:, 1:].argmax(-1))
            if k:
                values, low, high = shortlist
                values_host = values.to(device="cpu", dtype=torch.float32)
                ids = low.cpu().long() + high.cpu().long() * 512
                reference = resident.flatten(0, 1)
                assert torch.equal(values_host, reference.topk(k, dim=-1).values)
                assert torch.equal(values_host, reference.gather(-1, ids))
                assert all(len(set(row)) == k for row in ids.tolist())
                selected_parts[:] = [ids.view(batch, block, k)[:, 1:]]
            return greedy, shortlist

        checked_selector.pack = selector.pack
        gather = torch.compile(head.gather_readout, fullgraph=True, dynamic=False)
        host_readout = head.w2.weight.cpu()

        def checked_gather(ids):
            assert torch.equal(ids.cpu().long(), selected_parts[0].flatten())
            actual = gather(ids)
            assert torch.equal(actual.cpu(), host_readout[ids.cpu().long()])
            return actual

        select = torch.compile(_select_candidate_parts, fullgraph=True, dynamic=False)
        assemble = torch.compile(_assemble_feedback, fullgraph=True, dynamic=False)

        def checked_feedback(scores, parts, order, previous):
            score_values = scores.to(device="cpu", dtype=torch.float32)
            ids = parts[0].cpu().long() + parts[1].cpu().long() * 512
            assert torch.equal(ids, selected_parts[0])
            expected = ids.gather(-1, score_values.argmax(-1, keepdim=True)).squeeze(-1)
            low, high = select(scores, parts, order)
            draft, feedback = assemble(low, high, previous)
            assert torch.equal(draft.cpu().long(), expected)
            expected_feedback = torch.cat((anchors[:, None].expand(-1, 2), expected[:, :-1]), dim=1)
            assert torch.equal(
                feedback.cpu().long(), expected_feedback.reshape(-1, 1).expand(-1, 32)
            )
            return draft, feedback

        model = SimpleNamespace(
            xpress_head=head,
            xpress_num_passes=0,
            xpress_topc=topc,
            _vocabulary_selector=checked_selector,
            compute_device_logits=lambda _: base,
            _hidden_cache=torch.compile(head.project_hidden_cache, fullgraph=True, dynamic=False),
            _gather_readout=checked_gather,
            _refine_candidates=torch.compile(head.refine_candidates, fullgraph=True, dynamic=False),
            _assemble_feedback=assemble,
            _select_with_feedback=checked_feedback,
            _feedback_order=torch.arange(topc).half().to("spyre"),
        )
        for name in (
            "logits_seconds",
            "initial_device_selection_seconds",
            "initial_device_selection_calls",
            "host_to_device_seconds",
            "device_to_host_seconds",
            "candidate_gather_seconds",
            "refiner_seconds",
            "device_selection_seconds",
            "device_selection_calls",
            "candidate_id_transfer_bytes",
            "candidate_transfer_bytes",
            "proposal_id_transfer_bytes",
            "logits_transfer_bytes",
            "host_argmax_seconds",
            "host_topk_seconds",
        ):
            setattr(model, name, 0)
        model._propose_block_on_device = MethodType(
            SpyreDFlashQwen3ForCausalLM._propose_block_on_device, model
        )
        propose = MethodType(SpyreDFlashQwen3ForCausalLM.propose_block, model)
        assert propose(hidden, anchors) == resident[:, 1:].argmax(-1).tolist()
        model.xpress_num_passes = 2
        proposals = propose(hidden, anchors)
        assert len(proposals) == batch and all(len(row) == block - 1 for row in proposals)
        assert model.logits_transfer_bytes == 0
        assert model.host_argmax_seconds == model.host_topk_seconds == 0
        assert model.candidate_id_transfer_bytes == batch * (block - 1) * topc * 4
        assert model.device_selection_calls == 2
        checked_selector(base, topc, block)
