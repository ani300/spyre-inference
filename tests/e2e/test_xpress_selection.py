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

import warnings
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present
from torch.nn import functional as F

pytestmark = pytest.mark.model_quality


@pytest.mark.parametrize("batch,block,topc", [(1, 16, 128), (4, 8, 65), (4, 16, 512)])
def test_device_selection_drives_shared_feedback_without_full_logit_copies(
    monkeypatch, batch, block, topc
):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    import torch_spyre

    torch_spyre._autoload()
    required = ["topkvalue", "topk_payload"] + (["mask_by_index"] if topc > 128 else [])
    missing = [name for name in required if not hasattr(torch.ops.spyre, name)]
    if missing:
        pytest.skip(f"Requires optional torch-spyre selection operators: {missing}")
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM
    from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead
    from spyre_inference.v1.worker import compile_guard

    monkeypatch.setenv("SPYRE_XPRESS_DEVICE_SELECTION", "1")
    torch.manual_seed(918)
    vocab, passes = 262144, 2
    head = XPressRefinerHead(vocab, 128, block, rank=256, mlp_hidden=512, topc=topc).eval()
    head.fold_from_raw_(torch.randn_like(head.mix_L) * 0.01)
    head = head.half().to("spyre")
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        hidden = torch.randn(batch * block, 128).half().to("spyre")
        anchors = torch.arange(batch).int() * 20003 + 131073
        raw = torch.randn(batch * block, vocab + 512).half()
        maxima = (131073 + torch.arange(batch * block) * 997) % vocab
        raw.scatter_(1, maxima[:, None], 32.0)
        raw[:, vocab:] = 1000.0
        base = raw.to("spyre")
        resident = base.cpu().float()[:, :vocab].view(batch, block, vocab)
        draft = resident[:, 1:].argmax(-1)
        model = SimpleNamespace(
            model=SimpleNamespace(
                prepare_for_spyre=lambda: None, fc=torch.nn.Identity(), layers=[]
            ),
            config=SimpleNamespace(vocab_size=vocab),
            xpress_head=head,
            xpress_topc=topc,
            xpress_num_passes=0,
            compute_device_logits=lambda _: base,
            _copy_logits_to_cpu=Mock(side_effect=AssertionError("Full logit copy")),
            _argmax_on_cpu=Mock(side_effect=AssertionError("Host argmax")),
        )
        SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)
        selector = model._vocabulary_selector
        top = selector._top
        payloads_by_layout = {}
        state = {}

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
            ids = greedy[0].cpu().long() + greedy[1].cpu().long() * 512
            assert torch.equal(ids, draft)
            if k:
                values, low, high = shortlist
                values = values.cpu().float()
                ids = low.cpu().long() + high.cpu().long() * 512
                reference = resident.flatten(0, 1)
                assert torch.equal(values, reference.topk(k, dim=-1).values)
                assert torch.equal(values, reference.gather(-1, ids))
                assert all(len(set(row)) == k for row in ids.tolist())
                state["candidates"] = ids.view(batch, block, k)[:, 1:]
            return greedy, shortlist

        checked_selector.pack = selector.pack
        model._vocabulary_selector = checked_selector
        host_input, host_readout = head.w1.weight.cpu(), head.w2.weight.cpu()
        prepare, refine, select = (
            model._prepare_feedback,
            model._refine_feedback,
            model._select_feedback,
        )

        def checked_prepare(ids, previous):
            padded = F.pad(state["candidates"], (0, -topc % 64)).flatten()
            expected_ids = torch.cat((anchors[:, None].expand(-1, 2), draft[:, :-1]), dim=1)
            assert torch.equal(ids.cpu().long(), padded)
            assert torch.equal(previous.cpu().long(), expected_ids)
            embeddings, feedback, readout = prepare(ids, previous)
            assert torch.equal(embeddings.cpu(), host_input[padded])
            assert torch.equal(readout.cpu(), host_readout[padded])
            state["feedback"] = host_input[expected_ids]
            assert torch.equal(feedback.cpu(), state["feedback"])
            return embeddings, feedback, readout

        def checked_refine(base_scores, previous, cache, readout):
            assert torch.equal(previous.cpu(), state["feedback"])
            expected = resident[:, 1:].gather(-1, state["candidates"])
            assert torch.equal(base_scores.cpu()[..., :topc].float(), expected)
            return refine(base_scores, previous, cache, readout)

        def checked_select(scores, embeddings, order, previous):
            expected = scores.cpu()[..., :topc].argmax(-1)
            indices, feedback = select(scores, embeddings, order, previous)
            assert torch.equal(indices.cpu().long(), expected)
            state["draft"] = state["candidates"].gather(-1, expected.unsqueeze(-1)).squeeze(-1)
            expected_ids = torch.cat(
                (anchors[:, None].expand(-1, 2), state["draft"][:, :-1]), dim=1
            )
            state["feedback"] = host_input[expected_ids]
            assert torch.equal(feedback.cpu(), state["feedback"])
            return indices, feedback

        model._prepare_feedback = Mock(wraps=checked_prepare)
        model._refine_feedback = Mock(wraps=checked_refine)
        model._select_feedback = Mock(wraps=checked_select)
        propose = MethodType(SpyreDFlashQwen3ForCausalLM.propose_block, model)
        assert propose(hidden, anchors) == draft.tolist()
        model.xpress_num_passes = passes
        assert propose(hidden, anchors) == state["draft"].tolist()
        compile_guard.arm(compile_guard.CompileGuardLevel.ERROR)
        assert propose(hidden, anchors) == state["draft"].tolist()
        model._copy_logits_to_cpu.assert_not_called()
        model._argmax_on_cpu.assert_not_called()
        assert model._prepare_feedback.call_count == 2
        assert model._refine_feedback.call_count == model._select_feedback.call_count == 2 * passes
