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

from types import MethodType, SimpleNamespace

import pytest
import torch

from spyre_inference import envs
from spyre_inference.models.qwen3_dflash import (
    SpyreDFlashQwen3ForCausalLM,
    _assemble_feedback,
    _select_candidate_parts,
)
from spyre_inference.v1.spec_decode.selection import FP16VocabularySelector, argmax_parts
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


@pytest.mark.parametrize("feedback", [False, True])
def test_feedback_compilation_requires_opt_in(monkeypatch, feedback):
    head = XPressRefinerHead(1031, 128, 16, rank=64, mlp_hidden=128)
    model = SimpleNamespace(
        model=SimpleNamespace(prepare_for_spyre=lambda: None, fc=torch.nn.Identity(), layers=[]),
        xpress_head=head,
        xpress_topc=128,
        config=SimpleNamespace(vocab_size=1031),
    )
    compiled = []

    def compile(function, **kwargs):
        compiled.append(function)
        return function

    monkeypatch.setattr(head, "prepare_for_spyre", lambda: None)
    monkeypatch.setattr(torch, "compile", compile)
    monkeypatch.setattr(envs, "SPYRE_XPRESS_DEVICE_FEEDBACK", feedback)
    monkeypatch.setattr(envs, "SPYRE_XPRESS_DEVICE_SELECTION", False)
    SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)
    assert model._device_feedback_enabled is feedback
    assert (_assemble_feedback in compiled) is feedback
    assert (_select_candidate_parts in compiled) is feedback


@pytest.mark.parametrize("topc", [128, 512])
@pytest.mark.parametrize("missing", [None, "topkvalue", "topk_payload", "mask_by_index"])
def test_device_selection_requires_masking_only_for_multiple_rounds(monkeypatch, topc, missing):
    head = XPressRefinerHead(1031, 128, 16, rank=64, mlp_hidden=128, topc=topc)
    model = SimpleNamespace(
        model=SimpleNamespace(prepare_for_spyre=lambda: None, fc=torch.nn.Identity(), layers=[]),
        xpress_head=head,
        xpress_topc=topc,
        config=SimpleNamespace(vocab_size=1031),
    )
    ops = SimpleNamespace(
        **{
            name: lambda *args: None
            for name in ("topkvalue", "topk_payload", "mask_by_index")
            if name != missing
        }
    )
    monkeypatch.setattr(head, "prepare_for_spyre", lambda: None)
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setattr(torch.ops, "spyre", ops)
    monkeypatch.setattr(envs, "SPYRE_XPRESS_DEVICE_FEEDBACK", False)
    monkeypatch.setattr(envs, "SPYRE_XPRESS_DEVICE_SELECTION", True)
    if missing is None or (topc == 128 and missing == "mask_by_index"):
        SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)
        assert isinstance(model._vocabulary_selector, FP16VocabularySelector)
        assert model._device_feedback_enabled
    else:
        with pytest.raises(RuntimeError, match="requires patched torch-spyre and Deeptools"):
            SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)


@pytest.mark.parametrize("kind", ["random", "all_equal", "negative", "edge_ties"])
def test_vocabulary_argmax_preserves_large_ids_and_first_position_ties(kind):
    torch.manual_seed(817)
    vocab = 262144
    positions = torch.arange(vocab)
    scores = torch.randn(4, vocab).half()
    if kind == "all_equal":
        scores.zero_()
    elif kind == "negative":
        scores.fill_(-4)
        scores[torch.arange(4), torch.tensor([2049, 65537, 151935, 262143])] = -1
    elif kind == "edge_ties":
        scores.fill_(-4)
        for row, (first, second) in enumerate(
            [(511, 512), (65535, 65536), (131073, 151935), (262142, 262143)]
        ):
            scores[row, first] = scores[row, second] = 9
    low, high = argmax_parts(
        scores, (positions % 512).half(), (positions // 512).half(), torch.tensor([512.0]).half()
    )
    assert torch.equal(low.int() + high.int() * 512, scores.argmax(-1))


@pytest.mark.parametrize("kind", ["random", "all_equal", "negative", "edge_ties"])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("topc", [128, 512])
def test_device_feedback_selection_keeps_ties_large_ids_and_anchors(kind, batch, topc):
    torch.manual_seed(39)
    scores = torch.randn(batch, 15, topc).half()
    if kind == "all_equal":
        scores.zero_()
    elif kind == "negative":
        scores = -torch.arange(topc).view(1, 1, -1).expand(batch, 15, -1).half()
    elif kind == "edge_ties":
        scores.fill_(-4)
        for row in range(15):
            index = (31, 32, 63, 64, topc // 2 - 1, topc // 2, topc - 2, topc - 1)[row % 8]
            scores[:, row, index] = scores[:, row, topc - 1] = 9
    candidates = torch.randint(0, 262144, scores.shape)
    candidates[..., :8] = torch.tensor([0, 2049, 4097, 65537, 131073, 151935, 262143, 32769])
    parts = tuple(part.half() for part in (candidates % 512, candidates // 512))
    previous = (torch.arange(batch, dtype=torch.int32) * 30000 + 101).view(batch, 1, 1)
    previous = previous.expand(batch, 16, 32).reshape(-1, 32).contiguous()
    low, high = _select_candidate_parts(scores, parts, torch.arange(topc).half())
    draft, feedback = _assemble_feedback(low, high, previous)
    expected = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
    assert draft.tolist() == expected.tolist()
    expected_previous = torch.cat(
        (previous.view(batch, 16, 32)[:, :2, 0], expected[:, :-1]), dim=1
    ).int()
    torch.testing.assert_close(feedback, expected_previous.reshape(-1, 1).expand(-1, 32))


@pytest.mark.parametrize("passes", [1, 6])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("topc", [128, 512])
def test_device_feedback_serving_loop_copies_only_base_scores_and_final_ids(passes, batch, topc):
    torch.manual_seed(81)
    head = XPressRefinerHead(1031, 128, 16, rank=64, mlp_hidden=128, topc=topc).eval()
    hidden, base = torch.randn(batch * 16, 128), torch.randn(batch * 16, 1031)
    anchor = torch.arange(batch) + 1025
    model = SimpleNamespace(
        xpress_head=head,
        xpress_num_passes=passes,
        xpress_topc=topc,
        _device_feedback_enabled=True,
        _feedback_order=torch.arange(topc).half(),
        compute_device_logits=lambda _: base,
        _hidden_cache=head.project_hidden_cache,
        _refine=head.refine_full,
        _gather_readout=head.gather_readout,
        _refine_candidates=head.refine_candidates,
        _select_candidate_parts=_select_candidate_parts,
        _assemble_feedback=_assemble_feedback,
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
        "device_selection_calls",
        "device_selection_seconds",
        "proposal_id_transfer_bytes",
    ):
        setattr(model, name, 0)
    for name in ("_copy_logits_to_cpu", "_argmax_on_cpu", "_select_with_feedback"):
        setattr(model, name, MethodType(getattr(SpyreDFlashQwen3ForCausalLM, name), model))
    with torch.inference_mode():
        expected = head(
            base.view(batch, 16, 1031), hidden.view(batch, 16, 128), anchor, anchor, passes
        ).tolist()
        actual = SpyreDFlashQwen3ForCausalLM.propose_block(model, hidden, anchor)
    assert actual == expected
    assert model.selection_calls == 1
    assert model.device_selection_calls == passes
    assert model.logits_transfer_bytes == base.numel() * 4
    assert model.proposal_id_transfer_bytes == batch * 15 * 4
