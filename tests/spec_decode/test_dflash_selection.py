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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F
from xpress_reference import merged_refine

from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM
from spyre_inference.v1.spec_decode.selection import (
    FP16VocabularySelector,
    _trim_shortlist,
    argmax_parts,
)
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead, select_feedback
from spyre_inference.v1.worker import compile_guard


@pytest.mark.parametrize(
    "topc,available,missing",
    [
        (65, ["topkvalue", "topk_payload"], None),
        (128, ["topkvalue", "topk_payload"], None),
        (512, ["topkvalue", "topk_payload", "mask_by_index"], None),
        (128, ["topkvalue"], "topk_payload"),
        (129, ["topkvalue", "topk_payload"], "mask_by_index"),
    ],
)
def test_selection_requires_only_the_operators_it_uses(monkeypatch, topc, available, missing):
    monkeypatch.setenv("SPYRE_XPRESS_DEVICE_SELECTION", "1")
    monkeypatch.setattr(torch.ops, "spyre", SimpleNamespace(**dict.fromkeys(available)))
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    head = XPressRefinerHead(1031, 128, 8, rank=64, mlp_hidden=128, topc=topc)
    monkeypatch.setattr(head, "prepare_for_spyre", lambda: None)
    model = SimpleNamespace(
        model=SimpleNamespace(prepare_for_spyre=lambda: None, fc=torch.nn.Identity(), layers=[]),
        xpress_head=head,
        xpress_topc=topc,
        config=SimpleNamespace(vocab_size=1031),
    )
    if missing:
        with pytest.raises(RuntimeError, match=missing):
            SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)
    else:
        SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(model)
        assert isinstance(model._vocabulary_selector, FP16VocabularySelector)


def test_selection_rejects_dflash_without_an_xpress_head(monkeypatch):
    monkeypatch.setenv("SPYRE_XPRESS_DEVICE_SELECTION", "1")
    with pytest.raises(ValueError, match="XPress checkpoint"):
        SpyreDFlashQwen3ForCausalLM.prepare_for_spyre(SimpleNamespace(xpress_head=None))


@pytest.mark.parametrize("vocab", [127, 262145])
def test_selection_rejects_vocabulary_outside_payload_range(vocab):
    with pytest.raises(ValueError, match="128 to 262144"):
        FP16VocabularySelector(vocab)


def test_selection_registers_its_compiled_helpers(monkeypatch):
    compiled, watched = [], []

    def compile(function, **kwargs):
        compiled.append(function)
        return function

    monkeypatch.setattr(torch, "compile", compile)
    monkeypatch.setattr(compile_guard, "watch", lambda function, label: watched.append(function))
    FP16VocabularySelector(262144)
    assert compiled and compiled == watched


def test_argmax_digits_preserve_large_ids_and_first_ties():
    vocab = 262144
    ids = torch.arange(vocab)
    scores = torch.full((7, vocab), -5.0, dtype=torch.float16)
    first = torch.tensor([0, 511, 512, 65537, 131073, 262143, 200001])
    scores.scatter_(1, first[:, None], 5)
    scores[:5, -1] = 5
    low, high = argmax_parts(scores, (ids % 512).half(), (ids // 512).half(), 512.0)
    assert torch.equal(low.long() + high.long() * 512, scores.argmax(-1))


@pytest.mark.parametrize("batch,block,topc", [(1, 8, 1), (4, 8, 65), (4, 16, 128), (2, 32, 512)])
@pytest.mark.parametrize("passes", [0, 2])
def test_selection_uses_shared_feedback_without_full_logit_copies(batch, block, topc, passes):
    torch.manual_seed(81)
    vocab = 2051
    head = XPressRefinerHead(vocab, 128, block, rank=64, mlp_hidden=128, topc=topc).eval()
    head.fold_from_raw_(torch.randn_like(head.mix_L) * 0.01)
    hidden, base = torch.randn(batch * block, 128), torch.randn(batch * block, vocab)
    anchors = torch.arange(batch) + 2047

    class HostSelector:
        def __call__(self, scores, k, block_size):
            assert torch.equal(scores, base)
            draft = scores.view(batch, block_size, vocab)[:, 1:].argmax(-1)
            greedy = (draft % 512).half(), (draft // 512).half()
            if not k:
                return greedy, None
            values, candidates = scores.topk(k, dim=-1)
            return greedy, (values, (candidates % 512).half(), (candidates // 512).half())

        def pack(self, shortlist, block_size):
            return _trim_shortlist(*shortlist, block_size)

    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=vocab),
        xpress_head=head,
        xpress_topc=topc,
        xpress_num_passes=passes,
        _vocabulary_selector=HostSelector(),
        compute_device_logits=lambda _: F.pad(base, (0, 509), value=100),
        _copy_logits_to_cpu=Mock(side_effect=AssertionError("Full logit copy")),
        _argmax_on_cpu=Mock(side_effect=AssertionError("Host argmax")),
        _hidden_cache=head.project_hidden_cache,
        _prepare_feedback=Mock(wraps=head.prepare_feedback),
        _refine_feedback=Mock(wraps=head.refine_feedback),
        _select_feedback=Mock(wraps=select_feedback),
        _feedback_order=F.pad(torch.arange(topc).half(), (0, -topc % 64), value=-1),
    )
    with torch.inference_mode():
        expected = merged_refine(
            head, base.view(batch, block, vocab), hidden.view(batch, block, 128), anchors, passes
        ).tolist()
        actual = SpyreDFlashQwen3ForCausalLM.propose_block(model, hidden, anchors)
    assert actual == expected
    model._copy_logits_to_cpu.assert_not_called()
    model._argmax_on_cpu.assert_not_called()
    assert model._prepare_feedback.call_count == bool(passes)
    assert model._refine_feedback.call_count == model._select_feedback.call_count == passes
