# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Device regression for batched candidate-ID layouts and selector buffer lifetimes."""

import warnings

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present

pytestmark = pytest.mark.model_quality


def test_batched_native_feedback_preserves_ids_and_request_anchors():
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    import torch_spyre

    torch_spyre._autoload()
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.models.qwen3_dflash import _assemble_feedback, _select_candidate_parts

    select = torch.compile(_select_candidate_parts, fullgraph=True, dynamic=False)
    assemble = torch.compile(_assemble_feedback, fullgraph=True, dynamic=False)
    torch.manual_seed(39)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        order = torch.arange(512).half().to("spyre")
        for batch in (1, 2, 4):
            for kind in ("random", "all_equal", "edge_ties", "negative"):
                scores = torch.randn(batch, 15, 512).half()
                if kind == "all_equal":
                    scores.zero_()
                elif kind == "negative":
                    scores = -torch.arange(512).view(1, 1, -1).expand(batch, 15, -1).half()
                elif kind == "edge_ties":
                    scores.fill_(-4)
                    for row in range(15):
                        index = (31, 32, 63, 64, 255, 256, 510, 511)[row % 8]
                        scores[:, row, index] = scores[:, row, 511] = 9
                candidates = torch.randint(0, 262144, scores.shape)
                parts = tuple(
                    part.half().to("spyre") for part in (candidates % 512, candidates // 512)
                )
                anchors = (torch.arange(batch).int() * 30000 + 101).view(batch, 1)
                previous = anchors[:, :, None].expand(batch, 16, 32).reshape(-1, 32).contiguous()
                device_scores = scores.to("spyre")
                # DLF16 upload can round close scores into ties; use resident values.
                expected = candidates.gather(
                    -1, device_scores.cpu().argmax(-1, keepdim=True)
                ).squeeze(-1)
                expected_feedback = torch.cat(
                    (anchors.expand(-1, 2), expected[:, :-1]), dim=1
                ).int()
                low, high = select(device_scores, parts, order)
                draft, feedback = assemble(low, high, previous.to("spyre"))
                assert torch.equal(draft.cpu(), expected), (batch, kind)
                assert torch.equal(
                    feedback.cpu(), expected_feedback.reshape(-1, 1).expand(-1, 32)
                ), (batch, kind)
