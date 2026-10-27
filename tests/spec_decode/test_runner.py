# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.mark.parametrize("zeros_only", [False, True])
def test_host_drafts_and_context_limit_clear_do_not_need_cuda(zeros_only):
    runner = SimpleNamespace(
        _draft_token_ids=torch.tensor([[3, 5, 7]], dtype=torch.int32),
        input_batch=SimpleNamespace(req_ids=["request"]),
    )
    TorchSpyreModelRunner._copy_draft_token_ids_to_cpu(runner, None, zeros_only=zeros_only)
    assert runner._draft_token_ids == ([[0, 0, 0]] if zeros_only else [[3, 5, 7]])
    assert runner._draft_token_req_ids == ["request"]
    assert runner.prev_num_spec_tokens == 3
