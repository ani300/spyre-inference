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
