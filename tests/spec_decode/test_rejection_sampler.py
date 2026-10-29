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
from vllm.v1.sample import rejection_sampler as upstream

from spyre_inference.v1.spec_decode import rejection_sampler as portable


@pytest.fixture(autouse=True)
def portable_kernels(monkeypatch):
    for name, kernel in (
        ("_rejection_greedy_sample", portable.greedy_sample),
        ("_rejection_random_sample", portable.random_sample),
        ("_expand", portable.expand),
        ("_sample_recovered_tokens", portable.recover),
    ):
        monkeypatch.setattr(upstream, name, kernel)


def test_greedy_prefix_correction_bonus_and_empty_request():
    draft = torch.tensor([2, 3, 4, 1, 5, 0, 6])
    target = torch.tensor([2, 7, 4, 1, 5, 2, 6])
    result = upstream.rejection_sample(
        draft,
        [0, 3, 2, 2],
        3,
        torch.tensor([0, 3, 5, 7]),
        None,
        torch.nn.functional.one_hot(target, 8).float(),
        torch.tensor([[4], [0], [7], [1]]),
        SimpleNamespace(all_greedy=True, all_random=False),
    )
    assert result.dtype == torch.int32
    assert result.tolist() == [[4, -1, -1, -1], [2, 7, -1, -1], [1, 5, 7, -1], [2, -1, -1, -1]]


def test_random_acceptance_and_greedy_rows_are_untouched():
    output = torch.full((4, 4), -1, dtype=torch.int32)
    output[1, 0] = 8
    portable.random_sample(
        output,
        torch.tensor([3, 4, 5, 6]),
        torch.tensor([1, 0, 2, 1, -1, 1]),
        torch.tensor([[0.2, 0.6, 0.2]] * 6),
        torch.tensor([[0.3, 0.3, 0.4]] * 6),
        torch.tensor([2, 2, 2, 2]),
        torch.tensor([0, 1, 0, 0, 2, 0]),
        torch.tensor([0.4, 0.9, 0.9, 0.9, 0.1, 0.6]),
        torch.tensor([False, True, False, False]),
        3,
        None,
    )
    assert output.tolist() == [[1, 0, 2, 2], [8, -1, -1, -1], [2, -1, -1, -1], [0, -1, -1, -1]]


@pytest.mark.parametrize("no_draft", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_recovery_residual_and_first_index_ties(no_draft, dtype):
    p = torch.tensor([[0.25, 0.25, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    q = torch.tensor([[0.25, 0.25, 0.5], [0.5, 0.1, 0.4], [0.2, 0.3, 0.5]])
    ids = torch.tensor([2, -1, 1])
    noise = torch.tensor([[2.0, 2.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype)
    result = torch.empty(3, dtype=torch.long)
    portable.recover(
        result,
        torch.tensor([2, 3]),
        ids,
        None if no_draft else q,
        p,
        noise,
        max_spec_len=2,
    )
    assert result.tolist() == ([0, 1, 2] if no_draft else [0, 1, 0])


def test_expand_sampling_parameters_with_zero_draft_requests():
    actual = upstream.expand_batch_to_tokens(
        torch.tensor([0.0, 0.5, 2.0, 1.0]),
        torch.tensor([0, 2, 2, 5]),
        5,
        replace_from=1,
        replace_to=9,
    )
    assert actual.tolist() == [0.5, 0.5, 9, 9, 9]


def test_synthetic_acceptance_never_accepts_placeholder():
    output = torch.full((1, 4), -1, dtype=torch.int32)
    portable.greedy_sample(
        output,
        torch.tensor([3]),
        torch.tensor([1, -1, 2]),
        torch.tensor([0, 0, 0]),
        torch.tensor([3]),
        None,
        3,
        torch.tensor([0.1, 0.1, 0.1]),
        torch.ones(3),
    )
    assert output.tolist() == [[1, 0, -1, -1]]


@pytest.mark.parametrize("with_draft_probs", [False, True])
def test_random_rejection_preserves_target_distribution(with_draft_probs):
    torch.manual_seed(129)
    n = 12000
    p = torch.tensor([0.1, 0.3, 0.6])
    q = torch.tensor([0.6, 0.2, 0.2]) if with_draft_probs else torch.tensor([0.0, 1.0, 0.0])
    draft = torch.multinomial(q, n, replacement=True)
    result = upstream.rejection_sample(
        draft,
        [1] * n,
        1,
        torch.arange(1, n + 1),
        q.expand(n, -1).contiguous() if with_draft_probs else None,
        p.log().expand(n, -1).contiguous(),
        torch.zeros(n, 1, dtype=torch.int32),
        SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.ones(n),
            generators={},
        ),
    )
    frequency = result[:, 0].bincount(minlength=3).float() / n
    torch.testing.assert_close(frequency, p, atol=0.015, rtol=0)
