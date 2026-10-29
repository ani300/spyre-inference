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

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Host replacements for vLLM 0.32's Triton rejection kernels."""

import torch


class _TorchKernel:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def register_warmup(self, **kwargs):
        # Host operations have no Triton program to register for startup warmup.
        pass


@_TorchKernel
def greedy_sample(
    output,
    cumulative,
    draft_ids,
    target_ids,
    bonus_ids,
    is_greedy,
    max_spec_len,
    uniform,
    conditional_rates,
):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        if is_greedy is None or is_greedy[request]:
            for position, index in enumerate(range(start, end)):
                draft = draft_ids[index]
                accepted = (
                    draft >= 0 and uniform[index] < conditional_rates[position]
                    if conditional_rates is not None
                    else draft == target_ids[index]
                )
                output[request, position] = (
                    draft if conditional_rates is not None and accepted else target_ids[index]
                )
                if not accepted:
                    break
            else:
                output[request, end - start] = bonus_ids.reshape(-1)[request]
        start = end


@_TorchKernel
def random_sample(
    output,
    cumulative,
    draft_ids,
    draft_probs,
    target_probs,
    bonus_ids,
    recovered_ids,
    uniform,
    is_greedy,
    max_spec_len,
    conditional_rates,
):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        if not is_greedy[request]:
            for position, index in enumerate(range(start, end)):
                draft = int(draft_ids[index])
                if draft < 0:
                    accepted = False
                elif conditional_rates is not None:
                    accepted = uniform[index] < conditional_rates[position]
                else:
                    q = 1.0 if draft_probs is None else draft_probs[index, draft]
                    accepted = q > 0 and target_probs[index, draft] / q >= uniform[index]
                output[request, position] = draft if accepted else recovered_ids[index]
                if not accepted:
                    break
            else:
                output[request, end - start] = bonus_ids.reshape(-1)[request]
        start = end


@_TorchKernel
def expand(output, values, cumulative, replace_from, replace_to):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        value = values[request]
        output[start:end] = replace_to if value == replace_from else value
        start = end


@_TorchKernel
def recover(
    output,
    cumulative,
    draft_ids,
    draft_probs,
    target_probs,
    inverse_noise,
    *,
    max_spec_len,
):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        if end > start:
            if draft_probs is None:
                residual = target_probs[start:end].clone()
                ids = draft_ids[start:end].long()
                valid = ids >= 0
                rows = torch.arange(end - start)[valid]
                residual[rows, ids[valid]] = 0
            else:
                residual = (target_probs[start:end] - draft_probs[start:end]).clamp_min(0)
            # torch.argmax chooses the first index on ties, including all-zero rows.
            output[start:end] = (residual * inverse_noise[request]).argmax(dim=-1)
        start = end


def install_rejection_sampler_kernels() -> None:
    from vllm.v1.sample import rejection_sampler

    rejection_sampler._rejection_greedy_sample = greedy_sample
    rejection_sampler._rejection_random_sample = random_sample
    rejection_sampler._expand = expand
    rejection_sampler._sample_recovered_tokens = recover
