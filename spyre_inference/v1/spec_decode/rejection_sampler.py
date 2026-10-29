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

"""TEMPORARY / vendored: PyTorch ports of vLLM 0.32.0's rejection sampler kernels.

Source: vllm/v1/sample/rejection_sampler.py: rejection_greedy_sample_kernel,
rejection_random_sample_kernel, expand_kernel and sample_recovered_tokens_kernel.
Remove when upstream CPU ops are loadable without vllm._C or supplied by csrc/ (#1417).
"""

import torch


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
    SYNTHETIC_MODE,
):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        if is_greedy is None or is_greedy[request]:
            for position, index in enumerate(range(start, end)):
                draft = draft_ids[index]
                accepted = (
                    draft >= 0 and uniform[index] < conditional_rates[position]
                    if SYNTHETIC_MODE
                    else draft == target_ids[index]
                )
                output[request, position] = (
                    draft if SYNTHETIC_MODE and accepted else target_ids[index]
                )
                if not accepted:
                    break
            else:
                output[request, end - start] = bonus_ids.reshape(-1)[request]
        start = end


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
    vocab_size,
    conditional_rates,
    NO_DRAFT_PROBS,
    SYNTHETIC_MODE,
):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        if not is_greedy[request]:
            for position, index in enumerate(range(start, end)):
                draft = int(draft_ids[index])
                if draft < 0:
                    accepted = False
                elif SYNTHETIC_MODE:
                    accepted = uniform[index] < conditional_rates[position]
                else:
                    q = 1.0 if NO_DRAFT_PROBS else draft_probs[index, draft]
                    accepted = q > 0 and target_probs[index, draft] / q >= uniform[index]
                output[request, position] = draft if accepted else recovered_ids[index]
                if not accepted:
                    break
            else:
                output[request, end - start] = bonus_ids.reshape(-1)[request]
        start = end


def expand(output, values, cumulative, replace_from, replace_to, MAX_NUM_TOKENS):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        value = values[request]
        output[start:end] = replace_to if value == replace_from else value
        start = end


def recover(
    output,
    cumulative,
    draft_ids,
    draft_probs,
    target_probs,
    inverse_noise,
    vocab_size,
    BLOCK_SIZE,
    NO_DRAFT_PROBS,
    USE_FP64_GUMBEL,
):
    counts = cumulative.diff(prepend=cumulative.new_zeros(1))
    widths, group_sizes = counts.unique_consecutive(return_counts=True)
    start = request = 0
    # Group equal draft lengths to broadcast one noise row per request, avoiding
    # a full [num_tokens, vocab_size] copy of the noise with repeat_interleave.
    for width, group_size in zip(widths.tolist(), group_sizes.tolist(), strict=True):
        end = start + width * group_size
        if width:
            target = target_probs[start:end].view(group_size, width, vocab_size)
            noise = inverse_noise[request : request + group_size, None, :]
            if NO_DRAFT_PROBS:
                scores = (target * noise).view(-1, vocab_size)
                ids = draft_ids[start:end].long()
                valid = (ids >= 0) & (ids < vocab_size)
                rows = torch.arange(end - start, device=ids.device)[valid]
                # Mask after multiplying to avoid a residual copy; preserve 0 * inf.
                scores[rows, ids[valid]] = 0 * inverse_noise[request + rows // width, ids[valid]]
            else:
                draft = draft_probs[start:end].view_as(target)
                residual = (target - draft).clamp_min_(0)
                scores = residual.to(torch.promote_types(residual.dtype, noise.dtype))
                scores.mul_(noise)
                scores = scores.view(-1, vocab_size)
            # torch.argmax chooses the first index on ties, including all-zero rows.
            output[start:end] = scores.argmax(dim=-1)
        start = end
        request += group_size
