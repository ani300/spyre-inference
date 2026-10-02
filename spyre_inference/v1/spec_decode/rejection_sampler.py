# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Spyre-Inference Authors.

"""CPU kernels for vLLM 0.28's rejection sampler (including the empty build).

The upstream sampler still owns logits processors, RNG, bonus sampling and
logprobs. Only its four Triton launches are replaced; all buffers stay on CPU.
"""

import torch


class _TorchKernel:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        return self.function


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


@_TorchKernel
def expand(output, values, cumulative, replace_from, replace_to, MAX_NUM_TOKENS):
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
    vocab_size,
    BLOCK_SIZE,
    NO_DRAFT_PROBS,
    USE_FP64_GUMBEL,
):
    start = 0
    for request, end in enumerate(cumulative.tolist()):
        if end > start:
            if NO_DRAFT_PROBS:
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

    rejection_sampler.rejection_greedy_sample_kernel = greedy_sample
    rejection_sampler.rejection_random_sample_kernel = random_sample
    rejection_sampler.expand_kernel = expand
    rejection_sampler.sample_recovered_tokens_kernel = recover
