# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as NS

import pytest
import torch

from spyre_inference.v1.spec_decode.config import (
    normalize_speculative_config,
    validate_dflash_config,
)
from spyre_inference.v1.spec_decode.dflash import accepted_context_length


@pytest.mark.parametrize(
    "scheduled,drafts,sampled,expected",
    [(64, 0, 0, 64), (5, 0, 1, 5), (16, 15, 1, 1), (16, 15, 4, 4), (16, 15, 16, 16)],
)
def test_context_excludes_rejected_rows_and_uncomputed_correction(
    scheduled, drafts, sampled, expected
):
    assert accepted_context_length(scheduled, drafts, sampled) == expected


def test_xpress_bridge_preserves_checkpoint_and_disables_only_default_prefix_cache():
    original = dict(
        method="xpress", model="checkpoint", num_speculative_tokens=15, xpress_num_passes=0
    )
    args = NS(
        speculative_config=original, additional_config={"unrelated": 1}, enable_prefix_caching=None
    )
    normalize_speculative_config(args)
    assert original["method"] == "xpress"
    assert args.speculative_config == dict(
        method="dflash",
        model="checkpoint",
        num_speculative_tokens=15,
        disable_padded_drafter_batch=True,
    )
    assert args.additional_config == {"unrelated": 1, "spyre_xpress": {"num_passes": 0}}
    assert args.enable_prefix_caching is False
    args.speculative_config = original
    args.enable_prefix_caching = True
    normalize_speculative_config(args)
    assert args.enable_prefix_caching is True


@pytest.fixture
def config():
    target = NS(model_type="qwen3", hidden_size=4096, vocab_size=151936, num_hidden_layers=36)
    draft = NS(
        hidden_size=4096,
        vocab_size=151936,
        layer_types=["full_attention"] * 5,
        dflash_config={"target_layer_ids": [1, 9, 17, 25, 33]},
        xpress_block_size=16,
        xpress_rank=256,
        sample_from_anchor=False,
    )
    return NS(
        speculative_config=NS(
            use_dflash=lambda: True,
            draft_model_config=NS(hf_config=draft, quantization=None, dtype=torch.float16),
            num_speculative_tokens=15,
            draft_sample_method="greedy",
        ),
        model_config=NS(
            hf_text_config=target,
            is_multimodal_model=False,
            dtype=torch.float16,
            enforce_eager=False,
            quantization=None,
        ),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=64),
        cache_config=NS(enable_prefix_caching=False),
        additional_config={"spyre_xpress": {}},
    )


def test_published_contract_is_accepted(config):
    validate_dflash_config(config)
    assert config.speculative_config.disable_padded_drafter_batch


@pytest.mark.parametrize(
    "invalid",
    [
        "block",
        "taps",
        "vocab",
        "missing_refiner",
        "prefix_cache",
        "batch",
        "bf16",
        "draft_bf16",
        "quantized",
        "eager",
    ],
)
def test_incompatible_checkpoint_or_serving_configuration_fails_early(config, invalid):
    draft = config.speculative_config.draft_model_config.hf_config
    if invalid == "block":
        config.speculative_config.num_speculative_tokens = 7
    elif invalid == "taps":
        draft.dflash_config["target_layer_ids"] = [1, 36]
    elif invalid == "vocab":
        draft.vocab_size = 100
    elif invalid == "missing_refiner":
        del draft.xpress_rank
    elif invalid == "prefix_cache":
        config.cache_config.enable_prefix_caching = True
    elif invalid == "batch":
        config.scheduler_config.max_num_seqs = 2
    elif invalid == "bf16":
        config.model_config.dtype = torch.bfloat16
    elif invalid == "draft_bf16":
        config.speculative_config.draft_model_config.dtype = torch.bfloat16
    elif invalid == "quantized":
        config.model_config.quantization = "fp8"
    elif invalid == "eager":
        config.model_config.enforce_eager = True
    with pytest.raises(ValueError):
        validate_dflash_config(config)
