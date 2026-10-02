# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bridge XPress onto the DFlash contracts of the pinned vLLM 0.28 runner."""

import torch


def normalize_speculative_config(engine_args) -> None:
    spec = engine_args.speculative_config
    if not isinstance(spec, dict) or spec.get("method") != "xpress":
        return
    spec = dict(spec)
    passes = spec.pop("xpress_num_passes", None)
    if passes is not None and (type(passes) is not int or passes < 0):
        raise ValueError("xpress_num_passes must be a nonnegative integer")
    spec["method"] = "dflash"
    spec["disable_padded_drafter_batch"] = True
    engine_args.speculative_config = spec
    engine_args.additional_config = dict(engine_args.additional_config or {}) | {
        "spyre_xpress": {"num_passes": passes}
    }
    if engine_args.enable_prefix_caching is None:
        engine_args.enable_prefix_caching = False


def validate_dflash_config(config) -> None:
    spec = config.speculative_config
    if spec is None or not spec.use_dflash():
        return
    draft = spec.draft_model_config.hf_config
    target = config.model_config.hf_text_config
    parallel = config.parallel_config
    if config.model_config.dtype != torch.float16 or spec.draft_model_config.dtype != torch.float16:
        raise ValueError("Spyre DFlash/XPress currently requires float16 serving weights")
    if config.model_config.enforce_eager:
        raise ValueError("Spyre DFlash/XPress currently requires compiled execution")
    if (
        config.model_config.quantization is not None
        or spec.draft_model_config.quantization is not None
    ):
        raise ValueError(
            "Spyre DFlash/XPress currently requires unquantized target and draft weights"
        )
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or config.scheduler_config.max_num_seqs != 1
    ):
        raise ValueError("Spyre DFlash/XPress currently requires TP=1, PP=1 and max_num_seqs=1")
    if config.cache_config.enable_prefix_caching:
        raise ValueError(
            "Disable prefix caching for Spyre DFlash/XPress: draft context needs every target state"
        )
    if target.model_type != "qwen3" or config.model_config.is_multimodal_model:
        raise ValueError("Spyre DFlash/XPress currently supports the Qwen3 text target")
    if spec.draft_sample_method != "greedy":
        raise ValueError("Spyre DFlash/XPress currently uses greedy proposals")
    if config.scheduler_config.max_num_batched_tokens < spec.num_speculative_tokens + 1:
        raise ValueError("max_num_batched_tokens must accommodate a complete draft block")
    if draft.hidden_size != target.hidden_size or draft.vocab_size != target.vocab_size:
        raise ValueError("The draft must share the target hidden size and full vocabulary")
    if getattr(draft, "draft_vocab_size", draft.vocab_size) not in (None, target.vocab_size):
        raise ValueError("Spyre DFlash/XPress requires the full target vocabulary")
    if getattr(draft, "logit_scale", 1.0) != 1.0 or getattr(draft, "attention_bias", False):
        raise ValueError("Spyre DFlash/XPress requires unscaled logits and no attention bias")
    if getattr(draft, "is_causal", False) or draft.dflash_config.get("causal", False):
        raise ValueError("Spyre DFlash/XPress requires non-causal draft attention")
    if draft.dflash_config.get("use_swa", False) or draft.dflash_config.get(
        "attention_sink_bias", False
    ):
        raise ValueError(
            "Spyre DFlash/XPress does not support sliding attention or attention sinks"
        )
    if any(t != "full_attention" for t in getattr(draft, "layer_types", [])):
        raise ValueError(
            "Spyre DFlash currently supports full non-causal attention in every draft layer"
        )
    layers = draft.dflash_config.get("target_layer_ids", [])
    if (
        not layers
        or layers != sorted(set(layers))
        or not all(0 <= i < target.num_hidden_layers for i in layers)
    ):
        raise ValueError("The draft checkpoint must identify ordered, distinct target layers")
    block_size = getattr(draft, "xpress_block_size", getattr(draft, "block_size", None))
    if block_size != spec.num_speculative_tokens + 1:
        raise ValueError("num_speculative_tokens must equal the checkpoint block size minus one")
    xpress = "spyre_xpress" in config.additional_config
    if xpress and not hasattr(draft, "xpress_rank"):
        raise ValueError(
            "method='xpress' requires an XPress checkpoint, including its refiner weights"
        )
    if getattr(draft, "sample_from_anchor", False):
        raise ValueError("Spyre XPress requires a fixed anchor in slot zero")
    spec.disable_padded_drafter_batch = True
