# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Published Qwen3 DFlash/XPress checkpoint contract, version 1."""

from collections.abc import Mapping
from typing import Any

import torch

from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead

CHECKPOINT_VERSION = 1


def validate_dflash_weights(config, weights: Mapping[str, torch.Tensor]) -> None:
    if getattr(config, "spyre_xpress_checkpoint_version", CHECKPOINT_VERSION) != CHECKPOINT_VERSION:
        raise ValueError("Unsupported Spyre XPress checkpoint version")
    if getattr(config, "xpress_mixer_format", "raw") != "raw":
        raise ValueError("Checkpoint XPress mixer must be raw, before the serving fold")
    hidden = config.hidden_size
    head = config.head_dim
    query = config.num_attention_heads * head
    kv = config.num_key_value_heads * head
    intermediate = config.intermediate_size
    expected = {
        "fc.weight": (hidden, hidden * len(config.dflash_config["target_layer_ids"])),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
    }
    layer_shapes = {
        "self_attn.q_proj.weight": (query, hidden),
        "self_attn.k_proj.weight": (kv, hidden),
        "self_attn.v_proj.weight": (kv, hidden),
        "self_attn.o_proj.weight": (hidden, query),
        "self_attn.q_norm.weight": (head,),
        "self_attn.k_norm.weight": (head,),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
    }
    for layer in range(config.num_hidden_layers):
        expected.update({f"layers.{layer}.{name}": shape for name, shape in layer_shapes.items()})
    optional = {"embed_tokens.weight", "lm_head.weight"}
    present = {name for name in weights if not name.startswith("xpress_head.")}
    missing = expected.keys() - present
    unexpected = present - expected.keys() - optional
    if missing or unexpected:
        raise ValueError(
            f"Invalid DFlash backbone weights: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name in present & optional:
        expected[name] = (config.vocab_size, hidden)
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(
                f"Invalid {name} shape: expected {shape}, got {tuple(weights[name].shape)}"
            )


def convert_speculators_checkpoint(
    source: Mapping[str, Any],
    weights: Mapping[str, torch.Tensor],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Convert the pinned Speculators XPress schema to the published raw schema.

    Full-vocabulary Qwen3 backbones already use the published tensor names.
    Speculators auxiliary IDs index HF hidden_states (embedding at zero),
    while DFlash stores zero-based decoder layer IDs. No mixer fold is applied.
    """
    from transformers import Qwen3Config

    if source.get("speculators_model_type") != "xpress":
        raise ValueError("Expected a Speculators XPress checkpoint")
    transformer = source.get("transformer_layer_config", {})
    if transformer.get("model_type", "qwen3") != "qwen3" or target.get("model_type") != "qwen3":
        raise ValueError("XPress conversion currently supports Qwen3 text checkpoints")
    config = Qwen3Config(**transformer).to_dict()
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    if (hidden, vocab) != (target["hidden_size"], target["vocab_size"]):
        raise ValueError("Target and draft hidden size and vocabulary must match")
    if source.get("draft_vocab_size") != vocab or any(k in weights for k in ("t2d", "d2t")):
        raise ValueError("XPress conversion requires the full, unmapped target vocabulary")
    if source.get("target_hidden_size") not in (None, hidden):
        raise ValueError("Different target and draft hidden sizes are unsupported")
    if source.get("sample_from_anchor", False):
        raise ValueError("XPress conversion requires a fixed anchor in slot zero")
    if config.get("attention_bias", False) or config.get("hidden_act") != "silu":
        raise ValueError("XPress conversion requires bias-free Qwen3 attention and SwiGLU")
    if any(t != "full_attention" for t in config.get("layer_types", [])):
        raise ValueError("XPress conversion currently requires full draft attention")
    taps = source.get("aux_hidden_state_layer_ids")
    if (
        not isinstance(taps, list)
        or not taps
        or any(type(i) is not int or not 0 < i < target["num_hidden_layers"] for i in taps)
        or taps != sorted(set(taps))
    ):
        raise ValueError(
            "Auxiliary IDs must be ordered HF hidden-state taps, excluding embedding/final norm"
        )
    block = source.get("block_size")
    rank = source.get("xpress_rank", 256)
    ratio = source.get("xpress_mlp_ratio", 2)
    passes = source.get("num_jacobi_passes", 6)
    if (
        type(block) is not int
        or block < 2
        or any(type(n) is not int or n <= 0 for n in (rank, ratio))
    ):
        raise ValueError("Invalid XPress block, rank or MLP ratio")
    if type(passes) is not int or passes < 0:
        raise ValueError("num_jacobi_passes must be a nonnegative integer")
    mask = source.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < vocab:
        raise ValueError("An explicit in-vocabulary mask_token_id is required")

    config.pop("auto_map", None)
    config.update(
        architectures=["Qwen3XPressModel"],
        block_size=block,
        xpress_block_size=block,
        xpress_rank=rank,
        xpress_mlp_hidden=rank * ratio,
        xpress_num_passes=passes,
        sample_from_anchor=False,
        num_target_layers=target["num_hidden_layers"],
        dflash_config={"mask_token_id": mask, "target_layer_ids": [i - 1 for i in taps]},
        spyre_xpress_checkpoint_version=CHECKPOINT_VERSION,
        xpress_mixer_format="raw",
        xpress_source_config=dict(source),
    )
    with torch.device("meta"):
        head = XPressRefinerHead(vocab, hidden, block, rank, rank * ratio)
    expected = head.state_dict()
    training = XPressRefinerHead._TRAINING_KEYS
    published = {dest: name for name, dest in XPressRefinerHead._PUBLISHED_KEYS.items()}
    names = {k.removeprefix("refiner_head.") for k in weights if k.startswith("refiner_head.")}
    if names != training.keys() or any(k.startswith("xpress_head.") for k in weights):
        raise ValueError("Expected exactly the complete raw Speculators refiner_head weights")
    converted = {k: v for k, v in weights.items() if not k.startswith("refiner_head.")}
    for name, destination in training.items():
        weight = weights["refiner_head." + name]
        if weight.shape != expected[destination].shape:
            raise ValueError(f"Invalid refiner_head.{name} shape")
        converted["xpress_head." + published[destination]] = weight
    validate_dflash_weights(Qwen3Config(**config), converted)
    return config, converted
