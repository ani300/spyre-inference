# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from spyre_inference.v1.spec_decode.checkpoint import (
    convert_speculators_checkpoint,
    validate_dflash_weights,
)
from spyre_inference.v1.spec_decode.convert_checkpoint import convert_checkpoint
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


@pytest.fixture
def checkpoint():
    config = SimpleNamespace(
        hidden_size=128,
        head_dim=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        num_hidden_layers=1,
        vocab_size=256,
        dflash_config={"target_layer_ids": [0, 2]},
    )
    shapes = {
        "fc.weight": (128, 256),
        "hidden_norm.weight": (128,),
        "norm.weight": (128,),
        "layers.0.self_attn.q_proj.weight": (128, 128),
        "layers.0.self_attn.k_proj.weight": (64, 128),
        "layers.0.self_attn.v_proj.weight": (64, 128),
        "layers.0.self_attn.o_proj.weight": (128, 128),
        "layers.0.self_attn.q_norm.weight": (64,),
        "layers.0.self_attn.k_norm.weight": (64,),
        "layers.0.mlp.gate_proj.weight": (256, 128),
        "layers.0.mlp.up_proj.weight": (256, 128),
        "layers.0.mlp.down_proj.weight": (128, 256),
        "layers.0.input_layernorm.weight": (128,),
        "layers.0.post_attention_layernorm.weight": (128,),
    }
    return config, {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}


def test_backbone_without_shared_embedding_and_head_is_complete(checkpoint):
    validate_dflash_weights(*checkpoint)


@pytest.mark.parametrize(
    "name", ["fc.weight", "layers.0.self_attn.k_proj.weight", "layers.0.mlp.up_proj.weight"]
)
def test_missing_weight_including_part_of_fused_parameter_fails(checkpoint, name):
    config, weights = checkpoint
    del weights[name]
    with pytest.raises(ValueError, match="missing="):
        validate_dflash_weights(config, weights)


def test_feature_width_mismatch_fails_before_loading(checkpoint):
    config, weights = checkpoint
    weights["fc.weight"] = torch.empty(128, 128, device="meta")
    with pytest.raises(ValueError, match="fc.weight shape"):
        validate_dflash_weights(config, weights)


@pytest.fixture
def training_checkpoint(checkpoint):
    body_config, body = checkpoint
    transformer = vars(body_config) | {
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "layer_types": ["full_attention"],
    }
    transformer.pop("dflash_config")
    source = dict(
        speculators_model_type="xpress",
        transformer_layer_config=transformer,
        draft_vocab_size=256,
        aux_hidden_state_layer_ids=[2, 4],
        block_size=4,
        mask_token_id=200,
        sample_from_anchor=False,
        xpress_rank=8,
        xpress_mlp_ratio=2,
        num_jacobi_passes=6,
        speculators_config={"verifier": {"name_or_path": "test/qwen"}},
    )
    shapes = {
        "token_embed.weight": (256, 8),
        "down_h.weight": (8, 128),
        "down_g.weight": (8, 128),
        "in_proj.weight": (8, 24),
        "mix_l": (8, 4, 4),
        "mlp_gate.weight": (16, 8),
        "mlp_up.weight": (16, 8),
        "mlp_down.weight": (8, 16),
        "readout.weight": (256, 8),
    }
    generator = torch.Generator().manual_seed(19)
    weights = {name: torch.randn(t.shape, generator=generator) for name, t in body.items()}
    weights.update(
        {
            "refiner_head." + name: torch.randn(shape, generator=generator)
            for name, shape in shapes.items()
        }
    )
    target = transformer | {"num_hidden_layers": 6, "layer_types": ["full_attention"] * 6}
    return source, weights, target


def test_training_export_maps_taps_and_preserves_raw_mixer(training_checkpoint):
    source, weights, target = training_checkpoint
    untouched = deepcopy(source)
    config, converted = convert_speculators_checkpoint(source, weights, target)
    assert source == untouched
    assert config["dflash_config"] == {"mask_token_id": 200, "target_layer_ids": [1, 3]}
    assert config["xpress_block_size"] == 4
    assert config["xpress_mlp_hidden"] == 16
    assert config["xpress_source_config"]["speculators_config"] == source["speculators_config"]
    assert torch.equal(converted["xpress_head.mix.L"], weights["refiner_head.mix_l"])
    training, serving = [XPressRefinerHead(256, 128, 4, 8, 16) for _ in range(2)]
    training.load_checkpoint_weights(weights)
    serving.load_checkpoint_weights(converted)
    for name, tensor in training.state_dict().items():
        assert torch.equal(tensor, serving.state_dict()[name])
    assert torch.equal(serving.mix_L, weights["refiner_head.mix_l"].tril() + torch.eye(4))


@pytest.mark.parametrize("sharded", [False, True])
def test_training_export_safetensors_round_trip(training_checkpoint, tmp_path, sharded):
    source, weights, target = training_checkpoint
    original = tmp_path / "training"
    original.mkdir()
    (original / "config.json").write_text(json.dumps(source))
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target))
    if sharded:
        items = list(weights.items())
        mapping = {}
        for i, group in enumerate((items[::2], items[1::2])):
            filename = f"model-{i}.safetensors"
            save_file(dict(group), original / filename)
            mapping.update({name: filename for name, _ in group})
        (original / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_file(weights, original / "model.safetensors")
    output = tmp_path / "serving"
    convert_checkpoint(original, target_path, output)
    expected_config, expected_weights = convert_speculators_checkpoint(source, weights, target)
    assert json.loads((output / "config.json").read_text()) == json.loads(
        json.dumps(expected_config)
    )
    actual = load_file(output / "model.safetensors")
    assert actual.keys() == expected_weights.keys()
    assert all(torch.equal(actual[name], tensor) for name, tensor in expected_weights.items())
    assert json.loads((output / "conversion.json").read_text())["format_version"] == 1
    with pytest.raises(FileExistsError):
        convert_checkpoint(original, target_path, output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("draft_vocab_size", 128, "full, unmapped"),
        ("sample_from_anchor", True, "fixed anchor"),
        ("aux_hidden_state_layer_ids", [0, 2], "Auxiliary IDs"),
        ("aux_hidden_state_layer_ids", [2, 6], "Auxiliary IDs"),
        ("aux_hidden_state_layer_ids", [4, 2], "Auxiliary IDs"),
        ("mask_token_id", None, "mask_token_id"),
        ("num_jacobi_passes", -1, "num_jacobi_passes"),
    ],
)
def test_training_export_rejects_incompatible_contract(training_checkpoint, field, value, message):
    source, weights, target = training_checkpoint
    source[field] = value
    with pytest.raises(ValueError, match=message):
        convert_speculators_checkpoint(source, weights, target)


def test_training_export_rejects_incomplete_head(training_checkpoint):
    source, weights, target = training_checkpoint
    del weights["refiner_head.down_h.weight"]
    with pytest.raises(ValueError, match="complete raw"):
        convert_speculators_checkpoint(source, weights, target)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("spyre_xpress_checkpoint_version", 2, "version"), ("xpress_mixer_format", "folded", "raw")],
)
def test_unknown_format_or_prefolded_checkpoint_is_rejected(checkpoint, field, value, message):
    config, weights = checkpoint
    setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        validate_dflash_weights(config, weights)
