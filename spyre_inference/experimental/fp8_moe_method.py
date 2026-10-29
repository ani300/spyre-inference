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

"""Engine bridge for Gemma's FP8 experts, loaded directly into QFP8WT."""

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import QuantizationStrategy
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod,
)

from spyre_inference.experimental.fp8_native_moe import native_moe

logger = init_logger(__name__)


def _pad_rows(x, logits):
    # Keep the standalone persistent kernel's token axis stick aligned, including
    # decode. These dummy rows never leave the MoE and do not touch attention/KV.
    pad = -x.shape[0] % 128
    return F.pad(x, (0, 0, 0, pad)), F.pad(logits, (0, 0, 0, pad))


def _trim_rows(x, tokens):
    return x[:tokens].clone()


pad_rows = torch.compile(_pad_rows, fullgraph=True, dynamic=False)
trim_rows = torch.compile(_trim_rows, fullgraph=True, dynamic=False)


class SpyreFp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
    requires_device_loading = False
    supports_pre_processed_weights = False
    spyre_fp8_experts = True

    def __init__(self, weight_quant, input_quant, moe):
        super().__init__(weight_quant, input_quant, moe)
        self.supports_pre_processed_weights = False

    @property
    def is_monolithic(self):
        return True

    def process_weights_after_loading(self, layer):
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        if (
            self.weight_quant.strategy != QuantizationStrategy.CHANNEL
            or not self.input_quant.dynamic
        ):
            raise NotImplementedError(
                "Spyre FP8 MoE requires channel weights and dynamic token activations"
            )
        recipe = layer.spyre_moe_recipe
        if recipe.activation != "gelu_tanh" or recipe.routing != "full_softmax":
            raise NotImplementedError(
                "The FP8 MoE bridge currently implements Gemma's GELU/full-softmax recipe"
            )
        if layer.top_k != 8 or layer.local_num_experts != 128 or self.moe.tp_size != 1:
            raise NotImplementedError("The experimental FP8 MoE bridge requires Gemma TP1")
        w13, w2 = layer.w13_weight.detach(), layer.w2_weight.detach()
        if w13.device.type != "cpu" or w2.device.type != "cpu":
            raise RuntimeError(
                "FP8 expert post-load processing must start with CPU checkpoint tensors"
            )
        if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
            raise TypeError("Expert checkpoint weights must remain E4M3 FP8")
        inter = w13.shape[1] // 2
        pad = -inter % 128
        scales = {
            "gate": layer.w13_weight_scale[:, :inter].detach().float(),
            "up": layer.w13_weight_scale[:, inter:].detach().float(),
            "down": layer.w2_weight_scale.detach().float(),
        }
        if recipe.prepare_down_weight is not None:
            scales["down"] = recipe.prepare_down_weight(scales["down"])
        weights = {"gate": w13[:, :inter], "up": w13[:, inter:], "down": w2}
        for name in ("gate", "up", "down"):
            w = weights[name]
            if pad:
                w = F.pad(w.view(torch.uint8), (0, pad) if name == "down" else (0, 0, 0, pad)).view(
                    torch.float8_e4m3fn
                )
                if name != "down":
                    scales[name] = F.pad(scales[name], (0, 0, 0, pad), value=1)
            pool = _dma_to_spyre_fp8_kernel(w.transpose(-1, -2))
            layer.register_buffer("spyre_fp8_" + name, pool, persistent=False)
            layer.register_buffer(
                "spyre_fp8_" + name + "_scale",
                scales[name].transpose(-1, -2).contiguous().half().to("spyre"),
                persistent=False,
            )
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale, layer.w2_weight_scale
        self.moe_quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn, per_act_token_quant=True, per_out_ch_quant=True
        )
        layer._ensure_moe_quant_config_init()
        logger.info(
            "Spyre: directly loaded FP8 experts for %s (%d experts, I=%d padded to %d)",
            layer.layer_name,
            w13.shape[0],
            inter,
            inter + pad,
        )

    def get_fused_moe_quant_config(self, layer):
        return self.moe_quant_config

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "FP8 MoE must use the opaque MoERunner entry to preserve kernel boundaries"
            )
        tokens = x.shape[0]
        xp, logits = pad_rows(x, router_logits)
        pools = {p: getattr(layer, "spyre_fp8_" + p) for p in ("gate", "up", "down")}
        scales = {p: getattr(layer, "spyre_fp8_" + p + "_scale") for p in pools}
        value = native_moe(
            xp, logits, pools, scales, path="persistent", chunks=1, progress=lambda _: None
        )
        return trim_rows(value, tokens)
