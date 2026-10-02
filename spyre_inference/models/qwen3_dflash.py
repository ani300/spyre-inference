# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Spyre-Inference Authors.

"""DFlash context projection and XPress weights for the legacy Spyre runner."""

import time
from typing import cast

import torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from spyre_inference.custom_ops.parallel_lm_head import SpyreUnquantizedLMHeadMethod
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl, SpyrePagedKVCache
from spyre_inference.v1.spec_decode.checkpoint import validate_dflash_weights
from spyre_inference.v1.spec_decode.xpress_head import XPressRefinerHead


def _project_context(attention, context, positions):
    # Spyre transposes qkv_proj weights after loading. Use its live linear
    # method instead of the upstream fused buffer built from raw CPU weights.
    qkv, _ = attention.qkv_proj(context)
    _, key, value = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
    key = attention.k_norm(key.view(-1, attention.num_kv_heads, attention.head_dim))
    key, _ = attention.rotary_emb(positions, key.flatten(1), None)
    return key.view(-1, attention.num_kv_heads, attention.head_dim), value.view(
        -1, attention.num_kv_heads, attention.head_dim
    )


class SpyreDFlashQwen3Model(DFlashQwen3Model):
    fc: ReplicatedLinear

    def _build_fused_kv_buffers(self) -> None:
        # The portable projectors are built after final device/dtype placement.
        pass

    def prepare_for_spyre(self) -> None:
        self._context_norm = torch.compile(self.hidden_norm, fullgraph=True, dynamic=False)
        self._context_project = torch.compile(_project_context, fullgraph=True, dynamic=False)

    def precompute_and_store_context_kv(
        self,
        context_states,
        context_positions,
        context_slot_mapping=None,
    ) -> None:
        context = self._context_norm(context_states)
        indices = None
        for layer in self.layers:
            layer = cast(DFlashQwen3DecoderLayer, layer)
            attention = layer.self_attn
            key, value = self._context_project(attention, context, context_positions)
            if context_slot_mapping is not None:
                attn = attention.attn
                impl = cast(SpyreAttentionImpl, attn.impl)
                if indices is None:
                    indices = impl.kv_write_index(context_slot_mapping.clamp(min=0), key.device)
                impl.do_kv_cache_update(
                    None, key, value, cast(SpyrePagedKVCache, attn.kv_cache), indices
                )


class SpyreDFlashQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = SpyreDFlashQwen3Model
    model: SpyreDFlashQwen3Model

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.model.layers:
            attention = cast(DFlashQwen3DecoderLayer, layer).self_attn
            attention.attn.spyre_causal = attention.causal  # ty: ignore[invalid-assignment]
        self.xpress_head = None
        self.xpress_num_passes = 0
        self.device_to_host_seconds = 0.0
        self.host_argmax_seconds = 0.0
        self.host_to_device_seconds = 0.0
        self.refiner_seconds = 0.0
        self.logits_seconds = 0.0
        self.selection_calls = 0
        self.logits_transfer_bytes = 0
        if hasattr(self.config, "xpress_rank"):
            self.xpress_head = XPressRefinerHead(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.xpress_block_size,
                self.config.xpress_rank,
                self.config.xpress_mlp_hidden,
            )
            passes = vllm_config.additional_config.get("spyre_xpress", {}).get("num_passes")
            self.xpress_num_passes = self.config.xpress_num_passes if passes is None else passes

    def load_weights(self, weights):
        weights = dict(weights)
        validate_dflash_weights(self.config, weights)
        if self.xpress_head is not None:
            self.xpress_head.load_checkpoint_weights(weights)
            weights = {
                name: weight
                for name, weight in weights.items()
                if not name.startswith("xpress_head.")
            }
        elif any(name.startswith("xpress_head.") for name in weights):
            raise ValueError("XPress weights require an XPress head in the checkpoint config")
        super().load_weights(weights.items())

    def prepare_for_spyre(self) -> None:
        self.model.prepare_for_spyre()
        self._combine = torch.compile(self.model.fc, fullgraph=True, dynamic=False)
        for layer in self.model.layers:
            layer.compile(fullgraph=True, dynamic=False)
        head = self.xpress_head
        if head is not None:
            head.prepare_for_spyre()
            self._hidden_cache = torch.compile(head.hidden_cache, fullgraph=True, dynamic=False)
            self._refine = torch.compile(
                lambda base, previous, cache: base + head.refine_bias(previous, cache),
                fullgraph=True,
                dynamic=False,
            )

    def combine_hidden_states(self, hidden_states):
        return self._combine(hidden_states)

    def compute_device_logits(self, hidden_states):
        # TP=1 and the shared full vocabulary are checked before model loading.
        method = cast(SpyreUnquantizedLMHeadMethod, self.lm_head.quant_method)
        return method.apply(self.lm_head, hidden_states)

    def _select_on_cpu(self, logits):
        # aten.argmax falls back on this stack. topk returns inexact FP16
        # indices and has different tie ordering even with FP32 indices.
        started = time.perf_counter()
        # Direct D2H conversion preserves lanes; a materialized FP16->FP32
        # device cast permutes them on the current torch-spyre stack.
        host_logits = logits.to(device="cpu", dtype=torch.float32)
        self.device_to_host_seconds += time.perf_counter() - started
        self.logits_transfer_bytes += host_logits.numel() * host_logits.element_size()
        started = time.perf_counter()
        result = host_logits[:, 1:].argmax(-1)
        self.host_argmax_seconds += time.perf_counter() - started
        self.selection_calls += 1
        return result

    def propose_block(self, hidden_states, anchor_ids):
        started = time.perf_counter()
        base = self.compute_device_logits(hidden_states).unsqueeze(0)
        self.logits_seconds += time.perf_counter() - started
        draft = self._select_on_cpu(base)
        if self.xpress_head is not None and self.xpress_num_passes:
            started = time.perf_counter()
            cache = self._hidden_cache(hidden_states.unsqueeze(0))
            self.refiner_seconds += time.perf_counter() - started
            for _ in range(self.xpress_num_passes):
                # Match the pinned serving PR: the anchor is its own predecessor.
                previous = torch.cat(
                    (anchor_ids[:, None], anchor_ids[:, None], draft[:, :-1]), dim=1
                )
                started = time.perf_counter()
                previous = convert(previous, device=hidden_states.device)
                self.host_to_device_seconds += time.perf_counter() - started
                started = time.perf_counter()
                logits = self._refine(base, previous, cache)
                self.refiner_seconds += time.perf_counter() - started
                draft = self._select_on_cpu(logits)
        return draft.tolist()
