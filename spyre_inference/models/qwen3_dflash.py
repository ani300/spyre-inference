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


def _select_candidate_parts(scores, candidate_parts, order):
    maxima = scores.amax(dim=-1, keepdim=True)
    first = torch.where(scores == maxima, order, scores.shape[-1]).amin(dim=-1, keepdim=True)
    # Two base-512 digits preserve vocabulary IDs without widening the mask,
    # whose FP32 layout cannot be combined with a host-uploaded ID table.
    low = torch.where(order == first, candidate_parts[0], 0.0).sum(dim=-1)
    high = torch.where(order == first, candidate_parts[1], 0.0).sum(dim=-1)
    return low, high


def _assemble_feedback(low, high, previous):
    draft = (low.float() + high.float() * 512).to(torch.int32)
    # Whole int32 sticks let concat write token rows without an offset-two
    # mutation inside a stick. Keep this layout from the initial upload.
    batch, slots = draft.shape
    tail = draft.unsqueeze(-1).expand(-1, -1, 32)[:, :-1]
    anchors = previous.view(batch, slots + 1, 32)[:, :2]
    previous = torch.cat((anchors, tail), dim=1).reshape(-1, 32)
    return draft, previous


def _project_context(attention, context, positions):
    kv = context @ attention._context_kv_weight_t
    key, value = kv.split(attention.kv_size, dim=-1)
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
        for layer in self.layers:
            attention = cast(DFlashQwen3DecoderLayer, layer).self_attn
            # The live unquantized weight is [input, Q|K|V] after Spyre's
            # loading transform. Context needs only K/V; pack it once on load.
            weight_t = cast(torch.Tensor, attention.qkv_proj.weight)
            attention.register_buffer(
                "_context_kv_weight_t",
                weight_t.detach().cpu()[:, attention.q_size :].contiguous().to(weight_t.device),
                persistent=False,
            )
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
        self.xpress_topc = 0
        self.device_to_host_seconds = 0.0
        self.host_argmax_seconds = 0.0
        self.host_topk_seconds = 0.0
        self.host_to_device_seconds = 0.0
        self.candidate_gather_seconds = 0.0
        self.candidate_transfer_bytes = 0
        self.refiner_seconds = 0.0
        self.logits_seconds = 0.0
        self.selection_calls = 0
        self.logits_transfer_bytes = 0
        self.device_selection_calls = 0
        self.device_selection_seconds = 0.0
        self.proposal_id_transfer_bytes = 0
        self._device_feedback_enabled = False
        self.register_buffer("_feedback_order", None, persistent=False)
        if hasattr(self.config, "xpress_rank"):
            self.xpress_topc = getattr(self.config, "xpress_topc", 512)
            self.xpress_head = XPressRefinerHead(
                self.config.vocab_size,
                self.config.hidden_size,
                self.config.xpress_block_size,
                self.config.xpress_rank,
                self.config.xpress_mlp_hidden,
                topc=self.xpress_topc,
            )
            self.xpress_num_passes = getattr(self.config, "xpress_num_passes", 6)

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
            self._hidden_cache = torch.compile(
                head.project_hidden_cache, fullgraph=True, dynamic=False
            )
            self._refine = torch.compile(head.refine_full, fullgraph=True, dynamic=False)
            self._gather_readout = torch.compile(head.gather_readout, fullgraph=True, dynamic=False)
            self._refine_candidates = torch.compile(
                head.refine_candidates, fullgraph=True, dynamic=False
            )
            self._device_feedback_enabled = (
                head.block_size == 16
                and self.xpress_topc == 512
                and 512 <= self.config.vocab_size <= 512 * 512
            )
            if self._device_feedback_enabled:
                self._feedback_order = torch.arange(512, dtype=torch.float16).to(
                    head.w1.weight.device
                )
                # Fusing digit reduction with feedback assembly can alias the
                # low/high temporaries on batched refiner output layouts.
                self._select_candidate_parts = torch.compile(
                    _select_candidate_parts, fullgraph=True, dynamic=False
                )
                self._assemble_feedback = torch.compile(
                    _assemble_feedback, fullgraph=True, dynamic=False
                )

    def _select_with_feedback(self, scores, parts, order, previous):
        low, high = self._select_candidate_parts(scores, parts, order)
        return self._assemble_feedback(low, high, previous)

    def combine_hidden_states(self, hidden_states):
        return self._combine(hidden_states)

    def compute_device_logits(self, hidden_states):
        # TP=1 and the shared full vocabulary are checked before model loading.
        method = cast(SpyreUnquantizedLMHeadMethod, self.lm_head.quant_method)
        return method.apply(self.lm_head, hidden_states)

    def _copy_logits_to_cpu(self, logits):
        # aten.argmax falls back on this stack. topk returns inexact FP16
        # indices and has different tie ordering even with FP32 indices.
        started = time.perf_counter()
        # Direct D2H conversion preserves lanes; a materialized FP16->FP32
        # device cast permutes them on the current torch-spyre stack.
        host_logits = logits.to(device="cpu", dtype=torch.float32)
        self.device_to_host_seconds += time.perf_counter() - started
        self.logits_transfer_bytes += host_logits.numel() * host_logits.element_size()
        return host_logits

    def _argmax_on_cpu(self, host_logits, candidates=None):
        started = time.perf_counter()
        result = host_logits.argmax(-1)
        if candidates is not None:
            result = candidates.gather(-1, result.unsqueeze(-1)).squeeze(-1)
        self.host_argmax_seconds += time.perf_counter() - started
        self.selection_calls += 1
        return result

    def propose_block(self, hidden_states, anchor_ids):
        batch = anchor_ids.numel()
        hidden_blocks = hidden_states.view(batch, -1, hidden_states.shape[-1])
        started = time.perf_counter()
        base = self.compute_device_logits(hidden_states).view(batch, hidden_blocks.shape[1], -1)
        self.logits_seconds += time.perf_counter() - started
        host_base = self._copy_logits_to_cpu(base)[:, 1:]
        draft = self._argmax_on_cpu(host_base)
        device_feedback = False
        if self.xpress_head is not None and self.xpress_num_passes:
            started = time.perf_counter()
            cache = self._hidden_cache(hidden_blocks)
            self.refiner_seconds += time.perf_counter() - started
            candidates = readout = None
            base = base[:, 1:]
            if self.xpress_topc:
                started = time.perf_counter()
                base_c, candidates = host_base.topk(
                    min(self.xpress_topc, host_base.shape[-1]), dim=-1
                )
                self.host_topk_seconds += time.perf_counter() - started
                started = time.perf_counter()
                # A flat index avoids Spyre gather page splits across request axes.
                candidate_ids = convert(candidates.flatten(), device=hidden_states.device)
                base = convert(base_c, device=hidden_states.device, dtype=hidden_states.dtype)
                self.host_to_device_seconds += time.perf_counter() - started
                self.candidate_transfer_bytes += candidates.numel() * 4 + base.numel() * 2
                started = time.perf_counter()
                readout = self._gather_readout(candidate_ids)
                self.candidate_gather_seconds += time.perf_counter() - started
            device_feedback = getattr(self, "_device_feedback_enabled", False) and (
                self.xpress_topc == 512
            )
            if device_feedback:
                assert candidates is not None
                started = time.perf_counter()
                # Separate buffers avoid offset views into a stacked ID table.
                parts = tuple(
                    convert(part, hidden_states.device, dtype=hidden_states.dtype)
                    for part in (candidates % 512, candidates // 512)
                )
                previous = torch.cat(
                    (anchor_ids[:, None], anchor_ids[:, None], draft[:, :-1]), dim=1
                )
                feedback = convert(
                    previous.reshape(-1, 1).expand(-1, 32).contiguous(),
                    hidden_states.device,
                    dtype=torch.int32,
                )
                self.host_to_device_seconds += time.perf_counter() - started
                self.candidate_transfer_bytes += sum(p.numel() * p.element_size() for p in parts)
            for _ in range(self.xpress_num_passes):
                # Match the pinned serving PR: the anchor is its own predecessor.
                if device_feedback:
                    previous = feedback[:, 0].view(batch, -1)
                else:
                    previous = torch.cat(
                        (anchor_ids[:, None], anchor_ids[:, None], draft[:, :-1]), dim=1
                    )
                    started = time.perf_counter()
                    previous = convert(previous, device=hidden_states.device)
                    self.host_to_device_seconds += time.perf_counter() - started
                started = time.perf_counter()
                logits = (
                    self._refine(base, previous, cache)
                    if candidates is None
                    else self._refine_candidates(base, previous, cache, readout)
                )
                self.refiner_seconds += time.perf_counter() - started
                if device_feedback:
                    started = time.perf_counter()
                    draft, feedback = self._select_with_feedback(
                        logits, parts, self._feedback_order, feedback
                    )
                    self.device_selection_seconds += time.perf_counter() - started
                    self.device_selection_calls += 1
                else:
                    draft = self._argmax_on_cpu(self._copy_logits_to_cpu(logits), candidates)
        if device_feedback:
            started = time.perf_counter()
            draft = draft.cpu()
            self.device_to_host_seconds += time.perf_counter() - started
            self.proposal_id_transfer_bytes += draft.numel() * draft.element_size()
        return draft.tolist()
