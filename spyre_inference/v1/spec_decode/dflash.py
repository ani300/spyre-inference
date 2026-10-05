# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU bookkeeping and device-resident DFlash context for the Spyre runner."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer

if TYPE_CHECKING:
    from spyre_inference.models.qwen3_dflash import SpyreDFlashQwen3ForCausalLM


@contextmanager
def use_spyre_dflash_proposer():
    # The pinned runner constructs DFlash directly, with no proposer factory.
    from vllm.v1.worker import gpu_model_runner

    original = gpu_model_runner.DFlashProposer
    gpu_model_runner.DFlashProposer = SpyreDFlashProposer  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        gpu_model_runner.DFlashProposer = original


def accepted_context_length(scheduled: int, drafts: int, sampled: int) -> int:
    """Target input rows whose hidden states belong to the committed prefix.

    The correction/bonus token is sampled from the last valid input row; it
    has no target hidden state yet. Partial prefills commit every input row.
    """
    if drafts == 0:
        return scheduled
    if not 1 <= sampled <= drafts + 1 or scheduled < drafts + 1:
        raise ValueError("Invalid speculative acceptance counts")
    return scheduled - drafts + sampled - 1


class SpyreDFlashProposer(DFlashProposer):
    model: SpyreDFlashQwen3ForCausalLM
    dtype: torch.dtype

    def _raise_if_padded_drafter_batch_disabled(self):
        # propose_spyre handles accepted CPU token lists and pads device inputs.
        pass

    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config, device, runner)
        self.spyre_device = runner._spyre_device
        self.draft_block_size = self.num_speculative_tokens + 1
        self.proposal_seconds = 0.0
        self.proposal_calls = 0
        self.context_tokens = 0
        self.context_seconds = 0.0
        self.draft_forward_seconds = 0.0

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        # Upstream constructs each builder before appending the other layers.
        # Spyre binds KV write holders at construction, so bind the complete group.
        for group in self.draft_attn_groups:
            group.create_metadata_builders(
                self.vllm_config, self.device, kernel_block_size=self.block_size
            )

    def dummy_run(
        self, num_tokens, use_cudagraphs=True, is_graph_capturing=False, slot_mappings=None
    ):
        taps = len(self.dflash_config["target_layer_ids"])
        features = torch.zeros(num_tokens, taps * self.hidden_size, dtype=self.dtype)
        positions = torch.zeros(num_tokens, dtype=torch.int64)
        context = self.model.combine_hidden_states(convert(features, self.spyre_device))
        self.model.precompute_and_store_context_kv(context, convert(positions, self.spyre_device))
        block = self.draft_block_size
        attn_layer.publish_null_slots(block)
        ids = torch.full((block,), self.parallel_drafting_token_id, dtype=torch.int64)
        ids[0] = 0
        with set_forward_context(None, self.vllm_config, num_tokens=block, slot_mapping={}):
            hidden = self.model(
                input_ids=convert(ids, self.spyre_device),
                positions=convert(torch.arange(block), self.spyre_device),
            )
        self.model.propose_block(hidden, ids[:1])

    def propose_spyre(
        self,
        sampled_token_ids,
        aux_hidden_states,
        target_positions,
        spec_decode_metadata,
        common_attn_metadata,
        *,
        skip_proposal: bool = False,
    ):
        """Commit accepted context even when the request needs no further proposals."""
        started = time.perf_counter()
        if len(sampled_token_ids) != 1 or aux_hidden_states is None:
            raise ValueError("Spyre DFlash needs one request and its auxiliary hidden states")
        sampled = sampled_token_ids[0]
        scheduled = target_positions.shape[0]
        drafts = spec_decode_metadata.num_draft_tokens[0] if spec_decode_metadata else 0
        valid = accepted_context_length(scheduled, drafts, len(sampled))

        # Pad positions/slots to the target body bucket. Rejected and padded
        # rows write only the reserved null page, never the request's suffix.
        features = torch.cat(aux_hidden_states, dim=-1)
        rows = features.shape[0]
        positions = F.pad(target_positions[:scheduled], (0, rows - scheduled))
        slots = torch.zeros(rows, dtype=torch.int64)
        slots[:valid] = common_attn_metadata.slot_mapping[:valid]
        context = self.model.combine_hidden_states(features)
        self.model.precompute_and_store_context_kv(
            context, convert(positions, self.spyre_device), slots
        )
        self.context_tokens += valid
        self.context_seconds += time.perf_counter() - started
        if not sampled or skip_proposal:
            return [[]]

        block = self.draft_block_size
        context_end = int(target_positions[valid - 1]) + 1
        positions = torch.arange(context_end, context_end + block, dtype=torch.int64)
        block_table = common_attn_metadata.block_table_tensor[:1]
        slots = block_table[0, positions // self.block_size].long() * self.block_size
        slots += positions % self.block_size
        query_start = torch.tensor([0, block], dtype=torch.int32)
        seq_lens = torch.tensor([context_end + block], dtype=torch.int32)
        metadata = CommonAttentionMetadata(
            query_start_loc=query_start,
            query_start_loc_cpu=query_start,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens,
            num_reqs=1,
            num_actual_tokens=block,
            max_query_len=block,
            max_seq_len=context_end + block,
            block_table_tensor=block_table,
            slot_mapping=slots,
            causal=False,
        )
        _, per_layer = self.build_per_group_and_layer_attn_metadata(metadata)
        ids = torch.full((block,), self.parallel_drafting_token_id, dtype=torch.int64)
        ids[0] = sampled[-1]
        forward_started = time.perf_counter()
        with set_forward_context(
            per_layer,
            self.vllm_config,
            num_tokens=block,
            slot_mapping={name: slots for name in self._draft_attn_layer_names},
        ):
            hidden = self.model(
                input_ids=convert(ids, self.spyre_device),
                positions=convert(positions, self.spyre_device),
            )
        self.draft_forward_seconds += time.perf_counter() - forward_started
        proposals = self.model.propose_block(hidden, ids[:1])
        self.proposal_calls += 1
        self.proposal_seconds += time.perf_counter() - started
        return proposals
