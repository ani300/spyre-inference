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

"""Per-sequence paged attention over the KV cache."""

import torch
from torch_spyre._inductor.wsr import for_each_tile


def page_attn_kernel(
    query,
    query_row_index,
    k_pages,
    v_pages,
    page_index_table,
    mask_tiles,
    scale,
    num_blocks,
    padded_query_len,
    num_heads,
    num_kv_heads,
    head_size,
    logits_soft_cap=0.0,
    alibi_bias_tiles=None,
    out=None,
):
    """Online softmax attention over ``num_blocks`` KV pages.

    The page walk is a ``for_each_tile`` reduction. Keeping one page's work in a
    loop body makes compile cost independent of ``num_blocks``. Each iteration
    reads one page-table row and selects the physical K/V page without first
    materializing the sequence's whole cache.

    Expected shapes:
        query: [num_tokens, num_heads, head_size], the whole batch's query
        query_row_index: [padded_query_len] int32 device tensor of this
            sequence's absolute query rows.
        k_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        v_pages: [num_blocks_total, block_size, num_kv_heads, head_size]
        page_index_table: [num_blocks, INT32_ELEMS_PER_STICK] int32 device
            tensor, row i holding the i-th active block's page index at
            column 0.
        mask_tiles: [num_blocks]
        alibi_bias_tiles: list of [num_kv_heads, num_queries_per_kv, 1, block_size],
            or None for no ALiBi. The query-axis dim is 1 because softmax absorbs
            per-query-row constants — see the derivation at the bias-tile
            construction site in _online_softmax_attention.
        out: buffer to store into, or None to return the result instead.

    Returns [padded_query_len, num_heads, head_size], or ``out`` when this
    kernel stored the result itself.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    # A compiled region reads a view from offset 0, ignoring storage_offset
    # (torch-spyre#3770), so the rows are gathered here rather than sliced outside.
    q_rows = query.index_select(0, query_row_index)
    q = (
        q_rows.unsqueeze(0)
        .transpose(1, 2)
        .reshape(num_kv_heads, num_queries_per_kv, padded_query_len, head_size)
    )

    def page_body(carry, tiles):
        tile_max, tile_sum, tile_output = carry
        table_row, mask_tile, q_whole, k_whole, v_whole, *bias_tiles = tiles
        # A tiled 1-D index is read as element zero on every device iteration.
        # A stick-wide row instead gives this body an offset-zero [1, 32]
        # tensor, from which index_select can read the current page number.
        page_idx = table_row[0, 0:1]
        k_page = k_whole.index_select(0, page_idx)
        v_page = v_whole.index_select(0, page_idx)
        # Token-major page to head-major for the matmuls; permutes on device.
        k_page_4d = k_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)
        v_page_4d = v_page.squeeze(0).permute(1, 0, 2).unsqueeze(1)

        scores = torch.matmul(q_whole, k_page_4d.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Pull logits into (-cap, +cap) before the mask add so masked
            # positions still map cleanly to -inf. Applied before the ALiBi
            # bias so the positional term is not squashed by the tanh.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        if alibi_bias_tiles is not None:
            # ALiBi bias slope[h] * (kv_pos - context_len). The additive
            # mask_tile below uses finfo.min for masked positions, so this
            # bias cannot un-mask them.
            scores = scores + bias_tiles[0].squeeze(0)
        scores = scores + mask_tile.squeeze(0)
        scores_max = torch.amax(scores, dim=-1, keepdim=True)
        new_max = torch.maximum(tile_max, scores_max)
        rescale = torch.exp(tile_max - new_max)
        tile_probs = torch.exp(scores - new_max)
        new_output = tile_output * rescale + torch.matmul(tile_probs, v_page_4d)
        new_sum = tile_sum * rescale + tile_probs.sum(dim=-1, keepdim=True)
        return (new_max, new_sum, new_output), None

    mask = torch.stack(mask_tiles)
    loop_operands = (page_index_table, mask, q, k_pages, v_pages)
    loop_dims = (0, 0, None, None, None)
    if alibi_bias_tiles is not None:
        loop_operands += (torch.stack(alibi_bias_tiles),)
        loop_dims += (0,)

    acc_shape = (*q.shape[:-1], 1)
    init = (
        torch.full(acc_shape, float("-inf"), device=q.device, dtype=q.dtype),
        torch.zeros(acc_shape, device=q.device, dtype=q.dtype),
        torch.zeros_like(q),
    )
    (_, tile_sum, tile_output), _ = for_each_tile(
        page_body,
        loop_operands,
        dims=loop_dims,
        tile_size=1,
        init=init,
    )

    attn = tile_output / tile_sum
    attn = attn.reshape(1, num_heads, padded_query_len, head_size).transpose(1, 2)
    attn = attn.reshape(padded_query_len, num_heads, head_size)
    if out is not None:
        # `out` and `query` are both indexed by absolute token row. Storing the
        # full padded extent keeps the sequence's real query_len out of the
        # arguments, so it is not specialized on; rows past it duplicate the
        # sequence's last row, so index_copy_'s undefined write order for
        # duplicate indices is harmless.
        out.index_copy_(0, query_row_index, attn)
        return out
    return attn
