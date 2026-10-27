# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""FP16 vocabulary selection with exact base-512 token-ID payloads."""

import torch
from torch.nn import functional as F


def argmax_parts(scores, low, high, invalid):
    maxima = scores.amax(dim=-1, keepdim=True)
    selected_high = torch.where(scores == maxima, high, invalid).amin(dim=-1, keepdim=True)
    selected_low = torch.where((scores == maxima) & (high == selected_high), low, invalid).amin(
        dim=-1
    )
    return selected_low, selected_high.squeeze(-1)


def _block_argmax_parts(scores, low, high, invalid, block_size):
    scores = scores.view(-1, block_size, scores.shape[-1])[:, 1:]
    return argmax_parts(scores, low, high, invalid)


def _top128(scores, low, high):
    # The optional operators have no typed schema in the upstream torch-spyre pin.
    return (
        torch.ops.spyre.topkvalue(scores, 128, -1),  # ty: ignore[invalid-argument-type]
        torch.ops.spyre.topk_payload(scores, low, 128, -1),  # ty: ignore[invalid-argument-type]
        torch.ops.spyre.topk_payload(scores, high, 128, -1),  # ty: ignore[invalid-argument-type]
    )


def _tile_indices(low, high, tiles):
    return torch.where(high[:, None, :] == tiles[:, :, None], low[:, None, :], -1.0)


def _pad_scores(scores):
    return F.pad(scores, (0, (-scores.shape[-1]) % 512), value=-float("inf"))


def _join_topk(blocks, k):
    return tuple(torch.cat([block[i] for block in blocks], dim=-1)[:, :k] for i in range(3))


def _trim_shortlist(values, low, high, block_size):
    return tuple(
        part.view(-1, block_size, part.shape[-1])[:, 1:].clone() for part in (values, low, high)
    )


class FP16VocabularySelector:
    """Resident argmax and repeated top-128 + mask_by_index selection.

    Requires finite scores and at least k scores above the native mask fill.
    Native top-k prefers later vocabulary positions on ties; argmax preserves
    PyTorch's first-position semantics. Tables are uploaded once per row shape.
    """

    def __init__(self, vocab_size: int):
        if not 128 <= vocab_size <= 512 * 512:
            raise ValueError("FP16 vocabulary selection supports 128 to 262144 tokens")
        self.vocab_size = vocab_size
        self.padded_size = (vocab_size + 511) // 512 * 512
        self._tables = {}
        self._masked_tables = {}
        self._argmax = torch.compile(argmax_parts, fullgraph=True, dynamic=False)
        self._block_argmax = torch.compile(_block_argmax_parts, fullgraph=True, dynamic=False)
        self._top = torch.compile(_top128, fullgraph=True, dynamic=False)
        self._indices = torch.compile(_tile_indices, fullgraph=True, dynamic=False)
        self._mask = torch.compile(torch.ops.spyre.mask_by_index, fullgraph=True, dynamic=False)
        self._pad = torch.compile(_pad_scores, fullgraph=True, dynamic=False)
        self._join = torch.compile(_join_topk, fullgraph=True, dynamic=False)
        self._trim = torch.compile(_trim_shortlist, fullgraph=True, dynamic=False)

    def pack(self, shortlist, block_size):
        from torch_spyre._C import SpyreTensorLayout

        parts = []
        for part in shortlist:
            layout = SpyreTensorLayout(list(part.shape), list(part.stride()), part.dtype, [0, 1])
            parts.append(part.to(device=part.device, device_layout=layout))
        values, low, high = parts
        return self._trim(values, low, high, block_size)

    def prepare(self, rows, device):
        from torch_spyre._C import SpyreTensorLayout

        key = (rows, device)
        if key not in self._tables:

            def upload(tensor):
                tensor = tensor.contiguous()
                layout = SpyreTensorLayout(
                    list(tensor.shape), list(tensor.stride()), tensor.dtype, [1, 0]
                )
                return tensor.to(device, device_layout=layout)

            positions = torch.arange(self.padded_size, device="cpu")
            low = (positions % 512).half()
            high = (positions // 512).half()
            low_table, high_table = (upload(part[None, :].expand(rows, -1)) for part in (low, high))
            tiles = (
                torch.arange(self.padded_size // 512, device="cpu").half()[None, :].expand(rows, -1)
            )
            self._tables[key] = (
                low[: self.vocab_size].to(device),
                high[: self.vocab_size].to(device),
                torch.tensor([512.0], dtype=torch.float16, device="cpu").to(device),
                low_table,
                high_table,
                upload(tiles),
            )
        return self._tables[key]

    def __call__(self, scores, k, block_size=None):
        if scores.ndim != 2 or scores.shape[-1] != self.vocab_size:
            raise ValueError("Expected one full vocabulary row per draft position")
        if scores.dtype != torch.float16 or scores.device.type != "spyre":
            raise ValueError("Expected resident FP16 Spyre scores")
        if not 0 <= k <= self.vocab_size:
            raise ValueError("k must be between zero and the vocabulary size")
        rows = scores.shape[0]
        low, high, invalid, low_table, high_table, tiles = self.prepare(rows, scores.device)
        greedy = (
            self._argmax(scores, low, high, invalid)
            if block_size is None
            else self._block_argmax(scores, low, high, invalid, block_size)
        )
        if not k:
            return greedy, None
        work = self._pad(scores) if self.padded_size != self.vocab_size else scores
        # Repack while still 2-D: the mask cannot move a stick through the
        # subsequent vocabulary-to-tile view. This is a device-to-device copy.
        work = work.to(device=work.device, device_layout=low_table.device_tensor_layout())
        positions = low_table.view(rows, -1, 512)
        selected_low, selected_high = low_table, high_table
        blocks = []
        for offset in range(0, k, 128):
            block = self._top(work, selected_low, selected_high)
            blocks.append(block)
            if offset + 128 < k:
                indices = self._indices(block[1], block[2], tiles)
                work = self._mask(work.view(rows, -1, 512), indices, positions, -1).view(
                    rows, self.padded_size
                )
                # Masking changes the score layout. Cache matching ID tables so
                # later rounds stream all three inputs in the same order.
                layout = work.device_tensor_layout()
                key = (rows, work.device, tuple(layout.device_size), tuple(layout.stride_map))
                if key not in self._masked_tables:
                    self._masked_tables[key] = tuple(
                        part.to(device=work.device, device_layout=layout)
                        for part in (low_table, high_table)
                    )
                selected_low, selected_high = self._masked_tables[key]
        return greedy, self._join(blocks, k)
