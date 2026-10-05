# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class XPressRefinerHead(nn.Module):
    """Causal block refiner, with the raw checkpoint mixer folded at load time."""

    _PUBLISHED_KEYS = {
        "w1.weight": "w1.weight",
        "down_h.weight": "down_h.weight",
        "down_g.weight": "down_g.weight",
        "in_proj.weight": "in_proj.weight",
        "mix.L": "mix_L",
        "mlp.gate_proj.weight": "mlp_gate.weight",
        "mlp.up_proj.weight": "mlp_up.weight",
        "mlp.down_proj.weight": "mlp_down.weight",
        "w2.weight": "w2.weight",
    }
    _TRAINING_KEYS = {
        "token_embed.weight": "w1.weight",
        "down_h.weight": "down_h.weight",
        "down_g.weight": "down_g.weight",
        "in_proj.weight": "in_proj.weight",
        "mix_l": "mix_L",
        "mlp_gate.weight": "mlp_gate.weight",
        "mlp_up.weight": "mlp_up.weight",
        "mlp_down.weight": "mlp_down.weight",
        "readout.weight": "w2.weight",
    }

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        block_size: int,
        rank: int = 256,
        mlp_hidden: int = 512,
        topc: int = 512,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_size, rank, mlp_hidden) <= 0 or block_size < 2:
            raise ValueError(
                "XPress requires positive dimensions and a block of at least two tokens"
            )
        self.block_size = block_size
        self.rank = rank
        if type(topc) is not int or topc < 0:
            raise ValueError("topc must be a nonnegative integer")
        self.topc = topc
        self.w1 = nn.Embedding(vocab_size, rank)
        self.down_h = nn.Linear(hidden_size, rank, bias=False)
        self.down_g = nn.Linear(hidden_size, rank, bias=False)
        self.in_proj = nn.Linear(3 * rank, rank, bias=False)
        self.mix_L = nn.Parameter(torch.eye(block_size).expand(rank, -1, -1).clone())
        self.mlp_gate = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, rank, bias=False)
        self.w2 = nn.Linear(rank, vocab_size, bias=False)
        self.register_buffer("_mix_kjc", None, persistent=False)
        self.register_buffer("_readout_t", None, persistent=False)
        self.register_buffer("_hidden_proj_t", None, persistent=False)
        self.register_buffer("_token_proj_t", None, persistent=False)

    def prepare_for_spyre(self) -> None:
        """Build device layouts after the checkpoint is loaded in its serving dtype."""
        from spyre_inference.custom_ops.utils import convert, row_outermost_layout

        device = self.mix_L.device
        weight = self.w1.weight.detach().cpu()
        self.w1.weight = nn.Parameter(
            convert(weight, device, device_layout=row_outermost_layout(weight.shape, weight.dtype)),
            requires_grad=False,
        )
        self._mix_kjc = self.mix_L.detach().cpu().permute(1, 2, 0).contiguous().to(device)
        projection = self.in_proj.weight.detach().cpu()
        self._hidden_proj_t = projection[:, : 2 * self.rank].t().contiguous().to(device)
        self._token_proj_t = projection[:, 2 * self.rank :].t().contiguous().to(device)
        weight = self.w2.weight.detach().cpu()
        self.w2.weight = nn.Parameter(
            convert(weight, device, device_layout=row_outermost_layout(weight.shape, weight.dtype)),
            requires_grad=False,
        )
        # Match the target LM head's output tiling; trim only full sticks.
        padded = F.pad(weight, (0, 0, 0, (-weight.shape[0]) % 2048))
        self._readout_t = padded.t().contiguous().to(device)

    def load_checkpoint_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Load either published XPress or Speculators training head weights strictly.

        ``state_dict`` uses the folded serving representation. Loading a serving
        state_dict directly therefore does not apply the fold a second time.
        """
        published = any(name.startswith("xpress_head.") for name in weights)
        training = any(name.startswith("refiner_head.") for name in weights)
        if published == training:
            raise ValueError("Expected exactly one XPress checkpoint head representation")
        prefix, mapping = (
            ("xpress_head.", self._PUBLISHED_KEYS)
            if published
            else ("refiner_head.", self._TRAINING_KEYS)
        )
        names = {name[len(prefix) :] for name in weights if name.startswith(prefix)}
        if names != mapping.keys():
            raise ValueError(
                f"Invalid XPress head weights: missing={sorted(mapping.keys() - names)}, "
                f"unexpected={sorted(names - mapping.keys())}"
            )
        state = {dest: weights[prefix + source] for source, dest in mapping.items()}
        raw = state["mix_L"].to(dtype=self.mix_L.dtype)
        state["mix_L"] = raw.tril() + torch.eye(self.block_size, dtype=raw.dtype, device=raw.device)
        self.load_state_dict(state, strict=True)

    def hidden_cache(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, block, _ = hidden_states.shape
        local = self.down_h(hidden_states.flatten(0, 1)).view(batch, block, self.rank)
        summary = self.down_g(hidden_states.mean(dim=1))[:, None, :].expand_as(local)
        return torch.cat((local, summary), dim=-1)

    def refine_bias(self, previous_ids: torch.Tensor, hidden_cache: torch.Tensor) -> torch.Tensor:
        return self._refine_embeddings(self.w1(previous_ids), hidden_cache)

    def _refine_embeddings(self, previous_embeddings, hidden_cache):
        x = self.in_proj(torch.cat((hidden_cache, previous_embeddings), dim=-1))
        x = self._mix_and_mlp(x)
        if self._readout_t is not None:
            return (x @ self._readout_t)[..., : self.w2.out_features]
        return self.w2(x)

    def _mix_and_mlp(self, x):
        # Keep rank as the contiguous dimension for Spyre's stick layout.
        mixer = self.mix_L.permute(1, 2, 0) if self._mix_kjc is None else self._mix_kjc
        x = (x[:, None, :, :] * mixer[None, :, :, :]).sum(dim=2)
        return x + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))

    def project_hidden_cache(self, hidden_states):
        cache = self.hidden_cache(hidden_states)
        if self._hidden_proj_t is not None:
            return cache @ self._hidden_proj_t
        return F.linear(cache, self.in_proj.weight[:, : 2 * self.rank])

    def refine_latent(self, previous_ids, projected_cache):
        embeddings = self.w1(previous_ids)
        token_projection = (
            embeddings @ self._token_proj_t
            if self._token_proj_t is not None
            else F.linear(embeddings, self.in_proj.weight[:, 2 * self.rank :])
        )
        return self._mix_and_mlp(projected_cache + token_projection)[:, 1:]

    def gather_readout(self, candidates):
        return F.embedding(candidates, self.w2.weight)

    def refine_full(self, base, previous_ids, projected_cache):
        latent = self.refine_latent(previous_ids, projected_cache)
        if self._readout_t is not None:
            bias = (latent @ self._readout_t)[..., : self.w2.out_features]
        else:
            bias = self.w2(latent)
        return base + bias

    def refine_candidates(self, base, previous_ids, projected_cache, readout):
        latent = self.refine_latent(previous_ids, projected_cache)
        batch, slots, candidates = base.shape
        bias = torch.bmm(
            readout.reshape(batch * slots, candidates, self.rank),
            latent.reshape(batch * slots, self.rank, 1),
        ).view_as(base)
        return base + bias

    def forward(
        self,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_ids: torch.Tensor,
        predecessor_ids: torch.Tensor,
        num_passes: int = 6,
    ) -> torch.Tensor:
        """Return the B-1 greedy proposals while keeping slot zero fixed as the anchor."""
        if base_logits.shape[1] != self.block_size or hidden_states.shape[1] != self.block_size:
            raise ValueError("XPress input block size must match the checkpoint's learned mixer")
        if num_passes < 0:
            raise ValueError("The number of refinement passes cannot be negative")
        base = base_logits[:, 1:]
        draft = base.argmax(dim=-1)
        if not num_passes:
            return draft
        cache = self.project_hidden_cache(hidden_states)
        candidates = readout = None
        if self.topc:
            base, candidates = base.topk(min(self.topc, base.shape[-1]), dim=-1)
            readout = self.gather_readout(candidates)
        for _ in range(num_passes):
            previous = torch.cat(
                (predecessor_ids[:, None], anchor_ids[:, None], draft[:, :-1]), dim=1
            )
            if candidates is None:
                draft = self.refine_full(base, previous, cache).argmax(dim=-1)
            else:
                logits = self.refine_candidates(base, previous, cache, readout)
                draft = candidates.gather(-1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)
        return draft
