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

"""Experimental FP8 versions of DL16 gathered and persistent expert kernels.

The engine bridge uses the persistent path. Gathered and sharded paths remain
standalone probes for compiler bring-up and comparison.
"""

import torch
import torch.nn.functional as F

from spyre_inference.experimental.fp8_grouped_moe import (
    _activate_chunks,
    _chunk_inputs,
    _combine_chunks,
    _scaled_grouped_mm,
)
from spyre_inference.moe import (
    _compiler_scopes,
    _moe_persistent_routing,
    _name_persistent_dims,
    _probs,
    _reset_named_dims,
    _routing_weights,
)


def _gathered_row(x, logits, pools, scales, chunk_scale, offsets):
    routing, raw_ids = _routing_weights(logits, 8, "full_softmax", torch.float32)
    inputs, entries, sx = _chunk_inputs(x, raw_ids.reshape(-1), chunk_scale, offsets)
    gate = _scaled_grouped_mm(
        inputs, pools["gate"].index_select(0, entries), scales["gate"].index_select(0, entries), sx
    )
    up = _scaled_grouped_mm(
        inputs, pools["up"].index_select(0, entries), scales["up"].index_select(0, entries), sx
    )
    activated = _activate_chunks(gate, up, offsets.shape[1])
    parts = _scaled_grouped_mm(
        activated, pools["down"].index_select(0, entries), scales["down"].index_select(0, entries)
    )
    return _combine_chunks(parts, routing, x.shape[-1])


def _gathered_tokens(x, logits, pools, scales, chunk_scale, offsets):
    rows = [
        _gathered_row(x[t : t + 1], logits[t : t + 1], pools, scales, chunk_scale, offsets)
        for t in range(x.shape[0])
    ]
    return torch.cat(rows)


def _route(probs, identity):
    return _moe_persistent_routing(probs, identity, 8, 64)


def _persistent(x, route, gate, up, down, gate_scale, up_scale, down_scale):
    from torch_spyre._inductor.propagate_hints import spyre_hint
    from torch_spyre._inductor.wsr import for_each_tile

    with spyre_hint(named_dims=["E", "T", "ONE"]):
        route = route.permute(1, 0, 2).contiguous().clone()

    def expert_body(acc, tiles):
        x, r, g, u, d, sg, su, sd = tiles
        xb = x.unsqueeze(0)
        sx = torch.ops.spyre.quantscalepertokenfp8(xb, 448.0)  # ty: ignore[invalid-argument-type]
        gv = _scaled_grouped_mm(xb, g, sg, sx)
        uv = _scaled_grouped_mm(xb, u, su, sx)
        z = F.gelu(gv, approximate="tanh") * uv
        return acc + (_scaled_grouped_mm(z, d, sd) * r).squeeze(0), None

    result, _ = for_each_tile(
        expert_body,
        (x, route, gate, up, down, gate_scale, up_scale, down_scale),
        dims=(None, 0, 0, 0, 0, 0, 0, 0),
        tile_size=1,
        init=torch.zeros_like(x),
    )
    return result


gathered = torch.compile(_gathered_tokens, fullgraph=True, dynamic=False)
probabilities = torch.compile(_probs, fullgraph=True, dynamic=False)
route_tokens = torch.compile(_route, fullgraph=True, dynamic=False)
persistent = torch.compile(_persistent, fullgraph=True, dynamic=False)


def native_moe(x, logits, pools, scales, *, path, chunks, progress, route_observer=None):
    moe_scope, persistent_scope = _compiler_scopes()
    if path == "native":
        chunk_scale = torch.full((1, 1, 1), float(chunks), dtype=torch.float32).to(x.device)
        offsets = torch.arange(chunks, dtype=torch.float32).reshape(1, chunks, 1).to(x.device)
        progress("native_gathered")
        with moe_scope:
            return gathered(x, logits, pools, scales, chunk_scale, offsets)
    identity = torch.eye(64, dtype=torch.float16).to(x.device)
    with moe_scope:
        progress("persistent_routing")
        probs = probabilities(logits, torch.float32)
        route = route_tokens(probs, identity)[:, : pools["gate"].shape[0]]
        if route_observer is not None:
            route_observer(route)
        return run_persistent(x, route, pools, scales, progress)


def run_persistent(x, route, pools, scales, progress):
    _, persistent_scope = _compiler_scopes()
    weights = [pools[p] for p in ("gate", "up", "down")]
    _name_persistent_dims(x, *weights)
    from torch_spyre._inductor.wsr.propagate_named_dims import name_tensor_dims

    for p in ("gate", "up", "down"):
        name_tensor_dims(scales[p], ["E", "ONE", "H" if p == "down" else "M"])
    try:
        progress("persistent_fp8_experts")
        with persistent_scope:
            return persistent(x, route, *weights, *(scales[p] for p in ("gate", "up", "down")))
    finally:
        _reset_named_dims()


add_outputs = torch.compile(torch.add, fullgraph=True, dynamic=False)


def persistent_shards(x, logits, pool_shards, scale_shards, *, progress, route_observer=None):
    """Persistent FP8 prefill with statically partitioned resident expert pools.

    Routing is global across all 128 experts. The four load-time pool partitions
    avoid bulk QFP8WT packing's per-core span limit; no tokens move to the host.
    """
    moe_scope, _ = _compiler_scopes()
    identity = torch.eye(64, dtype=torch.float16).to(x.device)
    with moe_scope:
        progress("persistent_sharded_routing")
        route = route_tokens(probabilities(logits, torch.float32), identity)
        if route_observer is not None:
            route_observer(route)
        output = None
        offset = 0
        for i, (pools, scales) in enumerate(zip(pool_shards, scale_shards, strict=True)):
            experts = pools["gate"].shape[0]
            progress(f"persistent_shard_{i}")
            value = run_persistent(x, route[:, offset : offset + experts], pools, scales, progress)
            output = value if output is None else add_outputs(output, value)
            offset += experts
        return output
