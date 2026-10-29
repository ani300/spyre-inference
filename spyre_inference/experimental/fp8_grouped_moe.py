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

"""FP8-resident expert pools and FP8 grouped GEMMs, with explicit layout staging.

The grouped kernels consume [expert/chunk group, capacity, hidden] inputs.
The token wrapper follows the DL16 chunk equations with explicit compilation
boundaries and a Python launch loop; all tensor routing stays on device.
"""

import torch
import torch.nn.functional as F

from spyre_inference.moe import _routing_weights
from spyre_inference.v1.attention.ops.layout import head_major_kv_layout


def _pack_channel_pool(raw, destination):
    destination.copy_(torch.ops.spyre.qfp8ch(raw))
    return destination


def _pack_weight(raw):
    return torch.ops.spyre.qfp8wt(raw.transpose(-1, -2))


def _expand_selected(pool, ids):
    # Values are still the checkpoint's unscaled FP8 codes. No weight-scale
    # multiplication happens in DL16; compensation follows the FP8 GEMM.
    return pool.index_select(0, ids).to(torch.float16)


def _scaled_grouped_mm(x, weight, weight_scale, scale=None):
    if scale is None:
        scale = torch.ops.spyre.quantscalepertokenfp8(x, 448.0)  # ty: ignore[invalid-argument-type]
    quantized = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
    raw = torch.ops.spyre.scaled_bmm(quantized, weight, out_dtype=torch.float16)  # ty: ignore[invalid-argument-type]
    return raw * (scale * weight_scale)


def _native_grouped_mm(x, pool, ids, weight_scale, scale=None):
    return _scaled_grouped_mm(x, pool.index_select(0, ids), weight_scale, scale)


def _activate(gate, up):
    z = F.gelu(gate, approximate="tanh") * up
    return F.pad(z, (0, -z.shape[-1] % 128))


def _chunk_inputs(x, ids, chunk_scale, offsets):
    # Same e*chunks+c entries as _moe_gathered, with FP8-stick-aligned chunks.
    chunks = offsets.shape[1]
    padded = F.pad(x, (0, -x.shape[-1] % (chunks * 128)))
    hidden = padded.shape[-1]
    top_k = ids.numel()
    wide = ids[:, None].expand(top_k, 64).contiguous().to(torch.float32)
    entries = (wide[:, None, :] * chunk_scale + offsets).reshape(top_k * chunks, 64)
    entries = entries.amax(-1).to(torch.int32)
    activation = padded.reshape(1, 1, chunks, hidden // chunks).expand(
        1, top_k, chunks, hidden // chunks
    )
    return (
        activation.contiguous().reshape(top_k * chunks, 1, hidden // chunks),
        entries,
        torch.ops.spyre.quantscalepertokenfp8(x, 448.0),  # ty: ignore[invalid-argument-type]
    )


def _activate_chunks(gate, up, chunks):
    top_k = gate.shape[0] // chunks
    gate = gate.reshape(top_k, chunks, 1, gate.shape[-1]).sum(1)
    up = up.reshape(top_k, chunks, 1, up.shape[-1]).sum(1)
    z = F.gelu(gate, approximate="tanh") * up
    return (
        z[:, None]
        .expand(top_k, chunks, 1, z.shape[-1])
        .contiguous()
        .reshape(top_k * chunks, 1, z.shape[-1])
    )


def _combine_chunks(parts, weights, hidden):
    values = parts.reshape(1, weights.shape[1], -1)
    return (values * weights[:, :, None]).sum(1)[:, :hidden]


_pack_channel = torch.compile(_pack_channel_pool, fullgraph=True, dynamic=False)
_pack = torch.compile(_pack_weight, fullgraph=True, dynamic=False)
_expand = torch.compile(_expand_selected, fullgraph=True, dynamic=False)
_gemm = torch.compile(_scaled_grouped_mm, fullgraph=True, dynamic=False)
_native = torch.compile(_native_grouped_mm, fullgraph=True, dynamic=False)
_gelu = torch.compile(_activate, fullgraph=True, dynamic=False)
chunk_inputs = torch.compile(_chunk_inputs, fullgraph=True, dynamic=False)
activate_chunks = torch.compile(_activate_chunks, fullgraph=True, dynamic=False)
combine_chunks = torch.compile(_combine_chunks, fullgraph=True, dynamic=False)


def _dispatch_device(x, router_logits, top_k):
    weights, raw_ids = _routing_weights(router_logits, top_k, "full_softmax", torch.float32)
    rows = x.shape[0] * top_k
    # Mirror the existing MoE index path: a complete source stick before the
    # FP32 conversion, then a reduction reads its interleaved lanes whole.
    wide_ids = raw_ids.reshape(-1)[:, None].expand(rows, 64).contiguous().to(torch.float32)
    ids = wide_ids.amax(-1).to(torch.int32)
    grouped = x[:, None, :].expand(x.shape[0], top_k, x.shape[-1]).contiguous()
    return grouped.reshape(rows, 1, x.shape[-1]), ids, weights


def _select_scales(pool, ids):
    return pool.index_select(0, ids)


def _combine_device(grouped, weights):
    values = grouped.reshape(weights.shape[0], weights.shape[1], grouped.shape[-1])
    return (values * weights[:, :, None]).sum(1)


dispatch_device = torch.compile(_dispatch_device, fullgraph=True, dynamic=False)
select_scales = torch.compile(_select_scales, fullgraph=True, dynamic=False)
combine_device = torch.compile(_combine_device, fullgraph=True, dynamic=False)


def load_expert_pool(weight, *, path):
    """Load checkpoint [E, output, input] FP8 codes into an FP8 device pool.

    The temporary DL16 upload contains exact FP8 code values, not dequantized
    weights. Device quantization supplies the packing that generic FP8 H2D lacks.
    No DL16 copy of the expert pool is retained.
    """
    if path not in ("staged", "native"):
        raise ValueError(f"Unknown expert path: {path}")
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 3:
        raise ValueError("Expected checkpoint E4M3 [expert, output, input] weights")
    raw = weight.half().to("spyre")
    if path == "native":
        return _pack(raw)
    e, n, k = weight.shape
    if k % 128:
        raise ValueError("QFP8CH expert rows must be padded to 128 values")
    layout = head_major_kv_layout(e, n, k, torch.float8_e4m3fn)
    destination = torch.empty(  # ty: ignore[no-matching-overload]
        weight.shape, dtype=torch.float8_e4m3fn, device=torch.device("spyre"), device_layout=layout
    )
    return _pack_channel(raw, destination)


def prepare_experts(host, scales, router_scale, chunks=4):
    """Pad and materialize the same expert/chunk ordering as DL16 MoE.

    This is load-time CPU layout preparation, independent of token routing.
    Chunk views are materialized; sharing their storage with persistent pools
    through QFP8WT aliases remains a compiler integration task.
    """
    experts, inter, hidden = host["gate"].shape
    hp = (hidden + chunks * 128 - 1) // (chunks * 128) * chunks * 128
    ip = (inter + 127) // 128 * 128
    padded, padded_scales = {}, {}
    for name in ("gate", "up", "down"):
        pad = (0, hp - hidden, 0, ip - inter) if name != "down" else (0, ip - inter, 0, hp - hidden)
        w = F.pad(host[name].half(), pad).to(torch.float8_e4m3fn)
        s = scales[name].clone()
        if name == "down":
            s = s * router_scale[:, None, None]
        s = F.pad(s, (0, 0, 0, pad[-1]), value=1)
        if name != "down":
            w = w.reshape(experts, ip, chunks, hp // chunks).permute(0, 2, 1, 3).contiguous()
            s = s[:, None].expand(experts, chunks, ip, 1).contiguous()
        else:
            w = w.reshape(experts, chunks, hp // chunks, ip)
            s = s.reshape(experts, chunks, hp // chunks, 1)
        padded[name] = w.reshape(experts * chunks, *w.shape[-2:])
        padded_scales[name] = s.reshape(experts * chunks, -1, 1).transpose(-1, -2).contiguous()
    return padded, padded_scales


def grouped_experts(x, pools, ids, scales, *, path, progress=None, chunks=1, input_scale=None):
    """Three grouped FP8 GEMMs; only activation/scaling arithmetic uses DL16."""

    def project(activation, name, scale=None):
        if progress:
            progress(name + "_" + path)
        if path == "native":
            return _native(activation, pools[name], ids, scales[name], scale)
        if path != "staged":
            raise ValueError(f"Unknown expert path: {path}")
        selected16 = _expand(pools[name], ids)
        if progress:
            progress(name + "_repack")
        weight8 = _pack(selected16)
        if progress:
            progress(name + "_fp8_gemm")
        return _gemm(activation, weight8, scales[name], scale)

    gate = project(x, "gate", input_scale)
    up = project(x, "up", input_scale)
    if progress:
        progress("gelu")
    z = activate_chunks(gate, up, chunks) if chunks > 1 else _gelu(gate, up)
    return project(z, "down")


def _extract_row(x, router_logits, row):
    return x.index_select(0, row), router_logits.index_select(0, row)


def _write_row(output, value, row):
    output.index_copy_(0, row, value)
    return output


_extract = torch.compile(_extract_row, fullgraph=True, dynamic=False)
_write = torch.compile(_write_row, fullgraph=True, dynamic=False)


def moe_tokens(
    x, router_logits, pools, scale_pools, row_ids, *, path, progress=None, observer=None, chunks=4
):
    """Device-only tensor path, launched per token to bound scratch and bypass
    the compiler's multi-token top-k layout problem. Each token still runs
    grouped FP8 GEMMs over its eight selected experts. Python controls launches;
    only the prebuilt row-index metadata originates on the host.
    """
    output = torch.empty_like(x)
    chunk_scale = torch.full((1, 1, 1), float(chunks), dtype=torch.float32).to(x.device)
    offsets = torch.arange(chunks, dtype=torch.float32).reshape(1, chunks, 1).to(x.device)
    for t, row in enumerate(row_ids):
        if progress:
            progress(f"token_{t}_dispatch")
        xr, lr = _extract(x, router_logits, row)
        grouped, ids, routing = dispatch_device(xr, lr, 8)
        if observer:
            observer(t, ids, routing)
        grouped, entries, sx = chunk_inputs(xr, ids, chunk_scale, offsets)
        scales = {name: select_scales(scale, entries) for name, scale in scale_pools.items()}
        result = grouped_experts(
            grouped,
            pools,
            entries,
            scales,
            path=path,
            chunks=chunks,
            input_scale=sx,
            progress=progress if t == 0 else None,
        )
        value = combine_chunks(result, routing, x.shape[-1])
        _write(output, value, row)
    return output
