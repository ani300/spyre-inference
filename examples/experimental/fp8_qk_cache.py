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

"""Reproduce staged and native packed-cache FP8 attention on one Spyre device.

Every failure writes a JSON result and exits nonzero, including expected native
compiler failures. No CPU fallback or automatic substitution of the staged path.
"""

import argparse
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import torch
import torch_spyre  # noqa: F401
from torch_spyre.ops.fallbacks import FallbackWarning

from spyre_inference.v1.attention.ops.fp8_qk_cache import (
    allocate_fp8_qk_cache,
    pack_fp8_key,
    packed_fp8_attention,
    staged_fp8_attention,
    store_packed_kv_page_compiled,
)
from spyre_inference.v1.attention.ops.reshape_and_cache_head_major import (
    reshape_and_cache_head_major_kernel,
)


def relative_error(actual, expected):
    difference = actual - expected
    return {
        "finite": bool(actual.isfinite().all()),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-12)),
    }


def run(args, result):
    warnings.filterwarnings("error", category=FallbackWarning)
    torch.manual_seed(29)
    b, h, g, m, n, d = args.batch, 8, 4, args.query_len, 128, 128
    ks, vs = 0.02, 0.015
    q = torch.randn(b, m, h * g, d, dtype=torch.float16)
    k = torch.randn(b, h, n, d, dtype=torch.float16)
    v = torch.randn_like(k)
    page_ids_cpu = torch.arange(b - 1, -1, -1, dtype=torch.int32)
    page_ids = page_ids_cpu.to("spyre")
    result["stage"] = "allocate"
    k_pages, v_pages = allocate_fp8_qk_cache(b, h, n, d, "spyre", path=args.path)
    result["allocated_k_layout"] = str(k_pages.device_tensor_layout())
    result["allocated_v_layout"] = str(v_pages.device_tensor_layout())
    print(json.dumps(result), flush=True)
    store_staged = torch.compile(reshape_and_cache_head_major_kernel, fullgraph=True)

    if args.operation != "full" or args.path == "staged":
        result["stage"] = "seed_production_cache"
        seed_k = k_pages
        if args.path == "packed":
            seed_k, _ = allocate_fp8_qk_cache(b, h, n, d, "spyre", path="staged")
        rows = [
            (page_ids_cpu[:, None] * h * n + head * n + torch.arange(n)[None])
            .flatten()
            .to(dtype=torch.int32)
            .to("spyre")
            for head in range(h)
        ]
        store_staged(
            k.transpose(1, 2).reshape(b * n, h, d).contiguous().to("spyre"),
            v.transpose(1, 2).reshape(b * n, h, d).contiguous().to("spyre"),
            seed_k.view(-1, d),
            v_pages.view(-1, d),
            rows,
            ks,
            vs,
        )
        if args.path == "packed":
            result["stage"] = "seed_packed_cache"
            # Bypass only the writer, to isolate read/update support. Pack on device.
            physical_k = k.flip(0).transpose(-1, -2).contiguous().to("spyre")
            k_pages = torch.compile(pack_fp8_key, fullgraph=True)(physical_k, ks)
            result["seeded_k_layout"] = str(k_pages.device_tensor_layout())
    else:
        result["stage"] = "packed_prefill_store"
        offsets = torch.arange(n, dtype=torch.int32).to("spyre")
        for seq, page in enumerate(page_ids_cpu.tolist()):
            store_packed_kv_page_compiled(
                k[seq].transpose(0, 1).contiguous().to("spyre"),
                v[seq].transpose(0, 1).contiguous().to("spyre"),
                k_pages,
                v_pages,
                page,
                offsets,
                ks,
                vs,
            )

    if args.operation == "update":
        for offset in (63, 64, 127):
            result["stage"] = f"{args.path}_single_token_store_{offset}"
            new_k = torch.randn(b, h, d, dtype=torch.float16)
            new_v = torch.randn_like(new_k)
            k[:, :, offset] = new_k
            v[:, :, offset] = new_v
            if args.path == "staged":
                rows = [(page_ids_cpu * h * n + head * n + offset).to("spyre") for head in range(h)]
                store_staged(
                    new_k.to("spyre"),
                    new_v.to("spyre"),
                    k_pages.view(-1, d),
                    v_pages.view(-1, d),
                    rows,
                    ks,
                    vs,
                )
            else:
                index = torch.tensor([offset], dtype=torch.int32).to("spyre")
                for seq, page in enumerate(page_ids_cpu.tolist()):
                    store_packed_kv_page_compiled(
                        new_k[seq : seq + 1].contiguous().to("spyre"),
                        new_v[seq : seq + 1].contiguous().to("spyre"),
                        k_pages,
                        v_pages,
                        page,
                        index,
                        ks,
                        vs,
                    )

    mask = torch.zeros(m, n, dtype=torch.float16)
    mask.masked_fill_(torch.arange(n)[None] > torch.arange(m)[:, None] + n - m, -float("inf"))
    mask = mask[None, None, None].expand(b, h, g, m, n).reshape(b, h, g * m, n).contiguous()
    result["stage"] = f"{args.path}_attention"
    print(json.dumps(result), flush=True)
    attention = staged_fp8_attention if args.path == "staged" else packed_fp8_attention
    actual = (
        attention(q.to("spyre"), k_pages, v_pages, page_ids, mask.to("spyre"), ks, vs, d**-0.5)
        .cpu()
        .float()
    )

    q_fold = q.transpose(1, 2).reshape(b, h, g * m, d).float()
    qs = (q_fold.abs().amax(-1, keepdim=True) / 448).clamp_min(1.1920928955078125e-7)
    q8 = (q_fold / qs).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    k8 = (k.float() / ks).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    v8 = (v.float() / vs).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    scores = (q8 @ k8.transpose(-1, -2)) * (qs * (ks * d**-0.5))
    expected = (
        (torch.softmax(scores + mask.float(), -1) @ (v8 * vs))
        .reshape(b, h * g, m, d)
        .transpose(1, 2)
    )
    result["attention_vs_quantized_reference"] = relative_error(actual, expected)
    result["stage"] = "check_attention"
    assert actual.isfinite().all(), "Non-finite attention output"
    assert result["attention_vs_quantized_reference"]["relative_l2"] < 0.06, result
    result["stage"] = "complete"
    result["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("staged", "packed"), required=True)
    parser.add_argument(
        "--operation",
        choices=("full", "read", "update"),
        default="full",
        help="full: write/read; read: prepack K; update: seed then write tokens 63/64/127",
    )
    parser.add_argument("--batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--query-len", type=int, choices=(1, 128), default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "args": {**vars(args), "output": str(args.output)},
        "torch": torch.__version__,
        "torch_spyre": version("torch-spyre"),
        "status": "running",
        "stage": "setup",
    }
    try:
        run(args, result)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
