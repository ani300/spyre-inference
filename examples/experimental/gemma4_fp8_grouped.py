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

"""Assess real Gemma FP8 expert pools and grouped FP8 gate/up/down matmuls."""

import argparse
import json
import os
import time
import traceback
import warnings
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from spyre_inference.experimental.fp8_grouped_moe import (
    load_expert_pool,
    moe_tokens,
    prepare_experts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        choices=[
            "native",
            "staged",
            "persistent",
            "persistent-sharded",
            "persistent-dma",
            "rank3-dma",
        ],
        required=True,
    )
    parser.add_argument("--batch", type=int, choices=[1, 4], default=1)
    parser.add_argument("--query-len", type=int, choices=[1, 128], default=1)
    parser.add_argument(
        "--experts",
        type=int,
        choices=[32, 128],
        default=128,
        help="32 isolates downstream compiler failures; full checkpoint uses 128",
    )
    parser.add_argument("--chunks", type=int, choices=[1, 4], default=4)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/models/hf_cache")
    from torch_spyre.ops.fallbacks import FallbackWarning

    warnings.filterwarnings("error", category=FallbackWarning)
    result = {
        **vars(args),
        "output": str(args.output),
        "passed": False,
        "versions": {p: version(p) for p in ("torch", "torch-spyre", "vllm")},
        "dispatch": "on-device routing, token-expert expansion and reduction",
        "weight_storage": "FP8",
        "matmul": "spyre.scaled_bmm FP8 x FP8 -> DL16",
    }
    stage = "checkpoint"

    def progress(value):
        nonlocal stage
        stage = value
        print("Stage:", stage, flush=True)

    try:
        root = Path(
            snapshot_download(
                "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
                revision="ed35d7abe5d940da41b4ff06eb482feb0be8cb44",
                local_files_only=True,
            )
        )
        prefix = f"model.language_model.layers.{args.layer}"
        host, scales = {}, {}
        with safe_open(root / "model.safetensors", framework="pt") as sf:
            for proj in ("gate", "up", "down"):
                host[proj] = torch.stack(
                    [
                        sf.get_tensor(f"{prefix}.experts.{e}.{proj}_proj.weight")
                        for e in range(args.experts)
                    ]
                )
                scales[proj] = torch.stack(
                    [
                        sf.get_tensor(f"{prefix}.experts.{e}.{proj}_proj.weight_scale")
                        for e in range(args.experts)
                    ]
                ).float()
            router_scale = sf.get_tensor(prefix + ".router.per_expert_scale").float()[
                : args.experts
            ]
        if args.path == "rank3-dma":
            stage = "rank3_dma"
            from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

            pool = _dma_to_spyre_fp8_kernel(host["gate"].transpose(-1, -2).contiguous())
            result["layout"] = str(pool.device_tensor_layout())
            torch.testing.assert_close(
                pool.cpu().view(torch.uint8),
                host["gate"].transpose(-1, -2).view(torch.uint8),
                rtol=0,
                atol=0,
            )
            result["byte_exact"] = True
        else:
            torch.manual_seed(813)
            tokens = args.batch * args.query_len
            x = (torch.randn(tokens, 2816) * 0.5).half()
            logits = torch.randn(tokens, 128).half()
            logits[:, args.experts :] = -1000
            probs = torch.softmax(logits.float(), -1)
            ids = probs.topk(8, -1).indices
            prepared, prepared_scales = prepare_experts(
                host, scales, router_scale, 1 if args.path.startswith("persistent") else args.chunks
            )
            pools, layouts = {}, {}
            pool_shards, scale_shards = [], []
            if args.path == "persistent-sharded":
                for start in range(0, args.experts, 32):
                    shard = {}
                    for proj in ("gate", "up", "down"):
                        progress(f"load_shard_{start}_{proj}")
                        shard[proj] = load_expert_pool(
                            prepared[proj][start : start + 32], path="native"
                        )
                        assert shard[proj].dtype == torch.float8_e4m3fn
                        layouts[f"{start}_{proj}"] = str(shard[proj].device_tensor_layout())
                    pool_shards.append(shard)
                    scale_shards.append(
                        {
                            p: s[start : start + 32].half().to("spyre")
                            for p, s in prepared_scales.items()
                        }
                    )
                result["pool_logical_bytes"] = sum(
                    w.numel() for shard in pool_shards for w in shard.values()
                )
            else:
                for proj in ("gate", "up", "down"):
                    progress(f"load_{proj}_pool")
                    if args.path == "persistent-dma":
                        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

                        pools[proj] = _dma_to_spyre_fp8_kernel(prepared[proj].transpose(-1, -2))
                    else:
                        pools[proj] = load_expert_pool(
                            prepared[proj], path="staged" if args.path == "staged" else "native"
                        )
                    assert pools[proj].dtype == torch.float8_e4m3fn
                    layouts[proj] = str(pools[proj].device_tensor_layout())
                result["pool_logical_bytes"] = sum(w.numel() for w in pools.values())
            result["pool_layouts"] = layouts
            scale_pools = {p: s.half().to("spyre") for p, s in prepared_scales.items()}
            selected_ids = torch.empty(tokens, 8, dtype=torch.int64)
            device_weights = torch.empty(tokens, 8)

            def observe(t, ids, weights):
                # Test instrumentation only; these copies do not drive execution.
                selected_ids[t] = ids.cpu().long()
                device_weights[t] = weights.cpu().float()[0]

            def observe_route(route):
                weights, ids = route.cpu().float().squeeze(-1)[:tokens].topk(8, -1)
                selected_ids[:] = ids
                device_weights[:] = weights

            row_ids = [torch.tensor([t], dtype=torch.int32).to("spyre") for t in range(tokens)]
            progress("moe_tokens")
            started = time.monotonic()
            if args.path == "staged":
                actual = (
                    moe_tokens(
                        x.to("spyre"),
                        logits.to("spyre"),
                        pools,
                        scale_pools,
                        row_ids,
                        path=args.path,
                        progress=progress,
                        observer=observe,
                        chunks=args.chunks,
                    )
                    .cpu()
                    .float()
                )
            elif args.path == "persistent-sharded":
                from spyre_inference.experimental.fp8_native_moe import persistent_shards

                actual = (
                    persistent_shards(
                        x.to("spyre"),
                        logits.to("spyre"),
                        pool_shards,
                        scale_shards,
                        progress=progress,
                        route_observer=observe_route,
                    )
                    .cpu()
                    .float()
                )
            else:
                from spyre_inference.experimental.fp8_native_moe import native_moe

                xd, ld = x.to("spyre"), logits.to("spyre")
                if args.path == "persistent-dma":
                    from spyre_inference.experimental.fp8_moe_method import pad_rows, trim_rows

                    xd, ld = pad_rows(xd, ld)
                value = native_moe(
                    xd,
                    ld,
                    pools,
                    scale_pools,
                    path=args.path,
                    chunks=args.chunks,
                    progress=progress,
                    route_observer=observe_route,
                )
                if args.path == "persistent-dma":
                    value = trim_rows(value, tokens)
                actual = value.cpu().float()
                if args.path == "native":
                    selected_ids[:] = ids
                    ref_probs = probs.gather(1, ids)
                    device_weights[:] = ref_probs / ref_probs.sum(-1, keepdim=True)
                    result["routing_validation"] = (
                        "native output compared against CPU routing and expert reference"
                    )
            result["cold_compute_seconds"] = time.monotonic() - started
            progress("routing_validation")
            assert ((selected_ids >= 0) & (selected_ids < args.experts)).all()
            assert all(row.unique().numel() == 8 for row in selected_ids)
            selected_probs = probs.gather(1, selected_ids)
            expected_routing = selected_probs / selected_probs.sum(-1, keepdim=True)
            if args.path != "native":
                result["routing_set_match"] = bool(
                    torch.equal(ids.sort(-1).values, selected_ids.sort(-1).values)
                )
                torch.testing.assert_close(device_weights, expected_routing, rtol=0.02, atol=0.002)
                result["routing_different_rows"] = int(
                    (ids.sort(-1).values != selected_ids.sort(-1).values).any(-1).sum()
                )
                cutoff = probs.topk(8, -1).values[:, -1:]
                result["routing_max_cutoff_gap"] = float(
                    (cutoff - selected_probs).clamp_min(0).max()
                )
                assert (selected_probs >= cutoff - 0.0005).all()
            result["selected_expert_count"] = int(selected_ids.unique().numel())
            expected = torch.zeros(tokens, 2816)
            for e in selected_ids.unique():
                token_rows, slots = (selected_ids == e).nonzero(as_tuple=True)
                gate, up, down = [
                    (host[p][e].float() * scales[p][e]).t() for p in ("gate", "up", "down")
                ]
                z = F.gelu(x[token_rows].float() @ gate, approximate="tanh") * (
                    x[token_rows].float() @ up
                )
                expected.index_add_(
                    0,
                    token_rows,
                    (z @ down) * router_scale[e] * expected_routing[token_rows, slots, None],
                )
            stage = "numerical_validation"
            assert torch.isfinite(actual).all()
            result["relative_l2"] = float(
                torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
            )
            row_error = torch.linalg.vector_norm(
                actual - expected, dim=-1
            ) / torch.linalg.vector_norm(expected, dim=-1)
            result["max_row_relative_l2"] = float(row_error.max())
            result["max_abs"] = float((actual - expected).abs().max())
            result["reference_norm"] = float(torch.linalg.vector_norm(expected))
            # Predeclared norm tolerance for three FP8 GEMMs plus gated GELU.
            assert result["relative_l2"] < 0.10, result["relative_l2"]
            assert result["max_row_relative_l2"] < 0.10, result["max_row_relative_l2"]
        result["passed"] = True
    except Exception as exc:
        result.update(
            stage=stage,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
