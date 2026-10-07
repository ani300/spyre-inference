# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Small speculative query tiles preserve requests, masks and unused output rows."""

import warnings
from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import pytest
import torch
from spyre_testing_plugin.attn_helpers import assert_close_outliers
from spyre_testing_plugin.pytest_plugin import spyre_available
from test_jagged_page_attn import dense_reference, jagged_device_inputs

from spyre_inference.v1.attention.jagged_plan import build_jagged_tile_plan
from spyre_inference.v1.attention.ops.jagged_tile_attn import jagged_tile_attn_kernel

pytestmark = [pytest.mark.attention, pytest.mark.usefixtures("counted_spyre_loops")]


@pytest.mark.parametrize("target", ["cpu", "spyre"])
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("causal", [False, True])
def test_small_tiles_preserve_mixed_requests_and_reuse_graph(target, width, causal):
    if target == "spyre" and not spyre_available():
        pytest.skip("Requires a Spyre accelerator")
    from torch._dynamo.utils import counters
    from torch_spyre.ops.fallbacks import FallbackWarning

    dtype = torch.float16 if target == "spyre" else torch.float32
    generator = torch.Generator().manual_seed(1067)
    kernel = partial(jagged_tile_attn_kernel, scale=128**-0.5, head_major=True)
    if target == "spyre":
        kernel = torch.compile(kernel, fullgraph=True, dynamic=False)
    cpu_context = patch("torch.accelerator.is_available", return_value=False)
    with (
        torch.inference_mode(),
        warnings.catch_warnings(),
        cpu_context if target == "cpu" else nullcontext(),
    ):
        warnings.simplefilter("error", FallbackWarning)
        for batch in (1, 3, 4):
            for repeat in range(2):
                counts = torch.tensor([16, 9, 12, 16][:batch], dtype=torch.int32)
                starts = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()))
                lengths = torch.tensor([503, 377, 251, 125][:batch], dtype=torch.int32) + repeat
                pages = (torch.randperm(32, generator=generator) + 1).int().view(4, 8)[:batch]
                q = torch.randn(129, 32, 128, generator=generator).to(dtype)
                k = torch.randn(33, 128, 8, 128, generator=generator).to(dtype)
                v = torch.randn(k.shape, generator=generator).to(dtype)
                k[0] = v[0] = float("nan")
                plan = build_jagged_tile_plan(
                    starts,
                    lengths,
                    pages,
                    128,
                    query_capacity=129,
                    query_tile_size=width,
                    causal=causal,
                )
                out = torch.full((129 + width, 32, 128), float("nan"), dtype=dtype)
                if target == "spyre":
                    from spyre_inference.custom_ops.utils import convert, row_outermost_layout

                    inputs = jagged_device_inputs(q, k, v, plan, True)
                    q = inputs[0].cpu()
                    k, v = (x.cpu().transpose(1, 2).contiguous() for x in inputs[1:3])
                    out = convert(
                        out,
                        torch.device("spyre"),
                        device_layout=row_outermost_layout(out.shape, dtype),
                    )
                else:
                    inputs = (
                        q,
                        k.transpose(1, 2).contiguous(),
                        v.transpose(1, 2).contiguous(),
                        *(t.long() if t.dtype == torch.int32 else t for t in plan.tensors),
                    )
                expected = dense_reference(
                    q,
                    k,
                    v,
                    starts,
                    lengths,
                    pages,
                    causal=causal,
                    window=None,
                    cap=0.0,
                )
                before = counters["stats"]["unique_graphs"]
                actual = kernel(*inputs, out=out).cpu().double()
                if repeat:
                    assert counters["stats"]["unique_graphs"] == before
                live = int(starts[-1])
                if target == "spyre":
                    # Width64 has the same bounded FP16 outliers on these inputs.
                    assert_close_outliers(
                        actual[:live],
                        expected[:live],
                        max_outliers=5,
                        atol=0.002,
                        rtol=0.02,
                        outlier_atol=0.004,
                        outlier_rtol=0.04,
                    )
                    assert (actual[:live] - expected[:live]).norm() / expected[:live].norm() < 0.02
                else:
                    torch.testing.assert_close(actual[:live], expected[:live], atol=2e-6, rtol=2e-5)
                assert torch.isnan(actual[live:129]).all()
