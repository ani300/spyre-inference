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

import torch

from spyre_inference.v1.attention.ops.layout import INT32_ELEMS_PER_STICK
from spyre_inference.v1.attention.ops.page_attn import page_attn_kernel


def _capture_page_attn_graph(num_blocks: int):
    torch.manual_seed(0)
    padded_query_len = 4
    num_heads = 4
    num_kv_heads = 2
    head_size = 8
    block_size = 4
    pool_size = 16

    query = torch.randn(padded_query_len, num_heads, head_size)
    query_rows = torch.arange(padded_query_len, dtype=torch.int32)
    k_pages = torch.randn(pool_size, block_size, num_kv_heads, head_size)
    v_pages = torch.randn_like(k_pages)
    page_table = torch.zeros(num_blocks, INT32_ELEMS_PER_STICK, dtype=torch.int32)
    page_table[:, 0] = torch.randperm(pool_size)[:num_blocks]
    mask_tiles = [
        torch.zeros(num_kv_heads, num_heads // num_kv_heads, padded_query_len, block_size)
        for _ in range(num_blocks)
    ]
    args = (
        query,
        query_rows,
        k_pages,
        v_pages,
        page_table,
        mask_tiles,
        head_size**-0.5,
        num_blocks,
        padded_query_len,
        num_heads,
        num_kv_heads,
        head_size,
    )
    expected = page_attn_kernel(*args)

    graphs = []

    def capture_backend(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(
        page_attn_kernel,
        backend=capture_backend,
        dynamic=False,
        fullgraph=True,
    )
    actual = compiled(*args)
    torch.testing.assert_close(actual, expected)
    assert len(graphs) == 1
    return graphs[0]


def test_page_attn_uses_one_constant_size_page_loop():
    """Increasing the active-page count must resize the HOP, not unroll its body."""
    try:
        small = _capture_page_attn_graph(2)
        large = _capture_page_attn_graph(8)
    finally:
        torch._dynamo.reset()

    def executable_nodes(graph):
        return [node for node in graph.graph.nodes if node.op in ("call_function", "call_method")]

    small_nodes = executable_nodes(small)
    large_nodes = executable_nodes(large)
    assert sum(str(node.target) == "scan" for node in small_nodes) == 1
    assert sum(str(node.target) == "scan" for node in large_nodes) == 1
    assert len(large_nodes) == len(small_nodes)
