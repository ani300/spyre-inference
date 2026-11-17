# RFC 48 sparse MoE experiment

Set `SPYRE_MOE_IMPL=sparse_prefill` before loading the model to select the new
prefill region above `SPYRE_MOE_GATHERED_MAX_TOKENS`. Set it to `sparse` to also
select the whole-batch expert-union path for buckets of up to 32 tokens.
This is a token-bucket heuristic: a short prefill also takes that whole-batch
path. Both paths compute the same routed-expert function. The default,
`current`, selects the existing implementation.

`spyre_inference/sparse_moe.py` implements the operation order in
[RFC 48](https://github.com/torch-spyre/RFCs/pull/48). Its algorithm matches the
hf-adapters experiment. The bridge in `spyre_inference/moe.py` retains each
model's full-softmax or topk-softmax recipe, activation, and folded expert
scales. Token and expert id constants are prepared outside compilation.
When upstream traces MoE into the transformer block, the sparse path runs inline
and keeps full-softmax routing in the logits' dtype. The experiment supports
bias-free SiLU and GELU experts; upstream's other expert semantics remain
available through `current`.

Prefill uses `512 * ceil(E/K)` as the maximum chunk size (8192 at E=128, K=8).
Each chunk packs per-expert token ids and weights, appends a zero input row and
a sink output row, and uses `for_each_tile` for chunk, expert, and token-tile
loops. The inner trip count reads the FP32 rounded count tensor on device;
bounded symbolic narrowing exposes it to the shape-derived loop API. The
gather/add/scatter carry is functional to satisfy the loop's no-alias contract.
Decode uses `Gather` tile specifications for indirect expert weights and
routing-weight columns, running the full batch once per selected expert.

The new scan/count ops are called directly. This code supplies no replacement
kernels and performs no manual unstaggering or index-to-address conversion.
Dummy fp32 stick dimensions have 32 elements. The `aten.scatter` boundary casts
scan positions to logical int64 indices, which Spyre stores as int32.

The paths are for compiler bring-up: device-value trip counts and several
packing/indirect-access layouts are still unsupported in the tested stack.
The detailed report and standalone stage probe in the development workspace
are under `/mnt/devel/inductor_src/analysis/moe-rfc48/`.

TP expert sharding and reduction remain owned by vLLM. Gemma4 normalizes dense
and routed outputs separately, so its existing separate reductions must remain
before those normalizations. A single reduction of combined branch partials,
as in the RFC's simpler shared-expert example, would change model semantics.
The available workspace has one accelerator; 4-/8-way partial sums were checked
with an explicit CPU tile executor in the hf-adapters experiment, not with
multi-card collectives.
