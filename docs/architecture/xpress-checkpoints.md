# XPress checkpoint contract

The portable XPress head implements the recurrence merged in
[vLLM #54448](https://github.com/vllm-project/vllm/pull/54448) at
`ce9467b757c024a55fea4d0919ac2e682f28a894`. It can validate and convert
checkpoints on CPU independently of the Spyre runner integration.

## Refiner semantics

Position zero is a fixed, verified anchor. The serving caller supplies the same
anchor as its predecessor and receives positions 1 through B−1 as proposals.
The raw checkpoint mixer becomes `tril(L) + I` once on loading. A serving
`state_dict` already contains that fold and must use `load_state_dict` to restore it.

Each pass uses the block-wide hidden-state mean and the same cached hidden
projection. Causal Jacobi updates consume the preceding pass's tokens. Candidate
sets are selected per position from the initial logits and stay fixed across
passes; their readout rows are gathered once.

The Spyre head defaults to a shortlist of 128 candidates. An explicit 512 retains
the merged vLLM default, and 0 selects full-vocabulary refinement. Reducing the
shortlist can change proposals and acceptance; it is not an equivalence guarantee
between shortlist sizes. CPU tests compare each size against the merged equations.

## Conversion

The published Qwen3-8B checkpoint has five DFlash layers, a rank-256 refiner,
a 512-wide SwiGLU MLP and the full 151936-token vocabulary. Decoder layer IDs
`[1,9,17,25,33]` correspond to HF hidden-state indices `[2,10,18,26,34]`.
Target input embeddings and output-head weights are shared separately by the
serving model and may be omitted from the draft checkpoint.

The converter accepts the full-vocabulary Qwen3 schema from
[Speculators](https://github.com/ZKBig/speculators):

```bash
uv run --no-sync python -m spyre_inference.v1.spec_decode.convert_checkpoint \
  --input /path/to/speculators-checkpoint \
  --target-config /path/to/matching-target/config.json \
  --output /path/to/new-serving-checkpoint
```

It validates names and shapes, preserves raw mixers, translates head names and
hidden-state taps, and writes source hashes and shapes to a manifest. Unsupported
vocabulary mappings, shifted anchors, attention modes and incomplete weights fail
before creating the output directory. Existing output directories are refused.
A converted production checkpoint still needs model integration and acceptance
qualification. See the [plugin architecture](index.md) for the runner boundary.
