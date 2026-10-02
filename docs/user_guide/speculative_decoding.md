# XPress speculative decoding

The experimental XPress path supports the Qwen3 text target with the published
`UIUC-SSAIL/Qwen3-8B-XPress-b16` checkpoint. It runs the DFlash backbone once,
refines the proposed block, then uses vLLM's target verification and output
processing. Speculation is opt-in.

The first supported configuration uses compiled execution, unquantized FP16
target and draft weights, TP=1, PP=1, one active request, greedy target sampling,
and disabled prefix caching. The published
checkpoint has a fixed 16-position block: one anchor and 15 draft tokens.
Changing `num_speculative_tokens` alone does not create a smaller compatible
checkpoint. Continuous batching, stochastic target sampling, tensor parallelism,
and prefix-cache reuse need further integration and validation.

## Run Qwen3-8B

Use the normal Spyre development environment and check the Hugging Face cache
before fetching weights. In the development workspace the cache is
`/mnt/models/hf_cache`. Source
`/mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh` and put the selected
worktrees first on `PYTHONPATH`. The validated lower stack comes from
`/mnt/home/spyre/sentient`; senlib comes from `/opt/ibm/spyre/senlib`.

```bash
export SPYRE_ATTN_QUERY_BUCKETS=1,16,64
export SPYRE_ATTN_KV_LAYOUT=head_major
uv run --no-sync vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype float16 --tensor-parallel-size 1 \
  --max-num-seqs 1 --max-model-len 512 --max-num-batched-tokens 64 \
  --no-enable-prefix-caching \
  --compilation-config '{"compile_sizes": [1, 16, 64]}' \
  --speculative-config '{"method": "xpress", "model": "UIUC-SSAIL/Qwen3-8B-XPress-b16", "revision": "f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5", "num_speculative_tokens": 15, "xpress_num_passes": 6}'
```

Send requests with `temperature=0`. The same `speculative_config` dictionary
works with `vllm.LLM`. Set `xpress_num_passes` to `0` to run the same DFlash
backbone without refinement. Omit `speculative_config` for ordinary decoding.
Keep prompt formatting, thinking mode, output limits, dtype, and context length
identical when comparing these modes.

This adapter uses vLLM 0.28's legacy DFlash runner contract. It translates the
`xpress` method before vLLM constructs its speculative configuration; it does
not require the newer GPU runner from the upstream XPress PR.

## Checkpoint contract

The published checkpoint contains five draft layers, a rank-256 refiner with a
512-wide SwiGLU MLP, and a full 151936-token vocabulary. It shares the target's
embedding and LM head. Target decoder layer IDs `[1, 9, 17, 25, 33]` correspond
to HF hidden-state indices `[2, 10, 18, 26, 34]`, since HF index zero is the
embedding output. These auxiliary outputs stay on Spyre for context projection.

The serving loader checks every backbone projection and refiner tensor. Raw
checkpoint mixers are folded as `tril(L) + I` once, in the serving dtype.
Serialized `XPressRefinerHead.state_dict()` values already contain the fold and
must be restored with `load_state_dict`, not the raw checkpoint loader.

The pinned upstream serving implementation uses the anchor token as its own
predecessor. This implementation reproduces that convention. The training
pipeline can use the actual preceding token, so evaluate a change to that
convention separately when adapting a production checkpoint.

### Convert a Speculators checkpoint

XPress training is on the fork's `pr-a-xpress` branch, pinned for this integration
at `72fe5660e81d14c523961a186623243e3d3f4ae4`. Its default branch does not contain
the inspected XPress implementation. A saved full-vocabulary Qwen3 XPress
checkpoint can be converted locally:

```bash
uv run --no-sync python -m spyre_inference.v1.spec_decode.convert_checkpoint \
  --input /path/to/speculators-checkpoint \
  --target-config /path/to/matching-target/config.json \
  --output /path/to/new-serving-checkpoint
```

The converter handles single-file and indexed safetensors checkpoints, copies
the unchanged backbone, maps `refiner_head.*` into `xpress_head.*`, subtracts
one from HF auxiliary-state indices, and translates the refiner configuration.
It preserves the raw mixer. The output includes the original training config
and a conversion manifest with source hashes and tensor shapes. Unsupported
vocabulary mappings, anchor layouts, attention types, missing weights, and
ambiguous embedding/final-norm taps are rejected. The output directory must be
new. Use the exact target and tokenizer weights associated with training;
matching configuration dimensions alone cannot establish that pairing.

The format conversion has save/load coverage. A newly trained production
checkpoint still needs its own numerical, acceptance, and task-quality checks.
A Gemma target would also require a compatible drafter and model adapter.

## Precision and performance boundaries

The backbone and refiner math are compiled on Spyre. Each proposal selection
explicitly transfers logits to CPU for first-index `argmax`, then transfers the
chosen IDs back for the next pass. The model records D2H time, CPU selection
time, ID H2D time, refiner time, readout time, and transferred bytes. The proposer
also records context projection and draft-forward time. These counters are
available through the model runner for profiling; they are not a public metrics
API.

This boundary is intentional on the validated stack: device `argmax` falls back,
FP16 `topk` corrupts large token IDs, and FP32 `topk` uses a different tie order.
The transfer uses `logits.to(device="cpu", dtype=torch.float32)`. Materializing
a FP16-to-FP32 cast on Spyre before transferring can permute values.

Context projection can exceed IEEE FP16's range before normalization while
remaining representable in Spyre DLF16. Keep this intermediate on device.
A CPU numerical reference must use FP32 for the context FC and normalization
boundary before converting back to FP16.

Refinement magnifies close token choices: one measured CPU margin of 0.015625
became an equal-logit tie on Spyre and changed later proposals. Target
verification remains responsible for the output. Compare numerical errors and
token margins as well as proposal IDs.

No production speedup is claimed. Measure ordinary decoding, K=0, and K=6
after warmup on the same workload. Report accepted tokens per round alongside
draft, refinement, verification, transfer, and host time. Repeated full-vocabulary
transfers currently dominate refinement, making exact device token selection
the main optimization candidate.

## Validation

CPU contract, recurrence, and rejection tests:

```bash
uv run --no-sync pytest tests/spec_decode -m 'not upstream'
```

The hardware test requires both pinned model snapshots in the cache and runs
one engine on one card:

```bash
uv run --no-sync pytest tests/e2e/test_speculative_decoding.py \
  -m model_quality -s --basetemp=/path/to/new-validation-artifacts
```

It checks an empty-proposal target reference, zero/partial/full acceptance,
controlled rejected suffixes, real DFlash and XPress proposals, page/context
boundaries, and EOS/token/text/output-length stopping. It compares vLLM's public
acceptance counters with every verification round and cancels an active request
before reusing its pages. Cache snapshots cover all 36 target and five draft
layers, with exact layer-name and prefix-length checks. Identical acceptance
schedules with different rejected suffixes must preserve the committed cache
exactly, including after cancellation. Cross-shape deep-layer cache errors are
recorded separately because FP16 block and single-token execution can diverge
numerically. Use a separate engine without `speculative_config` when measuring
ordinary decoding's latency and memory.

Run Spyre tests serially. CPU-only green tests do not establish device support.
The integration was brought up against torch-spyre
`24c7b8c568c287315f6aa00450580be674b71285`, hf-adapters
`a138ad4d6b65a57b99005ced33f91b4f1848d407`, Torch `2.13.0+cpu`,
vLLM `0.28.0+empty`, and Transformers `5.16.1`.

References: [vLLM PR #54448](https://github.com/vllm-project/vllm/pull/54448),
[pinned serving head](https://github.com/vllm-project/vllm/blob/15ba199f070f4bb4d1d394108525b4101117c067/vllm/model_executor/models/qwen3_xpress.py),
[pinned training branch](https://github.com/ZKBig/speculators/tree/72fe5660e81d14c523961a186623243e3d3f4ae4).
