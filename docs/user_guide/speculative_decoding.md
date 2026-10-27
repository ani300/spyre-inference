# XPress speculative decoding

XPress runs a DFlash backbone once, refines a block of draft tokens, and lets
the target model verify the proposals. The experimental Spyre integration loads
the published `UIUC-SSAIL/Qwen3-8B-XPress-b16` checkpoint directly with
`Qwen/Qwen3-8B`; checkpoint conversion is not required.

Supported configuration: compiled FP16, unquantized weights, TP=1, PP=1,
the V1 runner, up to four active requests, greedy target sampling, and prefix caching
disabled. The checkpoint fixes the block at one anchor plus 15 proposals.

## Run Qwen3-8B

```bash
export SPYRE_MODEL_RUNNER=v1
export SPYRE_ATTN_QUERY_BUCKETS=1,16,64
uv run --no-sync vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype float16 --tensor-parallel-size 1 \
  --max-num-seqs 1 --max-model-len 512 --max-num-batched-tokens 64 \
  --no-enable-prefix-caching \
  --compilation-config '{"compile_sizes": [1, 16, 64]}' \
  --speculative-config '{"method": "dflash", "model": "UIUC-SSAIL/Qwen3-8B-XPress-b16", "revision": "f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5", "num_speculative_tokens": 15, "xpress_topc": 128}'
```

Use `temperature=0` in requests. The same speculative configuration works with
`vllm.LLM`. A full verification round needs at least 31 budgeted tokens because
vLLM reserves additional drafting slots; the adapter rejects a smaller budget.

For four concurrent requests, set `--max-num-seqs 4`,
`--max-num-batched-tokens 128`, and
`--compilation-config '{"compile_sizes": [1, 16, 32, 64, 128]}'`.
Set `SPYRE_ATTN_QUERY_BUCKETS=1,16,64,128` for this token budget.
The draft body uses batches of 1, 2, or 4; three active requests pad to four.
Each request retains its own accepted context, page table, and output limit.

| Maximum requests | Minimum token budget |
| --- | --- |
| 1 | 31 |
| 2 | 62 |
| 3 | 93 |
| 4 | 124 |

## Refinement and compatibility

The refiner follows [merged vLLM XPress](https://github.com/vllm-project/vllm/pull/54448):
a fixed anchor, causal Jacobi updates, a block-wide hidden summary, and candidate
sets fixed across passes. Target input embeddings and output-head weights are
shared separately. Accepted target states populate the draft KV cache; rejected
and padded rows write only the reserved null page.

`xpress_topc` overrides the checkpoint field. Spyre defaults to 128 when neither
supplies a value; explicit 512 retains the upstream default and 0 enables
full-vocabulary refinement. A smaller shortlist can change acceptance. The
checkpoint selects the pass count, with six as the default.

The pinned vLLM 0.32 lacks native XPress registration, so this adapter uses its V1
DFlash classes with a configuration bridge. The merged GPU speculator uses V2.
By default, initial selection and per-pass argmax run on CPU; readout and
refinement run on Spyre. Jagged attention remains an optional follow-up.

## Optional device selection

`SPYRE_XPRESS_DEVICE_FEEDBACK=1` keeps per-pass selection and proposal feedback
on Spyre. `SPYRE_XPRESS_DEVICE_SELECTION=1` also selects the initial shortlist
on Spyre and implies device feedback. Both default to zero.

These options require a compatible compiler stack: the current dependency pin
rejects the feedback layout join, and resident shortlist selection needs the
FP16 payload top-k extensions. They are qualification paths, not a supported
serving configuration on the current pin.

C128 uses one top-128 round and does not require masking. C512 uses four rounds
and needs position-aware masking between them. Token IDs use two base-512 FP16
digits regardless of shortlist size. Compact candidate IDs still pass through
the host before the readout gather; fully resident ID conversion remains open.

## Validation

```bash
uv run --no-sync pytest tests/spec_decode -m 'not upstream'
SPYRE_XPRESS_TEST_TOPC=128 uv run --no-sync pytest \
  tests/e2e/test_speculative_decoding.py tests/e2e/test_xpress_batch.py -m model_quality -s
```

CPU recurrence and lifecycle tests pass. Device qualification remains open:
the strict target-token comparison exposes a shape-dependent target score tie
between ordinary decoding and block verification. Equality assertions remain in
the single-request and batched tests. Batch-four and C512 qualification must be
repeated on the current dependency pins before this integration is ready.
