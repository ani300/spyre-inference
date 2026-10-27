# XPress speculative decoding

XPress runs a DFlash backbone once, refines a block of draft tokens, and lets
vLLM's target verification decide which tokens to accept. This experimental
adapter supports `Qwen/Qwen3-8B` with
`UIUC-SSAIL/Qwen3-8B-XPress-b16` on regular Spyre attention.

The supported scope is compiled FP16, unquantized target and draft weights,
TP=1, PP=1, the V1 model runner, one to four requests, greedy target sampling,
and prefix caching disabled. The checkpoint has a fixed block of 16 positions:
one verified anchor and 15 proposals. Changing the number of speculative tokens
requires a compatible checkpoint. Jagged attention is not a dependency.

## Run Qwen3-8B

Activate the normal development environment. In the shared development workspace,
source `/mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh`, then keep the selected
worktrees first on `PYTHONPATH`. Lower-stack components come from
`/mnt/home/spyre/sentient`, except senlib from `/opt/ibm/spyre/senlib`. Check the
existing model cache at `/mnt/models/hf_cache` before downloading weights.

```bash
export SPYRE_MODEL_RUNNER=v1
export SPYRE_ATTN_QUERY_BUCKETS=1,16,64
export SPYRE_ATTN_KV_LAYOUT=head_major
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
`vllm.LLM`. For four concurrent requests, use `--max-num-seqs 4`, a token budget
of at least 128, and `compile_sizes=[1,16,32,64,128]`. The minimum draft-buffer
capacity is 64 tokens, but vLLM also reserves 15 drafting slots per request when
admitting verification work. A full verification batch therefore needs a token
budget of at least 31, 62, 93 or 124 for one through four requests. The adapter
rejects smaller budgets: a 16-token budget for one request can run without ever
verifying drafts. `SPYRE_MAX_NUM_PARTIAL_PREFILLS=4` allows concurrent prefills.
Three active draft requests are padded to four blocks.

Each request commits only its accepted target states into its own draft KV
pages. Partial prefills commit context without proposing. Finished requests and
requests near the context limit stop proposing independently. Rejected and padded
rows write the reserved null page.

## Compatibility with merged vLLM XPress

The reference is [vLLM PR #54448](https://github.com/vllm-project/vllm/pull/54448),
merged as `ce9467b757c024a55fea4d0919ac2e682f28a894` on October 22, 2026.
Spyre's upstream dependency pin, vLLM 0.32, predates that merge. This adapter
retains the pin and provides the XPress checkpoint and configuration bridge for
the V1 runner; the merged GPU implementation uses `XPressSpeculator` in V2.

Both implementations use:

- `method="dflash"`, detecting XPress from `Qwen3XPressModel` or
  `DFlashQwen3XPressModel` in the checkpoint.
- A fixed anchor at position zero. The anchor also supplies its own predecessor;
  only positions 1 through 15 are returned as proposals.
- The block-wide hidden-state mean, a pass-invariant hidden projection, causal
  Jacobi refinement, and the raw mixer folded as `tril(L) + I` once at load.
- A separate candidate set for each predicted position, selected from the initial
  logits and held fixed across every refinement pass. Readout rows are gathered
  once. A candidate outside that set cannot be recovered during refinement.
- Target embeddings and the target output head, shared separately. An untied
  Qwen3 output head remains distinct from its input embeddings.

`xpress_topc` takes precedence over the checkpoint field. If neither supplies a
value, **Spyre defaults to 128**, while the merged vLLM head defaults to 512.
An explicit `0` keeps full-vocabulary scoring, and an explicit `512` retains the
larger shortlist. Six Jacobi passes are used unless the checkpoint overrides
`xpress_num_passes`. The historical `method="xpress"` and speculative-config
`xpress_num_passes` are Spyre compatibility aliases, not merged vLLM options.

Because vLLM 0.32 lacks `xpress_topc`, the bridge carries the option through
`additional_config` to the adapter. This shim can be removed when the dependency
pin includes the native field. Spyre keeps host request bookkeeping and target
sampling, with compiled PyTorch device kernels in place of the GPU refiner's
Triton kernels and CUDA graphs. CPU and GPU top-k tie ordering can differ;
proposal equivalence at ties is not an end-to-end output guarantee.

## Selection and transfer costs

The portable default copies initial vocabulary logits to CPU for argmax/top-k,
then uploads the candidate IDs and scores. Candidate readout and refinement run
on Spyre. Each refinement pass copies its candidate scores to CPU for argmax and
uploads predecessor IDs for the next pass. Full-vocabulary mode transfers each
pass's vocabulary scores instead of a compact shortlist.

`SPYRE_XPRESS_DEVICE_FEEDBACK=1` enables experimental resident predecessor
feedback for a 16-position block and C128/C512. After the initial CPU shortlist,
only final proposal IDs return to CPU. This path is **off by default**: the
upstream compiler pin rejects a mixed-element-arrangement join between decoded
IDs and uploaded anchors at both shortlist sizes. It needs a compatible compiler
and separate hardware qualification before use.

`SPYRE_XPRESS_DEVICE_SELECTION=1` enables the experimental resident FP16 shortlist.
It requires the separate torch-spyre/Deeptools payload top-k implementation,
which is **absent from the upstream torch-spyre pin**. The branch reports a clear
error if it is requested without those operators. Its base-512 token-ID encoding
is independent of shortlist size: reducing C to 128 must not change that radix.
Resident selection also enables the experimental feedback path and therefore
requires its compiler fixes.

On this path, C128 requires one top-128 round and no masking rounds. C512 requires
four top-128 rounds and three full-vocabulary masks. Candidate storage, readout
work and compact candidate-ID transfers also shrink by a factor of four. These
are work reductions, not a measured fourfold generation speedup: target execution,
draft execution, context projection and initial argmax remain. Smaller candidate
sets can change acceptance, so measure accepted tokens per round as well as time.
The experimental selector still bridges compact candidate-ID digits through CPU
before the gather; it does not download the full logits.

Per-stage counters expose host intervals and logical transferred bytes. They do
not measure exclusive device time: asynchronous work can be charged to a later
transfer. Use device profiling for CPU/device attribution. DLF16 intermediates
can exceed IEEE FP16's range, so references should normalize context projections
in FP32 before converting to FP16.

## Checkpoint conversion

The published checkpoint uses five DFlash layers, a rank-256 refiner, a
512-wide SwiGLU MLP and the full 151936-token vocabulary. Decoder layer IDs
`[1,9,17,25,33]` correspond to HF hidden-state indices `[2,10,18,26,34]`.

The loader checks backbone and refiner tensor names/shapes. A serving
`state_dict` already contains the folded mixer; restore it with `load_state_dict`
to avoid folding twice. The converter supports the inspected full-vocabulary
Speculators Qwen3 schema:

```bash
uv run --no-sync python -m spyre_inference.v1.spec_decode.convert_checkpoint \
  --input /path/to/speculators-checkpoint \
  --target-config /path/to/matching-target/config.json \
  --output /path/to/new-serving-checkpoint
```

It preserves raw mixers, translates names/taps and writes source hashes and
shapes to a manifest. Unsupported vocabulary mappings, shifted anchor layouts,
attention modes, or incomplete weights are rejected. A new production checkpoint
still needs its own acceptance and quality evaluation. Gemma targets require
additional model integration.

## Validation and measurement

This integration is still being qualified on the current dependency pins. CPU
parity and a real C128 generation smoke test pass; the strict target-output
comparison exposes a shape-dependent score tie. See the
[stack plan](../architecture/xpress-pr-stack.md#current-validation-october-27)
for the measured behavior and remaining submission gates.

```bash
uv run --no-sync pytest tests/spec_decode tests/runtime/test_shape_bucketer.py

SPYRE_XPRESS_TEST_TOPC=128 uv run --no-sync pytest \
  tests/e2e/test_speculative_decoding.py -m model_quality -s
SPYRE_XPRESS_TEST_TOPC=128 uv run --no-sync pytest \
  tests/e2e/test_xpress_batch.py -m model_quality -s
SPYRE_XPRESS_DEVICE_FEEDBACK=1 uv run --no-sync pytest \
  tests/e2e/test_xpress_feedback.py -m model_quality -s
```

Run hardware commands serially on a free card selected with `SPYRE_DEVICES`.
Repeat with C512 as the control and C0 for the full-vocabulary reference when
needed. CPU tests compare the refiner against the merged equations and exercise
vLLM 0.32's actual rejection dispatchers. Hardware tests cover acceptance,
rollback, stopping, cancellation, batch changes, cache ownership, and feedback
IDs. The optional `test_xpress_selection.py` also requires the payload operators.

Compare warmed requests at identical prompt/output lengths and batch sizes:

```bash
uv run --no-sync python scripts/microbench/xpress_latency.py \
  --topc 128 --output /tmp/xpress-c128-b1.json
uv run --no-sync python scripts/microbench/xpress_batch_latency.py \
  --topc 128 --output /tmp/xpress-c128-b4.json
```

Repeat in fresh processes with `--topc 512`. The batch harness also supports
`--mode ordinary` and a separate `--profile-output` run. Add `--device-selection`
only with the required compiler stack. Reports include acceptance, actual draft
batch sizes and compilation counts; timing claims must distinguish CPU selection
from the experimental resident selector. `--device-feedback` isolates resident
feedback while retaining CPU initial selection; it needs the feedback compiler
fix. Both optimizations default off in the benchmark harnesses.
