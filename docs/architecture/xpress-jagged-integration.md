# XPress and jagged attention integration

This branch combines the updated XPress proposal loop with experimental jagged
paged attention. Qwen3-8B remains the first target. The combined C512 end-to-end
test, CPU regressions, device attention checks and checkpoint head probe pass.
Broader model validation and production benchmarking remain separate gates. The optimization priorities
below distinguish measurements from calculated work and profiling hypotheses.

## Revisions and behavior

| Component | Pinned revision | Role |
| --- | --- | --- |
| [vLLM PR #54448](https://github.com/vllm-project/vllm/pull/54448) | `c768e7b6b1bf423bf276cf61b46a4b76f5ec7974` | XPress algorithm and public configuration contract |
| [spyre-inference PR #1118](https://github.com/torch-spyre/spyre-inference/pull/1118) | `5f085e6ec646784422eb2a7bcdcd86be8c132866` | Jagged attention implementation merged into this branch |
| [torch-spyre PR #5086](https://github.com/torch-spyre/torch-spyre/pull/5086) | `fcc50670cec3abbbd5790874ae428ed7c868b00a` | Companion lowering, pinned in `pyproject.toml` and `uv.lock` |
| vLLM runtime | `v0.28.0`, built with the empty backend | Existing Spyre runner contract |
| [Speculators training](https://github.com/ZKBig/speculators/tree/72fe5660e81d14c523961a186623243e3d3f4ae4) | `72fe5660e81d14c523961a186623243e3d3f4ae4` | Previously inspected `pr-a-xpress` training/export contract |

The upstream PR revisions were checked on October 5, 2026. This is an algorithm
port onto the existing Spyre runner, not an installation of the upstream GPU
runner. The standalone `feat/xpress-speculative-decoding` branch contains the
XPress update; `feat/xpress-jagged-attention` adds the attention merge and
dependency pin.

The latest jagged refresh includes the computed-mask realization fix before loop
splicing and explicitly exercises counted loops in the attention tests. The
inference PR also brings its current main-branch changes and RPM lock; local
validation uses workspace lower-stack artifacts rather than installing those RPMs.

The updated proposal loop follows the upstream head:

1. `method="dflash"` selects XPress from `Qwen3XPressModel` or
   `DFlashQwen3XPressModel`. The earlier `method="xpress"` alias still works.
2. `xpress_topc` uses an explicit override, then the checkpoint value, then 512.
   Zero enables full-vocabulary refinement. The adapter carries these options
   through `additional_config` because vLLM 0.28 predates the fields.
3. Each of the 15 predicted positions gets its own top-C set from the original
   base logits. Candidate IDs, base scores, and gathered readout weights remain
   fixed through every Jacobi pass.
4. The hidden half of `in_proj` runs once per proposal. Each pass updates the
   token-dependent latent and reads out only the predicted rows. Slot zero stays
   fixed, with the anchor serving as its own predecessor, as in the upstream PR.

Upstream fuses the latent update and selection on GPU. Spyre uses compiled
PyTorch kernels plus CPU selection. Top-C limits which draft tokens can be
proposed; it does not bypass target verification. CPU and GPU top-k tie ordering,
and device arithmetic, can produce different proposals.

The combination also preserves `causal=False` in jagged draft metadata and
passes each layer's causal setting into attention recording. Target verification
remains causal. The published 16-position draft block uses the jagged prefill
schedule: grouped decode only handles query widths up to eight. Jagged attention
does not reduce the surrounding model's token-bucket padding.

See [the XPress guide](../user_guide/speculative_decoding.md) for model pins and
checkpoint conversion, and [jagged attention](jagged-attention.md) for the
schedule, metadata lifetime, and unresolved model-level numerical comparisons.
The restrictions remain FP16, compiled execution, greedy sampling, TP=1, PP=1,
one active request, and no prefix caching.

## Evidence and limits

On October 5, the combined branch passed the Qwen3-8B end-to-end test on physical
card 2 with `SPYRE_JAGGED_ATTENTION=1`, C512, FP16, head-major KV, and counted
loops. It covered six cases, zero/partial/full acceptance, real DFlash and XPress
proposals, stopping, cancellation and page reuse. All 36 target and five draft
caches were checked; changing rejected suffixes under the same acceptance
schedule and cancellation/reuse preserved committed caches exactly. Warmup logs
show jagged attention recording for all 41 layers. The complete test took
657 seconds including model loading and compilation.

The instrumented six-case run produced:

| Measurement | DFlash, K=0 | XPress, K=6, C512 |
| --- | ---: | ---: |
| Sum of request wall times | 10.040 s | 9.527 s |
| Public speculative draft rounds | 32 | 29 |
| Accepted draft tokens | 86 | 91 |
| Draft score-copy wall time | 0.938 s | 0.909 s |
| Logical draft score payload on CPU | 291,717,120 bytes | 267,522,048 bytes |
| CPU draft argmax time | 0.044 s | 0.041 s |
| CPU draft top-k time | — | 0.058 s |

These are diagnostic measurements with cache inspection enabled, not a production
speedup comparison. They do not compare jagged versus regular attention or a
separate ordinary-decoding engine. Only C512/head-major/one card was tested
end to end in this refresh; the full-vocabulary and regular-attention matrix
remains future validation. Asynchronous device work can
be charged to a subsequent copy; dispatch and transfer timers are not exclusive
kernel durations and must not be added as if they were.

The original full-vocabulary branch's October 1 test and ordinary-decoding
baseline remain historical evidence under `xpress-validation`; they are not a
controlled performance comparison with this refreshed stack.

On October 5, the corrected-environment head probe used the pinned checkpoint
and one captured hidden-state/logit block, with ten warm repetitions per mode:

| Head-only mode | Median | Observed range | Logical score bytes per block |
| --- | ---: | ---: | ---: |
| K=0 | 7.32 ms | 7.04–12.46 ms | 9,723,904 |
| K=6, C=512 | 14.16 ms | 13.82–18.62 ms | 9,908,224 |
| K=6, full vocabulary | 62.12 ms | 55.80–63.23 ms | 64,420,864 |

This includes head execution and host selection, with precomputed base logits;
it excludes the LM head, draft backbone, target and acceptance effects. It is
not an end-to-end speedup measurement. Candidate readout gathers matched two
runtime ID tables exactly, including large vocabulary IDs. Against the pinned
upstream FP32 head, the six teacher-forced shortlist score comparisons had
relative L2 errors of 0.24–0.34%. Argmax agreed at 14 of 15 positions on the first
pass and all positions thereafter; exact proposals across numeric backends are
not guaranteed. Full-vocabulary execution was checked for repeated-proposal
stability; the FP32 score comparison covered shortlist mode.

The score-copy intervals averaged 9.0 ms per shortlist block versus 49.8 ms in
full-vocabulary mode. Shortlist CPU top-k and argmax averaged 1.2 ms and 1.6 ms
per block respectively. Copy intervals include waiting for preceding device work;
they cannot isolate DMA bandwidth or be added to dispatch time as exclusive costs.

Early device probes failed with `ComputeHardwareError 0x7b1b` and an
instruction-fetch page fault. The validation launcher caused this: activating a
second venv after `dev-env.sh` restored an older `PATH`, selecting `/opt` compiler
executables while loading workspace runtime libraries. Checking loaded libraries
alone missed the mismatch. Selecting the venv without another activation preserves
the workspace compiler paths. Compiled 64-by-64 addition and matmul then passed
in the original XPress venv with a fresh cache. A resident-table attention test
also passed on the private combined checkout. All eight packed-dispatch device
cases then passed, covering causal/noncausal attention, both cache layouts, and
both staging modes. These results supersede the earlier
baseline-hardware-fault interpretation; the failed runs do not establish an
XPress or jagged kernel defect.

Current results: 101 speculative-decoding CPU tests passed on the standalone
branch and again on the combined branch with jagged attention enabled. All 136
jagged CPU tests passed; 157 device cases were explicitly deselected in that
run. Three real vLLM `EngineArgs` checks passed using the cached model configs:
default shortlist, full-vocabulary override, and legacy alias. Repository
formatting, lint and type checks passed. The CPU tests exercise configuration precedence, checkpoint loading,
unfolded-refiner equivalence, fixed-shortlist recurrence, the actual serving
selection loop and its score payload, and jagged causal/noncausal dispatch. CPU
success cannot establish Spyre kernel correctness. PR #1118's existing full-model
numerical discrepancies also remain separate validation gates.

Local evidence is retained under `/mnt/home/spyre/xpress-jagged-validation`:
`pinned-sources/manifest.json`, `torch-spyre-build.log`,
`cpu-xpress-pinned.log`, `cpu-spec-combined.log`, `cpu-jagged-registered.log`,
`config-smoke.json`, `current-stack.json`, `head-probe.json`,
`sanity_probe.py`, `sanity-card1.log`, and `sanity-idle-cards.log`. The older
root-level head JSON is partial and superseded by the refreshed evidence below.
Historical measurements are in
`/mnt/home/spyre/xpress-validation/e2e4/test_qwen3_xpress_verification0/xpress-validation.json`
and `target-baseline.json`. The evolving investigation is preserved at
`spyre-inference-xpress/.claude/skills/debug-spyre/logs/xpress-topc-update/index.html`
under the workspace root.

The latest dependency refresh and corrected-environment results are in
`pr-refresh-20261005/`: `build.log`, `tool-paths.json`, `sanity-corrected.log`,
`control.log`, `dispatch-corrected.log` (8 passed, 8 CPU cases deselected), and
`head-registered.json` / `head-registered.log`, `cpu-spec.log`, `cpu-jagged.log`,
and `e2e-jagged-c512-run2.log`. The E2E report and all cache snapshots are in
`e2e-jagged-c512-run2/test_qwen3_xpress_verification0/`; `e2e-summary.json`
aggregates the diagnostic counters above.
Earlier failing logs are retained as
investigation history, with their environment mismatch recorded in the HTML log.

The initial jagged CPU run failed because the source-only backend had not
registered `tile_dim_marker`; explicit `torch_spyre._autoload()` in
`run_cpu_jagged.py` resolved the complete CPU selection without changing its
assertions. Backend registration and compiler-path preservation fix separate
setup issues. The shared-host launcher limits VFIO cleanup to subprocesses carrying
its invocation ID; unrelated sessions' card holders are excluded.

## Remaining performance work

Priority reflects measured host costs and the size of avoidable work, not a new
device profile of the combined branch. The kernel priorities need dedicated
measurements on the validated model path.

| Priority | Cost and code location | Evidence | Next implementation and measurement |
| --- | --- | --- | --- |
| 1 | Draft full-vocabulary D2H, CPU top-k, per-pass argmax and ID feedback in [`propose_block`](../../spyre_inference/models/qwen3_dflash.py) | Head-probe score-copy intervals average 9.0 ms/block with C512, including pending compute. The new path still has six serial CPU selection boundaries. | Implement exact device top-k/argmax with integer IDs, defined ties, and fused candidate-to-vocabulary selection. Check large IDs, ties, and finite-range behavior before moving the complete Jacobi loop onto device. Measure synchronized proposal latency and acceptance. |
| 2 | Target hidden-state D2H → CPU row selection → H2D, then full-vocabulary logit D2H and CPU sampling in [`_SpyreModelWrapper`](../../spyre_inference/v1/worker/spyre_model_runner.py) and [`SpyreLogitsProcessor`](../../spyre_inference/custom_ops/logits_processor.py) | Explicit serving transfers; unaffected by draft shortlisting. Auxiliary target taps already stay on device. | Keep final hidden states on device and gather only sampled rows there; implement device greedy verification and transfer accepted IDs/required logprobs. Preserve stopping, rollback and public metrics. Profile target and draft transfers separately. |
| 3 | Full-vocabulary draft and target LM-head projections in [`parallel_lm_head.py`](../../spyre_inference/custom_ops/parallel_lm_head.py) | The unpadded 4096×151936 FP16 weight table is about 1.16 GiB. Top-C narrows only the rank-256 refiner readout. | Measure projection bandwidth and row padding independently. Evaluate fusing projection with top-k or tiled selection to avoid materializing/transferring all logits; do not assume shortlisting eliminated this projection. |
| 4 | Padded context feature concatenation, 20480→4096 FC, and five layers of context QKV projection in [`dflash.py`](../../spyre_inference/v1/spec_decode/dflash.py) and [`_project_context`](../../spyre_inference/models/qwen3_dflash.py) | Only accepted target rows are committed, but the current path computes the whole target bucket. The full QKV projection computes Q and discards it; for this checkpoint Q is two-thirds of its output width. | Gather accepted rows into a smaller warmed bucket and use correctly laid-out KV-only projection weights. Measure useful versus padded rows and FC/KV time. Preserve DLF16 range through context FC→norm and all-layer cache semantics. |
| 5 | Small refiner embedding, projection, mixer, SwiGLU and candidate BMM kernels in [`xpress_head.py`](../../spyre_inference/v1/spec_decode/xpress_head.py) | Source-level work/dispatch remains after hoisting the invariant projection. No new kernel timing is available. | Profile launch count, layout conversions and BMM utilization; fuse the latent path and selection where supported. Reuse candidate and scratch buffers. Compare with the unfused numerical reference and target acceptance. |
| 6 | Jagged host plan construction and table/slot-index H2D in [`jagged_plan.py`](../../spyre_inference/v1/attention/jagged_plan.py) and [`spyre_attn.py`](../../spyre_inference/v1/attention/backends/spyre_attn.py) | Plans and device storage are reused; tables are uploaded once per group per step and shared across layers. Builder `.cpu()` calls usually receive already-CPU runner metadata. | Count actual device transfers, not occurrences of `.cpu()`. Profile host plan time, table bytes, upload count and KV-write indices; update only changing metadata or build it on device where profitable. |
| 7 | Per-layer Q/output staging copies and 64-row minimum jagged prefill tiles | A 16-row proposal or verification block has 4× row capacity in the 64-row attention tile, before page and power-of-two padding. Actual wasted device work needs measurement. | Profile staging D2D bytes and useful tile occupancy. Evaluate a 16-row speculative tile or wider grouped-decode support, and direct writes into output staging. Check sink rows, tails and both cache layouts. |
| 8 | Jagged page walks, repeated KV reads, narrow decode GEMMs and compensated softmax reductions in [`jagged_tile_attn.py`](../../spyre_inference/v1/attention/ops/jagged_tile_attn.py) and [`jagged_decode_attn.py`](../../spyre_inference/v1/attention/ops/jagged_decode_attn.py) | Algorithmic suspects, not a measured ranking. Page size, entry count, context length and request grouping change their cost. | Sweep page/entry budgets and compare grouped, tiled and split schedules with common inputs. Measure bandwidth, utilization and latency. Keep compensation/numerical gates until an alternative passes both attention and model checks. |

The historical device `argmax` fallback and incorrect large FP16 `topk` indices
explain the current explicit CPU boundary. They should be rechecked on the exact
pinned stack. Moving the same operations onto a nominal Spyre tensor without
removing their fallbacks would retain the transfers and obscure their cost.

### Calculated draft payload

For block size 16, vocabulary 151936, six passes and C=512, the score-copy counters
count FP32 host elements:

| Path | Calculation | Bytes per proposal |
| --- | --- | ---: |
| Original full-vocabulary implementation | `7 × 16 × 151936 × 4` | 68,067,328 |
| Updated full-vocabulary mode | `(16 + 6 × 15) × 151936 × 4` | 64,420,864 |
| Updated top-C mode | `16 × 151936 × 4 + 6 × 15 × 512 × 4` | 9,908,224 |

Top-C reduces this logical payload by 85.4% versus the original implementation.
It is a calculation, not measured wire traffic or a latency improvement, and
excludes target verification. The initial full-vocabulary copy alone accounts
for 9,723,904 bytes. Candidate IDs and base scores add an estimated 46,080 bytes
of H2D per proposal; the per-pass previous-ID uploads are small but synchronize
the loop.

The full rank-256 readout spans 74.2 MiB of FP16 weights. The 15 per-position
candidate tables occupy 3.75 MiB total, gathered once and reused across six
passes. Those are logical tensor sizes; cache behavior and kernel layouts
determine actual memory traffic.

## Verification and benchmark sequence

Use the workspace lower stack and a private checkout of torch-spyre at the pin
above. Source `/mnt/home/spyre/torch-spyre-docs/scripts/dev-env.sh`, then put the
chosen inference and torch-spyre worktrees first on `PYTHONPATH`. Check the
existing weights in `/mnt/models/hf_cache`. Choose a free card with
`SPYRE_DEVICES`; run our device commands serially and leave other sessions alone.

1. Verify a minimal compiled-add and matmul baseline with a fresh compile cache.
   Check both `command -v dbo-opt dxp_standalone` and the loaded Deeptools,
   runtime/Flex, libaiupti and spyre-comms paths point under
   `/mnt/home/spyre/sentient`, with only senlib under `/opt/ibm/spyre`. Preserve the
   canonical script's `PATH` when selecting a different venv.
2. Run CPU checks and the real-checkpoint head probe. Check candidate gather IDs
   and weights exactly, compare latent and score values with the upstream FP32
   reference, and examine close-token margins in full and shortlist modes.
3. Run the existing E2E suite for all four combinations of
   `SPYRE_JAGGED_ATTENTION=0/1` and `SPYRE_XPRESS_TEST_TOPC=0/512`. Confirm that
   target and draft layers actually execute the requested attention backend;
   preserve token, stopping, acceptance-counter and committed-cache checks.
4. Benchmark ordinary decoding in a separate engine, K=0, K=6/full and K=6/C512
   under both attention modes. Keep model revisions, prompts, thinking mode,
   output limits, context, cache layout, card and stack identical. Warm all shapes
   before measuring. Disable cache snapshots and debug logging for timings.
5. Report request/ITL percentiles, accepted tokens per round, context/draft/refiner/
   target time, actual transfer bytes, kernel launches, peak memory, and useful
   versus padded token/page counts. Use synchronization or device events for
   bounded microbenchmarks; retain normal asynchronous execution for E2E timing.

CPU commands from this worktree:

```bash
uv run --no-sync pytest tests/spec_decode -m 'not upstream'
uv run --no-sync pytest tests/attention/test_jagged_plan.py \
  tests/attention/test_jagged_page_attn.py tests/attention/test_jagged_backend.py \
  -m 'not upstream' -k 'not spyre'
```

One cell of the hardware matrix:

```bash
export SPYRE_JAGGED_ATTENTION=1
export DXP_LOOP_UNROLL=0
export SPYRE_XPRESS_TEST_TOPC=512
uv run --no-sync pytest tests/e2e/test_speculative_decoding.py \
  -m model_quality -s --basetemp=/path/to/new-jagged-topc512-artifacts
```

Use a fresh artifact directory for each invocation. Passing the primitive and
attention tests is a prerequisite, not a substitute for the model checks.
