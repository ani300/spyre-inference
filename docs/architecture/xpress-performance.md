# Qwen3-8B XPress performance measurements

The controlled attention comparison found no speedup from jagged attention on
this short-context, single-request workload. Request latency was about 3–5%
higher with jagged enabled. Generated tokens and public acceptance counters
matched in every run. This result does not cover long contexts, concurrent
requests or another model.

The follow-on performance branch reduced request latency by 5.3–6.6% against
that combined jagged baseline on the same workload. Output tokens and public
acceptance counts stayed identical. With the same optimizations, jagged
attention still measured 1.8–3.3% slower than regular attention.
Keep jagged disabled for this workload. The changes address proposal and transfer
costs around attention; the results do not establish a production speedup.

## Attention comparison

The October 5, 2026 comparison used the combined branch at `ee20818`, with the
model and dependency pins in [the integration report](xpress-jagged-integration.md).
The benchmark harness is committed as `926b9d6`. Both modes ran from the same
source tree; the changed setting was `SPYRE_JAGGED_ATTENTION`. This isolates the
attention option within the combined branch, rather than comparing all changes
between two different branch heads.

Settings: Qwen3-8B, the published XPress b16 checkpoint, FP16, TP=1, one active
request, K=6, C=512, head-major KV, maximum model length 512, scheduler budget 64,
and compile sizes `[1,16,64]`. Each request generated 64 tokens with greedy
sampling and EOS ignored. The prompts were 32, 127 and 383 tokens of repeated
prose; their high acceptance rate exercises full speculative blocks, but is not
a representative chat sample.

Fresh processes ran regular, jagged, jagged, regular on physical card 2. Each
process warmed all three prompts, then measured each twice in opposite case
orders. There are four samples per case per mode. Model loading and warmup are
excluded. Worker RPCs, file writes and KV-cache inspection were outside timed
requests. All 41 attention layers used the requested backend, including the
five noncausal draft layers. Torch used eight CPU threads. The compiler,
extension and loaded lower-stack hashes were unchanged across the comparison.

| Prompt tokens | Regular median (range), s | Jagged median (range), s | Jagged request-time change |
| ---: | ---: | ---: | ---: |
| 32 | 1.169 (1.144–1.193) | 1.214 (1.193–1.239) | +3.9% |
| 127 | 1.287 (1.259–1.300) | 1.324 (1.304–1.353) | +2.9% |
| 383 | 2.068 (2.058–2.070) | 2.175 (2.142–2.177) | +5.2% |

| Prompt tokens | Regular / jagged TTFT, ms | Regular / jagged effective TPOT, ms |
| ---: | ---: | ---: |
| 32 | 188.9 / 196.1 | 15.56 / 16.16 |
| 127 | 480.7 / 492.5 | 12.78 / 13.20 |
| 383 | 1239.4 / 1289.0 | 13.09 / 14.04 |

Effective TPOT divides elapsed time after the first token by the remaining
63 output tokens. Speculative tokens arrive in bursts; this is not a per-burst
latency. Ranges describe these observations, not confidence intervals.

The reproducible commands are in [the microbenchmark guide](../../scripts/microbench/README.md#xpress-request-latency).
Full local evidence is under
`/mnt/home/spyre/xpress-performance-validation/20261005-attention-ab`:
`regular-1.json`, `jagged-1.json`, `jagged-2.json`, `regular-2.json`,
`comparison.json`, and the before/after stack manifests. The benchmark retains
tokens, arrival times, acceptance counts, worker configuration and raw samples.

## Implementation changes

The performance branch adds three changes without changing the XPress recurrence:

- After CPU bookkeeping reaches `max_tokens`, the proposer commits the accepted
  context and skips the unused next draft forward, LM-head projection and
  refinement loop. In the baseline each mode ran 64 proposals for 52 verified
  draft rounds over 12 requests: one unused final proposal per request.
  At K6/C512, each skipped proposal avoids 9,908,224 bytes of logical score
  copies, seven CPU selections and the associated device work. EOS and text-stop
  detection remain with the existing stopping path.
- Context preparation uses a packed KV-only weight for each draft layer.
  The live Spyre weight has layout `[input, Q|K|V]`; packing its K/V columns once
  during preparation avoids computing and discarding Q. The draft forward
  retains the full QKV projection. This adds 80 MiB of logical device weights
  for the five layers and reduces each context projection's output width from
  6144 to 2048. The large context FC and padded context rows remain.
- For b16/C512 with vocabulary sizes from 512 through 262144, refinement selects
  candidates and assembles the next pass's predecessor IDs on Spyre. The initial
  full-vocabulary score copy, CPU argmax and CPU top-k remain. Six per-pass score
  copies and predecessor uploads become one final copy of 15 int32 IDs (60
  logical bytes). Other configurations retain CPU feedback, and K=0 still uses
  the base selection. Candidate selection keeps first-index ties and reconstructs
  vocabulary IDs from two exact base-512 digits. A `[16,32]` int32 predecessor
  buffer, populated in that layout from its first upload, avoids unsupported
  writes at an offset within a stick. This adds 30,720 logical candidate-upload
  bytes and a 2,048-byte initial predecessor upload per proposal.

A standalone projection probe used all five checkpoint layers with common
random context inputs at 1, 16 and 64 rows. Each path had 20 synchronized warm
samples per layer and shape, with alternating order. KV-only projection took
about 0.27 ms per layer versus 0.50–0.54 ms for QKV. This is a projection result,
not an end-to-end speedup. Outputs were exact at 1 and 16 rows; the two paths
differed by about 0.245% relative L2 at 64 rows. Both passed the independent
FP32 comparison with a 2% relative-L2 gate. Removing Q changes the GEMM schedule,
so model and cache validation remains necessary.

The 117 speculative CPU tests pass, including output-count boundaries,
overshooting the output limit in an accepted block, preserved accepted-context
writes, packing live transposed weights without serializing derived buffers,
large vocabulary IDs, tied scores, anchor preservation and the serving recurrence.
The projection probe and raw timings are in `context_kv_probe.py`,
`context_kv_probe.json` and `context_kv_probe.log` in the evidence directory.

A paired hardware probe called the edited serving `propose_block` with the
checkpoint head and one captured block, using precomputed base logits. Forty
alternating repetitions per path returned identical proposals. Median latency
was 15.287 ms with CPU feedback and 14.465 ms with device feedback, a 5.4%
reduction for this bounded head measurement. It excludes the draft backbone,
LM-head projection and target execution. Native selection also passed random,
all-equal, negative and edge-tie probes with fallback warnings treated as errors.
The exact integrated probe and raw samples are retained in
`spyre-inference-xpress/.claude/skills/debug-spyre/logs/debug-20261005-194727-xpress-selector-layout/`
under the workspace root as `head_feedback_integrated.py` and
`head_feedback_integrated.json`.

The optimized jagged/C512 full-model test passed in 555.25 seconds, including
model loading and warmup. All six prompts matched their target-only token
sequences across controlled rejection, full acceptance, real DFlash and real
XPress modes. It checked public acceptance counters, token/text stops, output
limits, all 36 target and five draft caches, and cancellation followed by page
reuse. Under the same acceptance schedule, changing rejected suffixes preserved
committed caches exactly; cancellation/reuse also matched all 82 K/V tensors
exactly. Cross-shape target-only versus speculative cache checks retain the
existing FP16 gates; this does not establish exact arithmetic across schedules
or production-model quality. The full evidence is in
`e2e-performance.log` and `e2e-performance/test_qwen3_xpress_verification0/`
under the evidence directory. No fallback warning was accepted.

## Request measurements after the changes

The optimized model code is `4ea65e8`. Each attention mode used a fresh process
with four measured requests per prompt, matching the baseline sample count.
The optimized jagged process ran first, followed by regular attention. Model
settings, card, stack and warmup procedure matched the attention comparison.
These are combined before/after results for all three optimizations, not an
attribution of request speedup to one kernel. The baseline used two processes
per mode; the optimized measurement used one. The sample size does not support
confidence intervals or production conclusions.

| Prompt tokens | Jagged baseline median, s | Optimized jagged median (range), s | Request-time change | Effective TPOT before / after, ms |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.214 | 1.134 (1.133–1.137) | −6.6% | 16.16 / 14.96 |
| 127 | 1.324 | 1.251 (1.245–1.270) | −5.5% | 13.20 / 12.17 |
| 383 | 2.175 | 2.060 (2.055–2.062) | −5.3% | 14.04 / 12.66 |

Regular attention also benefits from the proposal changes. Both optimized modes
matched every baseline output token and public acceptance count:

| Prompt tokens | Optimized regular median (range), s | Regular change from baseline | Optimized jagged overhead vs regular |
| ---: | ---: | ---: | ---: |
| 32 | 1.114 (1.107–1.119) | −4.7% | +1.8% |
| 127 | 1.219 (1.216–1.227) | −5.3% | +2.7% |
| 383 | 1.993 (1.987–2.008) | −3.6% | +3.3% |

Jagged remains slower in this comparison; its integration alone is not the
source of the improvement. Long contexts and concurrent requests need separate
measurements. Over 12 measured requests in each mode, the work counters matched:

| Work | Baseline | Optimized |
| --- | ---: | ---: |
| Public verified draft rounds | 52 | 52 |
| Actual proposal calls | 64 | 52 |
| CPU selections | 448 | 52 |
| Device refinement selections | 0 | 312 |
| Logical score bytes returned to CPU | 634,126,336 | 505,643,008 |
| Final native proposal-ID bytes returned to CPU | 0 | 3,120 |
| Candidate upload counter, logical bytes | 2,949,120 | 3,993,600 |

The score-copy payload fell 20.3%, largely by skipping 12 unused proposals.
The native selector uploads an additional exact ID-digit table, so candidate
upload bytes increase even as the number of CPU feedback boundaries falls.
The initial predecessor buffers add 106,496 logical H2D bytes across this run,
outside the candidate counter. These counts describe tensor payloads, not
measured PCIe traffic. Accepted-context processing remains present on the final
step: the context-token counter stayed at 2,992.

Steady allocated device memory increased from 19,042,037,376 to 19,126,107,776
bytes with jagged enabled, about 80 MiB. Peak allocated memory remained
19,758,510,592 bytes. This is an observed allocator result for this workload,
not a bound for another context length or batch size.

The [committed raw evidence](data/xpress-performance-20261005.json) contains all
48 measured requests, their input/output token sequences, arrival bursts,
acceptance, per-process counter deltas, source hashes, stack hashes and focused
probe samples. The local directory additionally retains
`performance-jagged.json`, `performance-regular.json`,
`performance-comparison.json` and `stack-performance-comparison.json`.
Torch, vLLM and Transformers versions were unchanged, as were the extension,
compiler and loaded lower-stack hashes, across all baseline and optimized
processes. Model-code changes are limited to the three optimizations described
above.

## Remaining bottlenecks

The baseline's proposer averaged 52–54 ms per call. Context preparation accounted
for about 8 ms in its host timer. Score-copy intervals averaged 32–34 ms, but
include waiting for earlier device work; they are not measurements of DMA alone.
Dispatch and copy timers overlap pending work and must not be summed as exclusive
kernel costs.

After optimization, the proposer averaged about 50 ms per actual proposal in
both modes. The combined score/final-ID copy intervals still averaged 31–32 ms,
including pending compute. The remaining initial vocabulary transfer and target
path therefore need direct profiling before ranking their exclusive costs.

| Work | Current evidence | Next step |
| --- | --- | --- |
| Initial draft full-vocabulary copy and CPU top-k | The initial copy is 9,723,904 logical bytes per proposal; C512 narrows subsequent readouts only. | Device selection with exact IDs and defined ties, then projection/selection fusion. |
| Target hidden-state and logit transfers | Final hidden states go to CPU for row selection, back to Spyre for the LM head, then logits return to CPU for verification. | Add a device row-selection path and exact device greedy verification. |
| Full-vocabulary LM-head projections | Draft and target still project through a roughly 1.16 GiB FP16 weight table. | Measure projection bandwidth and evaluate tiled selection without full-logit materialization. |
| Padded context FC | Auxiliary-state concatenation and the 20480→4096 FC still process the target bucket. | Compare accepted-row compaction and smaller warmed context buckets, including gather cost. |
| Refiner kernel overhead | Embedding, mixer, MLP and candidate BMM remain small kernels. | Fuse readout, selection and feedback where the compiler supports the required layouts. |
| Jagged metadata, staging and small tiles | The 16-token speculative block uses a minimum 64-row prefill tile; host plans and uploads remain. | Profile those costs separately and test a smaller speculative tile with the existing numerical gates. |
| Jagged KV reads and reductions | The request benchmark establishes a short-context regression, not its kernel-level cause. | Sweep context/page schedules and measure bandwidth and compensated reductions before changing them. |
| CPU refinement feedback outside b16/C512 | The native path covers the first checkpoint configuration; other block sizes and candidate counts still use CPU feedback. | Expand only with exact-ID, tie, layout and complete-head validation for each supported shape. |
