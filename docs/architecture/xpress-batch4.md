# Batch-four XPress and jagged attention

This branch extends the Qwen3 XPress adapter from one to four active requests.
It builds on `feat/xpress-performance` (`933699e`) and retains the combined
attention implementation from spyre-inference PR #1118 and the private
torch-spyre PR #5086 build at `fcc50670`.

For the measured batch-four workload, jagged attention increased median request
wave latency by 1.7% for ordinary decoding and 20.8% for XPress. The comparison
used heterogeneous prompt lengths and observed actual four-request proposals.

## Implementation

The proposer partitions packed target rows by request, commits each accepted
prefix to that request's draft KV pages, and selects eligible requests for one
packed draft forward. Positions, anchors and block tables remain independent.
Partial prefills commit context without proposing; output and context limits
disable proposals per request. Rejected rows and padding write the null page.
Active draft batches use one, two or four blocks, so three active requests
perform one padded block of extra work. All three shapes are warmed at startup.

Four full verification requests require 124 scheduler slots: four times the
16-token target block plus 15 extra drafting slots. Use a token budget of at
least 128 and matching compile sizes. The configuration minimum of 64 only
ensures that the padded draft buffers fit. A budget of 256 is used for the
longer-prompt request benchmark to keep the early requests active while the
remaining prompts finish prefill.

Batched candidate readout requires flattened gather indices on the tested
stack. Native feedback uses independent low/high base-512 ID buffers and two
compiled stages: select the digits, then reconstruct IDs and assemble the
predecessor sticks. Fusing those stages produced incorrect batched IDs on
refiner output layouts. This split preserves exact vocabulary IDs and anchors
without returning intermediate refinement scores to CPU.

## Validation and measurement method

The device selector passed random, negative, all-equal and edge-tie cases at
batches one, two and four. A checkpoint-head probe also matched CPU feedback
exactly over 20 alternating comparisons per batch. Its head-only medians were
12.269/12.180 ms, 20.353/19.902 ms and 38.767/38.148 ms for CPU/native feedback
at batches one, two and four. These timings exclude the draft backbone and
LM-head projection and do not measure request speedup. The speculative CPU
suite passed 182 tests before full-model validation.

The complete Qwen3 batch-four test passed on October 7 with regular attention
(526.40 seconds including initialization) and jagged attention (570.39 seconds).
Both checked mixed acceptance, partial prefills, context/output limits,
cancellation of one request followed by the remaining three, page reuse,
batch-one regression and token stopping. All 328 committed target/draft K/V
prefix tensors matched exactly when rejected tails changed and again after
cancellation and reuse. Validation allowed first-use attention compilation;
those durations are not latency measurements.

The benchmark runs separate engines for
ordinary and XPress decoding and for regular and jagged attention. Both use
the same Qwen3 target snapshot, device, CPU thread count and lower-stack builds.
Request admission pauses scheduling until all four requests are queued.
Without that barrier the background engine can begin prefill after the first
submission, changing the actual batches between runs.

The eight-token prose prompt exposed a target numerical boundary at output
position 21: ordinary decoding ranked token 1084 (" It") above token 576
(" The") by 0.03125 in log probability; block verification tied them exactly
and selected 576. Jagged attention reversed the same 0.03125 margin between
ordinary and block execution. Its prompt caches matched exactly; the final
layer's K/V relative errors over the shared input prefix were 0.30%/0.84%.
The test retains this case with a near-tie gate: both executions' gaps must be
at most 0.03125. Other prompt outputs must match exactly, and all emitted tokens must
have maximal reported target probability. Cross-shape cache comparison stops
at the first different input token. Changed rejected tails and cancellation
with the same acceptance schedule still require bit-exact outputs and caches.
Greedy output identity across all FP16 execution shapes is therefore not a
general guarantee of this integration.

Measurements record token sequences, acceptance, proposal batch histograms,
arrival bursts and graph counts. State RPCs and artifact writes are outside the
timed region. Timed repetitions must create no new graphs and must observe four
concurrent requests. Source and loaded-library hashes must match across the
attention comparison. Different acceptance or arrival schedules are reported
explicitly rather than attributed entirely to attention.

## Batch-four request results

Measurements on October 7 used Qwen3-8B on card 2, FP16, TP1, eight CPU threads,
head-major KV pages, greedy decoding, ignored EOS and disabled prefix caching.
Four prompts of 32, 127, 255 and 383 tokens were admitted together; each generated
128 tokens, for 512 output tokens per wave. The context limit was 512 and the
scheduler token budget was 256. XPress used six passes and top-512 candidates.
Each configuration ran in a fresh process, with one workload warmup and three
timed repetitions. Attention pre-recording was disabled, and all timed
repetitions compiled zero new graphs. Initialization is excluded.

| Decoding | Attention | Median wave, s | Output tokens/s |
| --- | --- | ---: | ---: |
| Ordinary | Regular | 19.183 | 26.690 |
| Ordinary | Jagged | 19.512 | 26.240 |
| XPress K6/C512 | Regular | 4.037 | 126.821 |
| XPress K6/C512 | Jagged | 4.877 | 104.973 |

All four configurations produced identical output tokens. Each regular/jagged
pair also matched acceptance, proposal batch counts and the step/token counts
of arrival bursts. Source and loaded-binary hashes, CPU affinity and thread
counts matched across all four configurations.

Each XPress repetition verified 33 drafts: 493 of 495 proposed tokens were
accepted (99.6%). It issued two one-request, three two-request, three
three-request and four four-request proposals. Thus the comparison includes
varying active batch sizes as well as distinct request lengths. Ordinary
decoding emitted tokens for all four requests together in 125 engine steps.

XPress was 4.75 times faster than ordinary decoding with regular attention and
4.00 times faster with jagged attention on this workload. These are
high-acceptance results on repeated prose. They do not establish production
speedup or performance at long contexts. Jagged's 1.7% ordinary and 20.8% XPress
latency increases describe these short-context runs; larger and more uneven
contexts still need measurement.

The following counters are sums over the three timed XPress repetitions.
Intervals are host wall time, can overlap, and include waiting for asynchronous
device work. They are not exclusive kernel or DMA durations.

| Counter | Regular | Jagged |
| --- | ---: | ---: |
| Proposal calls | 36 | 36 |
| Actual / padded request blocks | 99 / 108 | 99 / 108 |
| Accepted context tokens committed | 3969 | 3969 |
| Proposal interval, s | 3.208 | 4.100 |
| Draft score download interval, s | 1.981 | 2.648 |
| CPU argmax + top-k, s | 0.275 | 0.327 |
| Context projection/store interval, s | 0.239 | 0.301 |
| Draft-forward interval, s | 0.430 | 0.398 |
| Logical draft-logit download bytes | 1,050,181,632 | 1,050,181,632 |
| Logical candidate upload bytes | 8,294,400 | 8,294,400 |
| Logical final proposal-ID download bytes | 6,480 | 6,480 |

The initial draft score download is the largest instrumented interval inside
the proposer. It also waits for earlier compute, so a device trace is needed to
separate projection and attention execution from transfer cost. The equal work
counters support the attention comparison, but do not attribute its entire
latency difference to a particular kernel or transfer.

[Recorded validation and measurement data](data/xpress-batch4-20261007.json)
includes all four reports, both attention comparisons, per-request timings,
arrival bursts and source/library fingerprints. The recorded Python source
hash identifies the measured runtime; the Git manifest records its pre-commit
state over `933699e`.

## Attention-only baseline

The earlier pure-decode probe used four queries of one token each, 32 query
heads, eight KV heads, head size 128, 128-token pages and varying context
lengths `[L, 7L/8-13, 3L/4+7, 5L/8+17]`. Each path had 20 warm samples;
independent FP32 correctness checks passed and no compilation occurred during
timing.

| Maximum context | Regular median, ms | Jagged median, ms |
| ---: | ---: | ---: |
| 1024 | 0.474260 | 0.817324 |
| 2048 | 0.762571 | 1.150704 |
| 4096 | 1.293466 | 1.433685 |

Jagged was slower for these one-token queries. This probe does not cover the
multi-token verification and noncausal draft attention used by XPress.

## Remaining performance work

| Work | Batch-four impact | Next measurement or change |
| --- | --- | --- |
| Draft full-vocabulary transfer and CPU top-k | A full padded batch copies 38,895,616 logical FP32 score bytes to CPU before refinement, about 37.1 MiB. | Exact-ID device argmax/top-k, followed by tiled projection and selection. |
| Target hidden-state and logit transfers | Target states return to CPU for row selection, go back to Spyre for the LM head, then logits return for rejection sampling. | Device row selection and greedy verification; measure actual DMA separately from waiting on compute. |
| Full-vocabulary projection | Target and draft use the roughly 1.16 GiB FP16 LM-head weight table. | Measure bandwidth and avoid materializing the complete vocabulary where a selection kernel can consume tiles. |
| Padded context work | The 20480-to-4096 context FC and five KV projectors process padded and rejected target rows. | Measure accepted-row compaction, including its gather and metadata costs. |
| Three-request padding | Three drafts execute four model blocks, including projection, top-k and refinement. | Compare a supported 48-row path against current warm shape buckets. |
| Six sequential refiner passes | Small embedding, mixer, MLP and candidate-readout operations remain, followed by two selector stages per pass. | Profile launch and memory costs; fix the compiler layout issue before fusing selection and feedback. |
| CPU scheduling and metadata | Request packing, acceptance bookkeeping, draft-slot reservations and plan uploads remain on the host. | Trace host critical-path time and minimize repeated metadata construction. |
| Jagged staging and kernel efficiency | A 16-token draft/verification block currently uses a minimum 64-row prefill tile; plan staging and joins can dominate small batches. | Profile the batch-four workload and evaluate smaller tiles with the same numerical checks. |

Byte counters describe logical tensor payloads, not measured PCIe traffic.
Host timers can include pending asynchronous device work and must not be summed
as exclusive kernel times. The request measurements identify the draft logit
download as a priority for tracing; exclusive device timings are still needed
to rank kernel work against transfers and host scheduling.

Commands are in the [microbenchmark guide](../../scripts/microbench/README.md#batched-xpress-request-latency).
Local evidence is under `/mnt/home/spyre/xpress-batch4-validation/20261006` and
`20261007`; the latter contains the completed full-model validation and request
measurements.
