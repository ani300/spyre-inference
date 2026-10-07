# XPress jagged scheduling experiments

Batch-four request latency improved with both jagged scheduling experiments on
the rebuilt profiler stack. Request/page decode parallelism reduced resident
decode latency by 12–41%; smaller speculative tiles reduced XPress request
latency by 4.43%. Combined settings improved ordinary and XPress latency by
about 3.1% versus their jagged controls. XPress still trails regular attention,
and profiles identify attention execution as the main remaining regression.

Measurements were collected on 2026-10-07. Both settings remain opt-in.
[Machine-readable evidence](data/xpress-jagged-scheduling-20261007.json) includes
raw samples, validation, work comparisons, stack hashes and profile summaries.

## Implementation and stack

Inference code: `perf/xpress-jagged-scheduling`, commit `4e42155`, based on the
batch-four branch at `24afce0`. The two controls are:

| Setting | Default | Experiment |
| --- | ---: | ---: |
| `SPYRE_JAGGED_BATCH_PARALLEL` | 0 | 1 |
| `SPYRE_JAGGED_MIN_QUERY_TILE` | 64 | 16 |

Decode planning previously limited parallel entries using the page bucket alone.
For four requests this left four serial page-loop iterations at every tested
length. The new option uses request-tile capacity times page capacity, capped by
the existing 64-entry serving limit. It reduces those iterations to one through
context 2048, and two at 4096, without increasing total padded page work.

The small-tile option changes automatic query widths above eight tokens to allow
16 or 32 rows. Shorter groups still use the decode kernel. Warmup enumeration and
runtime share both controls. Width-16 output stores preserve each short int32
index row's separate stick instead of flattening incompatible layouts.

Reproduction requires the local torch-spyre branch
`perf/xpress-jagged-profile-stack`: jagged companion `fcc50670`, the upstream host-compute/H2D split from [torch-spyre PR #5073](https://github.com/torch-spyre/torch-spyre/pull/5073)
(`de03533`, cherry-picked from `8744437`), and the host-capacity fix `87e91315`.
The extension was built with `USE_SPYRE_PROFILER=1`. Runtime, Deeptools, libaiupti
and spyre-comms are loaded from `/mnt/home/spyre/sentient`; senlib comes from
`/opt/ibm/spyre/senlib`. Commands source the workspace's canonical `dev-env.sh`,
put the private worktrees first on `PYTHONPATH`, and use `uv run --no-sync`.
All device runs are sequential on card 2, with eight CPU threads and loop unrolling
disabled. Weights are taken from the local Hugging Face cache.

The attention matrix used the profiler extension at `de03533`. Full-model testing
then exposed a compatibility issue with the rebuilt runtime's D2H bounds checks:
the old caller omitted the host buffer capacity for FP16-to-FP32 conversion.
Commit `87e91315` passes the actual CPU storage capacity; it does not disable the
checks or insert a device cast. The request matrix uses this final extension.
Comparisons within each matrix use identical binaries. The two phases have
different extension hashes and are not a paired before/after extension benchmark.

## Correctness

The planner, warmup and native small-tile selection passed 44 tests. Native cases
cover widths 16/32/64, causal/noncausal attention, batches 1/3/4, changed inputs
without recompilation, masked/null pages and unused output rows. A diagnostic
compared all 36 matched-input width cases: outputs were bit-for-bit equal across
16/32/64. The FP32 comparison retains a 2% relative-L2 gate and the repository's
bounded-outlier policy; maximum observed relative L2 was 0.333%.

Eight direct-copy regressions passed after the runtime compatibility fix, covering
ordinary and FP32-widening copies with aligned and padded device shapes. Separate
diagnostic probes found unsupported arbitrary CPU destination-view behavior:
offset copies clear storage before the view, and strided/transposed destinations
receive incorrect values. Those failures are preserved in the artifacts and are
not claimed fixed. The logits path uses a contiguous destination.

The full Qwen3 batch-four test passed in 799.59 seconds including loading,
compilation and validation. It checks controlled rejection, changed rejected
suffixes, real proposals, partial prefill, context limits, cancellation/page reuse,
single-request regression and token stopping. Its existing short-prompt FP16
near-tie occurred at position 21 in the controlled cases, with logprob gap
0.03125012. The real-proposal wave matched its ordinary reference exactly. No
full-model tolerance or expectation was changed.

## Attention-only experiment

Geometry: 32 query heads, eight KV heads, head size 128, 128-token pages and four
unequal context lengths `[L, 7L/8-13, 3L/4+7, 5L/8+17]`. Decode uses one query per
request; verification and draft use 16 each, with causal and noncausal masks
respectively. Q/KV and metadata are resident. Each comparison has three warmups,
20 unprofiled samples and three separately profiled calls per backend/case.

All 120 rows passed their independent correctness checks, before and after timing,
with zero new graphs during measurement. Latency includes backend dispatch,
output stores/copies and synchronization. It excludes metadata preparation/upload,
query staging, KV insertion and the model. Four jagged arms isolate the changes:
control `(0,64)`, parallel `(1,64)`, small `(0,16)` and combined `(1,16)`.

Median unprofiled backend latency in milliseconds:

| Case / maximum context | Control | Parallel | Small | Combined |
| --- | ---: | ---: | ---: | ---: |
| decode4 / 256 | 0.5298 | 0.3568 | 0.5642 | 0.3232 |
| decode4 / 512 | 0.4934 | 0.3976 | 0.4951 | 0.3820 |
| decode4 / 1024 | 0.7289 | 0.5214 | 0.8596 | 0.5323 |
| decode4 / 2048 | 1.1992 | 0.7092 | 1.0790 | 0.6918 |
| decode4 / 4096 | 1.4110 | 1.2467 | 1.3505 | 1.2562 |
| verify4 / 256 | 1.4710 | 1.6249 | 1.3166 | 1.3162 |
| verify4 / 512 | 2.3456 | 2.3408 | 2.2702 | 2.3213 |
| verify4 / 1024 | 4.1181 | 4.2836 | 4.1831 | 3.9819 |
| verify4 / 2048 | 7.6409 | 7.6629 | 7.6145 | 7.6033 |
| verify4 / 4096 | 14.7409 | 14.6843 | 14.8084 | 14.7317 |
| draft4 / 256 | 1.4770 | 1.6063 | 1.3027 | 1.3136 |
| draft4 / 512 | 2.3553 | 2.3366 | 2.2566 | 2.1954 |
| draft4 / 1024 | 4.1140 | 4.2695 | 3.9572 | 3.9929 |
| draft4 / 2048 | 7.6041 | 7.7184 | 7.5899 | 7.6424 |
| draft4 / 4096 | 14.7529 | 14.6668 | 14.6330 | 14.7291 |

| Maximum context | Decode kernel, control → parallel, µs | Reduction | Verify kernel, control → small, µs | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 259.7 → 111.8 | 56.9% | 1179.9 → 1001.6 | 15.1% |
| 512 | 252.3 → 171.8 | 31.9% | 2032.2 → 1845.8 | 9.2% |
| 1024 | 482.6 → 240.8 | 50.1% | 3741.7 → 3496.8 | 6.5% |
| 2048 | 790.7 → 417.9 | 47.2% | 7150.3 → 7113.7 | 0.5% |
| 4096 | 1015.7 → 898.0 | 11.6% | 14148.8 → 13672.4 | 3.4% |

Parallel-only decode reduced wall latency by 12–41%, with a matching reduction
in actual device kernel duration. Width-16 verification reduced kernel duration
by about 0.5–15%; the benefit shrinks with longer contexts. Small wall-time
differences can reflect host noise: unchanged kernel cases sometimes vary by
around 10% in wall time. A separately profiled sample is not a latency sample.

The regular attention rows in this harness use query buckets `[1,512]`, so their
16-token calls are padded to 512. They are retained as measurements of that
configuration, but do not establish a regular-serving versus jagged speedup for
XPress. The request matrix uses query buckets `[1,16,64,256]` for the serving
comparison.

## Small-tile core utilization

The emitted SDSC descriptions show a concrete remaining inefficiency:

| Tile width | Q×K cores | Q×K head split | P×V cores |
| ---: | ---: | --- | ---: |
| 16 | 1 | all eight KV heads and four query groups on one core | 32 |
| 32 | 1 | all eight KV heads and four query groups on one core | 32 |
| 64 | 32 | eight KV heads × four query groups | 32 |

The width-16 Q×K operation really has `N_.y_=16`; it is not simply padded back to
64 rows. Reducing query work also changes its core partition. This is an observed
compiler inefficiency; the exact decision causing the one-core assignment has
not been isolated. The scratchpad joint solver's memory-only objective is a
candidate to investigate, not a proven cause. Do not blindly pin all FP16 matmuls:
layout compatibility and addressing require their own numerical validation.

## Request matrix

Median unprofiled latency for the complete four-request wave, in seconds.
Positive reduction means faster than the corresponding jagged control.

| Configuration | Median, s | Sample range, s | Reduction vs jagged control |
| --- | ---: | ---: | ---: |
| Ordinary / regular | 19.2623 | 19.0909–19.3437 | +0.65% |
| Ordinary / jagged control | 19.3890 | 19.3825–19.4449 | — |
| Ordinary / combined | 18.7976 | 18.7751–18.8802 | +3.05% |
| XPress / regular | 3.7966 | 3.7613–3.8248 | +14.24% |
| XPress / jagged control | 4.4269 | 4.3487–4.4689 | — |
| XPress / parallel only | 4.3560 | 4.2864–4.3625 | +1.60% |
| XPress / small tiles only | 4.2308 | 4.2296–4.2401 | +4.43% |
| XPress / combined | 4.2896 | 4.2563–4.3205 | +3.10% |

Ordinary combined latency is 3.05% below jagged control and 2.41% below regular.
Its effective inter-token time improves across all four requests; last-prompt
TTFT remains slightly behind regular, 767 versus 745 ms.

For XPress, parallel only reduces the median by 1.60%, with overlapping sample
ranges. Small tiles alone give the lowest observed jagged latency, 4.231 s
(4.43% below control); combined gives 4.290 s (3.10% below control).
The improvements do not add at request level. Combined is 1.39% slower than
small only in these samples; its inclusive CPU argmax counter is also higher,
70 versus 24 ms. That variation does not establish a decode-kernel regression.
Regular remains faster: small only is 11.44% above regular latency, and combined
is 12.99% above it.

All eight configurations have identical output token IDs. Within each mode,
acceptance, proposal/output batch histograms, and token-arrival steps match.
All 24 timed waves have zero new graphs and unchanged source/binary hashes.
The three profiled waves also match their unprofiled work signatures and compile
no new graphs. Every XPress wave accepts 493 of 495 proposed tokens over 33 draft
blocks. Twelve proposal calls have batch histogram `[0,2,3,3,4]`, meaning two
calls with one request, three with two, three with three, and four with four.

These are three repeats of one synthetic, high-acceptance workload, in sequential
configuration order. They demonstrate an improvement here, not a production
speedup or a statistically established ranking between nearby configurations.
Keep both controls opt-in while improving attention core utilization and testing
more representative context lengths and acceptance rates.

The workload queues four synthetic prompts of 32/127/255/383 tokens while
scheduling is paused, then produces 128 tokens per request. It uses cached Qwen3-8B and the
published XPress b16 checkpoint, FP16, greedy decoding, EOS ignored, TP1,
context 512 and token budget 256. XPress uses six refinement passes and 512
candidates. One warmup wave is excluded; three waves per configuration are timed.
Ordinary decoding uses a separate engine without the drafter. Regular, jagged
control and jagged combined are measured for ordinary decoding; XPress additionally
measures parallel-only and small-only arms. Fresh processes share frozen sources,
loaded-library/compiler hashes, card and CPU affinity.

## Profiling and remaining bottlenecks

| Profiled wave | Regular | Jagged control | Jagged combined |
| --- | ---: | ---: | ---: |
| Profiled wave, s | 4.016 | 4.685 | 4.405 |
| All device kernels, s | 2.438 | 2.896 | 2.832 |
| Attention kernels, s | 0.395 | 0.858 | 0.793 |
| Attention kernel count | 1641 | 708 | 780 |
| Output/MLP kernels, s | 1.466 | 1.461 | 1.462 |
| Input/QKV/cache kernels, s | 0.273 | 0.272 | 0.272 |
| Standalone matrix projections, s | 0.256 | 0.255 | 0.256 |
| DMA, ms | 37.200 | 37.634 | 37.247 |
| Device memsets, ms | 37.605 | 41.812 | 40.580 |
| All kernel launches | 3580 | 2647 | 2719 |
| DMA operations | 5579 | 4369 | 4451 |
| Host format conversion, s | 0.627 | 0.745 | 0.657 |
| CPU top-k, s | 0.050 | 0.071 | 0.058 |
| Profiler clock calibration, s | 0.199 | 0.202 | 0.234 |

The original jagged regression is concentrated in attention execution. Attention
adds 0.464 s versus regular, accounting for essentially all of the 0.458 s
increase in device-kernel time. Jagged launches fewer attention kernels, but
those kernels take longer; the other large kernel families are nearly unchanged.
The combined settings reduce attention time by 7.56%, from 0.858 to 0.793 s,
still about twice regular's 0.395 s. This isolates the kernel family responsible;
it does not measure how much of its time is matmul, masking, reduction or page
loop overhead. The one-core small-tile finding limits this new optimization and
does not explain the original width-64 control by itself.

All three captures have 50 D2H operations and exactly 392,171,008 device-format
bytes returned. D2H DMA itself takes 14–17 ms; total DMA takes about 37 ms.
CPU format conversion occupies 0.627–0.745 s and remains a substantial common
cost. Removing round trips can avoid that conversion, full-score storage and
synchronization even though raw transfer bandwidth is not the main regression.

The profiler supplies named kernels and DMA events. Device events are associated
with host submissions through correlation IDs. Some traces have host/device clock
offsets; absolute timestamp subtraction does not establish queue delay. Durations
and unions of device intervals remain useful. Memsets have no correlation ID and
are included only in whole-trace totals. Callback IDs 79–82 still appear unnamed
in the raw Kineto trace; the analyzer resolves them from the matching Flex enum
as response-worker fetch/completion/parse/iteration spans. Fetch includes waiting
for hardware and must not be treated as CPU computation. Host counters include
waits for earlier device work and overlap; they are not exclusive DMA or compute
time. Full-wave profile measurements are separate from the unprofiled request latency summary.

For attention-only calls, measured copies take approximately 4–14 microseconds;
the decode improvement appears in the attention kernel, not in a reduced transfer
payload. The full request traces above establish the larger attention and
conversion costs that remain after this change.

| Priority / work | Evidence | Next change or measurement |
| --- | --- | --- |
| Jagged speculative attention | Control attention takes 0.858 s versus 0.395 s for regular in matched XPress traces, despite 708 versus 1,641 attention kernels. Combined lowers this to 0.793 s. Other large kernel families are nearly unchanged. | Reduce padded/serial request-page work while preserving useful head/core partitions. Validate target and noncausal draft attention together. |
| Small-tile Q×K partition | The 16- and 32-row jagged variants put Q×K on one core; width 64 uses 32. P×V uses 32 at every width. This limits the smaller-tile experiment. | Identify where the compiler chooses that division; test compatible head/core partitions and retain the existing exact-width and full-model checks. The scratchpad objective is a hypothesis, not an established cause. |
| Host format conversion | `aiuDataConvert` occupies 0.627–0.745 s in the regular/control profiles. Total DMA is only about 0.038 s. | Avoid transferring/materializing full scores where device selection can consume them; profile and optimize the CPU conversion implementation for remaining copies. |
| Target hidden/logit round trips | Final target hidden states return to CPU for row selection, go back to the device for the LM head, and logits return for verification. Each trace has 50 D2H operations and the same 392,171,008 device-format bytes in total, including the draft. | Keep row selection and greedy verification on device with exact token IDs and defined ties; preserve rejection/stop behavior and cache isolation. |
| Initial draft vocabulary selection | Twelve proposal calls transfer 350,060,544 logical FP32 score bytes and run CPU argmax/top-k. Top-k alone is about 0.050–0.071 s in the traces. | Exact device argmax/top-k, followed by tiled projection/selection. The main opportunity also includes avoided conversion, synchronization and full-score storage. |
| Transformer output/MLP kernels | This family takes about 1.46 s in all three profiles, approximately 60% of regular XPress device-kernel time. It is a large common cost; these experiments do not prove that its implementation is inefficient. | Measure matrix tiling and weight bandwidth before selecting a kernel or precision change. |
| Full-vocabulary and context projections | Thirty-eight standalone matrix kernels take about 0.255 s. This family includes target/draft LM heads and the 20480→4096 context FC. LM-head weights are padded to 153600 vocabulary rows. | Profile the individual projection shapes; evaluate selection without materializing all vocabulary scores. Compact accepted context rows only if the gather cost and compilation behavior justify it. |
| CPU metadata and small submissions | The profiles contain 4,369–5,579 DMA operations and 2,647–3,580 compute launches. Metadata construction, conversion, graph lookup and response handling remain. | Reuse metadata/program inputs and fuse small stages where possible. Treat response fetch and synchronization spans as waits, not exclusive CPU compute. |
| Three-request padding | Each timed XPress wave runs 33 actual request blocks as 36 padded blocks. | Evaluate a 48-row draft shape against current power-of-two buckets, including projection, selection and warmup costs. |
| Refiner/selector dispatch | Six passes per proposal produce 72 device selections per wave. Refiner kernel families total about 0.020 s, and selector/feedback kernels are smaller still; launch costs can exceed their computation. | Fuse readout, selection and feedback after resolving the existing batched layout restrictions. Lower priority than the measured attention and conversion costs. |

Kernel-family times are from separate profiled waves. Host spans overlap device
work and one another; the table is not an additive decomposition of unprofiled
request latency. Profile clock calibration itself takes roughly 0.2 s per wave.

## Reproduction and evidence

Use `scripts/microbench/jagged_attn_latency.py` for resident attention and
`scripts/microbench/xpress_batch_latency.py` for queued request waves. Both expose
the two scheduling options and separate profiler outputs. Commands and scope are
in the microbenchmark README.

Local artifacts are under
`/mnt/home/spyre/xpress-jagged-tuning-validation/20261007`: the four attention
JSON files, attention traces/summary, eight request JSON files, request traces,
`nested-matmul-core-layouts.json`, test logs, matrix drivers and analyzers.
`torch-spyre-host-capacity.patch` preserves the required local compatibility fix.
The committed evidence includes raw latency samples, correctness and work
comparisons, source/binary fingerprints, and summarized profiler durations.
Large trace files remain in the local artifact directory.
