# XPress upstream alignment and proposed PR stack

Working branch: `feat/xpress-upstream`, based on spyre-inference
`bf00e99274ec53d20b97cd2e31e0e060803b6974` (October 27, 2026).
The source was `perf/xpress-jagged-scheduling` at `188bcff4`, including the
uncommitted FP16 selector and digits-layout cache work. The original branch and
worktree are preserved. This document plans the split; no PRs have been opened.

The dependency pins remain upstream's vLLM 0.32 and torch-spyre `64870622`.
The vLLM XPress reference is merge commit
[`ce9467b757c0`](https://github.com/vllm-project/vllm/commit/ce9467b757c024a55fea4d0919ac2e682f28a894).
That merge is newer than the v0.32 release branch, so the Spyre adapter still
needs architecture registration and a configuration bridge.

## Proposed order

Each PR should contain its own tests and the documentation needed to use it.
Do not postpone correctness tests to a final test-only PR.

| PR | Scope | Review boundary and validation | Dependencies |
|---|---|---|---|
| 1. Portable XPress head and checkpoint contract | `xpress_head.py`, `checkpoint.py`, `convert_checkpoint.py`; head, checkpoint and upstream-equation tests | Raw mixer folding, weight naming, full-vocabulary and fixed-shortlist Jacobi recurrence; C128 default, explicit C512/C0; CPU tests require no card | Upstream main |
| 2. Host verification and non-causal attention support | `rejection_sampler.py`; small changes to attention metadata/recording and logits-row bucketing | vLLM 0.32 dispatcher signatures, greedy prefix/correction/bonus, synthetic/random helper behavior, non-causal draft masks, verification-row warmup | PR 1, or independently on main if useful |
| 3. Qwen3 DFlash/XPress integration | Architecture/config bridge, DFlash context projector/proposer, runner loading, auxiliary outputs, single-request lifecycle | Use the portable CPU shortlist path first. Real-checkpoint loading, distinct target embedding/head sharing, accepted-prefix KV writes, rejection rollback, stopping, cancellation and C128 generation | PRs 1–2 |
| 4. Batched drafting | Batch padding, per-request proposal eligibility/limits, context ownership, batch counters and batch-four tests | Exercise 1/2/3/4 active requests, mixed prefills, shrinking batches, cancellation/page reuse and logprobs. Preserve ordinary target-only behavior | PR 3 |
| 5. Resident selection and feedback | `selection.py`, device feedback helpers and model dispatch; selector/feedback tests and focused benchmarks | C128 must use one native selection round and no masks; compare C512's four rounds/three masks. Validate exact IDs, ties, gathers, repeated feedback and transfer counters at batch 1/4 | PR 4 plus the required torch-spyre/Deeptools operations |
| Optional top PR: jagged attention integration | Reapply only after the independent attention work is ready | Repeat ragged-length batch-four acceptance and throughput comparisons. Avoid coupling base XPress config, checkpoint loading or verification to jagged APIs | The XPress stack plus the eventual jagged PRs |

The first usable XPress milestone is PR 3. PR 4 adds concurrency. PR 5 must
remain optional so compiler work does not block the functional integration.
The batched-request and serial-request benchmark scripts can accompany PRs 4/5;
historical profiler traces and large JSON dumps stay outside the PR diffs.

Use the existing `24afce0` batch commit as a guide for PR 4's hunks. Use
`4ea65e8` and the saved uncommitted selector work as guides for PR 5. Avoid
cherry-picking their surrounding merge commits: those bring in jagged kernels,
compiler pins and historical benchmark data.

## Changes needed for current upstream

- Preserve the current model-load, FP8-cache and input-embedding-buffer refactors
  while adding the drafter load and device preparation hooks.
- Patch the vLLM 0.32 rejection **dispatchers** before sampler construction.
  Patching only the old Triton kernel globals no longer changes execution.
- Keep all auxiliary hidden-state outputs on Spyre, and warm the target LM head
  for verification rows as well as one sampled row per request.
- Clear draft-probability state on each greedy proposal call, matching the
  current runner's lifecycle.
- Account for vLLM's per-request drafting reservation when validating the token
  budget. A block of 16 needs 31 budgeted slots per active request to verify all
  15 proposals; a 16-token budget can silently schedule no verification at all.
- Reject V2 for this adapter until a separate Spyre V2 speculator is implemented.
  The merged GPU implementation's V2 integration cannot run unchanged on Spyre.

## Merged XPress contract

| Contract | This branch |
|---|---|
| Public method and architecture detection | `method="dflash"`; both published and DFlash-prefixed XPress architecture names |
| Anchor and predecessor | Position zero fixed; the anchor is its own predecessor; return positions 1 through B−1 |
| Mixer | Raw lower-triangular mixer plus identity, folded once at load |
| Refinement | Block-wide mean, pass-invariant hidden projection, causal Jacobi updates and fixed initial candidate sets |
| Checkpoint sharing | Target input embeddings and target output head are shared separately |
| Top-C configuration | Explicit serving option, then checkpoint field, then Spyre default 128 |
| Deliberate default difference | Merged vLLM defaults to 512; C128 is a Spyre policy change requested for this integration |
| Full-vocabulary control | `xpress_topc=0`; preserve explicit 512 as the larger-shortlist control |
| Pass count | HF `xpress_num_passes`, default six; the old Spyre CLI pass override remains a compatibility alias |
| Execution | Compiled PyTorch on Spyre and host bookkeeping; GPU Triton/CUDA-graph implementation remains upstream-specific |

Top-C reduces the search space and can change acceptance. A shorter shortlist is
not expected to reproduce C512 proposal IDs. Comparisons must hold C fixed when
checking recurrence parity and must measure acceptance when comparing throughput.

## Compiler dependencies and performance gates

The upstream torch-spyre pin lacks `topk_payload` and the experimental
`topkvalue`/mask selector surface. Do not turn `SPYRE_XPRESS_DEVICE_SELECTION` on
by default or change the compiler pin just to land PRs 1–4. Before PR 5:

1. Upstream the FP16 payload-selection correctness work and required multi-output
   scheduling/serialization support in torch-spyre and Deeptools.
2. Requalify the optional fused digits template on that same lower stack. Keep its
   compiler flag opt-in until comparisons pass. A one-round C128 shortlist does
   not need the repeated-mask path; C512 compatibility still does.
3. Validate feedback ID construction and the join with uploaded anchors on the
   new element-arrangement rules. The October 27 hardware check rejected the old
   combined FP32 decode/concatenation at both C128 and C512; this is separate from
   the shortlist-size change. `SPYRE_XPRESS_DEVICE_FEEDBACK` is off by default;
   resident selection enables it implicitly. PRs 1–4 use CPU selection between
   refinement passes, leaving this compiler dependency in PR 5.
4. Measure C128/C512 with resident logits, then complete proposals and full
   generation. Report compile/warmup separately, device time, transfer volume,
   accepted tokens per round and tokens/second at batch 1 and batch 4.

The expected work reduction is concrete: four top-128 rounds become one, three
full-vocabulary masks disappear, and candidate readout/storage becomes one
quarter of C512. The transformer forward, context projection, initial argmax,
verification and residual host bookkeeping remain. No end-to-end speedup follows
from the round count alone.

## Current validation (October 27)

- 430 CPU cases pass, including merged-equation parity, real vLLM 0.32 rejection
  dispatchers, C128/C512 feedback and proposal loops, batch accounting, V2 rejection,
  and the scheduler's drafting-slot reservation. Ruff, type checks, spelling and
  Markdown checks pass.
- The cached Qwen3-8B target and XPress checkpoint load, compile and generate on
  device 2 with regular attention and C128. A 24-token smoke run executes four
  real verification rounds; every returned token has maximal target logprob.
- The full single-request equality test still fails in its scripted rejection
  control. A captured comparison at identical token prefixes localizes the first
  difference to the target logits at output offset 12: Spain/Germany score
  `24.375/24.28125` under single-token execution, but tie at `24.5/24.5` under
  16-row verification. The host sampler correctly chooses the smaller ID on the
  tie. This precedes XPress proposal quality; it needs numerical qualification
  before PR 3 is ready. The strict test has not been relaxed.
- Batch-four and C512 end-to-end qualification remain to be repeated on these
  pins. The resident-feedback compiler failures described above remain PR 5
  dependencies. No new C128/C512 performance comparison is claimed.

## Qualification before submitting the stack

- CPU suite: checkpoint/config guards, merged-equation parity at C0/C128/C512,
  feedback IDs/anchors, batched accepted-prefix accounting and real vLLM 0.32
  rejection dispatchers.
- Model load: the cached Qwen3 target/drafter pair using the upstream dependency
  pins, including untied LM-head sharing and auxiliary-layer selection.
- Device: regular attention, C128, batches 1 and 4, with output/cancellation/cache
  checks; repeat C512 as control. CPU-only passes are not device qualification.
- Performance: same prompts, token limits, dtype, card and warmup. Keep an
  ordinary-decoding baseline and a K0 drafter baseline. Profile separately from
  wall-clock timing.
- Every stacked PR must build and test against its declared parent. Open PRs only
  after the integration branch is split and each intermediate revision is checked.

The task's detailed logs, source snapshots and initial-worktree backup are in
`/mnt/home/spyre/xpress-upstream-validation/20261027`. Historical jagged and digits
measurements remain in their original validation directories and are not evidence
for the newly pinned stack.
