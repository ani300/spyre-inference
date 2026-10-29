# Gemma 4: FP8 experts through the full inference engine

Model: [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic), revision `ed35d7abe5d940da41b4ff06eb482feb0be8cb44`. Assessment: 2026-10-29, TP1, text backbone.

**The integrated B path generates text with FP8 expert storage and FP8 expert matmuls.** The single- and four-sentence FP16 KV-cache cases passed. The default tiled attention path fails to compile an FP8 cache read at Gemma’s head geometry; the existing Python attention page loop passes all eight focused Gemma cache checks. Full-model FP8-cache generation passes for both one and four sentences with that workaround. All 30 layers load, and the default compiled transformer blocks remain enabled.

## What changed

- [FP8 MoE method](../../../spyre_inference/experimental/fp8_moe_method.py): replaces the unsupported compressed-tensors method with a Spyre implementation, reuses checkpoint expert loaders, folds Gemma's learned expert multiplier into the down-channel scale, and uploads gate/up/down directly as QFP8WT.
- [torch-spyre loader branch](https://github.com/ani300/torch-spyre/tree/experiment/gemma4-fp8-moe-loader): extends `_dma_to_spyre_fp8_kernel` and its C++ data-conversion descriptor to rank-3 expert pools (also tested at rank 4). Source strides and offsets are respected; the destination uses canonical dense strides.
- [Persistent expert computation](../../../spyre_inference/experimental/fp8_native_moe.py): on-device routing followed by `for_each_tile` over resident experts. Gate, up and down use FP8 × FP8 `scaled_bmm`, with DL16 output, scale compensation, GELU and reduction. No tokens are packed on CPU.
- [Engine bridge](../../../spyre_inference/moe.py): preserves the separate compiled routing and persistent-expert regions through the opaque MoE call inside each compiled transformer block. Prefill and decode share one FP8 expert representation. Short batches are zero-padded to 128 rows inside the MoE and trimmed before returning.
- [Dense FP8 linear padding](../../../spyre_inference/custom_ops/fp8_linear_kernel.py): Gemma's shared MLP down projection has K=2112, which occupies 16.5 FP8 sticks. Pad its checkpoint bytes and activation to 2176 before FP8 matmul. Compiler-inserted padding otherwise loses QFP8WT and the matmul planner fails.

The expert intermediate width is padded from 704 to 768. Checkpoint FP8 codes are never dequantized for storage or repacked on device by the integrated path. The original source expert parameters are released after upload. Per-output-channel scales remain separate from the FP8 codes. Attention retains its existing DL16 computation, dequantizing an FP8 KV cache on read when that cache format is selected.

```text
At load time:
checkpoint FP8 bytes → zero padding → rank-3 QFP8WT DMA → resident FP8 experts
                                                               ↓
Each forward:
DL16 tokens → device routing → dynamic FP8 activations → FP8 grouped matmuls
                                                               ↓
                                     DL16 scales / GELU / weighted reduction

Attention / KV-cache writes and reads remain in the attention backend.
The MoE sees only hidden states and router logits.
```

## Validation so far

| Check | Result | Evidence |
| --- | --- | --- |
| Full model, FP16 KV, one sentence | Generation passed, 16 output tokens | [JSON](results/e2e-final/fp16-b1.json) |
| Full model, FP16 KV, four sentences | Generation passed, 16 output tokens per sentence | [JSON](results/e2e-final/fp16-b4.json) |
| Full model, FP8 KV, one sentence, unrolled attention | Generation passed; same 16 token IDs as FP16 for this prompt | [JSON](results/e2e-unrolled/fp8-b1.json) |
| Full model, FP8 KV, four sentences, unrolled attention | Generation passed, 16 output tokens per sentence | [JSON](results/e2e-unrolled/fp8-b4.json) |
| Full model, FP8 KV, one sentence, default attention | Fails in FP8-page dequantization during attention prefill compilation | [log](results/e2e-final/fp8-b1.log) |
| Gemma FP8 cache, unrolled attention, both head geometries × prefill/decode × one/four sequences | 8 numerical passes | [probe](results/e2e/attention-unrolled-probe.log), [remaining cases](results/e2e/attention-unrolled-matrix.log) |
| MoE configuration and opaque dispatch selection | 18 passed | [log](results/regression-final/moe-config.log) |
| CPU dispatch, padding, chunk equations and reduction | 8 passed | [log](results/e2e/grouped-cpu-tests.log) |
| Full FP8 loader suite, including rank-2/3/4 DMA, source views/offsets and grouped matmul | 31 passed | [log](results/regression-final/loader-fp8.log) |
| Full dense FP8 linear suite, including Gemma K=2112 padding at 1/4/128 rows | 39 passed | [log](results/regression-final/dense-fp8.log) |
| Real layer 0, 128 experts, 1 / 4 / 128 / 512 tokens, direct DMA + persistent FP8 | All pass; global relative L2 2.57% / 3.63% / 3.90% / 3.97%; worst row ≤7.85% | [1/4/512 JSON summary](results/regression-final/layers/summary.json), [128 JSON](results/e2e/persistent-dma-b1-q128.json) |
| Full model, first attempt | All experts load; shared dense MLP's unaligned K loses QFP8WT during compilation | [diagnostic log](results/e2e/fp16-b1-attempt2.log) |

The layer reference dequantizes the original checkpoint weights to FP32 and independently validates routing, including near ties. The 10% global and per-row norm thresholds were declared before these runs. These layer checks are not an end-to-end quality benchmark. Persistent paths use one whole-expert representation; the layer runner’s historical `chunks` field records the requested CLI value, while `effective_chunks` records the representation used by new runs (always 1 for persistent paths).

Scoped Ruff and runner type checks pass. The [torch-spyre pre-commit run](results/e2e/torch-spyre-precommit-review.log) passes its other checks but retains two existing Python 3.11 mypy errors for `warnings.warn(skip_file_prefixes=...)` at `model_utils.py:216`.

## Why the earlier packing issue was avoidable

The earlier prototype uploaded FP8 code values through temporary DL16 tensors and called `qfp8wt` on device. A whole pool exceeded that conversion kernel's per-core address span; four smaller pools worked. That was a limitation of the prototype's conversion route, not a requirement of checkpoint loading. The extended loader now packs the checkpoint bytes directly, exactly where FP8 linear weights are packed. The earlier claim that full-pool device packing blocked engine integration was too broad.

A roundtrip alone is insufficient: an incorrect layout can be inverted by the same incorrect descriptor. The loader tests therefore also consume the uploaded weights in an FP8 grouped matmul against an exact independent oracle. Source tensors with holes additionally exposed the need for canonical destination strides.

## Reproduce

Use the two experiment branches with the current upstream bases: spyre-inference `d69bc3f`, torch-spyre `f2bb1f8d`. Both were re-fetched from upstream main for this assessment. [Integration provenance](integration-provenance.json) records the implementation commits and lower-stack binary hashes. Build and install the patched torch-spyre into the prepared inference venv; do not let `uv sync` restore the repository pins. Runtime/Flex, Deeptools, libaiupti and comms come from `sentient/`; only senlib comes from `/opt/ibm/spyre/senlib`.

From this spyre-inference checkout:

```bash
export DTI_PROJECT_ROOT=/mnt/devel/inductor_src
source /mnt/devel/inductor_src/torch-spyre-docs/scripts/dev-env.sh
source .venv/bin/activate
export SENLIB_INSTALL_DIR=/opt/ibm/spyre/senlib
export LD_LIBRARY_PATH="$SPYRE_COMMS_INSTALL_DIR/lib:$LIBAIUPTI_INSTALL_DIR/lib:$RUNTIME_INSTALL_DIR/lib:$DEEPTOOLS_INSTALL_DIR/lib:$SENLIB_INSTALL_DIR/lib"
export PYTHONPATH="$PWD:$PWD/tests/plugin:$SENTIENT_BASE_INSTALL_DIR/runtime/lib"
export HF_HOME=/mnt/models/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE SPYRE_NUM_CPUS=8
export SPYRE_DEVICES=0 AIU_WORLD_SIZE=1 USE_SPYRE_PROFILER=0 SPYRE_KERNEL_CACHE=0
export SPYRE_ATTN_RECORD=0  # Compile attention variants on demand for these smoke runs.
export TORCHINDUCTOR_CACHE_DIR="$PWD/.gemma-fp8-cache"
export VLLM_CACHE_ROOT="$PWD/.gemma-vllm-cache"

# Build the paired torch-spyre branch against sentient/ into this venv:
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --no-build-isolation \
  /mnt/devel/inductor_src/torch-spyre-gemma4-fp8

# Serial full-model cases, each with single and four sentences.
python examples/experimental/run_gemma4_fp8_e2e.py --kv-cache fp16 \
  --output-dir results/gemma-e2e-fp16
python examples/experimental/run_gemma4_fp8_e2e.py --kv-cache fp8 \
  --attention-page-loop unrolled --output-dir results/gemma-e2e-fp8

# One case; normal compiled transformer blocks remain enabled.
python examples/experimental/gemma4_fp8_model.py --batch 4 --kv-cache fp8 \
  --attention-page-loop unrolled --output results/gemma-fp8-b4.json

# Real benchmark-trace inputs; actual Gemma token lengths are checked, not truncated.
python examples/experimental/run_gemma4_fp8_e2e.py --batch 4 --max-model-len 2048 \
  --dataset /mnt/models/online_benchmarking_data_reordered/aiops_results_2025.11.03_e2ee1b0_correct_order.jsonl \
  --attention-page-loop unrolled --output-dir results/gemma-aiops

# Isolated attention comparison; omit the flag to reproduce the tiled compiler error.
SPYRE_ATTN_FOR_EACH_TILE=0 python -m pytest examples/experimental/gemma4_fp8_attention.py -q

# Numerical layer checks and focused dense-linear regression.
python examples/experimental/run_gemma4_fp8_matrix.py --path persistent-dma \
  --output-dir results/gemma-layer-matrix
python -m pytest tests/test_fp8_linear.py -q -m 'not upstream'
python -m pytest --noconftest -o addopts= tests/models/test_fp8_grouped_moe.py -q

# From the paired torch-spyre checkout, with the same environment:
python -m pytest tests/inductor/test_fp8_prequantized_load.py -q
```

Run only one hardware workload at a time. When using a torch-spyre source worktree directly, keep it first on `PYTHONPATH` after sourcing the environment, and use its rebuilt extension. `--eager-blocks` is an optional diagnostic that leaves FP8 kernels compiled while using eager transformer-block orchestration. JSON records failures and output tokens; the matrix preserves commands, logs and nonzero exits. The smoke test forces 16 output tokens (`ignore_eos=True`), so text after an end-of-turn marker is not part of the natural answer. Successful generation is a smoke result, not a claim of model-quality parity or speedup. The AIOps trace uses its original prompts and a fixed generated-token budget, so this is integration coverage rather than the full serving benchmark.

The interactive [HTML report](report.html) is generated from this file by `python docs/experiments/gemma4-fp8-moe/render_report.py` (requires `markdown-it-py`).

## Remaining limitations

The default tiled FP8-cache prefill fails with `no mechanism to resolve stick incompatibility` while converting a gathered `[1,8,128,256]` FP8 page into the DL16 matmul layout. `--attention-page-loop unrolled` selects the existing `SPYRE_ATTN_FOR_EACH_TILE=0` attention implementation; expert `for_each_tile` stays enabled. The focused tests validate both Gemma head geometries and cache writes against the existing CPU reference. Longer page walks remain unmeasured.

TP1, Gemma's GELU/full-softmax routing, 128 experts and top-k 8 only. The bridge rejects unsupported recipes and parallel configurations. The model's vision/audio towers are outside this experiment.

Persistent decode computes all experts over padded rows, so it is a correctness workaround with substantial extra work. The native gathered FP8 decode path and shared chunk aliases still need compiler work. Neither the FP16 nor FP8 KV-cache choice removes that limitation, because expert storage is independent of the attention cache.

The [original layer assessment](layer-assessment.md) preserves the staged decode path, four-pool persistent path, failing native compiler probes and their numerical results. Those baseline files predate the direct-loader and engine integration changes.
