# Gemma 4 MoE with FP8 expert storage and FP8 matmuls

Assessment date: 2026-10-29. Model: [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic), revision `ed35d7abe5d940da41b4ff06eb482feb0be8cb44`.

**Result:** FP8 expert storage and FP8 grouped matmuls work at layer level with two workarounds: selected-weight repacking for gathered decode, and four resident QFP8WT pools for persistent prefill. Stock Gemma initialization still rejects quantized experts.

This branch contains experimental layer kernels, native-path compiler probes, and a stock-model startup test. It does not register a new vLLM quantization method or enable end-to-end Gemma generation. All numerical results below use real layer-0 expert weights and synthetic activations/router logits, not generated sentences or a quality benchmark. Batch/query dimensions represent independent token rows at a MoE layer; they do not exercise attention, a KV cache, or request scheduling.

## Measured layer results

All rows below store all 128 real layer-0 experts in FP8 and execute FP8 matmuls. Query length is a synthetic token-row count, not an end-to-end generation test.

| Path | Batch | Query length | Global relative L2 | Worst-row relative L2 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Gathered, staged | 1 | 1 | 2.50% | 2.50% | [pass](results/staged/staged-b1-q1.json) · [log](results/staged/staged-b1-q1.log) |
| Gathered, staged | 4 | 1 | 3.59% | 4.84% | [pass](results/staged/staged-b4-q1.json) · [log](results/staged/staged-b4-q1.log) |
| Gathered, staged | 1 | 128 | 3.90% | 5.31% | [pass](results/staged/staged-b1-q128.json) · [log](results/staged/staged-b1-q128.log) |
| Gathered, staged | 4 | 128 | 3.97% | 7.69% | [pass](results/staged/staged-b4-q128.json) · [log](results/staged/staged-b4-q128.log) |
| Persistent, four pools | 1 | 128 | 3.90% | 5.29% | [pass](results/persistent-sharded/persistent-sharded-b1-q128.json) · [log](results/persistent-sharded/persistent-sharded-b1-q128.log) |
| Persistent, four pools | 4 | 128 | 3.97% | 7.85% | [pass](results/persistent-sharded/persistent-sharded-b4-q128.json) · [log](results/persistent-sharded/persistent-sharded-b4-q128.log) |

## Current support

Stock `spyre-inference` refuses this checkpoint in `configure_spyre_moe_layer`:

```
NotImplementedError: Spyre MoE backend does not support quantized experts.
```

Fresh stock-engine runs reproduced this error for both KV-cache formats, before generation. The existing path requires `UnquantizedFusedMoEMethod`. vLLM's FP8 MoE backend selection also has no Spyre implementation. Dense FP8 linears already have a Spyre kernel; their support does not cover quantized experts.

| KV cache | Full model with current native MoE | This layer experiment |
| --- | --- | --- |
| FP16 | [Blocked by quantized expert initialization](results/native-model/fp16.log) | Independent of KV format |
| FP8 | [Same quantized expert initialization failure](results/native-model/fp8.log) | Independent of KV format |

The checkpoint has 30 layers, 128 experts per layer, top-k 8, hidden width 2816, and intermediate width 704. Each expert has E4M3 gate/up `[704,2816]` and down `[2816,704]` weights, with BF16 per-output-channel scales and dynamic per-token activation quantization. Gemma's learned per-expert multiplier is folded into down's output-channel scale without altering FP8 codes. Routing follows the existing Gemma DL16 recipe: full softmax, top-k, renormalization, then GELU(tanh) × up.

## Follow the DL16 implementation

The proposed native code is [fp8_native_moe.py](../../../spyre_inference/experimental/fp8_native_moe.py). It mirrors the existing gathered decode and persistent prefill equations in [moe.py](../../../spyre_inference/moe.py):

- Decode: gather entry `expert * chunks + chunk`, grouped gate/up, sum K-chunk partial results before GELU, repeat the activated vector across down output chunks, grouped down, concatenate chunks, weighted expert sum. A traced per-token loop handles multiple rows, as in DL16.
- Prefill: reuse DL16's on-device dense top-k routing, persistent whole-expert pools, and `for_each_tile` over experts. Replace matmuls with FP8 `scaled_bmm`, dynamic per-token activation quantization, and output-channel scale compensation.

For the four-chunk gathered path, pad H=2816→3072 and I=704→768. Each chunk is 768 wide, or six 128-value QFP8CH sticks. QFP8WT uses its own 2×64 stick arrangement; these padded dimensions satisfy both layouts. Eight selected experts × four chunks gives 32 grouped entries. Gate/up share a scale computed from the full input token before splitting its K dimension. Down quantizes each selected expert's activated vector. Added weights and activation channels are zero, so padding does not change the mathematical result.

```
Token x[2816] → zero-pad x[3072] → four chunks of 768
                                         │
Selected expert e: entries 4e,4e+1,4e+2,4e+3
  FP8 x_chunk @ FP8 gate_chunk → DL16 partials ┐
  FP8 x_chunk @ FP8 up_chunk   → DL16 partials ├→ sum four chunks → GELU(gate) × up
                                             ┘
  activated[768] @ FP8 down_chunk → four output slices of 768
  concatenate → [3072] → route-weighted expert sum → trim to [2816]
```

This prototype materializes chunk pools. It does not yet construct zero-copy QFP8WT chunk aliases sharing the persistent whole-expert pool. That is additional layout integration work required to reproduce DL16's memory management, beyond reproducing its equations.

## Native compiler results

The full 128-expert native probes fail during runtime QFP8WT pool packing: the compiler plans a 576 MiB per-core input span for the gathered chunk pool, or 528 MiB for the persistent whole-expert pool, above its 256 MiB limit. This is a limitation of this bulk pack implementation, not a claim that the model cannot fit device memory. B1/B4 gathered and B1/B4 prefill probes fail at this same load stage. Direct rank-3 QFP8WT DMA separately raises `FP8 KERNEL layout is for 2D weights only`.

To reach the downstream kernels independently of bulk packing, `--experts 32` loads the first 32 real experts while retaining a 128-wide router, masking experts 32–127. This is a reduced compiler probe, not a full Gemma layer. Native gathered decode reaches `dbo-opt` but fails with `newPds.stickDimOrder_.size() == newPds.stickSize_.size()` in `designSpaceConfig.cpp:7018`. It produces no numerical output. See [native gathered error](results/native-reduced/native-b1-q1.json), [full-pool failure](results/native-full/native-b1-q1.json), and [rank-3 DMA failure](results/native-full/rank3-dma-b1-q1.json). The observed failure is in the combined native gather/FP8 pipeline; the exact offending descriptor has not been minimized further.

The reduced persistent path **compiles and passes** on this revision when it uses DL16's separate softmax and routing compilation boundaries. With 32 real experts, B1×128 has 4.24% global / 5.17% maximum-row relative L2 error; B4×128 has 4.31% / 5.52%. The reference uses the observed on-device routing and independently checks probabilities/top-k eligibility. Comparing strictly to FP32 top-k initially produced a 22% outlier; observing routing identified near-cutoff expert changes, not a 22% FP8 arithmetic error. No tolerance was relaxed. Reduced-probe evidence: [B1×128](results/persistent-reduced/persistent-b1-q128.json), [B4×128](results/persistent-reduced/persistent-b4-q128.json).

Thus, historical issue #5423 is not a blanket blocker for this persistent kernel on the tested main. A fused softmax+route experiment failed in the lower compiler's scheduler; splitting those regions as DL16 does allowed execution.

## B path

[fp8_grouped_moe.py](../../../spyre_inference/experimental/fp8_grouped_moe.py) keeps the resident expert pools in FP8 QFP8CH and computes gate/up/down with FP8 × FP8 grouped matmuls. It has explicit compilation boundaries:

1. At load time, upload the exact, unscaled FP8 code values through a temporary DL16 tensor and pack the device pool to FP8. Release the temporary; no persistent dequantized expert pool is retained.
2. Route each token on-device and gather its 32 expert/chunk entries from the FP8 pool.
3. Expand only those selected codes to DL16, transpose, then repack them into QFP8WT. This is a layout conversion workaround; weight scales are applied after the FP8 GEMM.
4. Dynamically scale/quantize activations, execute `spyre.scaled_bmm`, compensate activation and weight scales, then perform GELU and reductions in DL16.
5. Write the output row on-device.

CPU token packing is unnecessary. The host prepares static row indices and controls a per-token launch loop; tensor routing, gathers, output assembly, and computation run on Spyre. The probe copies selected IDs/weights back solely to validate the result; those copies do not drive computation. `FallbackWarning` is an error.

The four-chunk resident pools contain 905,969,664 logical FP8 bytes (864 MiB) per layer, versus 726 MiB for the checkpoint's unpadded expert codes. This excludes scales, allocator overhead, activations and selected-weight scratch. The whole-expert I-only-padded pool is 792 MiB. No performance or full-model memory-fit claim follows from these figures.

## Persistent prefill with all 128 experts

`--path persistent-sharded` stores all 128 experts in four QFP8WT pools of 32 experts. Each pool fits the current runtime packer's per-core span check. It computes the global 128-expert router once, slices each pool's routing contributions on-device, runs the same FP8 `for_each_tile` body, and adds the four output tensors on-device. Both B1×128 and B4×128 pass with all 128 experts: 3.90% and 3.97% global relative L2 error, with maximum-row errors of 5.29% and 7.85%. It uses I=768 padding; H=2816 already spans whole FP8 sticks when it is not divided into four K chunks.

This path keeps the DL16 persistent structure, with four static pool partitions as the workaround. Unlike staged gathered decode, it does not expand/repack selected weights during inference or launch once per token. The resident pools total 792 MiB/layer, excluding scales and scratch. Load-time packing still passes exact FP8 code values through temporary DL16 tensors.

The gathered and persistent paths are tested separately. A combined serving implementation must share/reuse compatible weight layouts or explicitly budget for both pool copies. This branch does not yet have DL16's shared whole-expert/chunk aliases; keeping both current representations would erode the FP8 storage saving.

## Reproduce

Use the current upstream stack, not the repository's dependency pins. The tested revisions and lower-stack binary hashes are in [provenance.json](provenance.json). Torch-spyre `f2bb1f8d875cc0389058384f544145128e5366be` was re-fetched from upstream main over HTTPS and matched the installed `0.6.0.dev374+head.gf2bb1f8d` build. Inference base: `d69bc3fec204a311b00a0e255e8f7dc0f39406f7`. No torch-spyre source patch is required by the B paths.

On this workspace, from the branch checkout:

```bash
export DTI_PROJECT_ROOT=/mnt/devel/inductor_src
source /mnt/devel/inductor_src/torch-spyre-docs/scripts/dev-env.sh
source .venv/bin/activate
export SENLIB_INSTALL_DIR=/opt/ibm/spyre/senlib
export LD_LIBRARY_PATH="$SPYRE_COMMS_INSTALL_DIR/lib:$LIBAIUPTI_INSTALL_DIR/lib:$RUNTIME_INSTALL_DIR/lib:$DEEPTOOLS_INSTALL_DIR/lib:$SENLIB_INSTALL_DIR/lib"
export PYTHONPATH="$PWD:$SENTIENT_BASE_INSTALL_DIR/runtime/lib"
export HF_HOME=/mnt/models/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export SPYRE_DEVICES=0
export AIU_WORLD_SIZE=1
export USE_SPYRE_PROFILER=0
export SPYRE_KERNEL_CACHE=0
export TORCHINDUCTOR_CACHE_DIR="$PWD/.gemma-fp8-cache"
export VLLM_CACHE_ROOT="$PWD/.gemma-vllm-cache"
```

The existing experiment venv contains the tested upstream wheel and reuses unchanged dependencies from the earlier FP8-QK venv through a `.pth` file. On a fresh checkout, install the project and a torch-spyre wheel built from the recorded commit against `sentient/`, then use that venv. Do not copy a `.pth` reference to a nonexistent environment. Do not use plain `uv run` or `uv sync`: those restore the pinned torch-spyre. Use `python` in the prepared venv or `uv run --no-sync`.

Run hardware jobs serially:

```bash
python examples/experimental/run_gemma4_fp8_matrix.py --path staged --output-dir results/gemma-fp8-staged
python examples/experimental/run_gemma4_fp8_matrix.py --path persistent-sharded --case persistent-sharded-b1-q128 --case persistent-sharded-b4-q128 --output-dir results/gemma-fp8-sharded
python examples/experimental/run_gemma4_fp8_matrix.py --path native --output-dir results/gemma-fp8-native
python examples/experimental/run_gemma4_fp8_matrix.py --path persistent --output-dir results/gemma-fp8-persistent
python examples/experimental/run_gemma4_fp8_matrix.py --path rank3-dma --output-dir results/gemma-fp8-dma
python -m pytest --noconftest -o addopts= tests/models/test_fp8_grouped_moe.py -q
```

Each matrix covers B=1/4 and query length=1/128 (1, 4, 128, 512 token rows), except rank-3 DMA which needs only one case. `--case staged-b4-q1` selects one case and can be repeated. Failing probes return nonzero and retain command JSON, logs and result JSON. A process abort stops the matrix for a device health check. Captured text logs have trailing whitespace normalized for version control. Native and persistent compiler probes are allowed to fail as part of this assessment; failures are not converted into passes.

Stock-model reproduction, without experimental hooks:

```bash
python examples/experimental/gemma4_fp8_model.py --kv-cache fp16 --output results/native-fp16.json
python examples/experimental/gemma4_fp8_model.py --kv-cache fp8 --output results/native-fp8.json
```

The startup script requests the text backbone only. If initialization eventually succeeds on a newer stack, it will also attempt short deterministic generation. It currently cannot establish generation quality.

## Validation and limits

The test checks finite outputs, global and per-row relative L2 error below 10%, eight distinct valid expert IDs per row, routing weights against CPU probabilities (`rtol=0.02`, `atol=0.002`), and selected probabilities within 0.0005 of the top-k cutoff. The oracle dequantizes the original checkpoint weights to FP32; it does not reuse the padded/chunked device representation. Reported arithmetic error uses the device-selected experts after independently validating routing. It is not strict end-to-end equality to FP32 routing when top-k values are nearly tied.

Compiler excerpts for [gate/up](compiler/gate-up.excerpt.json), [down](compiler/down.excerpt.json), and [persistent prefill](compiler/persistent.excerpt.json) contain `batchmatmulfp8`, with 128-value activation sticks, 2×64 weight sticks and 64-value DL16 output sticks. They establish the FP8 computation path; scheduling fields are not a performance measurement. Eight CPU tests verify dispatch, padding, chunk equations, scale folding and output reduction. Scoped Ruff and [type checks](results/typecheck-final.log) pass; [CPU test output](results/final-cpu-tests.log) is included.

An earlier unpadded probe triggered one PCI bus fence. A subsequent compiled device health check passed, then padded probes passed. There was no controlled unpadded rerun, so the fence cannot be attributed conclusively to padding. The reported results use the padded paths.

## Work still needed

- Add a Spyre FP8 expert method and checkpoint post-load handling, including channel scales, Gemma's learned multiplier, padding, TP partitioning and existing model recipe integration.
- Support direct rank-3 FP8 QFP8WT upload and preserve that arrangement through expert gathers and shared chunk aliases. The tested `_dma_to_spyre_fp8_kernel` explicitly accepts rank 2 only. This is a restriction of that helper, not a claim that all possible rank-3 FP8 uploads are impossible.
- Fix native gathered layout propagation/lowering. Preserve the demonstrated persistent FP8 loop and solve full-pool packing/aliases so prefill can use it without per-token launch staging.
- Integrate quantized experts into compiled transformer blocks and warmup, then run full-model single/multiple sentence batches and the performance dataset with FP16 and FP8 KV caches separately.

Additional downstream compiler probes:

```bash
python examples/experimental/run_gemma4_fp8_matrix.py --experts 32 --case native-b1-q1 --case persistent-b1-q128 --output-dir results/gemma-fp8-reduced
```

Related upstream work: [rank-3 FP8 weight upload #5633](https://github.com/torch-spyre/torch-spyre/issues/5633), [PR #5631](https://github.com/torch-spyre/torch-spyre/pull/5631), [scaled_bmm in for_each_tile #5423](https://github.com/torch-spyre/torch-spyre/issues/5423). These links provide context; local probe results determine the findings in this report.
