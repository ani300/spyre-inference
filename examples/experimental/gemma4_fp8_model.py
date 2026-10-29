# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma 4 FP8 checkpoint smoke test; errors are recorded and exit nonzero."""

import argparse
import json
import os
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[1, 4], default=1)
    parser.add_argument("--kv-cache", choices=["fp16", "fp8"], default="fp16")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/models/hf_cache")
    result = {**vars(args), "output": str(args.output), "passed": False}
    result["versions"] = {p: version(p) for p in ("torch", "torch-spyre", "vllm")}
    engine = None
    stage = "startup"
    started = time.monotonic()
    try:
        from vllm import LLM, SamplingParams

        from spyre_inference.models.gemma4 import GEMMA4_TEXT_BACKBONE_OVERRIDE

        model = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"
        revision = "ed35d7abe5d940da41b4ff06eb482feb0be8cb44"
        settings: dict[str, Any] = dict(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            dtype="float16",
            hf_overrides=GEMMA4_TEXT_BACKBONE_OVERRIDE,
            max_model_len=512,
            max_num_seqs=args.batch,
            max_num_batched_tokens=128,
            compilation_config={"compile_sizes": sorted({1, args.batch, 128})},
            kv_cache_dtype="auto" if args.kv_cache == "fp16" else "fp8",
            num_gpu_blocks_override=1 + args.batch * 4,
            enforce_eager=False,
        )
        result["settings"] = settings
        engine = LLM(**settings)
        result["startup_seconds"] = time.monotonic() - started
        stage = "generate"
        prompts = [
            "What is the capital of France? Answer briefly.",
            "Compute 7 multiplied by 8. Answer briefly.",
            "Write one sentence explaining why leaves look green.",
            "Name three common programming languages.",
        ][: args.batch]
        tokenizer = engine.get_tokenizer()
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        started = time.monotonic()
        outputs = engine.generate(
            formatted,
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        result["generation_seconds"] = time.monotonic() - started
        result["outputs"] = [
            {"prompt": prompts[i], "text": o.outputs[0].text, "token_ids": o.outputs[0].token_ids}
            for i, o in enumerate(outputs)
        ]
        assert len(outputs) == args.batch
        assert all(len(o.outputs[0].token_ids) == args.max_tokens for o in outputs)
        result["passed"] = True
    except Exception as exc:
        result.update(
            stage=stage,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown(timeout=60)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
