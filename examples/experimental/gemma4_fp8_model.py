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
import hashlib
import json
import os
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[1, 4], default=1)
    parser.add_argument("--kv-cache", choices=["fp16", "fp8"], default="fp16")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--dataset", type=Path, help="Spyre benchmark JSONL input trace")
    parser.add_argument("--dataset-offset", type=int, default=0)
    parser.add_argument(
        "--attention-page-loop",
        choices=["tiled", "unrolled"],
        help="Override only the attention page loop; FP8 MoE stays persistent",
    )
    parser.add_argument(
        "--eager-blocks",
        action="store_true",
        help="Keep block orchestration eager; FP8 kernels still compile",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.attention_page_loop:
        os.environ["SPYRE_ATTN_FOR_EACH_TILE"] = str(int(args.attention_page_loop == "tiled"))
    os.environ.setdefault("HF_HOME", "/mnt/models/hf_cache")
    os.environ.setdefault("SPYRE_NUM_CPUS", "8")
    result = {
        **vars(args),
        "output": str(args.output),
        "dataset": str(args.dataset) if args.dataset else None,
        "passed": False,
    }
    result["versions"] = {p: version(p) for p in ("torch", "torch-spyre", "vllm")}
    result["environment"] = {
        key: os.environ.get(key)
        for key in (
            "SPYRE_ATTN_RECORD",
            "SPYRE_ATTN_FOR_EACH_TILE",
            "SPYRE_KERNEL_CACHE",
            "SPYRE_NUM_CPUS",
            "PYTHONPATH",
        )
    }
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
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch,
            max_num_batched_tokens=128,
            compilation_config={"compile_sizes": [1, 128] if args.batch == 1 else [1, 2, 4, 128]},
            kv_cache_dtype="auto" if args.kv_cache == "fp16" else "fp8",
            num_gpu_blocks_override=1 + args.batch * ((args.max_model_len + 127) // 128),
            enforce_eager=args.eager_blocks,
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
                enable_thinking=False,
            )
            for p in prompts
        ]
        if args.dataset:
            rows = [
                json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()
            ]
            rows = rows[args.dataset_offset : args.dataset_offset + args.batch]
            assert len(rows) == args.batch, "Not enough dataset rows"
            formatted = [row["prompt"] for row in rows]
            prompts = formatted
            result["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        result["prompt_lengths"] = [len(tokenizer.encode(cast(str, p))) for p in formatted]
        assert max(result["prompt_lengths"]) + args.max_tokens <= args.max_model_len
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
