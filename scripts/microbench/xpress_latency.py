#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measure serial XPress generation with identical jagged/regular configurations.

Run each attention mode in a fresh process. Worker RPCs, JSON writes and cache
inspection stay outside timed requests. Speculative output arrives in bursts;
the reported TPOT is elapsed decode time per token, not a per-burst latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

TARGET = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT = "UIUC-SSAIL/Qwen3-8B-XPress-b16"
DRAFT_REVISION = "f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5"


def _worker_state(worker, prepare=False):
    import warnings

    import torch
    import torch_spyre
    from torch._dynamo.utils import counters as compile_counters
    from torch_spyre.ops.fallbacks import FallbackWarning

    import spyre_inference

    if prepare:
        warnings.simplefilter("error", FallbackWarning)
    runner = worker.model_runner
    layers = runner.compilation_config.static_forward_context
    attention = {
        name: {
            "jagged": layers[name].impl._jagged_enabled,
            "causal": bool(getattr(layers[name], "spyre_causal", True)),
        }
        for name in runner._spyre_kv_caches
    }
    memory = torch.spyre.memory.memory_stats(0)
    drafter = getattr(runner, "drafter", None)
    model = drafter.model if drafter is not None else None
    counters = {
        key: getattr(model, key)
        for key in (
            "device_to_host_seconds",
            "host_argmax_seconds",
            "host_topk_seconds",
            "host_to_device_seconds",
            "candidate_gather_seconds",
            "refiner_seconds",
            "logits_seconds",
            "selection_calls",
            "logits_transfer_bytes",
            "candidate_transfer_bytes",
            "device_selection_calls",
            "device_selection_seconds",
            "proposal_id_transfer_bytes",
        )
        if model is not None
    }
    return {
        "torch": torch.__version__,
        "torch_spyre": torch_spyre.__file__,
        "inference": spyre_inference.__file__,
        "attention": attention,
        "threads": torch.get_num_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "allocated_bytes": memory["allocated_bytes.all.current"],
        "peak_allocated_bytes": memory["allocated_bytes.all.peak"],
        "counters": counters,
        "unique_graphs": compile_counters["stats"]["unique_graphs"],
        "proposal_batch_sizes": list(drafter.proposal_batch_sizes) if drafter else [0] * 5,
        "drafter_counters": {
            key: getattr(drafter, key)
            for key in (
                "proposal_calls",
                "proposal_requests",
                "proposal_padded_requests",
                "proposal_seconds",
                "context_tokens",
                "context_seconds",
                "draft_forward_seconds",
            )
            if drafter is not None
        },
    }


def _acceptance(llm):
    wanted = {"num_drafts", "num_draft_tokens", "num_accepted_tokens"}
    return {
        metric.name.removeprefix("vllm:spec_decode_"): metric.value
        for metric in llm.get_metrics()
        if metric.name.removeprefix("vllm:spec_decode_") in wanted
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jagged", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.repeats < 1 or not 2 <= args.max_tokens <= 128:
        parser.error("use positive repeats and 2..128 generated tokens")
    os.environ["SPYRE_JAGGED_ATTENTION"] = str(args.jagged)
    os.environ["SPYRE_ATTN_KV_LAYOUT"] = "head_major"
    os.environ["SPYRE_ATTN_QUERY_BUCKETS"] = "1,16,64"
    # The benchmark sends its own trusted callable to the local worker.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    for model, revision in ((TARGET, TARGET_REVISION), (DRAFT, DRAFT_REVISION)):
        snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "jagged": bool(args.jagged),
        "model": TARGET,
        "revision": TARGET_REVISION,
        "draft": DRAFT,
        "draft_revision": DRAFT_REVISION,
        "passes": 6,
        "topc": 512,
        "max_tokens": args.max_tokens,
        "max_model_len": 512,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 1,
        "logprobs": None,
        "ignore_eos": True,
        "temperature": 0.0,
        "timing": (
            "Frontend request time; effective TPOT excludes first-token time. "
            "No in-request RPCs or cache snapshots."
        ),
        "inference_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "compilers": {name: shutil.which(name) for name in ("dbo-opt", "dxp_standalone")},
        "card": os.environ.get("SPYRE_DEVICES"),
        "warmup": [],
        "runs": [],
        "complete": False,
    }

    def save():
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    started = time.perf_counter()
    llm = LLM(
        model=TARGET,
        revision=TARGET_REVISION,
        tokenizer_revision=TARGET_REVISION,
        dtype="float16",
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=2026,
        compilation_config={"compile_sizes": [1, 16, 64]},
        speculative_config={
            "method": "dflash",
            "model": DRAFT,
            "revision": DRAFT_REVISION,
            "num_speculative_tokens": 15,
            "xpress_num_passes": 6,
            "xpress_topc": 512,
        },
    )
    try:
        report["initialization_seconds"] = time.perf_counter() - started
        save()
        print("INITIALIZED", report["initialization_seconds"], flush=True)
        state = llm.collective_rpc(_worker_state, args=(True,))[0]
        assert len(state["attention"]) == 41, state["attention"]
        assert all(row["jagged"] == bool(args.jagged) for row in state["attention"].values())
        assert sum(not row["causal"] for row in state["attention"].values()) == 5
        report["worker_before_warmup"] = state
        tokens = llm.get_tokenizer().encode(
            "IBM develops computers, software, and consulting services. " * 64
        )
        cases = [{"name": f"context_{length}", "ids": tokens[:length]} for length in (32, 127, 383)]
        engine = llm.llm_engine

        def run(case, request_id):
            before = _acceptance(llm)
            seen = 0
            arrivals = []
            start = time.perf_counter()
            engine.add_request(
                request_id,
                {"prompt_token_ids": case["ids"]},
                SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True),
            )
            final = None
            while engine.has_unfinished_requests():
                outputs = engine.step()
                now = time.perf_counter() - start
                for result in outputs:
                    assert result.request_id == request_id
                    count = len(result.outputs[0].token_ids)
                    if count > seen:
                        arrivals.append({"seconds": now, "new_tokens": count - seen})
                        seen = count
                    if result.finished:
                        final = result
            elapsed = time.perf_counter() - start
            assert final is not None and seen == args.max_tokens, (request_id, seen)
            after = _acceptance(llm)
            output = final.outputs[0]
            return {
                "case": case["name"],
                "prompt_ids": case["ids"],
                "token_ids": list(output.token_ids),
                "seconds": elapsed,
                "ttft_seconds": arrivals[0]["seconds"],
                "effective_tpot_seconds": (arrivals[-1]["seconds"] - arrivals[0]["seconds"])
                / (seen - 1),
                "arrival_bursts": arrivals,
                "acceptance": {key: after[key] - before[key] for key in after},
            }

        for case in cases:
            row = run(case, "warmup_" + case["name"])
            report["warmup"].append(row)
            save()
            print("WARMUP", case["name"], row["seconds"], flush=True)
        report["worker_after_warmup"] = llm.collective_rpc(_worker_state)[0]
        for repeat in range(args.repeats):
            for case in cases if repeat % 2 == 0 else reversed(cases):
                row = run(case, f"repeat_{repeat}_" + case["name"])
                row["repeat"] = repeat
                warmup = next(x for x in report["warmup"] if x["case"] == case["name"])
                assert row["token_ids"] == warmup["token_ids"], (case["name"], repeat)
                report["runs"].append(row)
                save()
                print(
                    "RUN",
                    case["name"],
                    repeat,
                    row["seconds"],
                    row["effective_tpot_seconds"],
                    flush=True,
                )
        report["worker_after_measurement"] = llm.collective_rpc(_worker_state)[0]
        report["summary"] = {
            case["name"]: {
                key: statistics.median(
                    row[key] for row in report["runs"] if row["case"] == case["name"]
                )
                for key in ("seconds", "ttft_seconds", "effective_tpot_seconds")
            }
            for case in cases
        }
        report["token_digest"] = hashlib.sha256(
            json.dumps([row["token_ids"] for row in report["warmup"]]).encode()
        ).hexdigest()
        report["complete"] = True
        save()
        print("COMPLETE", args.output, flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
