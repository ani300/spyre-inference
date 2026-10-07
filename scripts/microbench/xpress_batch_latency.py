#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Time concurrent Qwen3 requests; run each decoding/attention mode in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from xpress_latency import (
    DRAFT,
    DRAFT_REVISION,
    TARGET,
    TARGET_REVISION,
    _acceptance,
    _worker_state,
)


def _fingerprint(worker):
    import torch_spyre
    from jagged_attn_latency import binary_manifest, source_manifest

    import spyre_inference

    return dict(
        sources={
            "inference": source_manifest(spyre_inference),
            "torch_spyre": source_manifest(torch_spyre),
        },
        binaries=binary_manifest(torch_spyre),
    )


def _start_device_profile(worker):
    import torch

    worker._batch_latency_profile = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
    )
    worker._batch_latency_profile.start()


def _stop_device_profile(worker, path):
    import torch

    torch.spyre.synchronize()
    profile = worker._batch_latency_profile
    profile.stop()
    profile.export_chrome_trace(path)
    del worker._batch_latency_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jagged", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode", choices=("ordinary", "xpress"), default="xpress")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", nargs=4, type=int, default=[32, 127, 255, 383])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--token-budget", type=int, choices=(128, 256), default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jagged-batch-parallel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--jagged-min-query-tile", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--profile-output", type=Path)
    args = parser.parse_args()
    if min(args.prompt_lengths) < 1 or args.max_tokens < 2 or min(args.warmup, args.repeats) < 1:
        parser.error("positive lengths/repeats and at least two output tokens are required")
    if max(args.prompt_lengths) + args.max_tokens > args.max_model_len:
        parser.error("prompt plus output length must fit max-model-len")
    buckets = [1, 16, 32, 64] + ([128] if args.token_budget >= 128 else [])
    if args.token_budget == 256:
        buckets.append(256)
    os.environ["SPYRE_JAGGED_ATTENTION"] = str(args.jagged)
    os.environ["SPYRE_JAGGED_BATCH_PARALLEL"] = str(args.jagged_batch_parallel)
    os.environ["SPYRE_JAGGED_MIN_QUERY_TILE"] = str(args.jagged_min_query_tile)
    os.environ["SPYRE_ATTN_KV_LAYOUT"] = "head_major"
    os.environ["SPYRE_MAX_NUM_PARTIAL_PREFILLS"] = "4"
    os.environ["SPYRE_ATTN_QUERY_BUCKETS"] = ",".join(map(str, (1, 16, 64, args.token_budget)))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    models = [(TARGET, TARGET_REVISION)]
    if args.mode == "xpress":
        models.append((DRAFT, DRAFT_REVISION))
    for model, revision in models:
        snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        created_at=datetime.now(UTC).isoformat(),
        parameters=vars(args)
        | {
            "output": str(args.output),
            "profile_output": str(args.profile_output) if args.profile_output else None,
        },
        model=TARGET,
        revision=TARGET_REVISION,
        draft=DRAFT if args.mode == "xpress" else None,
        draft_revision=DRAFT_REVISION if args.mode == "xpress" else None,
        dtype="float16",
        seed=2026,
        temperature=0.0,
        ignore_eos=True,
        max_num_seqs=4,
        passes=6 if args.mode == "xpress" else None,
        topc=512 if args.mode == "xpress" else None,
        compile_sizes=buckets,
        card=os.environ.get("SPYRE_DEVICES"),
        execution_environment={
            key: os.environ.get(key)
            for key in (
                "SPYRE_ATTN_RECORD",
                "SPYRE_ATTN_PROFILING",
                "SPYRE_ATTN_QUERY_BUCKETS",
                "SPYRE_JAGGED_PARALLEL_ENTRIES",
                "SPYRE_JAGGED_BATCH_PARALLEL",
                "SPYRE_JAGGED_MIN_QUERY_TILE",
                "TORCHINDUCTOR_COMPILE_THREADS",
                "DXP_LOOP_UNROLL",
            )
        },
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        timing=(
            "Frontend wall time from resuming four queued requests. "
            "Scheduling is paused during admission; wake-up is included in timing. "
            "State RPCs, snapshots and report writes are outside timing. "
            "TPOT is per request; aggregate throughput includes all prefills and completion tails."
        ),
        warmup=[],
        runs=[],
        complete=False,
    )

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
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        max_num_batched_tokens=args.token_budget,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=2026,
        compilation_config={"compile_sizes": buckets},
        speculative_config=(
            dict(
                method="dflash",
                model=DRAFT,
                revision=DRAFT_REVISION,
                num_speculative_tokens=15,
                xpress_num_passes=6,
                xpress_topc=512,
            )
            if args.mode == "xpress"
            else None
        ),
    )
    try:
        report["initialization_seconds"] = time.perf_counter() - started
        report["stack_before"] = llm.collective_rpc(_fingerprint)[0]
        initial = llm.collective_rpc(_worker_state, args=(True,))[0]
        assert all(row["jagged"] == bool(args.jagged) for row in initial["attention"].values())
        assert len(initial["attention"]) == (41 if args.mode == "xpress" else 36)
        assert sum(not row["causal"] for row in initial["attention"].values()) == (
            5 if args.mode == "xpress" else 0
        )
        report["worker_before_warmup"] = initial
        engine = llm.llm_engine
        tokens = llm.get_tokenizer().encode(
            "IBM develops computers, software, and consulting services. "
            * (max(args.prompt_lengths) // 8 + 8)
        )
        prompts = [tokens[:length] for length in args.prompt_lengths]

        def run(label):
            llm.sleep(level=0, mode="keep")
            while not engine.engine_core.outputs_queue.empty():
                engine.step()
            before = llm.collective_rpc(_worker_state)[0]
            metrics_before = _acceptance(llm)
            seen = {}
            arrivals = {}
            ids = []
            final = {}
            step_index = 0
            output_batch_sizes = [0] * 5
            for index, prompt in enumerate(prompts):
                request = f"{label}_{index}"
                engine.add_request(
                    request,
                    {"prompt_token_ids": prompt},
                    SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True),
                )
                ids.append(request)
                seen[request] = 0
                arrivals[request] = []
            start = time.perf_counter()
            llm.wake_up(tags=["scheduling"])
            while engine.has_unfinished_requests():
                outputs = engine.step()
                output_batch_sizes[len(outputs)] += 1
                step_index += 1
                now = time.perf_counter() - start
                for output in outputs:
                    request = output.request_id
                    count = len(output.outputs[0].token_ids)
                    if count > seen[request]:
                        arrivals[request].append(
                            dict(step=step_index, seconds=now, new_tokens=count - seen[request])
                        )
                        seen[request] = count
                    if output.finished:
                        final[request] = output
            elapsed = time.perf_counter() - start
            assert set(final) == set(ids) and all(
                count == args.max_tokens for count in seen.values()
            )
            after = llm.collective_rpc(_worker_state)[0]
            metrics_after = _acceptance(llm)
            batch_counts = [
                a - b for a, b in zip(after["proposal_batch_sizes"], before["proposal_batch_sizes"])
            ]
            requests = []
            for request, prompt in zip(ids, prompts):
                bursts = arrivals[request]
                requests.append(
                    dict(
                        prompt_ids=prompt,
                        token_ids=list(final[request].outputs[0].token_ids),
                        ttft_seconds=bursts[0]["seconds"],
                        seconds=bursts[-1]["seconds"],
                        effective_tpot_seconds=(bursts[-1]["seconds"] - bursts[0]["seconds"])
                        / (args.max_tokens - 1),
                        arrival_bursts=bursts,
                    )
                )
            row = dict(
                label=label,
                seconds=elapsed,
                output_tokens_per_second=sum(seen.values()) / elapsed,
                requests=requests,
                proposal_batch_sizes=batch_counts,
                output_batch_sizes=output_batch_sizes,
                unique_graphs=after["unique_graphs"] - before["unique_graphs"],
                acceptance={key: metrics_after[key] - metrics_before[key] for key in metrics_after},
            )
            for group in ("counters", "drafter_counters"):
                row[group] = {key: after[group][key] - before[group][key] for key in before[group]}
            return row

        for repeat in range(args.warmup):
            row = run(f"warmup_{repeat}")
            report["warmup"].append(row)
            save()
            print(
                "WARMUP",
                row["seconds"],
                "proposal batch counts",
                row["proposal_batch_sizes"],
                flush=True,
            )
            batch_key = "proposal_batch_sizes" if args.mode == "xpress" else "output_batch_sizes"
            assert row[batch_key][4] > 0, "Scheduler never reached four simultaneous requests"
        expected = [r["token_ids"] for r in report["warmup"][-1]["requests"]]
        for repeat in range(args.repeats):
            row = run(f"repeat_{repeat}")
            report["runs"].append(row)
            save()
            assert row["unique_graphs"] == 0, row
            assert row[batch_key][4] > 0
            assert [r["token_ids"] for r in row["requests"]] == expected
            print(
                "RUN",
                repeat,
                row["seconds"],
                row["output_tokens_per_second"],
                row["acceptance"],
                flush=True,
            )
        report["stack_after"] = llm.collective_rpc(_fingerprint)[0]
        assert report["stack_before"] == report["stack_after"], "Stack changed during measurement"
        report["worker_after_measurement"] = llm.collective_rpc(_worker_state)[0]
        report["summary"] = {
            key: statistics.median(row[key] for row in report["runs"])
            for key in ("seconds", "output_tokens_per_second")
        }
        report["token_digest"] = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
        if args.profile_output:
            args.profile_output.parent.mkdir(parents=True, exist_ok=True)
            llm.collective_rpc(_start_device_profile)
            try:
                report["profile_run"] = run("profile")
            finally:
                llm.collective_rpc(_stop_device_profile, args=(str(args.profile_output.resolve()),))
            assert report["profile_run"]["unique_graphs"] == 0
            assert [r["token_ids"] for r in report["profile_run"]["requests"]] == expected
        report["complete"] = True
        save()
        print("COMPLETE", args.output, report["summary"], flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
