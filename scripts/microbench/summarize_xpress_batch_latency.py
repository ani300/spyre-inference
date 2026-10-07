#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare completed batch-four reports, preserving token and acceptance differences."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regular", type=Path, required=True)
    parser.add_argument("--jagged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {
        name: json.loads(path.read_text())
        for name, path in (("regular", args.regular), ("jagged", args.jagged))
    }
    regular, jagged = reports.values()
    assert all(report["complete"] for report in reports.values()), "Incomplete measurement"
    assert regular["parameters"]["jagged"] == 0 and jagged["parameters"]["jagged"] == 1
    settings = lambda report: {
        key: value
        for key, value in report["parameters"].items()
        if key not in ("jagged", "output", "warmup", "repeats")
    }
    assert settings(regular) == settings(jagged), "Workload settings differ"
    assert regular["card"] == jagged["card"]
    assert regular["execution_environment"] == jagged["execution_environment"]
    for key in ("torch", "threads", "cpu_affinity"):
        assert regular["worker_before_warmup"][key] == jagged["worker_before_warmup"][key], key
    assert (
        regular["stack_before"]
        == regular["stack_after"]
        == jagged["stack_before"]
        == jagged["stack_after"]
    )
    summary = {
        "parameters": settings(regular),
        "card": regular["card"],
        "worker": {
            key: regular["worker_before_warmup"][key]
            for key in ("torch", "threads", "cpu_affinity")
        },
        "stack": regular["stack_before"],
        "source_reports": {"regular": str(args.regular), "jagged": str(args.jagged)},
        "groups": {},
    }
    signatures = {}
    token_signatures = {}
    for mode, report in reports.items():
        runs = report["runs"]
        signatures[mode] = set()
        token_signatures[mode] = set()
        for run in runs:
            token_work = [
                {key: request[key] for key in ("prompt_ids", "token_ids")}
                for request in run["requests"]
            ]
            token_signatures[mode].add(
                hashlib.sha256(json.dumps(token_work, sort_keys=True).encode()).hexdigest()
            )
            bursts = [
                [
                    {key: burst[key] for key in ("step", "new_tokens")}
                    for burst in request["arrival_bursts"]
                ]
                for request in run["requests"]
            ]
            signatures[mode].add(
                hashlib.sha256(
                    json.dumps(
                        {
                            "requests": token_work,
                            "acceptance": run["acceptance"],
                            "batches": run["proposal_batch_sizes"],
                            "bursts": bursts,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
            )
            assert run["unique_graphs"] == 0
            batch_key = (
                "proposal_batch_sizes"
                if report["parameters"]["mode"] == "xpress"
                else "output_batch_sizes"
            )
            assert run[batch_key][4] > 0
        group = {
            "samples": len(runs),
            "seconds": [run["seconds"] for run in runs],
            "median_seconds": statistics.median(run["seconds"] for run in runs),
            "median_output_tokens_per_second": statistics.median(
                run["output_tokens_per_second"] for run in runs
            ),
            "acceptance": [run["acceptance"] for run in runs],
            "proposal_batch_sizes": [run["proposal_batch_sizes"] for run in runs],
            "output_batch_sizes": [run["output_batch_sizes"] for run in runs],
            "requests": [
                {
                    key: statistics.median(run["requests"][index][key] for run in runs)
                    for key in ("ttft_seconds", "seconds", "effective_tpot_seconds")
                }
                for index in range(4)
            ],
        }
        for namespace in ("counters", "drafter_counters"):
            group[namespace] = {
                key: sum(run[namespace][key] for run in runs) for key in runs[0][namespace]
            }
        summary["groups"][mode] = group
    summary["identical_tokens"] = (
        len(token_signatures["regular"]) == 1
        and token_signatures["regular"] == token_signatures["jagged"]
    )
    summary["identical_tokens_acceptance_batches_and_bursts"] = (
        len(signatures["regular"]) == 1 and signatures["regular"] == signatures["jagged"]
    )
    summary["jagged_latency_change_percent"] = 100 * (
        summary["groups"]["jagged"]["median_seconds"]
        / summary["groups"]["regular"]["median_seconds"]
        - 1
    )
    summary["jagged_throughput_ratio"] = (
        summary["groups"]["regular"]["median_seconds"]
        / summary["groups"]["jagged"]["median_seconds"]
    )
    summary["counter_scope"] = (
        "Host wall intervals include pending device work and are not exclusive kernel/DMA times."
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key.startswith(("identical", "jagged"))
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
