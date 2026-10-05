#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare completed xpress_latency.py runs, including whether they did equal work."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    config = None
    for item in args.group:
        label, path = item.split("=", 1)
        data = json.loads(Path(path).read_text())
        assert data["complete"], f"Incomplete run: {path}"
        settings = {
            key: data[key]
            for key in (
                "model",
                "revision",
                "draft",
                "draft_revision",
                "passes",
                "topc",
                "max_tokens",
                "max_model_len",
                "max_num_batched_tokens",
                "max_num_seqs",
                "temperature",
                "ignore_eos",
                "logprobs",
                "card",
            )
        }
        if config is None:
            config = settings
        assert settings == config, f"Configuration differs: {path}"
        groups[label].append((path, data))

    report = {
        "configuration": config,
        "scope": (
            "Warm serial requests; startup excluded. "
            "Ranges are observations, not confidence intervals."
        ),
        "groups": {},
        "comparisons": {},
    }
    signatures = defaultdict(dict)
    for label, files in groups.items():
        rows = [row for _, data in files for row in data["runs"]]
        cases = sorted({row["case"] for row in rows})
        summaries = {}
        for case in cases:
            matching = [row for row in rows if row["case"] == case]
            signatures[label][case] = {
                hashlib.sha256(
                    json.dumps(
                        {key: row[key] for key in ("prompt_ids", "token_ids", "acceptance")},
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                for row in matching
            }
            summaries[case] = {
                "samples": len(matching),
                "work_signatures": sorted(signatures[label][case]),
                "acceptance": [row["acceptance"] for row in matching],
                **{
                    key: {
                        "median": statistics.median(row[key] for row in matching),
                        "min": min(row[key] for row in matching),
                        "max": max(row[key] for row in matching),
                    }
                    for key in ("seconds", "ttft_seconds", "effective_tpot_seconds")
                },
            }
        counters = defaultdict(float)
        for _, data in files:
            before = data["worker_after_warmup"]
            after = data["worker_after_measurement"]
            for namespace in ("counters", "drafter_counters"):
                for key, value in after[namespace].items():
                    counters[key] += value - before[namespace][key]
        report["groups"][label] = {
            "files": [path for path, _ in files],
            "inference_heads": sorted({data["inference_head"] for _, data in files}),
            "jagged": sorted({data["jagged"] for _, data in files}),
            "cases": summaries,
            "counters": dict(counters),
            "counter_scope": (
                "Host wall intervals include pending device work; "
                "they are not exclusive kernel or DMA times."
            ),
            "milliseconds_per_proposal": {
                key: 1000 * value / counters["proposal_calls"]
                for key, value in counters.items()
                if key.endswith("_seconds")
            },
        }

    reference = next(iter(groups))
    for label in list(groups)[1:]:
        assert signatures[reference].keys() == signatures[label].keys(), "Different cases"
        comparison = {}
        for case in signatures[reference]:
            baseline = report["groups"][reference]["cases"][case]
            candidate = report["groups"][label]["cases"][case]
            comparison[case] = {
                "identical_tokens_and_acceptance": (
                    len(signatures[reference][case]) == 1
                    and signatures[reference][case] == signatures[label][case]
                ),
                "observed_latency_ratio_reference_over_candidate": (
                    baseline["seconds"]["median"] / candidate["seconds"]["median"]
                ),
                "effective_tpot_change_percent": 100
                * (
                    candidate["effective_tpot_seconds"]["median"]
                    / baseline["effective_tpot_seconds"]["median"]
                    - 1
                ),
            }
        report["comparisons"][f"{reference}_vs_{label}"] = comparison
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
