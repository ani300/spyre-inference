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

"""Serial hardware matrix; retain logs/JSON, including process-level failures."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        choices=[
            "staged",
            "native",
            "persistent",
            "persistent-sharded",
            "persistent-dma",
            "rank3-dma",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--experts", type=int, choices=[32, 128], default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Restrict to a case such as staged-b4-q1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        ("staged", "native", "persistent", "persistent-sharded", "persistent-dma", "rank3-dma")
        if args.path == "all"
        else (args.path,)
    )
    cases = [
        (p, b, q)
        for p in paths
        for b, q in (((1, 1),) if p == "rank3-dma" else ((1, 1), (4, 1), (1, 128), (4, 128)))
    ]
    if args.case:
        cases = [(p, b, q) for p, b, q in cases if f"{p}-b{b}-q{q}" in args.case]
        if not cases:
            parser.error("No matching case")
    summary = []
    for path, batch, query in cases:
        name = f"{path}-b{batch}-q{query}"
        result_path = args.output_dir / (name + ".json")
        result_path.unlink(missing_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("gemma4_fp8_grouped.py")),
            "--path",
            path,
            "--experts",
            str(args.experts),
            "--batch",
            str(batch),
            "--query-len",
            str(query),
            "--output",
            str(result_path),
        ]
        (args.output_dir / (name + ".command.json")).write_text(json.dumps(cmd, indent=2) + "\n")
        print("Running", name, flush=True)
        log_path = args.output_dir / (name + ".log")
        with log_path.open("w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if not result_path.exists():
            stages = [
                s
                for s in log_path.read_text(errors="replace").splitlines()
                if s.startswith("Stage:")
            ]
            result_path.write_text(
                json.dumps(
                    {
                        "passed": False,
                        "returncode": code,
                        "stage": stages[-1] if stages else "unknown",
                        "error": "Process exited without writing JSON; inspect log",
                    },
                    indent=2,
                )
                + "\n"
            )
        result = json.loads(result_path.read_text())
        summary.append(
            {
                "case": name,
                "returncode": code,
                "passed": code == 0 and result.get("passed", False),
                "stage": result.get("stage"),
                "relative_l2": result.get("relative_l2"),
            }
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(summary[-1], flush=True)
        # A hardware abort needs an explicit health check before another workload.
        if code < 0 or code == 255:
            break
    raise SystemExit(0 if all(r["passed"] for r in summary) and len(summary) == len(cases) else 1)


if __name__ == "__main__":
    main()
