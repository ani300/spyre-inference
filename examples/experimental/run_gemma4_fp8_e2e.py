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

"""Run Gemma FP8 full-model cases serially; preserve logs and nonzero exits."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kv-cache", nargs="+", choices=["fp16", "fp8"], default=["fp16", "fp8"])
    parser.add_argument("--batch", nargs="+", type=int, choices=[1, 4], default=[1, 4])
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--eager-blocks", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for kv in args.kv_cache:
        for batch in args.batch:
            name = f"{kv}-b{batch}"
            output = args.output_dir / f"{name}.json"
            output.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("gemma4_fp8_model.py")),
                "--kv-cache",
                kv,
                "--batch",
                str(batch),
                "--output",
                str(output),
                "--max-tokens",
                str(args.max_tokens),
                "--max-model-len",
                str(args.max_model_len),
            ]
            if args.dataset:
                command.extend(["--dataset", str(args.dataset)])
            if args.eager_blocks:
                command.append("--eager-blocks")
            (args.output_dir / f"{name}.command.json").write_text(
                json.dumps(command, indent=2) + "\n"
            )
            print("Running", name, flush=True)
            with (args.output_dir / f"{name}.log").open("w") as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            result = json.loads(output.read_text()) if output.exists() else {}
            summary.append(
                dict(
                    case=name,
                    returncode=code,
                    passed=code == 0 and result.get("passed", False),
                    stage=result.get("stage"),
                )
            )
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(summary[-1], flush=True)
            if code < 0 or code == 255:
                raise SystemExit(
                    "Hardware/process abort; inspect log and health-check before continuing"
                )
    raise SystemExit(0 if all(case["passed"] for case in summary) else 1)


if __name__ == "__main__":
    main()
