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

"""Run one Spyre process at a time; preserve failures and refresh a local HTML report."""

import argparse
import datetime
import html
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    probe = Path(__file__).with_name("fp8_qk_cache.py")
    cases = [
        ("staged", "full", 1, 128),
        ("packed", "full", 1, 128),
        ("packed", "read", 1, 128),
        ("packed", "update", 1, 1),
        ("staged", "full", 4, 128),
        ("staged", "update", 1, 1),
        ("staged", "update", 4, 1),
        ("packed", "full", 4, 128),
        ("packed", "read", 4, 128),
        ("packed", "update", 4, 1),
    ]
    results = []
    for path, operation, batch, query_len in cases:
        name = f"{path}-{operation}-b{batch}-q{query_len}"
        command = [
            sys.executable,
            "-u",
            str(probe),
            "--path",
            path,
            "--operation",
            operation,
            "--batch",
            str(batch),
            "--query-len",
            str(query_len),
            "--output",
            str(root / f"{name}.json"),
        ]
        print(f"RUN {name}", flush=True)
        with (root / f"{name}.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        result = {
            "name": name,
            "exit_code": process.returncode,
            "command": command,
            "time": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        results.append(result)
        (root / "matrix.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        rows = []
        for item in results:
            stem = item["name"]
            detail_path = root / f"{stem}.json"
            detail = json.loads(detail_path.read_text()) if detail_path.exists() else {}
            rows.append(
                f"<tr><td>{html.escape(stem)}</td><td>{item['exit_code']}</td>"
                f"<td>{html.escape(detail.get('stage', 'unknown'))}</td>"
                f'<td><a href="{stem}.log">log</a> · <a href="{stem}.json">JSON</a></td></tr>'
            )
        (root / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>FP8 QK cache paths</title>'
            "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}"
            "table{border-collapse:collapse}"
            "td,th{padding:10px;border:1px solid #aaa;text-align:left}</style>"
            "<h1>FP8 QK cache paths</h1><p>Native failures are retained, "
            "not replaced by the staged path. All probes ran serially. "
            "Each log contains the full traceback and numerical results.</p>"
            "<table><tr><th>Probe</th><th>Exit code</th><th>Last stage</th><th>Evidence</th></tr>"
            + "".join(rows)
            + "</table></html>"
        )
    return int(any(item["exit_code"] != 0 for item in results))


if __name__ == "__main__":
    sys.exit(main())
