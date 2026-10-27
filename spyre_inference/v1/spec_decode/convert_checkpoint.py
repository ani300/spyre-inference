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

"""Export a local Speculators XPress checkpoint without importing its training runtime."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from spyre_inference.v1.spec_decode.checkpoint import convert_speculators_checkpoint


def convert_checkpoint(source: Path, target_config: Path, output: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    target = json.loads(target_config.read_text())
    index = source / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        filenames = sorted(set(weight_map.values()))
    else:
        weight_map = None
        filenames = ["model.safetensors"]
    weights, sources = {}, {}
    for filename in filenames:
        path = source / filename
        shard = load_file(path)
        if weights.keys() & shard.keys():
            raise ValueError("Duplicate tensor names across checkpoint shards")
        weights.update(shard)
        with path.open("rb") as stream:
            sources[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    if weight_map is not None and weights.keys() != weight_map.keys():
        raise ValueError("Checkpoint shards do not match their safetensors index")
    converted_config, converted = convert_speculators_checkpoint(config, weights, target)
    output.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in converted.items()}, output / "model.safetensors")
    (output / "config.json").write_text(json.dumps(converted_config, indent=2) + "\n")
    manifest = dict(
        format_version=1,
        source_sha256=sources,
        source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        target_config=target,
        target_config_sha256=hashlib.sha256(target_config.read_bytes()).hexdigest(),
        tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in converted.items()},
        mixer="raw; folded exactly once by the serving loader",
        anchor_predecessor="anchor token, matching the pinned upstream serving implementation",
    )
    (output / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Local Speculators checkpoint")
    parser.add_argument(
        "--target-config", required=True, type=Path, help="Matching target config.json"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for the serving checkpoint"
    )
    args = parser.parse_args()
    convert_checkpoint(args.input, args.target_config, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
