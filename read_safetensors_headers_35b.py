import json
import struct
from pathlib import Path

model_path = Path("/Users/sbaruwal/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec")
index_path = model_path / "model.safetensors.index.json"

with open(index_path) as f:
    index_json = json.load(f)

weight_map = index_json["weight_map"]

targets = [
    "language_model.model.layers.11.mlp.switch_mlp.gate_proj.weight",
    "language_model.model.layers.11.mlp.switch_mlp.gate_proj.scales",
    "language_model.model.layers.11.mlp.switch_mlp.gate_proj.biases",
    "language_model.model.layers.11.mlp.switch_mlp.up_proj.weight",
    "language_model.model.layers.11.mlp.switch_mlp.up_proj.scales",
    "language_model.model.layers.11.mlp.switch_mlp.up_proj.biases",
    "language_model.model.layers.11.mlp.switch_mlp.down_proj.weight",
    "language_model.model.layers.11.mlp.switch_mlp.down_proj.scales",
    "language_model.model.layers.11.mlp.switch_mlp.down_proj.biases",
]

def read_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len

cache = {}

for name in targets:
    shard = weight_map[name]
    shard_path = model_path / shard
    if shard not in cache:
        header, data_start = read_safetensors_header(shard_path)
        cache[shard] = (header, data_start)

    header, data_start = cache[shard]
    meta = header[name]
    rel_begin, rel_end = meta["data_offsets"]
    size_bytes = rel_end - rel_begin

    print(f"\n{name}")
    print(f"  shard: {shard}")
    print(f"  dtype: {meta['dtype']}")
    print(f"  shape: {meta['shape']}")
    print(f"  rel_offsets: {meta['data_offsets']}")
    print(f"  size_bytes: {size_bytes}")