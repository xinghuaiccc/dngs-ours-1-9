#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <data_root> <output_root> [gpu_id]"
  echo "Example: $0 data/dtu output/dtu 0"
  exit 1
fi

data_root="$1"
output_root="$2"
gpu_id="${3:-0}"

if [ "${data_root#/}" != "$data_root" ]; then
  data_root_abs="$data_root"
elif [ -d "$(pwd)/$data_root" ]; then
  data_root_abs="$(pwd)/$data_root"
elif [ -d "$data_root" ]; then
  data_root_abs="$data_root"
else
  echo "Error: data_root not found: $data_root"
  exit 1
fi

if [ ! -d "${data_root_abs}/scan30" ]; then
  echo "Error: expected scan folders under: ${data_root_abs}"
  exit 1
fi

echo "=== DPT DEPTH (DTU) ==="
scans=(
  scan34 scan41 scan45 scan82 scan114 scan31 scan8
)

missing_depth=0
for scan_id in "${scans[@]}"; do
  depth_dir="${data_root_abs}/${scan_id}/depth_maps"
  if [ ! -d "$depth_dir" ]; then
    missing_depth=1
    break
  fi
  shopt -s nullglob
  depth_files=("${depth_dir}/depth_"*.png)
  shopt -u nullglob
  if [ ${#depth_files[@]} -eq 0 ]; then
    missing_depth=1
    break
  fi
done

if [ "$missing_depth" -eq 1 ]; then
  (cd dpt && python get_depth_map_for_llff_dtu.py --root_path "$data_root_abs" --benchmark DTU)
else
  echo "Depth maps found for all scans. Skipping DPT."
fi

for scan_id in "${scans[@]}"; do
  echo "=== TRAIN/RENDER/EVAL: ${scan_id} ==="
  bash scripts/run_dtu.sh "${data_root}/${scan_id}" "${output_root}/${scan_id}" "${gpu_id}"
done

export OUTPUT_ROOT="$output_root"
python - <<'PY'
import json
import os
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
scans = [
    "scan30","scan34","scan41","scan45","scan82","scan103","scan38","scan21","scan40",
    "scan55","scan63","scan31","scan8","scan110","scan114",
]

results = {}
for scan in scans:
    result_path = output_root / scan / "results_eval_mask.json"
    if not result_path.exists():
        continue
    with result_path.open("r") as f:
        data = json.load(f)
    # data structure: {scene_dir: {method: metrics}}
    # flatten to {scan: {method: metrics}}
    if isinstance(data, dict):
        for _scene_dir, methods in data.items():
            results[scan] = methods
            break

out_path = output_root / "dtu_metrics.json"
with out_path.open("w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {out_path}")
PY
