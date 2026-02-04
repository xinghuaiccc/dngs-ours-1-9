#!/usr/bin/env bash

# Robust mask copying script for DTU
# Usage: ./copy_mask_dtu.sh <output_base_dir> [mask_source_dir]

base="${1:-output/dtu/}"
mask_path="${2:-data/dtu/submission_data/idrmasks}"

echo "📦 Copying DTU masks to: $base"
echo "📂 Source masks: $mask_path"

for scan_id in scan30 scan34 scan41 scan45 scan82 scan103 scan38 scan21 scan40 scan55 scan63 scan31 scan8 scan110 scan114
do  
    target_dir="$base/$scan_id/mask"
    if [ -d "$base/$scan_id" ]; then
        mkdir -p "$target_dir"
        id=0
        
        # Check if sub-mask directory exists
        if [ -d "${mask_path}/$scan_id/mask" ]; then
            source_dir="${mask_path}/$scan_id/mask"
        elif [ -d "${mask_path}/$scan_id" ]; then
            source_dir="${mask_path}/$scan_id"
        else
            echo "⚠️  Warning: No masks found for $scan_id in $mask_path"
            continue
        fi

        echo "   -> Copying masks for $scan_id..."
        # Copy and rename files sequentially to 00000.png, 00001.png, etc.
        # We sort to maintain order
        for file in $(ls "$source_dir"/*.png | sort)
        do
            file_name=$(printf "%05d" $id).png
            cp "$file" "$target_dir/$file_name"
            ((id = id + 1))
        done
    fi
done
