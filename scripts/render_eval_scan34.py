import os
import subprocess
import sys
from pathlib import Path
import shutil

DATA_ROOT = "/root/DNGaussian/data/dtu"
SCENE = "scan34"
OUTPUT_ROOT = "output-ablation/dtu/scan34"
MASK_SOURCE = "/root/DNGaussian/data/dtu/submission_data/idrmasks"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_masks(model_path):
    model_path = Path(model_path)
    mask_dst = model_path / "mask"
    mask_dst.mkdir(parents=True, exist_ok=True)

    scan_mask_root = Path(MASK_SOURCE) / SCENE
    if (scan_mask_root / "mask").is_dir():
        src_dir = scan_mask_root / "mask"
    else:
        src_dir = scan_mask_root

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Mask source not found: {src_dir}")

    files = sorted(src_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No mask pngs in {src_dir}")

    for idx, src in enumerate(files):
        dst = mask_dst / f"{idx:05d}.png"
        if not dst.exists():
            shutil.copyfile(src, dst)


def main():
    scene_path = os.path.join(DATA_ROOT, SCENE)
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Missing scene: {scene_path}")

    if not os.path.isdir(OUTPUT_ROOT):
        raise FileNotFoundError(f"Missing output root: {OUTPUT_ROOT}")

    runs = sorted(
        d for d in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, d)) and d != "mask"
    )
    if not runs:
        print("No runs found under", OUTPUT_ROOT)
        return

    for run in runs:
        model_path = os.path.join(OUTPUT_ROOT, run)
        print("=" * 60)
        print("Rendering + metrics for:", model_path)
        print("=" * 60)

        ensure_masks(model_path)

        run_cmd([
            sys.executable,
            "render.py",
            "-s", scene_path,
            "--model_path", model_path,
            "-r", "4",
            "--iteration", "3000",
            "--skip_train",
        ])

        run_cmd([
            sys.executable,
            "metrics_dtu.py",
            "--model_path", model_path,
        ])


if __name__ == "__main__":
    main()
