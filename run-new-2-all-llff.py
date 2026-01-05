import os
import subprocess
import sys
import time

# ==================== Config ====================
DATA_ROOT = "/root/all-data/nerf_llff_data"
OUTPUT_ROOT = "output/nerf_llff_data"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SCENES = [
    "fern",
    "flower",
    "fortress",
    "horns",
    "leaves",
    "orchids",
    "room",
    "trex",
]

COMMON_TRAIN_ARGS = [
    "-r",
    "8",
    "--eval",
    "--n_sparse",
    "3",
    "--iterations",
    "7000",
    "--position_lr_init",
    "0.00016",
    "--position_lr_final",
    "0.0000016",
    "--position_lr_max_steps",
    "7000",
    "--densify_until_iter",
    "4000",
    "--densify_grad_threshold",
    "0.0002",
    "--lambda_dssim",
    "0.2",
    "--hard_depth_start",
    "8000",
    "--soft_depth_start",
    "8000",
    "--near",
    "0.1",
    "--prune_threshold",
    "0.01",
    "--percent_dense",
    "0.01",
    "--opacity_reset_interval",
    "30000",
]

# ===============================================


def run_all():
    python_exe = sys.executable
    print("🚀 Running LLFF all scenes with generate + train")
    print(f"📂 Data root: {DATA_ROOT}")
    print(f"💾 Output root: {OUTPUT_ROOT}\n")

    for scene in SCENES:
        scene_path = os.path.join(DATA_ROOT, scene)
        output_path = os.path.join(OUTPUT_ROOT, scene)

        if not os.path.exists(scene_path):
            print(f"⚠️  Missing scene: {scene_path}, skipping.")
            continue

        print("==================================================")
        print(f"▶️  Scene: {scene}")
        print("==================================================")

        gen_cmd = [
            python_exe,
            "generate_fused_pcd.py",
            "-s",
            scene_path,
            "--n_sparse",
            "3",
        ]

        train_cmd = [
            python_exe,
            "train_llff_new-2.py",
            "-s",
            scene_path,
            "--model_path",
            output_path,
        ] + COMMON_TRAIN_ARGS

        try:
            subprocess.run(gen_cmd, check=True)
            subprocess.run(train_cmd, check=True)
            print(f"\n✅ Scene {scene} done.\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Scene {scene} failed! Exit code: {e.returncode}")
            print("Continuing to next scene...\n")
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            sys.exit(0)

        time.sleep(3)

    print("🎉 All scenes completed.")


if __name__ == "__main__":
    run_all()
