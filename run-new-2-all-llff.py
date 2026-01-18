import os
import subprocess
import sys
import time

# ==================== Config ====================
DATA_ROOT = "/root/all-data/nerf_llff_data"
OUTPUT_ROOT = "output-new-2-new/nerf_llff_data"
ITERATION = "15000"

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
    ITERATION,
    "--position_lr_init",
    # "0.00016",
    "0.0009",
    "--position_lr_final",
    "0.0000016",
    "--position_lr_max_steps",
    # "7000",
    "15000",
    "--densify_until_iter",
    "4000",
    "--densify_grad_threshold",
    "0.0002",
    "--lambda_dssim",
    "0.2",
    "--hard_depth_start",
    "31000",
    "--soft_depth_start",
    "31000",
    "--near",
    "10",
    "--prune_threshold",
    "0.01",
    "--percent_dense",
    "0.01",
    "--opacity_reset_interval",
    "31000",
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

        train_cmd = [
            python_exe,
            "train_llff_new-2.py",
            "-s",
            scene_path,
            "--model_path",
            output_path,
        ] + COMMON_TRAIN_ARGS
        render_cmd = [
            python_exe,
            "render.py",
            "-s",
            scene_path,
            "--model_path",
            output_path,
            "-r",
            "8",
            "--iteration",
            ITERATION,
            "--near",
            "10",
            "--skip_train",
        ]
        metrics_cmd = [
            python_exe,
            "metrics.py",
            "-m",
            output_path,
        ]

        try:
            subprocess.run(train_cmd, check=True)
            subprocess.run(render_cmd, check=True)
            subprocess.run(metrics_cmd, check=True)
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
