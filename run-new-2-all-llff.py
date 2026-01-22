import os
import subprocess
import sys
import time
import argparse

# ==================== Config ====================
DATA_ROOT = "/root/all-data/nerf_llff_data"
OUTPUT_ROOT = "output-new-2-new/nerf_llff_data"
ITERATION = "7000"

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

# 默认训练参数
COMMON_TRAIN_ARGS = [
    "-r", "8",
    "--eval",
    "--n_sparse", "3",
    "--iterations", ITERATION,
    "--position_lr_init", "0.0009",
    "--position_lr_final", "0.0000016",
    "--position_lr_max_steps", "15000",
    "--densify_until_iter", "4000",
    "--densify_grad_threshold", "0.0001",  
    "--lambda_dssim", "0.2",
    "--hard_depth_start", "31000", # 保持禁用状态
    "--soft_depth_start", "31000", # 保持禁用状态
    "--near", "0",
    "--percent_dense", "0.01",
    "--opacity_reset_interval", "31000",
    "--shape_pena", "0", 
    "--scale_pena", "0",
]

def run_all(gen_fused_pcd=False):
    python_exe = sys.executable
    print("🚀 Running LLFF all scenes with generate + train")
    print(f"📂 Data root: {DATA_ROOT}")
    print(f"💾 Output root: {OUTPUT_ROOT}\n")
    if gen_fused_pcd:
        print("🔧 Mode: Generating Fused Point Cloud (Active)")
    else:
        print("🔧 Mode: Using Pre-computed MVS Point Cloud (Default)")

    for scene in SCENES:
        scene_path = os.path.join(DATA_ROOT, scene)
        output_path = os.path.join(OUTPUT_ROOT, scene)

        if not os.path.exists(scene_path):
            print(f"⚠️  Missing scene: {scene_path}, skipping.")
            continue

        print("==================================================")
        print(f"▶️  Scene: {scene}")
        print("==================================================")

        # 1. 如果启用了生成融合点云，先运行生成脚本
        if gen_fused_pcd:
            print("🔨 Generating fused point cloud...")
            gen_cmd = [
                python_exe,
                "generate_fused_pcd.py",
                "--source_path", scene_path,
                "--n_sparse", "3"
            ]
            try:
                subprocess.run(gen_cmd, check=True)
                print("✅ Generation complete.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Generation failed! Exit code: {e.returncode}")
                continue # Skip training if generation fails

        # 2. 构建训练命令
        # 根据模式选择初始化参数：
        # - gen_fused_pcd=True: 不传任何pcd参数，dataset_readers会自动加载 points3D_fused.ply
        # - gen_fused_pcd=False: 传 --mvs_pcd，加载 3_views/dense/fused.ply
        
        current_train_args = COMMON_TRAIN_ARGS.copy()
        if not gen_fused_pcd:
            current_train_args.append("--mvs_pcd")
        
        train_cmd = [
            python_exe,
            "train_llff_new-2.py",
            "-s", scene_path,
            "--model_path", output_path,
        ] + current_train_args

        render_cmd = [
            python_exe,
            "render.py",
            "-s", scene_path,
            "--model_path", output_path,
            "-r", "8",
            "--iteration", ITERATION,
            "--near", "0",
            "--skip_train",
        ]
        
        metrics_cmd = [
            python_exe,
            "metrics.py",
            "-m", output_path,
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
    parser = argparse.ArgumentParser()
    # 添加命令行开关，默认为 False (保持原有行为)
    parser.add_argument("--gen_fused_pcd", action="store_true", help="Generate fused point cloud from monocular depth before training")
    args = parser.parse_args()
    
    run_all(gen_fused_pcd=args.gen_fused_pcd)
