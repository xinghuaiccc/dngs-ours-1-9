import os
import subprocess
import sys
import time
import argparse

# ==================== Config ====================
DATA_ROOT = "/root/DNGaussian/data/dtu"
OUTPUT_ROOT = "output-new-2-new/dtu"
ITERATION = "7000"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 使用 scripts/run-all-dtu.sh 中定义的场景列表
SCENES = [
    "scan34", "scan41", "scan45", "scan82", "scan114", "scan31", "scan8"
]

def run_all(gen_fused_pcd=False, gen_mono_depth=False):
    python_exe = sys.executable
    print("🚀 Running DTU all scenes (Ours - LLFF Aligned Strategy)")
    print(f"📂 Data root: {DATA_ROOT}")
    print(f"💾 Output root: {OUTPUT_ROOT}\n")

    # 0. 生成单目深度 (Depth-Anything-V2)
    if gen_mono_depth:
        print("🔮 Generating Monocular Depth maps for DTU (Switching to FSGS environment)...")
        # Use conda run to switch to the FSGS environment for transformers compatibility
        mono_cmd = [
            "conda", "run", "-n", "FSGS", "python",
            "dpt/get_depth_map_for_dtu_depth_anything_v2.py",
            "-r", DATA_ROOT
        ]
        try:
            subprocess.run(mono_cmd, check=True)
            print("✅ Depth generation complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Depth generation failed! Exit code: {e.returncode}")
            return

    for scene in SCENES:
        scene_path = os.path.join(DATA_ROOT, scene)
        output_path = os.path.join(OUTPUT_ROOT, scene)

        if not os.path.exists(scene_path):
            print(f"⚠️  Missing scene: {scene_path}, skipping.")
            continue

        print("==================================================")
        print(f"▶️  Scene: {scene}")
        print("==================================================")

        # 1. 运行点云融合脚本 (创新点一)
        if gen_fused_pcd:
            fused_ply_path = os.path.join(scene_path, "points3D_fused.ply")
            if os.path.exists(fused_ply_path):
                print(f"✅ Fused point cloud already exists at {fused_ply_path}. Skipping generation.")
            else:
                print("🔨 Generating fused point cloud (DTU optimized)...")
                gen_cmd = [
                    python_exe,
                    "generate_fused_pcd_dtu.py",
                    "--source_path", scene_path,
                    "--n_sparse", "3"
                ]
                try:
                    subprocess.run(gen_cmd, check=True)
                    print("✅ Generation complete.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Generation failed! Exit code: {e.returncode}")
                    continue 

        # 2. 训练参数 
        # 核心策略：对齐 LLFF 的成功经验
        # - 使用融合点云初始化 (--mvs_pcd)
        # - 禁用训练中深度监督 (--hard_depth_start 31000)
        # - 禁用不透明度重置 (--opacity_reset_interval 31000)
        # - 使用较低的学习率 (0.00016)
        # - 开启 FFT (--lambda_fft 0.05)
        
        train_cmd = [
            python_exe,
            "train_dtu.py",
            "--dataset", "DTU",
            "-s", scene_path,
            "--model_path", output_path,
            "-r", "4",
            "--eval",
            "--n_sparse", "3",
            "--iterations", ITERATION,
            "--lambda_dssim", "0.6",
            "--densify_grad_threshold", "0.001",
            "--prune_threshold", "0.01",
            "--densify_until_iter", "6000",
            "--percent_dense", "0.1",
            "--position_lr_init", "0.0016", # [回调] 恢复标准学习率，防止过拟合
            "--position_lr_final", "0.000016",
            "--position_lr_max_steps", "30000", # 恢复标准衰减
            "--position_lr_start", "500",
            "--test_iterations", "100", "1000", "2000", "3000", "4500", "6000", "7000",
            "--save_iterations", "7000",
            "--error_tolerance", "0.01",
            "--opacity_lr", "0.05",
            "--scaling_lr", "0.003",
            "--shape_pena", "0.005", 
            "--scale_pena", "0.005",
            "--opa_pena", "0.001",
            
            # --- 创新点配置 ---
    "--mvs_pcd", # 创新1：使用融合点云
    "--lambda_fft", "10", # 创新2：开启 FFT (调低权重防止初期干扰)
    # "--hard_depth_start", "31000", # 回退：DTU 需要深度监督来纠正几何
    "--soft_depth_start", "31000",
    # "--opacity_reset_interval", "31000", # 回退：DTU 需要重置来清理噪声
]

        render_cmd = [
            python_exe,
            "render.py",
            "-s", scene_path,
            "--model_path", output_path,
            "-r", "4",
            "--iteration", ITERATION,
            "--skip_train",
        ]
        
        metrics_cmd = [
            python_exe,
            "metrics_dtu.py",
            "--model_path", output_path,
        ]

        try:
            print("🏋️ Training...")
            subprocess.run(train_cmd, check=True)
            
            print(f"📦 Copying masks for {scene}...")
            # Copy masks using the updated script
            subprocess.run(["bash", "scripts/copy_mask_dtu.sh", OUTPUT_ROOT], check=False)

            print("🖌️ Rendering...")
            subprocess.run(render_cmd, check=True)
            
            print("📊 Calculating Metrics...")
            subprocess.run(metrics_cmd, check=True)
            print(f"\n✅ Scene {scene} done.\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Scene {scene} failed! Exit code: {e.returncode}")
            print("Continuing to next scene...\n")
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            sys.exit(0)

        time.sleep(3)

    print("🎉 All DTU scenes completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_fused_pcd", action="store_true", default=True, help="Generate fused point cloud")
    parser.add_argument("--gen_mono_depth", action="store_true", help="Generate monocular depth")
    args = parser.parse_args()
    
    run_all(gen_fused_pcd=args.gen_fused_pcd, gen_mono_depth=args.gen_mono_depth)
