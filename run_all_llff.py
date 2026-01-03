import os
import subprocess
import time
import sys

# ==================== 配置区域 ====================

# 1. 数据集根目录 (请确认路径是否正确)
DATA_ROOT = "/root/all-data/nerf_llff_data"

# 2. 输出保存路径
OUTPUT_ROOT = "output/nerf_llff_data"

# 3. 指定使用的 GPU 编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 4. LLFF 8个场景列表
# 注意: 请检查你的文件夹里是 "trex" 还是 "t-rex"
SCENES = [
    "fern",
    "flower",
    "fortress",
    "horns",
    "leaves",
    "orchids",
    "room",
    "trex"
]

# 5. 训练参数配置 (根据你提供的命令)
# -r 8 是防止显存爆炸的关键
COMMON_ARGS = [
    "-r", "8",
    "--eval",
    "--n_sparse", "3",
    "--rand_pcd",
    "--iterations", "30000",
    "--lambda_dssim", "0.2",
    "--densify_grad_threshold", "0.0013",
    "--prune_threshold", "0.01",
    "--densify_until_iter", "15000",
    "--percent_dense", "0.01",
    "--position_lr_init", "0.016",
    "--position_lr_final", "0.00016",
    "--position_lr_max_steps", "15000",
    "--position_lr_start", "500",
    "--split_opacity_thresh", "0.1",
    "--error_tolerance", "0.00025",
    "--scaling_lr", "0.003",
    "--shape_pena", "0.002",
    "--opa_pena", "0.001",
    "--near", "10"
]


# =================================================

def run_training():
    # 获取当前 python 解释器路径 (确保使用 conda 环境)
    python_exe = sys.executable

    print(f"🚀 开始批量训练 LLFF 数据集 (稀疏视图模式)...")
    print(f"📂 数据路径: {DATA_ROOT}")
    print(f"💾 输出路径: {OUTPUT_ROOT}\n")

    for scene in SCENES:
        scene_path = os.path.join(DATA_ROOT, scene)
        output_path = os.path.join(OUTPUT_ROOT, scene)

        # 检查场景目录是否存在
        if not os.path.exists(scene_path):
            print(f"⚠️  警告: 找不到场景目录 {scene_path}，已跳过。")
            continue

        print(f"==================================================")
        print(f"▶️  正在训练场景: {scene}")
        print(f"==================================================")

        # 构建完整的命令
        # python train_llff.py -s [source] --model_path [output] [args...]
        cmd = [
                  python_exe, "train_llff.py",
                  "-s", scene_path,
                  "--model_path", output_path
              ] + COMMON_ARGS

        try:
            # 执行命令，check=True 表示如果报错则抛出异常
            # 这里的 subprocess 会启动一个新的进程，结束后完全释放显存
            subprocess.run(cmd, check=True)
            print(f"\n✅ 场景 {scene} 训练完成。\n")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ 场景 {scene} 训练失败！错误代码: {e.returncode}")
            print("继续尝试下一个场景...\n")

        except KeyboardInterrupt:
            print("\n🛑 用户手动停止脚本。")
            sys.exit(0)

        # 休息 3 秒，让 GPU 喘口气（降温/清理显存残余）
        time.sleep(3)

    print("🎉 所有任务执行完毕！")


if __name__ == "__main__":
    run_training()