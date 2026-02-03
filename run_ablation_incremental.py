import os
import subprocess
import sys
import time
import json
import re

# ==================== Config ====================
DATA_ROOT = "/root/all-data/nerf_llff_data"
SCENE = "fern"
SCENE_PATH = os.path.join(DATA_ROOT, SCENE)
OUTPUT_ROOT = "output-ablation-incremental"
ITERATION = "15000"
RESULTS_JSON_PATH = os.path.join(OUTPUT_ROOT, "ablation_results.json")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 基础参数 (Baseline)
BASE_ARGS = [
    "-r", "8",
    "--eval",
    "--n_sparse", "3",
    "--iterations", ITERATION,
    "--position_lr_final", "0.0000016",
    
    # 彻底禁用 DNGaussian 原版机制
    "--hard_depth_start", "31000",
    "--soft_depth_start", "31000",
    "--opacity_reset_interval", "31000",
    
    "--lambda_dssim", "0.2",
    "--percent_dense", "0.01",
    "--densify_until_iter", "15000",
]

EXPERIMENTS = [
    # 1. Baseline: 随机初始化, 无 FFT, 无正则
    # {
    #     "name": "A_Baseline",
    #     "desc": "Baseline (Random Init, No FFT, No Reg)",
    #     "args": [
    #         "--position_lr_init", "0.00016",
    #         "--lambda_fft", "0",
    #         "--shape_pena", "0", "--scale_pena", "0",
    #         "--near", "0",
    #     ]
    # },

    # 2. +Prior: 引入 V2 先验初始化
    {
        "name": "B_Plus_V2_Prior",
        "desc": "Baseline + V2 Prior Init (No FFT)",
        "args": [
            "--mvs_pcd",
            "--position_lr_init", "0.0009",
            "--lambda_fft", "0",
            "--shape_pena", "0", "--scale_pena", "0",
            "--near", "0",
        ]
    },
    
    # 3. +FFT: 引入频域损失 (Full Method)
    {
        "name": "C_Plus_FFT_Full",
        "desc": "V2 Prior Init + FFT (Full Method)",
        "args": [
            "--mvs_pcd",
            "--position_lr_init", "0.0009",
            "--lambda_fft", "0.05",
            "--shape_pena", "0", "--scale_pena", "0",
            "--near", "0",
        ]
    },
]

def parse_metrics_output(output_str):
    """从 metrics.py 的输出中提取 PSNR, SSIM, LPIPS"""
    metrics = {}
    # 假设输出格式为 "Scene: fern, PSNR: 25.123, SSIM: 0.854, LPIPS: 0.123" 类似格式
    # 或者 metrics.py 会打印特定的行。我们需要根据实际情况调整正则表达式。
    # 这里是一个通用的尝试匹配：
    psnr_match = re.search(r'PSNR\s*:\s*([0-9.]+)', output_str)
    ssim_match = re.search(r'SSIM\s*:\s*([0-9.]+)', output_str)
    lpips_match = re.search(r'LPIPS\s*:\s*([0-9.]+)', output_str)
    
    if psnr_match: metrics['PSNR'] = float(psnr_match.group(1))
    if ssim_match: metrics['SSIM'] = float(ssim_match.group(1))
    if lpips_match: metrics['LPIPS'] = float(lpips_match.group(1))
    
    return metrics

def run_ablation():
    python_exe = sys.executable
    print(f"🚀 Running Full Ablation Pipeline (Baseline -> Prior -> FFT) on scene: {SCENE}")
    print(f"📂 Data: {SCENE_PATH}")
    print(f"💾 Output Root: {OUTPUT_ROOT}\n")
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = {}

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        exp_desc = exp["desc"]
        exp_args = exp["args"]
        
        output_path = os.path.join(OUTPUT_ROOT, exp_name, SCENE)
        
        print("==================================================")
        print(f"🧪 Experiment: {exp_name}")
        print(f"📝 Description: {exp_desc}")
        print("==================================================")

        train_cmd = [
            python_exe,
            "train_llff_new-2.py",
            "-s", SCENE_PATH,
            "--model_path", output_path,
        ] + BASE_ARGS + exp_args
        
        render_cmd = [
            python_exe,
            "render.py",
            "-s", SCENE_PATH,
            "--model_path", output_path,
            "-r", "8",
            "--iteration", ITERATION,
            "--near", "10",
            "--skip_train",
        ]
        
        metrics_cmd = [
            python_exe,
            "metrics.py",
            "-m", output_path,
        ]

        try:
            print(f"Running Training...")
            subprocess.run(train_cmd, check=True)
            
            print(f"Running Rendering...")
            subprocess.run(render_cmd, check=True)
            
            print(f"Running Metrics...")
            # 捕获 stdout 以便解析指标
            result = subprocess.run(metrics_cmd, check=True, capture_output=True, text=True)
            print(result.stdout) # 打印出来以便调试
            
            # 解析指标
            metrics = parse_metrics_output(result.stdout)
            
            # 记录结果
            all_results[exp_name] = {
                "description": exp_desc,
                "metrics": metrics
            }
            
            print(f"\n✅ Experiment {exp_name} completed. Metrics: {metrics}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Experiment {exp_name} failed! Exit code: {e.returncode}")
            if e.stdout: print(e.stdout)
            if e.stderr: print(e.stderr)
            
        time.sleep(2)

    # 保存所有结果到 JSON
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"🎉 All ablation experiments completed. Results saved to {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    run_ablation()
