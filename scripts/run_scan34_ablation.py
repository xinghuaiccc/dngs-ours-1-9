import os
import subprocess
import sys
import time

# Single-scene ablation runner for DTU (scan34) at 3000 iterations

DATA_ROOT = "/root/DNGaussian/data/dtu"
SCENE = "scan34"
OUTPUT_ROOT = "output-ablation/dtu/scan34"

ITERATIONS = "3000"
CUDA_DEVICE = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE


ABLATIONS = [
    {
        "name": "baseline_fused_defaults",
        "use_fused": True,
        "fused_args": {
            "n_sparse": "3",
            "photo_thresh": "0.1",
            "dist_thresh": "0.1",
            "max_points_per_view": "3000",
        },
        "train_args": {
            "lambda_fft": "0",
        },
    },
    {
        "name": "fused_bg_fft_0p15",
        "use_fused": True,
        "fused_args": {
            "n_sparse": "3",
            "photo_thresh": "0.15",
            "dist_thresh": "0.08",
            "max_points_per_view": "10000",
        },
        "train_args": {
            "lambda_fft": "0.02",
        },
    },
    {
        "name": "fused_bg_nofft_0p15",
        "use_fused": True,
        "fused_args": {
            "n_sparse": "3",
            "photo_thresh": "0.15",
            "dist_thresh": "0.08",
            "max_points_per_view": "10000",
        },
        "train_args": {
            "lambda_fft": "0",
        },
    },
    {
        "name": "fused_relaxed_0p15",
        "use_fused": True,
        "fused_args": {
            "n_sparse": "3",
            "photo_thresh": "0.15",
            "dist_thresh": "0.08",
            "max_points_per_view": "10000",
        },
        "train_args": {
            "lambda_fft": "0",
        },
    },
    {
        "name": "fused_relaxed_0p20",
        "use_fused": True,
        "fused_args": {
            "n_sparse": "3",
            "photo_thresh": "0.2",
            "dist_thresh": "0.07",
            "max_points_per_view": "15000",
        },
        "train_args": {
            "lambda_fft": "0",
        },
    },
    {
        "name": "no_fused_colmap_init",
        "use_fused": False,
        "fused_args": {},
        "train_args": {
            "lambda_fft": "0",
        },
    },
    {
        "name": "origin_like_rand_pcd",
        "use_fused": False,
        "fused_args": {},
        "train_args": {
            "lambda_fft": "0",
            "rand_pcd": True,
            "densify_until_iter": "6000",
        },
    },
    {
        "name": "origin_like_fused_nofft",
        "use_fused": True,
        "fused_args": {
            "n_sparse": "3",
            "photo_thresh": "0.15",
            "dist_thresh": "0.08",
            "max_points_per_view": "10000",
        },
        "train_args": {
            "lambda_fft": "0",
            "densify_until_iter": "6000",
        },
    },
    {
        "name": "origin_like_fused_relaxed_nofft",
        "use_fused": True,
        "fused_args": {
            "n_sparse": "3",
            "photo_thresh": "0.2",
            "dist_thresh": "0.07",
            "max_points_per_view": "15000",
        },
        "train_args": {
            "lambda_fft": "0",
            "densify_until_iter": "6000",
        },
    },
]


def run_cmd(cmd, cwd=None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def build_train_cmd(scene_path, output_path, use_fused, extra_train_args):
    cmd = [
        sys.executable,
        "train_dtu.py",
        "--dataset", "DTU",
        "-s", scene_path,
        "--model_path", output_path,
        "-r", "4",
        "--eval",
        "--n_sparse", "3",
        "--iterations", ITERATIONS,
        "--lambda_dssim", "0.6",
        "--densify_grad_threshold", "0.001",
        "--prune_threshold", "0.01",
        "--densify_until_iter", "3000",
        "--percent_dense", "0.1",
        "--position_lr_init", "0.0016",
        "--position_lr_final", "0.000016",
        "--position_lr_max_steps", "15000",
        "--position_lr_start", "500",
        "--test_iterations", "100", "1000", "2000", "3000",
        "--save_iterations", "3000",
        "--error_tolerance", "0.01",
        "--opacity_lr", "0.05",
        "--scaling_lr", "0.003",
        "--shape_pena", "0.005",
        "--scale_pena", "0.005",
        "--opa_pena", "0.001",
    ]
    if use_fused:
        cmd.append("--mvs_pcd")
    if extra_train_args:
        for k, v in extra_train_args.items():
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k}")
            else:
                cmd.extend([f"--{k}", v])
    return cmd


def build_fused_cmd(scene_path, fused_args):
    cmd = [
        sys.executable,
        "generate_fused_pcd_dtu.py",
        "--source_path", scene_path,
    ]
    for k, v in fused_args.items():
        cmd.extend([f"--{k}", v])
    return cmd


def main():
    scene_path = os.path.join(DATA_ROOT, SCENE)
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Missing scene: {scene_path}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for cfg in ABLATIONS:
        name = cfg["name"]
        output_path = os.path.join(OUTPUT_ROOT, name)
        os.makedirs(output_path, exist_ok=True)

        print("=" * 60)
        print(f"Running ablation: {name}")
        print("=" * 60)

        fused_ply_path = os.path.join(scene_path, "points3D_fused.ply")
        if os.path.exists(fused_ply_path):
            os.remove(fused_ply_path)

        if cfg["use_fused"]:
            fused_cmd = build_fused_cmd(scene_path, cfg["fused_args"])
            run_cmd(fused_cmd)

        train_cmd = build_train_cmd(scene_path, output_path, cfg["use_fused"], cfg["train_args"])
        run_cmd(train_cmd)

        time.sleep(2)

    print("All ablations finished.")


if __name__ == "__main__":
    main()
