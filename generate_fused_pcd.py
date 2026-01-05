#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from scene.dataset_readers import readColmapSceneInfo


parser = argparse.ArgumentParser()
parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to LLFF dataset root")
parser.add_argument("--images", "-i", default="images", type=str)
parser.add_argument(
    "--dist_thresh",
    default=0.1,
    type=float,
    help="Distance threshold to keep point (bigger = filter more)",
)
parser.add_argument("--stride", default=4, type=int, help="Downsample stride (4 means 1/16 pixels)")
parser.add_argument("--max_points_per_view", default=100000, type=int, help="Max points to add per view")
parser.add_argument("--n_sparse", default=0, type=int, help="Use the same N sparse views as training")
parser.add_argument("--photo_thresh", default=0.1, type=float, help="Photometric error threshold")
args = parser.parse_args()


def save_ply(path, xyz, rgb):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, np.zeros_like(xyz), rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
    print(f"Saved PLY to {path} with {xyz.shape[0]} points")


def lstsq_align(mono_depth, sparse_depth, min_samples=50):
    valid = (sparse_depth > 0) & np.isfinite(sparse_depth)
    if valid.sum() < min_samples:
        return mono_depth, 1.0, 0.0, False

    mono_vals = mono_depth[valid].reshape(-1).astype(np.float32)
    sparse_vals = sparse_depth[valid].reshape(-1).astype(np.float32)

    A = np.stack([mono_vals, np.ones_like(mono_vals)], axis=1)
    scale, shift = np.linalg.lstsq(A, sparse_vals, rcond=None)[0]
    aligned = mono_depth * scale + shift
    return aligned, scale, shift, True


def project_sparse_depth(colmap_xyz, w2c, focal_x, focal_y, cx, cy, width, height):
    pts = torch.cat([colmap_xyz, torch.ones_like(colmap_xyz[:, :1])], dim=1)
    pts_cam = (w2c @ pts.T).T[:, :3]
    valid_z = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid_z]

    u = (pts_cam[:, 0] * focal_x / pts_cam[:, 2]) + cx
    v = (pts_cam[:, 1] * focal_y / pts_cam[:, 2]) + cy

    u = u.long().cpu().numpy()
    v = v.long().cpu().numpy()
    z = pts_cam[:, 2].cpu().numpy()

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]

    depth = np.full((height, width), np.inf, dtype=np.float32)
    flat = depth.reshape(-1)
    idx = v * width + u
    np.minimum.at(flat, idx, z)
    depth = flat.reshape(height, width)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def backproject_points(aligned_depth, focal_x, focal_y, cx, cy):
    height, width = aligned_depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xx = (xx - cx) / focal_x
    yy = (yy - cy) / focal_y

    valid_mask = (aligned_depth > 0.1) & (aligned_depth < 100.0) & np.isfinite(aligned_depth)

    z = aligned_depth[valid_mask]
    x = xx[valid_mask] * z
    y = yy[valid_mask] * z

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
    return pts_cam, valid_mask


def project_points(pts_world_h, w2c, focal_x, focal_y, cx, cy, width, height):
    pts_cam = (w2c @ pts_world_h.T).T
    z = pts_cam[:, 2]
    valid_z = z > 0.01
    u = (pts_cam[:, 0] * focal_x / z) + cx
    v = (pts_cam[:, 1] * focal_y / z) + cy
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid = valid_z & valid_uv
    return u, v, valid


def main():
    print(f"Loading scene from {args.source_path}...")

    use_eval_split = args.n_sparse > 0
    scene_info = readColmapSceneInfo(
        args.source_path,
        args.images,
        dataset="LLFF",
        eval=use_eval_split,
        rand_pcd=False,
        mvs_pcd=False,
        N_sparse=args.n_sparse,
    )
    cameras = scene_info.train_cameras

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    colmap_xyz = torch.tensor(scene_info.point_cloud.points).float().to(device)
    colmap_rgb = torch.tensor(scene_info.point_cloud.colors).float().to(device)

    print(f"Original COLMAP points: {colmap_xyz.shape[0]}")
    print("Building KDTree for COLMAP points...")
    kd_tree = cKDTree(colmap_xyz.cpu().numpy())

    depth_folder = os.path.join(args.source_path, "depths_npy")
    if not os.path.exists(depth_folder):
        raise FileNotFoundError(f"Depth folder not found at {depth_folder}")

    new_points_xyz = []
    new_points_rgb = []

    for idx, cam in enumerate(cameras):
        print(f"\nProcessing View {idx}: {cam.image_name} ...")

        npy_path = os.path.join(depth_folder, cam.image_name + ".npy")
        if not os.path.exists(npy_path):
            print(f"Skipping {cam.image_name}, no depth npy found.")
            continue

        image_ref = np.array(Image.open(cam.image_path).convert("RGB"))
        mono_depth = np.load(npy_path)

        stride = args.stride
        mono_depth = mono_depth[::stride, ::stride]
        image_ref = image_ref[::stride, ::stride]
        height, width = mono_depth.shape

        fovx = cam.FovX
        fovy = cam.FovY
        focal_y = height / (2.0 * np.tan(fovy / 2.0))
        focal_x = width / (2.0 * np.tan(fovx / 2.0))
        cx = width / 2.0
        cy = height / 2.0

        R = torch.tensor(cam.R).float().to(device).transpose(0, 1)
        w2c = torch.eye(4, device=device)
        w2c[:3, :3] = R
        w2c[:3, 3] = torch.tensor(cam.T).to(device)

        sparse_depth = project_sparse_depth(
            colmap_xyz, w2c, focal_x, focal_y, cx, cy, width, height
        )
        aligned_depth, scale, shift, ok = lstsq_align(mono_depth, sparse_depth)
        if not ok:
            print("  -> Not enough COLMAP points for alignment, skipping.")
            continue

        if scale < 0:
            print("  [WARNING] Negative scale detected! Trying inverse depth alignment...")
            valid = (sparse_depth > 0) & np.isfinite(sparse_depth)
            mono_vals = mono_depth[valid].reshape(-1).astype(np.float32)
            sparse_vals = sparse_depth[valid].reshape(-1).astype(np.float32)
            inv_mono_vals = 1.0 / (mono_vals + 1e-6)
            A_inv = np.stack([inv_mono_vals, np.ones_like(inv_mono_vals)], axis=1)
            scale_inv, shift_inv = np.linalg.lstsq(A_inv, sparse_vals, rcond=None)[0]
            print(f"  -> Inverse Alignment: scale={scale_inv:.4f}, shift={shift_inv:.4f}")
            aligned_depth = scale_inv * (1.0 / (mono_depth + 1e-6)) + shift_inv
            if scale_inv < 0:
                print("  [ERROR] Inverse alignment negative. Skipping view.")
                continue

        print(f"  -> Alignment: scale={scale:.4f}, shift={shift:.4f}")

        pts_cam, valid_mask = backproject_points(aligned_depth, focal_x, focal_y, cx, cy)
        c2w = torch.inverse(w2c).cpu().numpy()
        pts_world = (c2w @ pts_cam)[:3, :].T

        colors_ref = image_ref[valid_mask]

        if len(cameras) < 2:
            print("  -> Not enough views for photometric check, skipping.")
            continue

        src_idx = idx + 1 if idx + 1 < len(cameras) else idx - 1
        src_cam = cameras[src_idx]

        image_src = np.array(Image.open(src_cam.image_path).convert("RGB"))
        image_src = image_src[::stride, ::stride]
        src_h, src_w = image_src.shape[:2]

        src_fovx = src_cam.FovX
        src_fovy = src_cam.FoVY if hasattr(src_cam, "FoVY") else src_cam.FovY
        src_focal_y = src_h / (2.0 * np.tan(src_fovy / 2.0))
        src_focal_x = src_w / (2.0 * np.tan(src_fovx / 2.0))
        src_cx = src_w / 2.0
        src_cy = src_h / 2.0

        src_R = torch.tensor(src_cam.R).float().to(device).transpose(0, 1)
        src_w2c = torch.eye(4, device=device)
        src_w2c[:3, :3] = src_R
        src_w2c[:3, 3] = torch.tensor(src_cam.T).to(device)
        src_w2c = src_w2c.cpu().numpy()

        pts_world_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
        u_src, v_src, valid_src = project_points(
            pts_world_h, src_w2c, src_focal_x, src_focal_y, src_cx, src_cy, src_w, src_h
        )

        if valid_src.sum() == 0:
            print("  -> No valid projections into src view.")
            continue

        u_valid = u_src[valid_src].astype(np.int32)
        v_valid = v_src[valid_src].astype(np.int32)
        colors_src = image_src[v_valid, u_valid]
        colors_ref_valid = colors_ref[valid_src]

        diff = np.abs(colors_ref_valid.astype(np.float32) / 255.0 - colors_src.astype(np.float32) / 255.0)
        diff = diff.mean(axis=1)
        photo_mask = diff < args.photo_thresh

        pts_photo = pts_world[valid_src][photo_mask]
        colors_photo = colors_ref_valid[photo_mask]

        print(f"  -> Photometric pass: {pts_photo.shape[0]} / {pts_world.shape[0]}")

        if pts_photo.shape[0] == 0:
            continue

        dists, _ = kd_tree.query(pts_photo, k=1)
        keep_mask = dists > args.dist_thresh

        pts_filtered = pts_photo[keep_mask]
        colors_filtered = colors_photo[keep_mask]

        print(f"  -> After KDTree filter: {pts_filtered.shape[0]}")

        if pts_filtered.shape[0] > args.max_points_per_view:
            indices = np.random.choice(pts_filtered.shape[0], args.max_points_per_view, replace=False)
            pts_filtered = pts_filtered[indices]
            colors_filtered = colors_filtered[indices]
            print(f"  -> Capped to max {args.max_points_per_view}")

        new_points_xyz.append(pts_filtered)
        new_points_rgb.append(colors_filtered)

    if len(new_points_xyz) > 0:
        all_new_xyz = np.concatenate(new_points_xyz, axis=0)
        all_new_rgb = np.concatenate(new_points_rgb, axis=0)

        print(f"\nTotal new points added: {all_new_xyz.shape[0]}")

        final_xyz = np.concatenate([colmap_xyz.cpu().numpy(), all_new_xyz], axis=0)
        final_rgb = np.concatenate([colmap_rgb.cpu().numpy(), all_new_rgb], axis=0)

        save_ply(os.path.join(args.source_path, "points3D_fused.ply"), final_xyz, final_rgb)
    else:
        print("No new points generated.")


if __name__ == "__main__":
    main()
