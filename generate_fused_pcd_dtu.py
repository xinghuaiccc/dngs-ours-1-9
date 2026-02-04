#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

# We assume standard DTU loading logic
from scene.dataset_readers import readColmapSceneInfo

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to DTU scan root")
parser.add_argument("--stride", default=4, type=int, help="Downsample stride")
parser.add_argument("--max_points_per_view", default=300, type=int, help="Max points to add per view")
parser.add_argument("--n_sparse", default=3, type=int, help="Number of sparse views used in training")
parser.add_argument("--photo_thresh", default=0.1, type=float, help="Photometric error threshold")
parser.add_argument("--dist_thresh", default=0.1, type=float, help="Distance threshold for KDTree fusion")
args = parser.parse_args()

def save_ply(path, xyz, rgb):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
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
    return mono_depth * scale + shift, scale, shift, True

def project_sparse_depth(colmap_xyz, w2c, focal_x, focal_y, cx, cy, width, height):
    pts = torch.cat([colmap_xyz, torch.ones_like(colmap_xyz[:, :1])], dim=1)
    pts_cam = (w2c @ pts.T).T[:, :3]
    valid_z = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid_z]
    u = (pts_cam[:, 0] * focal_x / pts_cam[:, 2]) + cx
    v = (pts_cam[:, 1] * focal_y / pts_cam[:, 2]) + cy
    u, v = u.long(), v.long()
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[in_bounds].cpu().numpy(), v[in_bounds].cpu().numpy(), pts_cam[in_bounds, 2].cpu().numpy()
    depth = np.full((height, width), np.inf, dtype=np.float32)
    flat = depth.reshape(-1)
    np.minimum.at(flat, v * width + u, z)
    depth = flat.reshape(height, width)
    depth[~np.isfinite(depth)] = 0.0
    return depth

def backproject_points(depth, focal_x, focal_y, cx, cy):
    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xx = (xx - cx) / focal_x
    yy = (yy - cy) / focal_y
    mask = (depth > 0.1) & (depth < 100.0) & np.isfinite(depth)
    z = depth[mask]
    x, y = xx[mask] * z, yy[mask] * z
    return np.stack([x, y, z, np.ones_like(z)], axis=0), mask

def build_bg_mask(img, source_path):
    # Match DTU background heuristic from training: scan110 uses lower threshold.
    thresh = 15.0 / 255.0 if "scan110" in source_path else 30.0 / 255.0
    return (img.max(axis=2) < thresh)

def project_points(pts_world_h, w2c, focal_x, focal_y, cx, cy, width, height):
    pts_cam = (w2c @ pts_world_h.T).T
    z = pts_cam[:, 2]
    u = (pts_cam[:, 0] * focal_x / z) + cx
    v = (pts_cam[:, 1] * focal_y / z) + cy
    valid = (z > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid

def main():
    print(f"Generating DTU fused point cloud for {args.source_path}...")
    scene_info = readColmapSceneInfo(args.source_path, "images", dataset="DTU", eval=True, rand_pcd=False, mvs_pcd=False, N_sparse=args.n_sparse)
    cameras = scene_info.train_cameras
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colmap_xyz = torch.tensor(scene_info.point_cloud.points).float().to(device)
    colmap_rgb = torch.tensor(scene_info.point_cloud.colors).float().to(device)
    kd_tree = cKDTree(colmap_xyz.cpu().numpy())
    
    depth_folder = os.path.join(args.source_path, "depths_npy")
    new_points_xyz, new_points_rgb = [], []

    for idx, cam in enumerate(cameras):
        npy_path = os.path.join(depth_folder, cam.image_name + ".npy")
        if not os.path.exists(npy_path): continue
        
        img_raw = np.array(Image.open(cam.image_path).convert("RGB"))
        depth_raw = np.load(npy_path)
        
        # Consistent slicing
        depth_s = depth_raw[::args.stride, ::args.stride]
        img_s = img_raw[::args.stride, ::args.stride]
        h, w = depth_s.shape
        img_s = img_s[:h, :w] # Shape safety

        bg_mask = build_bg_mask(img_s, args.source_path)
        
        # Intrinsics
        focal_y = h / (2.0 * np.tan(cam.FovY / 2.0))
        focal_x = w / (2.0 * np.tan(cam.FovX / 2.0))
        cx, cy = w / 2.0, h / 2.0
        
        # W2C
        R = torch.tensor(cam.R).float().to(device).transpose(0, 1)
        w2c = torch.eye(4, device=device)
        w2c[:3, :3], w2c[:3, 3] = R, torch.tensor(cam.T).to(device)
        
        # Alignment
        sparse_d = project_sparse_depth(colmap_xyz, w2c, focal_x, focal_y, cx, cy, w, h)
        aligned_d, scale, shift, ok = lstsq_align(depth_s, sparse_d)
        if not ok: continue
        
        # Inverse alignment if scale negative
        if scale < 0:
            inv_mono = 1.0 / (depth_s + 1e-6)
            aligned_d, _, _, _ = lstsq_align(inv_mono, sparse_d)

        if bg_mask.shape != aligned_d.shape:
            h_min = min(bg_mask.shape[0], aligned_d.shape[0])
            w_min = min(bg_mask.shape[1], aligned_d.shape[1])
            bg_mask = bg_mask[:h_min, :w_min]
            aligned_d = aligned_d[:h_min, :w_min]
            img_s = img_s[:h_min, :w_min]

        valid_mask = (aligned_d > 0.1) & (aligned_d < 100.0) & np.isfinite(aligned_d)
        valid_mask = valid_mask & (~bg_mask)

        h, w = aligned_d.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        xx = (xx - cx) / focal_x
        yy = (yy - cy) / focal_y
        z = aligned_d[valid_mask]
        x, y = xx[valid_mask] * z, yy[valid_mask] * z
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
        
        pts_world = (torch.inverse(w2c).cpu().numpy() @ pts_cam)[:3, :].T
        
        # Since pts_cam length depends on mask count, we must ensure mask count 
        # is consistent with the indexing.
        try:
            colors_ref = img_s[valid_mask]
        except IndexError:
            # Fallback: manually match the shapes
            h_min = min(img_s.shape[0], mask.shape[0])
            w_min = min(img_s.shape[1], mask.shape[1])
            mask_aligned = mask[:h_min, :w_min]
            img_aligned = img_s[:h_min, :w_min]
            
            # Re-generate pts_cam to match aligned mask
            h, w = aligned_d.shape
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            xx = (xx - cx) / focal_x
            yy = (yy - cy) / focal_y
            # Apply common bounds
            valid_depth = (aligned_d > 0.1) & (aligned_d < 100.0) & np.isfinite(aligned_d)
            valid_depth = valid_depth[:h_min, :w_min]
            bg_mask_aligned = bg_mask[:h_min, :w_min]
            valid_depth = valid_depth & (~bg_mask_aligned)
            
            z = aligned_d[:h_min, :w_min][valid_depth]
            x, y = xx[:h_min, :w_min][valid_depth] * z, yy[:h_min, :w_min][valid_depth] * z
            pts_cam_aligned = np.stack([x, y, z, np.ones_like(z)], axis=0)
            pts_world = (torch.inverse(w2c).cpu().numpy() @ pts_cam_aligned)[:3, :].T
            colors_ref = img_aligned[valid_depth]
            
        # Photometric consistency (LLFF logic: single neighbor view + per-view intrinsics)
        if len(cameras) < 2:
            continue

        src_idx = idx + 1 if idx + 1 < len(cameras) else idx - 1
        src_cam = cameras[src_idx]
        img_src = np.array(Image.open(src_cam.image_path).convert("RGB"))[::args.stride, ::args.stride]
        src_h, src_w = img_src.shape[:2]

        src_focal_y = src_h / (2.0 * np.tan(src_cam.FovY / 2.0))
        src_focal_x = src_w / (2.0 * np.tan(src_cam.FovX / 2.0))
        src_cx, src_cy = src_w / 2.0, src_h / 2.0

        src_R = torch.tensor(src_cam.R).float().to(device).transpose(0, 1)
        src_w2c = torch.eye(4, device=device)
        src_w2c[:3, :3], src_w2c[:3, 3] = src_R, torch.tensor(src_cam.T).to(device)

        pts_world_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
        u_src, v_src, vld_src = project_points(
            pts_world_h,
            src_w2c.cpu().numpy(),
            src_focal_x,
            src_focal_y,
            src_cx,
            src_cy,
            src_w,
            src_h,
        )

        if vld_src.sum() == 0:
            continue

        u_valid = u_src[vld_src].astype(np.int32)
        v_valid = v_src[vld_src].astype(np.int32)
        bg_src = build_bg_mask(img_src, args.source_path)
        src_keep = ~bg_src[v_valid, u_valid]
        if src_keep.sum() == 0:
            continue
        colors_src = img_src[v_valid[src_keep], u_valid[src_keep]]
        colors_ref_valid = colors_ref[vld_src][src_keep]

        diff = np.abs(colors_ref_valid.astype(np.float32) / 255.0 - colors_src.astype(np.float32) / 255.0).mean(axis=1)
        photo_mask = diff < args.photo_thresh

        pts_p = pts_world[vld_src][src_keep][photo_mask]
        colors_p = colors_ref_valid[photo_mask]
        
        # KDTree Complementary Fusion + cap per-view points (LLFF logic)
        if pts_p.shape[0] > 0:
            dists, _ = kd_tree.query(pts_p, k=1)
            keep = dists > args.dist_thresh
            pts_keep = pts_p[keep]
            colors_keep = colors_p[keep]
            if pts_keep.shape[0] > args.max_points_per_view:
                indices = np.random.choice(pts_keep.shape[0], args.max_points_per_view, replace=False)
                pts_keep = pts_keep[indices]
                colors_keep = colors_keep[indices]
            new_points_xyz.append(pts_keep)
            new_points_rgb.append(colors_keep)

    if len(new_points_xyz) > 0:
        # Fused points from mono depth
        mono_xyz = np.concatenate(new_points_xyz, axis=0)
        mono_rgb = np.concatenate(new_points_rgb, axis=0)
        
        # Original SfM points
        sfm_xyz = colmap_xyz.cpu().numpy()
        sfm_rgb = colmap_rgb.cpu().numpy()
        
        # Innovation V3: Hybrid Initialization (Add Random Points to fill "Missing Chunks")
        # Use SfM bounds to define random cloud extent
        if sfm_xyz.shape[0] > 0:
            pcd_min = sfm_xyz.min(axis=0)
            pcd_max = sfm_xyz.max(axis=0)
            pcd_center = (pcd_min + pcd_max) / 2
            pcd_scale = (pcd_max - pcd_min).max()
            
            # Generate 10k random points in the bounding box
            num_rand = 10000
            rand_xyz = (np.random.random((num_rand, 3)) - 0.5) * pcd_scale * 1.5 + pcd_center
            rand_rgb = np.random.random((num_rand, 3)) * 255 # Random noise color? Or gray?
            # Let's make them gray/neutral to not disturb training too much, or just random
            rand_rgb = np.ones((num_rand, 3)) * 128
            
            print(f"Adding {num_rand} random points to fill geometric voids...")
            res_xyz = np.concatenate([sfm_xyz, mono_xyz, rand_xyz], axis=0)
            res_rgb = np.concatenate([sfm_rgb, mono_rgb, rand_rgb], axis=0)
        else:
            res_xyz = np.concatenate([sfm_xyz, mono_xyz], axis=0)
            res_rgb = np.concatenate([sfm_rgb, mono_rgb], axis=0)

        save_ply(os.path.join(args.source_path, "points3D_fused.ply"), res_xyz, res_rgb)

if __name__ == "__main__":
    main()
