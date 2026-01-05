import torch
import torch.nn.functional as F


def warp_image_based_on_depth(src_img, src_depth, src_K, src_w2c, dst_w2c):
    """
    Forward-warp src_img to dst view using src_depth.
    Returns (dst_img, dst_mask), where dst_mask indicates valid splatted pixels.
    """
    if src_img.ndim != 4 or src_img.shape[0] != 1 or src_img.shape[1] != 3:
        raise ValueError("src_img must be [1, 3, H, W]")
    if src_depth.ndim != 4 or src_depth.shape[0] != 1 or src_depth.shape[1] != 1:
        raise ValueError("src_depth must be [1, 1, H, W]")

    device = src_img.device
    dtype = src_img.dtype
    _, _, h, w = src_img.shape

    src_K = src_K.to(device=device, dtype=dtype)
    src_w2c = src_w2c.to(device=device, dtype=dtype)
    dst_w2c = dst_w2c.to(device=device, dtype=dtype)

    u = torch.arange(w, device=device, dtype=dtype)
    v = torch.arange(h, device=device, dtype=dtype)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    ones = torch.ones_like(grid_u)
    pix = torch.stack([grid_u, grid_v, ones], dim=0).reshape(3, -1)

    depth = src_depth[0, 0].reshape(1, -1)
    k_inv = torch.inverse(src_K)
    cam_dirs = k_inv @ pix
    cam_points = cam_dirs * depth

    cam_points_h = torch.cat(
        [cam_points, torch.ones((1, cam_points.shape[1]), device=device, dtype=dtype)],
        dim=0,
    )

    src_c2w = torch.inverse(src_w2c)
    world_points = src_c2w @ cam_points_h
    dst_cam_points = dst_w2c @ world_points

    x = dst_cam_points[0]
    y = dst_cam_points[1]
    z = dst_cam_points[2]

    valid = z > 1e-6
    x = x[valid]
    y = y[valid]
    z = z[valid]

    u_proj = (src_K[0, 0] * (x / z)) + src_K[0, 2]
    v_proj = (src_K[1, 1] * (y / z)) + src_K[1, 2]

    u_int = torch.round(u_proj).long()
    v_int = torch.round(v_proj).long()

    in_bounds = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
    u_int = u_int[in_bounds]
    v_int = v_int[in_bounds]
    z = z[in_bounds]

    dst_idx = v_int * w + u_int

    zbuf = torch.full((h * w,), float("inf"), device=device, dtype=dtype)
    if hasattr(zbuf, "scatter_reduce_"):
        zbuf.scatter_reduce_(0, dst_idx, z, reduce="amin", include_self=True)
    else:
        zbuf = torch.minimum(zbuf, zbuf.scatter(0, dst_idx, z))

    keep = z <= (zbuf[dst_idx] + 1e-6)

    src_colors = src_img[0].permute(1, 2, 0).reshape(-1, 3)
    src_colors = src_colors[valid][in_bounds][keep]

    dst_flat = torch.zeros((h * w, 3), device=device, dtype=dtype)
    dst_mask = torch.zeros((h * w, 1), device=device, dtype=dtype)

    dst_idx_keep = dst_idx[keep]
    dst_flat[dst_idx_keep] = src_colors
    dst_mask[dst_idx_keep] = 1.0

    dst_img = dst_flat.view(h, w, 3).permute(2, 0, 1).unsqueeze(0)
    dst_mask = dst_mask.view(1, h, w)

    return dst_img, dst_mask
