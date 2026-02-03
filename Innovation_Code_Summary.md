# 本项目核心创新代码整理 (Code Summary for Innovations)

本文件整理了本项目两个核心创新点（基于单目先验的密集初始化、FFT频域损失）的关键代码片段，方便查阅和展示。

---

## 1. 创新点一：基于单目深度先验的密集几何初始化
**(Dense Geometric Initialization via Monocular Depth Priors)**

该部分逻辑包含两个步骤：
1.  **生成**：利用 `Depth-Anything-V2` 的深度图，通过反向投影和多视角一致性校验生成密集点云。
2.  **加载**：修改数据加载器，优先使用生成好的 `fused.ply` 代替稀疏 SfM 点云。

### 代码片段 1.1: 深度图反向投影与融合 (Back-projection Logic)
*来源: `generate_fused_pcd.py`*

```python
# 核心函数：将单目深度图反向投影为3D点
def backproject_points(aligned_depth, focal_x, focal_y, cx, cy):
    height, width = aligned_depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xx = (xx - cx) / focal_x
    yy = (yy - cy) / focal_y

    # 过滤无效深度
    valid_mask = (aligned_depth > 0.1) & (aligned_depth < 100.0) & np.isfinite(aligned_depth)

    z = aligned_depth[valid_mask]
    x = xx[valid_mask] * z
    y = yy[valid_mask] * z

    # 生成相机坐标系下的点云 [x, y, z, 1]
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
    return pts_cam, valid_mask

# 主流程片段：对齐与融合
# ...
    # 将 COLMAP 稀疏点投影到当前视图，获得 sparse_depth
    sparse_depth = project_sparse_depth(colmap_xyz, w2c, focal_x, focal_y, cx, cy, width, height)
    
    # 将单目深度 (Depth-Anything) 与 稀疏深度 (COLMAP) 进行最小二乘对齐
    aligned_depth, scale, shift, ok = lstsq_align(mono_depth, sparse_depth)
    
    # 反向投影生成密集点云
    pts_cam, valid_mask = backproject_points(aligned_depth, focal_x, focal_y, cx, cy)
    
    # 转世界坐标系并融合
    c2w = torch.inverse(w2c).cpu().numpy()
    pts_world = (c2w @ pts_cam)[:3, :].T
# ...
```

### 代码片段 1.2: 优先加载融合点云 (Loader Priority Logic)
*来源: `scene/dataset_readers.py`*

```python
def readColmapSceneInfo(path, ... mvs_pcd, ...):
    # ...
    # 创新点逻辑：如果指定 mvs_pcd=True，则强制加载 3_views/dense/fused.ply
    # 这是由 generate_fused_pcd.py 生成的密集点云
    if mvs_pcd:
        ply_path = os.path.join(path, "3_views/dense/fused.ply")
        print(f"Init MVS point cloud from: {ply_path}")
        assert os.path.exists(ply_path)
        pcd = fetchPly(ply_path) # 加载此点云用于初始化 Gaussians
    # 否则才回退到随机初始化或原始 COLMAP 点云
    elif rand_pcd:
        print('Init random point cloud.')
        # ...
```

---

## 2. 创新点二：基于FFT的频域精化一致性约束
**(FFT-Refined Frequency Domain Consistency)**

该部分在训练过程中引入频域损失，约束幅值谱和相位谱。

### 代码片段 2.1: FFT 损失函数定义 (Loss Definition)
*来源: `train_llff_new-2.py`*

```python
def fft_loss(image, gt_image):
    # 1. 图像尺寸填充 (Padding) 至 32 的倍数，避免 FFT 边界效应
    H, W = image.shape[-2], image.shape[-1]
    target_H = ((H + 31) // 32) * 32
    target_W = ((W + 31) // 32) * 32
    # ... (interpolate code) ...

    # 2. 快速傅里叶变换 (RFFT2)
    fft_pred = torch.fft.rfft2(pred)
    fft_gt = torch.fft.rfft2(gt)

    # 3. 计算幅值谱 (Amplitude)
    amp_pred = torch.sqrt(fft_pred.real.pow(2) + fft_pred.imag.pow(2) + 1e-6)
    amp_gt = torch.sqrt(fft_gt.real.pow(2) + fft_gt.imag.pow(2) + 1e-6)
    loss_amp = torch.abs(amp_pred - amp_gt).mean()

    # 4. 计算相位谱 (Phase) - 使用实部和虚部的 L1 距离近似
    loss_phase = (
        torch.abs(fft_pred.real - fft_gt.real).mean()
        + torch.abs(fft_pred.imag - fft_gt.imag).mean()
    )

    # 5. 组合损失：幅值损失 + 0.5 * 相位损失
    return loss_amp + 0.5 * loss_phase
```

### 代码片段 2.2: 训练循环中的调用 (Training Loop Usage)
*来源: `train_llff_new-2.py`*

```python
# ... inside training loop ...
        # 计算常规 L1 和 SSIM 损失
        Ll1 = l1_loss(image, gt_image)
        d_ssim = 1.0 - ssim(image, gt_image)
        
        # [创新点] 计算 FFT 频域损失
        loss_fft = fft_loss(image, gt_image)
        
        # 总损失加权求和
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * d_ssim
            + fft_weight * loss_fft  # 引入频域约束
        )
        
        loss.backward()
# ...
```
