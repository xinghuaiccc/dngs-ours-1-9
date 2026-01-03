import cv2
import numpy as np
import matplotlib.pyplot as plt


def evaluate_error_map(gt_path, render_path, threshold=0.1):
    # 1. 读取并归一化到 [0, 1] (方便计算)
    gt = cv2.imread(gt_path).astype(np.float32) / 255.0
    render = cv2.imread(render_path).astype(np.float32) / 255.0

    if gt.shape != render.shape:
        render = cv2.resize(render, (gt.shape[1], gt.shape[0]))

    # 2. 计算绝对误差图 (Difference Map)
    diff = np.abs(gt - render)
    # 取三通道平均，变成单通道误差图
    diff_map = np.mean(diff, axis=2)

    # --- 核心评分机制 ---

    # 指标 1: MAE (平均绝对误差) - 越小越好
    # 代表全图平均错了多少
    mae = np.mean(diff_map)

    # 指标 2: RMSE (均方根误差) - 越小越好
    # 代表"严重的错误"有多少 (对红线很敏感)
    rmse = np.sqrt(np.mean(diff_map ** 2))

    # 指标 3: Bad Pixel Ratio (坏点率) - 越小越好 !!!
    # 统计误差超过 threshold (比如 0.1) 的像素占比
    # 这是证明你"消除了伪影"的最强证据
    bad_pixels = np.sum(diff_map > threshold)
    total_pixels = diff_map.size
    bpr = (bad_pixels / total_pixels) * 100.0  # 换算成百分比

    print(f"=== 误差图评分 (Error Map Metrics) ===")
    print(f"1. MAE  (平均误差): {mae:.5f} (越低越好)")
    print(f"2. RMSE (严重程度): {rmse:.5f} (越低越好)")
    print(f"3. BPR  (坏点比例): {bpr:.2f}%  (阈值>{threshold}, 越低越好)")
    print(f"======================================")

    return diff_map, mae, rmse, bpr


# 使用方法
# 记得用你之前保存的图，或者直接读原图
diff_map, score_mae, score_rmse, score_bpr = evaluate_error_map('00001_gt.jpg', '00001_render.jpg')

# 如果你想保存带分数的图
plt.figure(figsize=(6, 5))
plt.imshow(diff_map, cmap='jet', vmin=0, vmax=0.5)  # vmax可以控制热力图亮度，设小一点会让红线更明显
plt.title(f"Error Map\nMAE:{score_mae:.4f} | BPR:{score_bpr:.2f}%")
plt.axis('off')
plt.colorbar()
plt.savefig("scored_error_map.png")