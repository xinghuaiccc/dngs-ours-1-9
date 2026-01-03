import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_error_comparison(gt_path, render_path):
    # 1. 读取两张图
    gt = cv2.imread(gt_path)
    render = cv2.imread(render_path)

    if gt is None or render is None:
        print("图片读取失败，请检查文件名！")
        return

    # 确保尺寸一致（如果不一致，以GT为准进行缩放）
    if gt.shape != render.shape:
        render = cv2.resize(render, (gt.shape[1], gt.shape[0]))

    # 2. 计算绝对误差 (Absolute Difference)
    diff = cv2.absdiff(gt, render)

    # 转为灰度误差，方便做热力图
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # --- 新增：单独保存误差图 ---
    # 1. 保存为原始灰度误差图 (适合定量分析)
    cv2.imwrite('error_map_gray.png', diff_gray)

    # 2. 保存为彩色热力图 (使用 jet 映射，更直观，适合放进论文)
    # 我们先用 plt 将其转换并保存
    plt.imsave('error_map_heat.png', diff_gray, cmap='jet')

    print("成功保存：error_map_gray.png (灰度) 和 error_map_heat.png (彩色热力)")
    # --------------------------

    # 3. 可视化对比大图
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth (GT)")
    plt.imshow(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Rendered Image (Baseline)")
    plt.imshow(cv2.cvtColor(render, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Error Map (Where it fails)")
    plt.imshow(diff_gray, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    # 注意：在某些服务器环境下 plt.show() 会报错，建议直接用 savefig
    plt.savefig('error_comparison_result.png', dpi=300)
    print("对比总图已保存为 error_comparison_result.png")


# 使用示例
visualize_error_comparison('00001_gt.jpg', '00001_render.jpg')