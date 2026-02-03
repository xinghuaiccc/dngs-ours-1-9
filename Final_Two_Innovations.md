# 极稀疏视角合成论文：最终双核心创新点总结 (修正版)

本文针对 3 视图极稀疏场景下的重建难题，提出了双核心解决方案：

---

### 1. 创新点一：Robust Initialization via Advanced Monocular Priors (基于先进先验的稳健初始化)
**学术标题**：*Dense Geometric Initialization Guided by Depth-Anything-V2*

*   **问题**：DNGaussian 在极稀疏视角下退化为随机初始化，导致几何结构崩塌和收敛缓慢。
*   **方法**：
    *   引入 **Depth-Anything-V2** 高精度单目深度模型。
    *   构建 **3 视图 MVS 密集初始化** 流程，利用可见视角的深度一致性反向投影生成初始点云。
    *   **关键发现**：该先验**仅用于初始化 (Initial-Only)**。实验证明，无需在训练过程中施加额外的单目深度监督，仅凭借高质量的初始化，就能维持几何结构的稳定。
*   **贡献**：从根本上解决了“冷启动”难题，确立了更公平、高效且极简的初始化基线。

---

### 2. 创新点二：FFT-Refined Frequency Domain Consistency (频域精化一致性)
**学术标题**：*Frequency-Aware Texture Refinement via Fourier Transform Loss*

*   **问题**：仅依靠像素级损失（L1/SSIM），在稀疏监督下会导致纹理过度平滑（模糊）。
*   **方法**：
    *   **独立设计并实装**了 FFT 损失函数。
    *   将监督信号扩展至频域，协同优化 **幅值谱**（恢复高频细节）和 **相位谱**（锁定结构边缘）。
*   **贡献**：有效弥补了空间域损失的不足，显著提升了渲染结果的锐度和细节真实感。

---

### 总结
本文的方法论是**“初始化定乾坤”**：通过**Point 1 (V2 初始化)** 提供完美的几何起点，再通过 **Point 2 (FFT Loss)** 进行纹理精修。这种极简方案成功替代了原版 DNGaussian 复杂的深度重标定机制。
