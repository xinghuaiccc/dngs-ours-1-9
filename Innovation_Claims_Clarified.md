# 创新点核对说明（DNGaussian vs 本项目）

本文档用于澄清哪些内容是真正新增、哪些是原版已有但被沿用或调整的部分，
以便论文写作做到“可验证、可辩护、可追溯”。

---

## 范围说明

基线：`/root/DNGaussian`  
当前项目：本仓库（`train_llff_new-2.py`、`dpt/`、`scene/`）

目标：
1) 算法级新增（真正新模块/新损失）。  
2) 流程/系统级改造（组件替换或加载策略改变）。  
3) 原版已有（不可写成原创）。

---

## A. 已确认新增（可作为创新点）

### A1. FFT 频域损失（算法级新增）
证据（本仓库）：
- `train_llff_new-2.py:41` 定义 `fft_loss(...)`，包含幅值与相位一致性。
- `train_llff_new-2.py:230` 在训练中调用 `loss_fft = fft_loss(...)`。

证据（原版）：
- `/root/DNGaussian` 中找不到 `fft_loss`（无 `def fft_loss`）。

可写贡献：
- 引入 FFT 频域一致性损失，显著提升极稀疏视角下的高频纹理恢复能力。

### A2. Depth-Anything V2 单目深度先验（流程级替换）
证据（本仓库）：
- `dpt/get_depth_map_for_llff_dtu_depth_anything_v2.py` 使用 Depth-Anything V2 生成深度 `.npy`。

证据（原版）：
- `/root/DNGaussian` 无 `depth_anything` 相关脚本或引用。

可写贡献：
- 将单目深度先验升级为 Depth-Anything V2，提高稀疏视角下深度监督的鲁棒性。

---

## B. 有变化但不是新模块（需谨慎表述）

### B1. MVS 初始化使用策略（加载优先级调整）
证据（本仓库）：
- `scene/dataset_readers.py:231-237` 在 `if mvs_pcd:` 分支里优先加载
  `3_views/dense/fused.ply`。

证据（原版）：
- `/root/DNGaussian/scene/dataset_readers.py:247-250` 已支持 `--mvs_pcd`，
  只是分支顺序放在随机初始化之后。

变化点：
- 你将 `mvs_pcd` 作为优先分支，明确要求使用 3 视图 fused 点云初始化。

可写贡献（流程级）：
- 调整初始化策略，优先采用三视图 fused 点云以稳定冷启动。

---

## C. 原版已有（不能写成原创）

### C1. 形态正则（Shape/Scale Penalty）
证据（原版）：
- `/root/DNGaussian/arguments/__init__.py` 默认已有 `shape_pena`、`scale_pena`。

结论：
- 可写“保留/使用”，不可写“新增”。

### C2. 近场剪枝（Near-range Pruning）
证据（原版）：
- `/root/DNGaussian/train_llff.py` 已有 `near_range` 剪枝逻辑。

结论：
- 不可写为原创模块。

### C3. `--mvs_pcd` 开关本身
证据（原版）：
- `/root/DNGaussian/arguments/__init__.py`、`scene/dataset_readers.py`、
  `scripts/run_llff.sh` 均已包含。

结论：
- 参数不是新增，只能说你改变了使用策略。

---

## D. 论文可用的严谨表述（建议）

1) 算法级贡献：  
   - “引入 FFT 频域一致性损失，利用幅值与相位联合约束恢复高频纹理。”

2) 流程级贡献：  
   - “将单目深度先验升级为 Depth-Anything V2。”  
   - “调整初始化策略，优先采用三视图 fused 点云进行冷启动。”

3) 明确非原创：  
   - “形态正则与近场剪枝为原版已有机制，本项目沿用。”

---

## E. 最小准确创新列表（2-3 条）

1) FFT 频域损失（新增模块）。  
2) Depth-Anything V2 深度先验（流程替换）。  
3) 三视图 MVS 初始化优先策略（流程调整）。
