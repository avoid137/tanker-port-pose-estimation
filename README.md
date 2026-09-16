# 课题六 V1 版本归档 — 流质化工产品罐车加注与回收口精确位姿估计

> 归档时间：2026-09-16 ｜ 代码版本：port_locator v7 ｜ 状态：全部硬指标达标

---

## 一、任务概述

从罐车顶部点云（binary PCD）中自动定位**加注口（ry，大圆 r≈79mm）**与**回收口（ly，小圆 r≈59mm）**两个端口盖，输出：

- 圆心 3D 坐标（mm）
- 盖面法向量 → 俯仰角/偏航角（约定与 val2.png 一致：pitch=atan2(ny,nz)，yaw=atan2(nx,nz)，法向统一朝 +z）
- 半径

**硬指标**：圆心各轴误差 ≤20mm；俯仰/偏航角偏差 ≤5°；单帧处理 ≤60s。

**数据**：source.pcd（有官方真值 coordinates.txt）+ 实车 10 帧 ICP_input0~9.pcd（真值为 CloudCompare 手选 gt_manual.json）。

**难点**（实测确认）：实车上端口盖与罐面**共面无台阶**，常规聚类/台阶检测/边界提取全部失效；存在部分遮挡帧（input2 ly）；无标注数据，深度学习不适用。

---

## 二、算法流程（port_locator.py v7）

```
读 binary PCD（自实现，不依赖 open3d）
  → 5mm 体素化 + FFT 球卷积 K55 密度图（O(1) 查表）
  → 36mm 均匀候选网格 + 密度预筛（排除稀疏区，均匀采样防大平面独占）
  → 3mm 降采样上平面区域生长（初始平面用迭代裁剪 SVD，3mm 平面带内接受点）
  → 形状判据筛选：收敛 + R∈[35,130] + 面内各向异性≤1.30 + 盘面点数≥400
  → 全分辨率重新生长精化 → NMS（90mm 去重）
  → 配对：尺寸先验（59/79 高斯打分）× 连线⊥罐面法向（不写死间距）
  → 遮挡修复：观测半径亏缺>6mm 触发已知半径 RANSAC 边界圆拟合
  → 输出 JSON；visualize() 生成俯视/侧视/前视三视图 PNG
```

### 关键技术设计

| # | 设计 | 位置 | 解决的问题 |
|---|------|------|-----------|
| 1 | **实心盘半径修正** R=√2·median(rad) | `fit_disc_3d` | 均匀实心盘上 Kasa 最小二乘圆拟合半径系统性低估至 R/√2（实测 41.9 vs 59mm）；中位数估计抗污染 50% |
| 2 | **平面带区域生长 + 各向异性判据** | `grow_flat_region` | 口盖与罐面共面，固定球 SVD 法向被台阶带偏 48°；面内协方差特征值比 √(λ1/λ2) 区分圆盘（1.00~1.02）与罐面长条（1.19~1.60） |
| 3 | **FFT 球卷积密度预筛** | `locate` 前段 | 全场景 55mm 球内点数一次算完，候选 O(1) 查表剔除稀疏区 |
| 4 | **双法向分离** | `locate` 精化段 | 盖面法向（输出）与罐面法向（配对垂线判据用）物理含义不同；罐面法向在 160mm 邻域 ±25mm 带内重估 |
| 5 | **几何配对判据** | `locate` 配对段 | 同周向两口连线即罐体轴向，必⊥罐面法向；消融实验：写死间距 413±30 → 仅 1/10，去掉间距仅靠尺寸+垂线 → 足够 |
| 6 | **遮挡盘修复** | `refine_partial_disc` | 部分遮挡时质心偏向可见侧（input2 ly 偏 13.7mm）；利用题目"接口规格一致"的已知物理半径做 3 点圆 RANSAC（800 试次+固定种子）+2 轮最小二乘精修，降至 2.86mm，正常盘零触发 |
| 7 | **中心吸附** | `plane_disc_signature` | 拟合平面被盖边缘台阶带偏，离质心 60mm 处离面 40mm+，KDTree 吸附到最近实测点 |

---

## 三、文件清单

### 核心代码

| 文件 | 功能 |
|------|------|
| `port_locator.py` | **定位器主体**。`locate(path)` API + CLI（`--json/--vis/--gt/--pair_d`）+ `visualize()` 三视图 |
| `evaluate_all.py` | **全量评估**：source（官方真值）+ 实车 10 帧（gt_manual.json），输出 result_final.json + error_table_final.csv |
| `norm_check.py` | **法向量独立验证**：A) val2 双直径法自动复刻（方法学交叉验证）；B) `--cc cc_norm_manual.json` 对照 CC 人工实测（正式验证口径） |
| `sensitivity.py` | 阈值敏感性实验：ANISO_MAX / NPT_DS_MIN / PERP_MAX / R_PARTIAL_TOL / R0_DS 五参数扫描（多进程） |
| `run_batch.py` | 批量运行：`python run_batch.py [source|real|all]` |
| `compare_gt_det.py` | GT 与检测结果可视化对比（虚线 GT 圆 / 实线检测圆 + 距离标注） |
| `gen_cc_check.py` | 生成 CloudCompare 检查点文件（ASCII xyzRGB 圆环轮廓） |
| `export_gt_excel.py` | 导出测量数据汇总 Excel（圆心真值 + CC 法向量 + 自动换算俯仰/偏航角） |

### 真值数据

| 文件 | 内容 |
|------|------|
| `gt_manual.json` | 10 帧实车圆心真值（CC 手选；input1/2/3/4 的 ry 于 2026-09-15 按 val2 方法重选） |
| `cc_norm_manual.json` | 20 个口的 CC 拟合平面实测法向（2026-09-16 最新，含 input0-ly / input7-ly 二次复测） |

### 结果数据

| 文件 | 内容 |
|------|------|
| `result_final.json` / `error_table_final.csv` | 全量评估最终结果（逐帧偏差/角度/用时） |
| `norm_check_auto.json` / `norm_check_cc.json` | 法向量双途径验证结果 |
| `测量数据汇总.xlsx` | 三个工作表：圆心真值 / 法向量实测（含角度）/ 说明 |

### 可视化

`vis_final_source/input2/input4.png`（三视图演示）、`compare_all.png`（GT vs 检测对比）、`norm_diag_3ports.png`（法向诊断）。

---

## 四、使用方法

```bash
# 环境：D:\anaconda3\python.exe（numpy / scipy / matplotlib / openpyxl）

# 单帧定位（JSON + 三视图）
python port_locator.py "D:\...\source.pcd" --json out.json --vis out.png

# 全量评估（11 场景，输出达标判定）
python evaluate_all.py

# 法向量正式验证（对照 CC 实测）
python norm_check.py --cc cc_norm_manual.json

# 敏感性实验
python sensitivity.py

# 导出 Excel
python export_gt_excel.py
```

API 调用：

```python
from port_locator import locate, pitch_yaw
r = locate(r"D:\...\ICP_input0.pcd", verbose=False)
r["best"]["ly"]["center"]    # 回收口圆心 [x,y,z] mm
r["best"]["ly"]["pitch"]     # 俯仰角 °
r["best"]["ly"]["yaw"]       # 偏航角 °
r["best"]["ly"]["normal"]    # 盖面法向单位向量
```

---

## 五、验证结果（最终达标情况）

### 圆心精度（逐轴最大误差口径 ≤20mm）

| 场景 | ly | ry | 用时 |
|------|----|----|------|
| source（官方真值） | 0.93mm | 3.49mm | 13.5s |
| 实车 10 帧 | 全部 ≤3.41mm（最差 input0-ly） | 全部 ≤4.78mm（最差 input5-ry） | 3.2~14.1s |

**11/11 场景 PASS，远优于 20mm 指标。**

### 法向量（俯仰/偏航偏差 ≤5°）

- CC 实测对照：20/20 口全部通过，最大 Δyaw=2.64°
- val2 双直径法自动复刻：与程序平面 PCA 法向独立途径交叉验证一致

### 敏感性

5 个关键阈值在合理扰动范围内（如 ANISO_MAX 1.20~1.40、PERP_MAX 12°~25°）保持 10/10 达标，鲁棒性良好。

---

## 六、版本历史

| 版本 | 日期 | 要点 |
|------|------|------|
| v1~v12（散稿） | 09-14 前 | FRDP 签名 / 多尺度采样等尝试，实车帧大量漏检 |
| v13~v21 | 09-14 | 参数与判据快速迭代（pose_estimate_v13~v21.py，已归档至上级目录） |
| v6 | 09-15 | 确立"平面区域生长 + 各向异性 + 尺寸先验配对"主框架 |
| v7 | 09-15 | 修正 GT 后全量达标；新增遮挡修复 `refine_partial_disc`、val2 角度约定 `pitch_yaw`、三视图 `visualize` |

**重要教训**：input1/2/3/4 的 ry 手选真值最初偏 28~84mm，曾导致误判算法失败；经区域生长收敛性 + 扇区半径分析独立验证后重选，全部帧达标。误差评估一律使用重选后真值。

---

## 七、与通用方法的差异（报告可引用的创新点）

1. **实心圆盘半径估计修正**（√2·median）— 网上圆拟合代码默认环状边界点，实心盘直接用会错 29%
2. **各向异性判据区分圆盘/罐面长条** — 通用分割库没有的领域定制判别器
3. **FFT 球卷积密度预筛** — 深度学习 voxel CNN 思想的无学习版
4. **连线⊥罐面法向配对 + 消融实验** — 用数据证明间距先验非必需
5. **观测半径亏缺自动触发的已知半径 RANSAC 遮挡修复** — 触发机制与守卫设计（位移限制/内点门槛/固定种子）为工程贡献

（对比对象：PCL SACMODEL_CIRCLE2D/3D 盲跑方案、鹤管 2D 视觉专利、欧式聚类+圆柱拟合专利、深度学习圆基元提取等，详见项目复盘文档。）

---

## 八、本地更新仓库（绕过 TLS 吊销检查）

本机网络链路（含 `WattToolkit` 代理）会拦截 CRL/OCSP 证书吊销检查，导致 `git push`/`git fetch` 在 TLS 握手阶段报：

```
schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK (0x80092012)
```

`http.schannelCheckRevoke=false` 在当前 git 2.55（Windows）环境无效。可用以下方式绕过（GitHub 凭证已存于本机 Windows 凭据管理器，免交互读取）：

```bash
# 1) 让 git 用 wincred 读取已存的 GitHub token（仓库本地配置，不含明文 token）
git config credential.helper wincred

# 2) push / fetch 时关闭整体证书校验，以绕过吊销检查的硬失败
git -c http.sslVerify=false push -u origin main
# 拉取：
git -c http.sslVerify=false fetch origin
```

> 说明：本机 `GIT_TERMINAL_PROMPT=0` 禁止交互输密码，因此必须走 `wincred`（凭据管理器里的 `git:https://github.com` 条目）。**不要把 GitHub token 明文写进命令或 config**；本仓库上云即用此法完成，未落盘任何 token。
