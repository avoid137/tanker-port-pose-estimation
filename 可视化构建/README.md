# 可视化构建链说明

把「11 个场景的原始点云 + `result_final.json` 检测结果」合成为**单文件、离线可用**的三维交互页面
（上级目录的 `课题六-端口位姿三维可视化.html`）。

## 三步流程

```
原始 PCD + result_final.json
        │  ① export_web.py
        ▼
  viewer_data.json          （降采样 + int16 量化 + base64，约 1.76 MB / 21.9 万点）
        │  ② build_viewer.py   ← 需 three.min.js
        ▼
  port_viewer.html          （2.39 MB，双击即开，无需联网）
```

### ① 导出数据 `export_web.py`

- 读 `../result_final.json`（检测结果）、`../gt_manual.json`、`../cc_norm_manual.json`（真值与法向）、
  以及数据集的 `coordinates.txt`（source 官方真值）；
- 每个场景点云做 **12mm 体素降采样**，超 32000 点则自动放大体素直到达标；
- 以包围盒中点为原点做 **int16 毫米量化**（量化误差 ±0.5mm），base64 编码；
- 法向做**符号统一**（SVD 解出的法向符号任意，对俯仰/偏航角无影响，但可视化会误读，故统一朝 +z）。

> 脚本内路径写成绝对路径（构建时的真实位置），换机器需改 `V1` / `BASE` 两个常量。

### ② 合成单文件 `build_viewer.py`

把 `viewer_template.html` 里的 `/*__THREE__*/` 与 `/*__DATA__*/` 两处占位替换为
three.js 源码与数据 JSON，输出单文件 HTML。

- 查找顺序：**本目录同名文件优先**，缺失则回退到 `D:\k6data\`；
- 内嵌前检查内容是否含 `</script>`（会提前闭合 `<script>` 标签）；
- 用法：`python build_viewer.py [输出路径]`，省略则输出到本目录。

**依赖**：`three.min.js`（three.js **r128 UMD** 构建，取自 `unpkg.com/three@0.128.0/build/three.min.js`）。
因体积 603 KB，未纳入版本库；需要重新构建时按上面地址下载放到本目录即可。

### ③ 页面功能（`viewer_template.html`）

Three.js 渲染 + 自写轨道控制（不依赖 OrbitControls）。支持 URL 参数便于截图/演示：

| 参数 | 含义 | 取值 |
|---|---|---|
| `scene` | 场景序号 | `0`=source，`1~10`=实车 input0~9 |
| `view` | 预设视角 | `focus` / `port` / `side` / `top` / `fit` |
| `oth` | 是否显示其余候选盘 | `1` 显示 |
| `spin` | 自动旋转 | `1` 开启 |
| `dbg` | 调试输出（投影坐标） | `1` 开启 |

侧栏另有「自动巡场」（答辩轮播）与「保存当前视图为 PNG」。

## 重新构建

```bash
# 前提：本目录有 three.min.js
"D:/anaconda3/python.exe" export_web.py            # 生成 viewer_data.json
"D:/anaconda3/python.exe" build_viewer.py "D:/k6data/port_viewer.html"
```

构建产物已按 sha256 校验：本流程重跑结果与仓库内 `课题六-端口位姿三维可视化.html` **逐字节一致**。
