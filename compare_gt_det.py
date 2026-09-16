# -*- coding: utf-8 -*-
"""
手选真值(gt_manual.json) vs 代码检出(port_locator.locate) 对比可视化
=====================================================================
参考 view_ICP_input0.png 的风格：
  左图 X-Y 俯视（沿Z轴，端口圆盘在此视图中最清楚）
  中图 X-Z 侧视（沿Y轴）
  右图 X-Y 局部放大（两口周围 ±160mm，专看圆心对没对准盘心）
图例：
  红点=手选 GT ly   蓝点=手选 GT ry
  绿星=代码 ly      橙星=代码 ry
  绿/橙实线=代码检出圆盘轮廓(3D圆投影)；红/蓝虚线=以手选点为心、同半径的参考圆
  点间连线标注 Δ=两点距离（mm）
输出：compare_all.png（2×5 总览，仅 X-Y）
      compare_ICP_inputN.png（每帧三视图大图）
用法：python compare_gt_det.py
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_locator import locate, read_pcd, voxelize, _circle3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Windows 中文字体（与 view_ICP_input0.png 同款系统字体；缺失时回退英文标签）
_FP = r"C:\Windows\Fonts\simhei.ttf"
HAS_FONT = os.path.exists(_FP)
if HAS_FONT:
    font_manager.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def L(zh, en):
    return zh if HAS_FONT else en

BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"
HERE = os.path.dirname(os.path.abspath(__file__))
GT = json.load(open(os.path.join(HERE, "gt_manual.json")))
OUTDIR = os.path.join(HERE, "compare_views")
os.makedirs(OUTDIR, exist_ok=True)

C_GT_LY, C_GT_RY = "#d62728", "#1f77b4"     # 红 / 蓝
C_DET_LY, C_DET_RY = "#2ca02c", "#ff7f0e"   # 绿 / 橙


def draw_scene(ax, Pd, ai, bi, xlab, ylab, title):
    ax.scatter(Pd[:, ai], Pd[:, bi], s=0.25, c="0.80", rasterized=True, linewidths=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)


def draw_marker(ax, c, ai, bi, color, marker, size, label, z=8):
    ax.scatter([c[ai]], [c[bi]], c=color, marker=marker, s=size,
               edgecolors="black", linewidths=0.8, zorder=z, label=label)


def draw_pair(ax, ai, bi, g_ly, g_ry, d_ly, d_ry, n_ly, n_ry, r_ly, r_ry):
    # 代码检出圆盘轮廓（实线）
    for c, n, R, col in [(d_ly, n_ly, r_ly, C_DET_LY), (d_ry, n_ry, r_ry, C_DET_RY)]:
        circ = _circle3d(np.asarray(c, float), np.asarray(n, float), R)
        ax.plot(circ[:, ai], circ[:, bi], color=col, lw=1.4, zorder=5)
    # 手选点为心的同半径参考圆（虚线）——与实际点云盘不重合即手选偏心
    for c, n, R, col in [(g_ly, n_ly, r_ly, C_GT_LY), (g_ry, n_ry, r_ry, C_GT_RY)]:
        circ = _circle3d(np.asarray(c, float), np.asarray(n, float), R)
        ax.plot(circ[:, ai], circ[:, bi], color=col, lw=1.0, ls="--", zorder=4)
    # 点标记
    draw_marker(ax, g_ly, ai, bi, C_GT_LY, "o", 110, L("手选GT ly", "GT ly"))
    draw_marker(ax, g_ry, ai, bi, C_GT_RY, "o", 110, L("手选GT ry", "GT ry"))
    draw_marker(ax, d_ly, ai, bi, C_DET_LY, "*", 190, L("代码 ly", "Det ly"))
    draw_marker(ax, d_ry, ai, bi, C_DET_RY, "*", 190, L("代码 ry", "Det ry"))
    # GT-代码 连线 + 距离
    for g, d in [(g_ly, d_ly), (g_ry, d_ry)]:
        g = np.asarray(g, float); d = np.asarray(d, float)
        delta = np.linalg.norm(g - d)
        ax.plot([g[ai], d[ai]], [g[bi], d[bi]], color="magenta", lw=1.0,
                ls=":", zorder=6)
        if delta > 8.0:
            mx, my = (g[ai] + d[ai]) / 2, (g[bi] + d[bi]) / 2
            ax.annotate(L("Δ=%.0fmm" % delta, "d=%.0f" % delta),
                        (mx, my), color="magenta", fontsize=10, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", zorder=9)


def main():
    summary = {}
    cache = {}
    for i in range(10):
        name = "ICP_input%d" % i
        f = os.path.join(BASE, "实车点云", "%s.pcd" % name)
        r = locate(f, verbose=False)
        cache[name] = r
        if not r["best"]:
            print("%s 未配对，跳过" % name)
            continue
        P0 = read_pcd(f)
        P = P0[np.isfinite(P0).all(1)]
        Pd = voxelize(P, 6.0)
        cache[name + "_Pd"] = Pd

        g_ly = np.array(GT[name]["ly"]); g_ry = np.array(GT[name]["ry"])
        d_ly = np.array(r["best"]["ly"]["center"]); d_ry = np.array(r["best"]["ry"]["center"])
        n_ly = np.array(r["best"]["ly"]["normal"]); n_ry = np.array(r["best"]["ry"]["normal"])
        r_ly, r_ry = r["best"]["ly"]["radius"], r["best"]["ry"]["radius"]
        summary[name] = dict(dly=float(np.linalg.norm(g_ly - d_ly)),
                             dry=float(np.linalg.norm(g_ry - d_ry)), r=r)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        draw_scene(axes[0], Pd, 0, 1, "x (mm)", "y (mm)",
                   L("%s  X-Y 俯视（沿Z轴）" % name, "%s  X-Y top view" % name))
        draw_scene(axes[1], Pd, 0, 2, "x (mm)", "z (mm)",
                   L("X-Z 侧视（沿Y轴）", "X-Z side view"))
        # 局部放大：两口 bbox ± 160mm（X-Y）
        draw_scene(axes[2], Pd, 0, 1, "x (mm)", "y (mm)",
                   L("X-Y 局部放大（两口 ±160mm）", "X-Y zoom around ports"))
        # 左：X-Y 全景观测
        draw_pair(axes[0], 0, 1, g_ly, g_ry, d_ly, d_ry, n_ly, n_ry, r_ly, r_ry)
        # 中：X-Z 侧视
        draw_pair_xz(axes[1], g_ly, g_ry, d_ly, d_ry, n_ly, n_ry, r_ly, r_ry)
        # 右：X-Y 局部放大
        allc = np.array([g_ly, g_ry, d_ly, d_ry])
        xc, yc = allc[:, 0].mean(), allc[:, 1].mean()
        axes[2].set_xlim(xc - 340, xc + 340)
        axes[2].set_ylim(yc - 200, yc + 200)
        draw_pair(axes[2], 0, 1, g_ly, g_ry, d_ly, d_ry, n_ly, n_ry, r_ly, r_ry)
        for ax in axes:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        out = os.path.join(OUTDIR, "compare_%s.png" % name)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("%-12s ly Δ=%5.1fmm  ry Δ=%5.1fmm  -> %s"
              % (name, summary[name]["dly"], summary[name]["dry"], out))

    # ---------- 2×5 总览（X-Y 局部裁剪，每帧一个子图）----------
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    for i, ax in enumerate(axes.flat):
        name = "ICP_input%d" % i
        r = cache.get(name)
        if not r or not r["best"]:
            continue
        Pd = cache[name + "_Pd"]
        g_ly = np.array(GT[name]["ly"]); g_ry = np.array(GT[name]["ry"])
        d_ly = np.array(r["best"]["ly"]["center"]); d_ry = np.array(r["best"]["ry"]["center"])
        n_ly = np.array(r["best"]["ly"]["normal"]); n_ry = np.array(r["best"]["ry"]["normal"])
        draw_scene(ax, Pd, 0, 1, "x (mm)", "y (mm)", "")
        draw_pair(ax, 0, 1, g_ly, g_ry, d_ly, d_ry, n_ly, n_ry,
                  r["best"]["ly"]["radius"], r["best"]["ry"]["radius"])
        allc = np.array([g_ly, g_ry, d_ly, d_ry])
        xc, yc = allc[:, 0].mean(), allc[:, 1].mean()
        ax.set_xlim(xc - 340, xc + 340); ax.set_ylim(yc - 200, yc + 200)
        s = summary[name]
        flag = "" if max(s["dly"], s["dry"]) < 20 else L("  ★ry手选偏差大", "  ★GT-ry off")
        ax.set_title(L("%s  Δly=%.0f Δry=%.0fmm%s" % (name, s["dly"], s["dry"], flag),
                       "%s d_ly=%.0f d_ry=%.0f%s" % (name, s["dly"], s["dry"], flag)),
                     fontsize=11, color="crimson" if flag else "black")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12)
    fig.suptitle(L("手选GT（红/蓝圆点+虚线圆） vs 代码检出（绿/橙星+实线圆盘轮廓）  ——  X-Y俯视局部",
                   "Manual GT (red/blue dots + dashed) vs Detected (green/orange stars + solid)"),
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_all = os.path.join(HERE, "compare_all.png")
    fig.savefig(out_all, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\n总览图: %s" % out_all)
    print("逐帧大图目录: %s" % OUTDIR)


def draw_pair_xz(ax, g_ly, g_ry, d_ly, d_ry, n_ly, n_ry, r_ly, r_ry):
    ai, bi = 0, 2
    for c, n, R, col in [(d_ly, n_ly, r_ly, C_DET_LY), (d_ry, n_ry, r_ry, C_DET_RY)]:
        circ = _circle3d(np.asarray(c, float), np.asarray(n, float), R)
        ax.plot(circ[:, ai], circ[:, bi], color=col, lw=1.4, zorder=5)
    for c, n, R, col in [(g_ly, n_ly, r_ly, C_GT_LY), (g_ry, n_ry, r_ry, C_GT_RY)]:
        circ = _circle3d(np.asarray(c, float), np.asarray(n, float), R)
        ax.plot(circ[:, ai], circ[:, bi], color=col, lw=1.0, ls="--", zorder=4)
    draw_marker(ax, g_ly, ai, bi, C_GT_LY, "o", 110, L("手选GT ly", "GT ly"))
    draw_marker(ax, g_ry, ai, bi, C_GT_RY, "o", 110, L("手选GT ry", "GT ry"))
    draw_marker(ax, d_ly, ai, bi, C_DET_LY, "*", 190, L("代码 ly", "Det ly"))
    draw_marker(ax, d_ry, ai, bi, C_DET_RY, "*", 190, L("代码 ry", "Det ry"))
    for g, d in [(g_ly, d_ly), (g_ry, d_ry)]:
        g = np.asarray(g, float); d = np.asarray(d, float)
        delta = np.linalg.norm(g - d)
        ax.plot([g[ai], d[ai]], [g[bi], d[bi]], color="magenta", lw=1.0, ls=":", zorder=6)
        if delta > 8.0:
            ax.annotate(L("Δ=%.0fmm" % delta, "d=%.0f" % delta),
                        ((g[ai] + d[ai]) / 2, (g[bi] + d[bi]) / 2),
                        color="magenta", fontsize=10, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", zorder=9)


if __name__ == "__main__":
    main()
