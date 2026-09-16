# -*- coding: utf-8 -*-
"""
生成 CloudCompare 可直接打开的核对点文件（ASCII: x y z R G B）
对 input1/2/3/4 各生成一个 cc_check/ICP_inputN_check.txt：
  橙色圆环 = 代码检出的 ry 圆盘轮廓(r≈79)
  绿色小环 = 代码检出 ry 圆心
  蓝色小环 = gt_manual.json 手选 ry 点
CC 用法：先打开对应 ICP_inputN.pcd，再把 txt 拖入（导入向导选 x y z R G B，0行跳过）。
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_locator import _circle3d

HERE = os.path.dirname(os.path.abspath(__file__))
GT = json.load(open(os.path.join(HERE, "gt_manual.json")))
res = json.load(open(os.path.join(HERE, "result_final.json", ), encoding="utf-8"))["results"]
OUT = os.path.join(HERE, "cc_check")
os.makedirs(OUT, exist_ok=True)

ORANGE = (255, 127, 0)
GREEN = (0, 190, 0)
BLUE = (0, 90, 255)

def ring_pts(c, n, R, rgb, n_seg=160):
    cc = _circle3d(np.asarray(c, float), np.asarray(n, float), R, n_seg)
    out = np.zeros((n_seg, 6))
    out[:, :3] = cc
    out[:, 3:] = rgb
    return out

def disk_fill(c, n, R, rgb, n_ring=4, n_seg=80):
    """实心小圆盘填充点（让中心标记在CC里更醒目）"""
    pts = [np.array([[c[0], c[1], c[2], rgb[0], rgb[1], rgb[2]]])]
    for k in range(1, n_ring + 1):
        pts.append(ring_pts(c, n, R * k / n_ring, rgb, n_seg))
    return np.vstack(pts)

print("%-12s %22s %22s %8s  偏差轴向" % ("帧", "手选GT ry (蓝)", "代码ry (绿)", "Δmm"))
for i in [1, 2, 3, 4]:
    name = "ICP_input%d" % i
    b = res[name]["best"]["ry"]
    c_det = np.array(b["center"]); n_det = np.array(b["normal"]); R = b["radius"]
    c_gt = np.array(GT[name]["ry"])
    d = c_det - c_gt
    dist = np.linalg.norm(d)
    ax = ["X", "Y", "Z"][int(np.argmax(np.abs(d)))]
    print("%-12s (%8.2f,%8.2f,%8.2f) (%8.2f,%8.2f,%8.2f) %7.1f  代码-手选: dX=%6.1f dY=%6.1f dZ=%6.1f (主%s)"
          % (name, c_gt[0], c_gt[1], c_gt[2], c_det[0], c_det[1], c_det[2],
             dist, d[0], d[1], d[2], ax))
    big = ring_pts(c_det, n_det, R, ORANGE, 200)          # 检出圆盘外轮廓
    cen_det = disk_fill(c_det, n_det, 14.0, GREEN)       # 检出中心
    cen_gt = disk_fill(c_gt, n_det, 14.0, BLUE)          # 手选点
    allpts = np.vstack([big, cen_det, cen_gt])
    fp = os.path.join(OUT, "%s_check.txt" % name)
    np.savetxt(fp, allpts, fmt="%.3f %.3f %.3f %d %d %d",
               header="x y z R G B   orange=detected rim(r~79) green=detected center blue=manual GT",
               comments="// ")
    print("             -> %s" % fp)
print("\nCC 打开方式：File->Open 先开 ICP_inputN.pcd，再拖入同名 _check.txt；")
print("导入向导里 0 行跳过，列依次选 X Y Z R G B。")
