# -*- coding: utf-8 -*-
"""
法向量独立验证（val2.png 方法的自动化复刻 + CC 人工实测录入接口）
=====================================================================
任务书要求：法向量【俯仰角、偏航角】与实际偏差 ≤5°。
val2 验证法：在 CloudCompare 中量圆盘两条正交直径端点的 ΔX,ΔY,ΔZ：
  · 一条直径向量 d1（近似沿 x 轴方向）-> 偏航角
  · 另一条直径向量 d2（近似沿 y 轴方向）-> 俯仰角
  · n = d1 × d2，再统一朝 +z
本脚本：
  A) 自动复刻：以 GT 圆心为心，从原始点云独立取环带边缘点，沿两个正交
     方向各取直径端点，叉乘得参考法向 —— 与 port_locator 的"内点平面PCA
     法向"数学途径不同，做方法学交叉验证。
  B) CC 实测：读取 cc_norm_manual.json（用户用测距工具量得的 Δ），
     计算"实际"俯仰/偏航角并与程序输出对比，这是 val2 规定的正式验证。

角度约定（与 port_locator.pitch_yaw 一致，朝 +z）：
  pitch = atan2(ny, nz)    yaw = atan2(nx, nz)

用法：
  python norm_check.py                 # A) 自动交叉验证（source + 10帧，共22个口）
  python norm_check.py --cc file.json  # B) 对照 CC 人工实测
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_locator import locate, read_pcd
from scipy.spatial import cKDTree

BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"
HERE = os.path.dirname(os.path.abspath(__file__))
GT = json.load(open(os.path.join(HERE, "gt_manual.json"), encoding="utf-8"))
GT_OFFICIAL = np.loadtxt(os.path.join(BASE, "coordinates.txt"))
R_REF = {"ly": 59.0, "ry": 79.0}


def pitch_yaw(n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    return float(np.degrees(np.arctan2(n[1], n[2]))), float(np.degrees(np.arctan2(n[0], n[2]))), n


def pca_normal(Q):
    C = np.cov((Q - Q.mean(0)).T)
    w, V = np.linalg.eigh(C)
    n = V[:, 0]
    return n if n[2] >= 0 else -n


def val2_diameters(P, tree, ctr, r_ref, band=4.0, frac=0.06):
    """以 ctr 为心独立估计两条正交直径向量 d1(沿平面内主方向)、d2，叉乘出法向。
    返回 n_ref, d1, d2, 端点有效率。"""
    idx = tree.query_ball_point(ctr, r_ref + 12.0)
    Q = P[idx]
    n0 = pca_normal(Q[np.linalg.norm(Q - ctr, axis=1) < min(50.0, r_ref * 0.8)])
    h = (Q - ctr) @ n0
    Qp = Q[np.abs(h) < band]                      # 平面 ±4mm 带
    q = Qp - ctr
    q = q - np.outer(q @ n0, n0)                  # 投影到平面
    rad = np.linalg.norm(q, axis=1)
    rim = q[rad > 0.62 * r_ref]                  # 环带边缘点（平面内向量）
    Qrim = Qp[rad > 0.62 * r_ref]
    if len(rim) < 60:
        return None
    # 平面内主方向（边缘点协方差最大轴），端点取最外 frac 点的三维均值
    C = np.cov(rim.T); _, V = np.linalg.eigh(C)
    ex = V[:, -1]; ex = ex - (ex @ n0) * n0; ex /= np.linalg.norm(ex)
    ey = np.cross(n0, ex); ey /= np.linalg.norm(ey)
    ends = []
    for e in (ex, ey):
        s = rim @ e
        k = max(8, int(frac * len(rim)))
        ip = Qrim[s >= np.sort(s)[-k]]
        im = Qrim[s <= np.sort(s)[k - 1]]
        p_plus, p_minus = ip.mean(0), im.mean(0)
        ends.append((p_plus, p_minus))
    d1 = ends[0][0] - ends[0][1]
    d2 = ends[1][0] - ends[1][1]
    n_ref = np.cross(d1, d2)
    nn = np.linalg.norm(n_ref)
    if nn < 1e-9:
        return None
    n_ref = n_ref / nn
    if n_ref[2] < 0:
        n_ref = -n_ref
    cov = min(np.linalg.norm(d1), np.linalg.norm(d2)) / (2 * r_ref)
    return n_ref, d1, d2, cov


def angle_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def run_auto():
    scenes = [("source", os.path.join(BASE, "source.pcd"),
               GT_OFFICIAL[0], GT_OFFICIAL[1])]
    for i in range(10):
        name = "ICP_input%d" % i
        scenes.append((name, os.path.join(BASE, "实车点云", "%s.pcd" % name),
                       np.array(GT[name]["ly"], float), np.array(GT[name]["ry"], float)))
    print("=" * 118)
    print("法向独立交叉验证：程序平面PCA法向 vs val2双直径法（同一GT圆心，独立数学途径）")
    print("=" * 118)
    print("%-10s %3s  %8s %8s | %8s %8s | %6s %6s %6s | %5s  判定" %
          ("场景", "口", "程序pitch", "程序yaw", "v2pitch", "v2yaw",
           "Δpitch", "Δyaw", "夹角°", "覆盖"))
    worst = 0.0
    rows = []
    for name, f, g_ly, g_ry in scenes:
        r = locate(f, verbose=False)
        P0 = read_pcd(f); P = P0[np.isfinite(P0).all(1)]
        tree = cKDTree(P)
        for key, g in (("ly", g_ly), ("ry", g_ry)):
            o = r["best"][key]
            n_det = np.array(o["normal"], float)
            p_d, y_d, n_det = pitch_yaw(n_det)
            v = val2_diameters(P, tree, np.array(g, float), R_REF[key])
            if v is None:
                print("%-10s %3s   参考法向估计失败" % (name, key))
                continue
            n_ref, d1, d2, cov = v
            p_r, y_r, _ = pitch_yaw(n_ref)
            dp, dy = angle_diff(p_d, p_r), angle_diff(y_d, y_r)
            ang = np.degrees(np.arccos(min(1.0, abs(np.dot(n_det, n_ref)))))
            worst = max(worst, dp, dy)
            ok = "PASS" if (dp <= 5 and dy <= 5) else "FAIL"
            print("%-10s %3s  %8.2f %8.2f | %8.2f %8.2f | %6.2f %6.2f %6.2f | %4.2f  %s"
                  % (name, key, p_d, y_d, p_r, y_r, dp, dy, ang, cov, ok))
            rows.append(dict(scene=name, port=key, pitch_det=round(p_d, 2), yaw_det=round(y_d, 2),
                             pitch_val2=round(p_r, 2), yaw_val2=round(y_r, 2),
                             dpitch=round(dp, 2), dyaw=round(dy, 2),
                             normal_angle=round(ang, 2), diameter_cov=round(cov, 2)))
    print("-" * 118)
    print("最大 Δpitch/Δyaw = %.2f°（限值5°）->  %s"
          % (worst, "全部PASS" if worst <= 5 else "存在FAIL"))
    json.dump(rows, open(os.path.join(HERE, "norm_check_auto.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("已写出 norm_check_auto.json")


def run_cc(cc_file):
    """cc_norm_manual.json 格式（两种二选一）：
    方式1 val2 直径法：
      {"ICP_input1": {"ly": {"d1":[dx,dy,dz], "d2":[dx,dy,dz]}, ...}}
      d1=CC测距近似沿x方向直径(平行直径, 取ΔXΔZ算偏航),
      d2=近似沿y方向直径(垂直直径, 取ΔYΔZ算俯仰)。
    方式2 CC 拟合平面（Tools > Fit > Plane）直接读法向量：
      {"ICP_input1": {"ly": {"n":[nx,ny,nz]}, ...}}
    source 帧的键名用 "source"。"""
    manual = json.load(open(cc_file, encoding="utf-8"))
    print("=" * 100)
    print("CC 人工实测（val2 正式验证） vs 程序法向")
    print("=" * 100)
    print("%-12s %3s | %8s %8s (实测) | %8s %8s (程序) | %6s %6s  判定" %
          ("场景", "口", "pitch", "yaw", "pitch", "yaw", "Δpit", "Δyaw"))
    worst = 0.0
    rows = []
    for name, ports in manual.items():
        if name == "_说明":
            continue
        if name == "source":
            f = os.path.join(BASE, "source.pcd")
        else:
            f = os.path.join(BASE, "实车点云", "%s.pcd" % name)
        r = locate(f, verbose=False)
        if not r["best"]:
            print("%-12s   该场景未检出端口，跳过" % name)
            continue
        for key, dd in ports.items():
            if "n" in dd:
                n_gt = np.array(dd["n"], float)
            else:
                d1, d2 = np.array(dd["d1"], float), np.array(dd["d2"], float)
                n_gt = np.cross(d1, d2)
            if np.linalg.norm(n_gt) < 1e-6:
                print("%-12s %3s   未填写或读数为零，跳过（请在 cc_norm_manual.json 中补全）"
                      % (name, key))
                continue
            p_g, y_g, _ = pitch_yaw(n_gt)
            n_det = np.array(r["best"][key]["normal"], float)
            p_d, y_d, _ = pitch_yaw(n_det)
            dp, dy = angle_diff(p_d, p_g), angle_diff(y_d, y_g)
            worst = max(worst, dp, dy)
            print("%-12s %3s | %8.2f %8.2f       | %8.2f %8.2f       | %6.2f %6.2f  %s"
                  % (name, key, p_g, y_g, p_d, y_d, dp, dy,
                     "PASS" if dp <= 5 and dy <= 5 else "FAIL"))
            rows.append(dict(scene=name, port=key, pitch_cc=round(p_g, 3), yaw_cc=round(y_g, 3),
                             pitch_det=round(p_d, 3), yaw_det=round(y_d, 3),
                             dpitch=round(dp, 3), dyaw=round(dy, 3),
                             pass_5deg=bool(dp <= 5 and dy <= 5)))
    print("-" * 100)
    print("最大角差 %.2f° -> %s" % (worst, "全部PASS" if worst <= 5 else "存在FAIL"))
    json.dump(rows, open(os.path.join(HERE, "norm_check_cc.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("已写出 norm_check_cc.json")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--cc":
        run_cc(sys.argv[2])
    else:
        run_auto()
