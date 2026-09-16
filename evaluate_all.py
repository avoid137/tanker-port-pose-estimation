# -*- coding: utf-8 -*-
"""
课题六 全量评估（任务口径）
---------------------------------------------------------------
硬指标：圆心【各轴】坐标误差 ≤20mm；法向俯仰角/偏航角偏差 ≤5°；单帧 ≤60s。
  - source.pcd：对照官方 coordinates.txt（唯一数值圆心真值）。
  - 实车 10 帧：对照 gt_manual.json（用户 CloudCompare 重选真值；
    input1/2/3/4 的 ry 已于 2026-09-15 按 val2 方法重选）。
  - 法向：数据集不含法向数值真值，按 val2 方法把法向换算为俯仰/偏航角，
    并用「两口法向夹角」「连线⊥罐面法向偏离」两项几何一致性交叉验证。
输出：控制台汇总表 + result_final.json + error_table_final.csv
用法：python evaluate_all.py
"""
import numpy as np, os, sys, json, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_locator import locate, read_pcd
from scipy.spatial import cKDTree

BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"
HERE = os.path.dirname(os.path.abspath(__file__))
GT_OFFICIAL = np.loadtxt(os.path.join(BASE, "coordinates.txt"))
GT = json.load(open(os.path.join(HERE, "gt_manual.json")))


def angle_between(n1, n2):
    return float(np.degrees(np.arccos(min(1.0, abs(np.dot(n1, n2))))))


def assess(name, f, gt_ly=None, gt_ry=None):
    r = locate(f, verbose=False)
    row = dict(scene=name, time=round(r["time"], 1), n_ports=r["n_ports"],
               n_pairs=r["n_pairs"], paired=bool(r["best"]))
    if not r["best"]:
        row["pass_axes20"] = False
        row["pass_time60"] = True
        return row, r
    for key, gt in (("ly", gt_ly), ("ry", gt_ry)):
        o = r["best"][key]
        c = np.array(o["center"])
        row["%s_r" % key] = round(o["radius"], 1)
        row["%s_pitch" % key] = round(o["pitch"], 2)
        row["%s_yaw" % key] = round(o["yaw"], 2)
        row["%s_refined" % key] = bool(o.get("refined"))
        if gt is not None:
            gt = np.array(gt, float)
            row["%s_dx" % key] = round(float(c[0] - gt[0]), 2)
            row["%s_dy" % key] = round(float(c[1] - gt[1]), 2)
            row["%s_dz" % key] = round(float(c[2] - gt[2]), 2)
            row["%s_emax" % key] = round(float(np.max(np.abs(c - gt))), 2)
            row["%s_e3d" % key] = round(float(np.linalg.norm(c - gt)), 2)
    n1 = np.array(r["best"]["ly"]["normal"])
    n2 = np.array(r["best"]["ry"]["normal"])
    row["normal_angle"] = round(angle_between(n1, n2), 2)
    row["perp_off"] = round(r["best"]["perp_off"], 2)
    row["distance"] = round(r["best"]["distance"], 1)
    if gt_ly is not None and gt_ry is not None:
        row["pass_axes20"] = bool(row["ly_emax"] <= 20 and row["ry_emax"] <= 20)
        row["pass_time60"] = bool(r["time"] <= 60)
    return row, r


def main():
    rows = []
    detail = {}
    print("=" * 110)
    print("一、source.pcd（官方真值 coordinates.txt）")
    print("=" * 110)
    row, r = assess("source", os.path.join(BASE, "source.pcd"), GT_OFFICIAL[0], GT_OFFICIAL[1])
    rows.append(row); detail["source"] = r
    print("  ly: 各轴偏差(dx,dy,dz)=(%6.2f,%6.2f,%6.2f) 最大轴误差=%5.2fmm 欧氏=%5.2fmm | "
          "r=%.1f 俯仰=%.2f° 偏航=%.2f°"
          % (row["ly_dx"], row["ly_dy"], row["ly_dz"], row["ly_emax"], row["ly_e3d"],
             row["ly_r"], row["ly_pitch"], row["ly_yaw"]))
    print("  ry: 各轴偏差(dx,dy,dz)=(%6.2f,%6.2f,%6.2f) 最大轴误差=%5.2fmm 欧氏=%5.2fmm | "
          "r=%.1f 俯仰=%.2f° 偏航=%.2f°"
          % (row["ry_dx"], row["ry_dy"], row["ry_dz"], row["ry_emax"], row["ry_e3d"],
             row["ry_r"], row["ry_pitch"], row["ry_yaw"]))
    print("  两口法向夹角=%.2f°  连线⊥罐面法向偏离=%.2f°  间距=%.1fmm  用时=%.1fs  -> %s"
          % (row["normal_angle"], row["perp_off"], row["distance"], row["time"],
             "PASS" if row["pass_axes20"] and row["pass_time60"] else "FAIL"))

    print("\n" + "=" * 110)
    print("二、实车 10 帧（对照用户 CloudCompare 重选真值 gt_manual.json）")
    print("=" * 110)
    hdr = ("场景       用时   ly最大轴  ry最大轴  ly欧氏  ry欧氏   ly俯仰/偏航       ry俯仰/偏航     "
           "法向夹角 垂线偏离  判定")
    print(hdr)
    n_pass = 0
    for i in range(10):
        name = "ICP_input%d" % i
        f = os.path.join(BASE, "实车点云", "%s.pcd" % name)
        row, r = assess(name, f, GT[name]["ly"], GT[name]["ry"])
        # 真值点到最近实际点的距离（手选点质量自检，单位mm）
        P0 = read_pcd(f)
        P = P0[np.isfinite(P0).all(1)]
        tree = cKDTree(P)
        d_gt_ly = tree.query(np.array(GT[name]["ly"], float))[0]
        d_gt_ry = tree.query(np.array(GT[name]["ry"], float))[0]
        rows.append(row); detail[name] = r
        ok = row["pass_axes20"] and row["pass_time60"]
        n_pass += int(ok)
        mark = "PASS" if ok else "FAIL"
        disp = name.replace("ICP_", "")
        if "ly_emax" in row:
            print("%-9s %5.1fs %7.2f %8.2f %7.2f %7.2f  (%6.2f/%6.2f) (%6.2f/%6.2f) %7.2f° %7.2f°  %s (GT落云 %.1f/%.1fmm)%s"
                  % (disp, row["time"], row["ly_emax"], row["ry_emax"], row["ly_e3d"], row["ry_e3d"],
                     row["ly_pitch"], row["ly_yaw"], row["ry_pitch"], row["ry_yaw"],
                     row["normal_angle"], row["perp_off"], mark, d_gt_ly, d_gt_ry,
                     "  [遮挡修复]" if r["best"]["ly"].get("refined") or r["best"]["ry"].get("refined") else ""))
        else:
            print("%-9s %5.1fs  ---- 该帧未配对/未检出，无法评估圆心 ----  %s" % (disp, row["time"], mark))
    print("\n  逐轴≤20mm 且 ≤60s 达标：%d/10" % n_pass)

    json.dump({"rows": rows, "results": detail},
              open(os.path.join(HERE, "result_final.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    keys = [k for k in rows[0].keys()]
    with open(os.path.join(HERE, "error_table_final.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("\n已写出 result_final.json / error_table_final.csv")


if __name__ == "__main__":
    main()
