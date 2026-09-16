# -*- coding: utf-8 -*-
"""配对准则消融实验（用当前修正版真值 gt_manual.json 重测）

对比三种配对策略下实车 10 帧的双口达标帧数（逐轴 ≤20mm）：
  A. 写死 source 间距 411.8±30mm
  B. 弱间距先验 356±60mm
  C. 不启用间距约束（pair_tol=1e9，本项目默认）
"""
import numpy as np, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import port_locator as pl

BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"
HERE = os.path.dirname(os.path.abspath(__file__))
GT = json.load(open(os.path.join(HERE, "gt_manual.json"), encoding="utf-8"))
FILES = [os.path.join(BASE, "实车点云", "ICP_input%d.pcd" % i) for i in range(10)]

CASES = [
    ("A 写死 source 间距 411.8±30mm", 411.8, 30.0),
    ("B 弱间距先验 356±60mm",          356.0, 60.0),
    ("C 不启用间距约束（默认）",        356.0, 1e9),
]


def main():
    print("配对准则消融（实车 10 帧，修正版真值，逐轴 ≤20mm）")
    print("%-34s %-8s %-10s %s" % ("策略", "达标帧", "最差误差", "明细(帧:ly/ry 最大轴误差)"))
    for label, pd, pt in CASES:
        ok = 0
        worst = 0.0
        det = []
        t0 = time.time()
        for i, f in enumerate(FILES):
            name = "ICP_input%d" % i
            r = pl.locate(f, pair_d=pd, pair_tol=pt, verbose=False)
            if not r["best"]:
                det.append("%s:未配对" % i)
                continue
            g = GT[name]
            ely = float(np.max(np.abs(np.array(r["best"]["ly"]["center"]) - np.array(g["ly"]))))
            ery = float(np.max(np.abs(np.array(r["best"]["ry"]["center"]) - np.array(g["ry"]))))
            em = max(ely, ery)
            worst = max(worst, em)
            if ely <= 20 and ery <= 20:
                ok += 1
            det.append("%d:%.1f/%.1f" % (i, ely, ery))
        print("%-34s %-8s %-10s %s"
              % (label, "%d/10" % ok, "%.2fmm" % worst, "  ".join(det)))
        print("      (用时 %.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
