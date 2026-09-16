# -*- coding: utf-8 -*-
"""
关键阈值敏感性实验（对照修正版真值，逐轴≤20mm 口径）
扰动：各向异性 ANISO_MAX、最少盘面点数 NPT_DS_MIN、垂线 PERP_MAX、
      遮挡修复触发 R_PARTIAL_TOL、起始球 R0_DS。
每格跑实车 10 帧，统计双口达标的帧数（默认参数为当前配置）。
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import port_locator as pl

BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"
HERE = os.path.dirname(os.path.abspath(__file__))
GT = json.load(open(os.path.join(HERE, "gt_manual.json"), encoding="utf-8"))
FILES = [os.path.join(BASE, "实车点云", "ICP_input%d.pcd" % i) for i in range(10)]


DEFAULTS = dict(ANISO_MAX=1.30, NPT_DS_MIN=400, PERP_MAX=20.0,
                R_PARTIAL_TOL=6.0, R0_DS=32.0)


def run_one(args):
    name, overrides = args
    for k, v in overrides.items():
        setattr(pl, k, v)
    try:
        r = pl.locate(FILES[int(name[-1])], verbose=False)
        if not r["best"]:
            return name, overrides, False, 999.0, r["time"]
        g = GT[name]
        ely = np.max(np.abs(np.array(r["best"]["ly"]["center"]) - np.array(g["ly"])))
        ery = np.max(np.abs(np.array(r["best"]["ry"]["center"]) - np.array(g["ry"])))
        return name, overrides, bool(ely <= 20 and ery <= 20 and r["time"] <= 60), \
            float(max(ely, ery)), r["time"]
    except Exception:
        return name, overrides, False, -1.0, 999.0


def sweep(pool, label, var, values):
    print("\n### %s" % label)
    tasks = []
    for val in values:
        ov = dict(DEFAULTS); ov[var] = val
        for i in range(10):
            tasks.append(("ICP_input%d" % i, ov))
    results = list(pool.imap_unordered(run_one, tasks))
    for val in values:
        sub = [x for x in results if x[1][var] == val]
        ok = sum(1 for x in sub if x[2])
        worst = max(x[3] for x in sub)
        tmax = max(x[4] for x in sub)
        star = " <- 默认" if val == DEFAULTS[var] else ""
        print("  %s=%-6s  达标 %2d/10  最差最大轴误差=%6.2fmm  最慢=%.1fs%s"
              % (var, val, ok, worst, tmax, star))


if __name__ == "__main__":
    import multiprocessing as mp
    print("阈值敏感性（实车10帧，修正版真值，逐轴≤20mm；多进程并行）")
    with mp.Pool(6, maxtasksperchild=1) as pool:
        sweep(pool, "各向异性上限", "ANISO_MAX", [1.20, 1.25, 1.30, 1.40])
        sweep(pool, "降采样盘面最少点数", "NPT_DS_MIN", [200, 300, 400, 600])
        sweep(pool, "垂线判据上限(°)", "PERP_MAX", [12.0, 16.0, 20.0, 25.0])
        sweep(pool, "遮挡修复触发(mm)", "R_PARTIAL_TOL", [4.0, 6.0, 9.0])
        sweep(pool, "起始生长球(mm)", "R0_DS", [24.0, 32.0, 40.0])
