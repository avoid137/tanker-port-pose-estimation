# -*- coding: utf-8 -*-
"""对一批场景跑 port_locator.locate，输出汇总表 + JSON"""
import numpy as np, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_locator import locate

BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"

def run_batch(name, files, gt=None, out_json=None, pair_d=None):
    print("=" * 96)
    print(f"### {name}")
    print("=" * 96)
    rows = []
    for f in files:
        t0 = time.time()
        kw = {} if pair_d is None else dict(pair_d=pair_d[0], pair_tol=pair_d[1])
        r = locate(f, verbose=False, **kw)
        rows.append(r)
        tag = os.path.basename(f)
        if r['best']:
            ly = np.array(r['best']['ly']['center']); ry = np.array(r['best']['ry']['center'])
            d = r['best']['distance']
            msg = (f"{tag:26s} 用时{r['time']:5.1f}s 候选盘={r['n_ports']:3d}配对={r['n_pairs']:3d} | "
                   f"ly r={r['best']['ly']['radius']:5.1f} ry r={r['best']['ry']['radius']:5.1f} 间距={d:6.1f}mm")
            if gt is not None:
                e0 = np.linalg.norm(ly - gt[0]); e1 = np.linalg.norm(ry - gt[1])
                verdict = "OK" if (e0 <= 20 and e1 <= 20 and r['time'] <= 60) else "!!"
                msg += f" | 误差 ly={e0:6.2f} ry={e1:6.2f} mm  {verdict}"
            print(msg)
        else:
            print(f"{tag:26s} 用时{r['time']:5.1f}s 候选盘={r['n_ports']:3d}  **未形成配对**")
    if out_json:
        json.dump(rows, open(out_json, "w"), indent=2, ensure_ascii=False)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "source"
    if which == "source":
        GT = np.loadtxt(os.path.join(BASE, "coordinates.txt"))
        run_batch("source.pcd（官方真值 coordinates.txt）", [os.path.join(BASE, "source.pcd")],
                  gt=GT, out_json=r"D:\k6data\batch_source.json")
    elif which == "real":
        fs = [os.path.join(BASE, "实车点云", f"ICP_input{i}.pcd") for i in range(10)]
        run_batch("实车点云 ICP_input0-9（无官方真值，看鲁棒性/间距一致性）", fs,
                  out_json=r"D:\k6data\batch_real.json")
    elif which == "all":
        GT = np.loadtxt(os.path.join(BASE, "coordinates.txt"))
        run_batch("source.pcd（官方真值）", [os.path.join(BASE, "source.pcd")], gt=GT)
        print()
        fs = [os.path.join(BASE, "实车点云", f"ICP_input{i}.pcd") for i in range(10)]
        run_batch("实车点云 ICP_input0-9", fs, out_json=r"D:\k6data\batch_real.json")
