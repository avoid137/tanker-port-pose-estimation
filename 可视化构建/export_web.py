# -*- coding: utf-8 -*-
"""
课题六 · Web 可视化数据导出
================================================================
把 11 个场景（source + 实车 10 帧）的点云降采样 + int16 量化 + base64，
连同 result_final.json 里的检测结果（圆心/法向/半径/误差）一起打包成
一个紧凑 JSON，供单文件 HTML（Three.js）实时渲染。

坐标处理：每场景以 bbox 中点为原点，int16 存毫米偏移（量化误差 ±0.5mm）。
输出：D:\\k6data\\viewer_data.json
"""
import sys, os, json, base64, time
import numpy as np

sys.path.insert(0, r"D:\智能工程训练项目\课题六代码\V1")
from port_locator import read_pcd, voxelize

V1 = r"D:\智能工程训练项目\课题六代码\V1"
BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"

VOX_TARGET = 12.0      # 目标体素
NPT_CAP = 32000        # 单场景点数上限

RES = json.load(open(os.path.join(V1, "result_final.json"), encoding="utf-8"))
DETAIL = RES["results"]
GT_REAL = json.load(open(os.path.join(V1, "gt_manual.json"), encoding="utf-8"))
GT_NORM = json.load(open(os.path.join(V1, "cc_norm_manual.json"), encoding="utf-8"))
GT_OFFICIAL = np.loadtxt(os.path.join(BASE, "coordinates.txt"))


def encode_points(P):
    """int16 量化 + base64。返回 (b64, origin, n)"""
    lo = P.min(0)
    hi = P.max(0)
    origin = (lo + hi) / 2.0
    q = np.round(P - origin).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii"), origin.tolist(), len(P)


def pack_port(o, gt_c=None, gt_n=None):
    def fixsign(n):
        """SVD 求出的法向符号是任意的（对俯仰/偏航角无影响），
        可视化时统一朝 +z，避免箭头方向随机导致误读。"""
        n = np.array(n, float)
        return (-n if n[2] < 0 else n)
    d = {
        "center": [round(float(x), 2) for x in o["center"]],
        "normal": [round(float(x), 5) for x in fixsign(o["normal"])],
        "radius": round(float(o["radius"]), 1),
        "pitch": round(float(o["pitch"]), 2),
        "yaw": round(float(o["yaw"]), 2),
        "aniso": round(float(o.get("aniso", 1.0)), 3),
        "refined": bool(o.get("refined", False)),
    }
    if gt_c is not None:
        gc = np.array(gt_c, float)
        c = np.array(o["center"], float)
        d["gt_center"] = [round(float(x), 2) for x in gc]
        d["err_axis"] = round(float(np.max(np.abs(c - gc))), 2)
        d["err_3d"] = round(float(np.linalg.norm(c - gc)), 2)
    if gt_n is not None:
        d["gt_normal"] = [round(float(x), 5) for x in fixsign(gt_n)]
    return d


def build_scene(sid, label, kind, pcd_path, gt_ly_c, gt_ry_c, gt_ly_n, gt_ry_n, metrics):
    t0 = time.time()
    P = read_pcd(pcd_path)
    P = P[np.isfinite(P).all(1)]
    v = VOX_TARGET
    Q = voxelize(P, v)
    while len(Q) > NPT_CAP:                      # 超上限则放大体素
        v *= 1.25
        Q = voxelize(P, v)
    b64, origin, n = encode_points(Q)
    det = DETAIL.get(sid)
    sc = {
        "id": sid, "label": label, "kind": kind, "vox": round(v, 1),
        "n": n, "raw_n": int(len(P)), "origin": [round(x, 2) for x in origin],
        "bbox": [[round(float(x), 1) for x in Q.min(0)], [round(float(x), 1) for x in Q.max(0)]],
        "points": b64,
        "metrics": metrics,
        "others": [[round(float(c), 1) for c in o["center"]] + [round(float(o["radius"]), 1)]
                   for o in det["ports"]] if det else [],
    }
    if det and det.get("best"):
        b = det["best"]
        sc["ly"] = pack_port(b["ly"], gt_ly_c, gt_ly_n)
        sc["ry"] = pack_port(b["ry"], gt_ry_c, gt_ry_n)
    print("  %-12s v=%.0fmm n=%6d (raw %7d)  %.1fs" % (sid, v, n, len(P), time.time() - t0))
    return sc


def main():
    rows = {r["scene"]: r for r in RES["rows"]}
    scenes = []

    print("导出场景点云 ...")
    # 1) source
    metrics = dict(rows["source"])
    scenes.append(build_scene(
        "source", "样例场景 source（官方真值）", "source",
        os.path.join(BASE, "source.pcd"),
        GT_OFFICIAL[0], GT_OFFICIAL[1], None, None, metrics))

    # 2) 实车 10 帧
    for i in range(10):
        sid = "ICP_input%d" % i
        f = os.path.join(BASE, "实车点云", "%s.pcd" % sid)
        gt = GT_REAL[sid]
        gn = GT_NORM.get(sid, {})
        scenes.append(build_scene(
            sid, "实车 %s" % sid.replace("ICP_", ""), "real", f,
            gt["ly"], gt["ry"],
            (gn.get("ly") or {}).get("n"), (gn.get("ry") or {}).get("n"),
            dict(rows[sid])))

    data = {"scenes": scenes, "r_l": 59.0, "r_r": 79.0}
    out = r"D:\k6data\viewer_data.json"
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    sz = os.path.getsize(out) / 1e6
    print("已写出 %s  (%.2f MB)" % (out, sz))
    print("场景数 %d，总点数 %d" % (len(scenes), sum(s["n"] for s in scenes)))


if __name__ == "__main__":
    main()
