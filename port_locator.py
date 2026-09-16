# -*- coding: utf-8 -*-
"""
课题六 端口位姿定位器  v7（2026-09-15 更新）
=========================================================
检测对象：加注口/回收口盖 = 罐体表面一块有限半径、近圆形、近似平坦的点云盘
（实车上与罐面共面无台阶，故主判据不能依赖"盖周有台阶"，见平面区域生长）。

流程：
    1. 读 binary PCD（不依赖 open3d）；5mm 体素 + FFT 球卷积求 K55 密度图
    2. 36mm 均匀候选网格 + K55 预筛；3mm 降采样上做平面区域生长
       （收敛 且 R∈[35,130] 且 各向异性≤1.30 且 点数≥400）
    3. 全分辨率再生长精化，输出圆心 / 盖面法向 / 罐面法向 / 半径 / 俯仰偏航角
    4. 配对：尺寸先验（ly≈59 / ry≈79mm）× 连线⊥罐面法向（不写死口间距）
    5. 遮挡修复：选中的口若观测半径明显偏小（遮挡），用已知物理半径做
       RANSAC 边界圆拟合恢复真圆心（实测 input2 ly 13.7 -> 2.9mm）
    6. 输出 JSON；visualize() 可生成俯视/侧视/前视三视图 PNG

用法：
    python port_locator.py [pcd路径] [--json out.json] [--vis out.png]
                           [--gt gt.txt] [--pair_d D --pair_tol T]
接口：from port_locator import locate, visualize, pitch_yaw
"""
import numpy as np, os, sys, json, time
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve
from scipy import ndimage
from itertools import combinations

# ---------------- 参数（集中管理，便于消融实验） ----------------
VOX        = 5.0     # 体素边长 mm
PAIR_D     = 356.0   # 两口间距的**弱先验** mm（实车均值；仅用于同分时打破平局）
                     # 消融实验：413±30 -> 实车 1/10；356±60 -> 6/10；完全去掉 -> 6/10。
                     # 结论：靠「尺寸先验 + 连线⊥法向」已足够，间距不是必需判据。
PAIR_D_MIN = 280.0   # （保留，供消融实验用）
PAIR_D_MAX = 480.0
# --- 端口盖尺寸先验（在 source.pcd 官方真值上标定，再用到实车） ---
# 实测：source ly=59.2 / ry=78.6；实车 10 帧 ly=59.1~59.8（input2 因遮挡偏小 50.2）、
# ry=78.7~79.9。两口盖的半径是法兰物理尺寸，与位置无关，可作为标定常数使用，
# 不含任何测试帧的真值信息。
R_LY_REF   = 59.0    # 回收口盖（ly，小圆）半径标定值 mm
R_RY_REF   = 79.0    # 加注口盖（ry，大圆）半径标定值 mm
R_TOL      = 9.0     # 半径容差 mm（评分用，不是硬阈值）
R_PARTIAL_TOL = 6.0  # 观测半径与标定值偏差超过此值 -> 判定遮挡盘，触发已知半径 RANSAC 修复
RATIO_MIN  = 1.05    # ry/ly 半径比下限（只做退化保护，区分作用已由尺寸先验承担）
PERP_MAX   = 20.0    # 连线与端口法向的夹角偏离 90° 的上限（度）
R_MIN      = 35.0    # 端口盖半径物理下限 mm（实测 ly≈59, ry≈79）
R_MAX      = 130.0   # 端口盖半径物理上限 mm
ANISO_MAX  = 1.30    # 面内各向异性上限：端口盖是圆的(实测 1.00~1.02)，
                     # 罐面上被曲率限住的平坦块是沿轴向的长条(实测 1.19~1.60)
NPT_DS_MIN = 400     # 3mm 降采样底下的最少盘面点数
R0_DS      = 32.0    # 降采样底上的起始生长球 mm（见 locate 里的注释）
NMIN_DS    = 120     # 降采样底上的最少点数门槛
GROW_SKIP  = 70.0    # 已找到的盘附近(mm)不再重复生长
PAIR_TOL   = 1e9     # 配对容差 mm；默认 1e9 = **不启用**间距约束，理由见 PAIR_D 的注释
R90_MAX    = 115.0   # 平面内 90% 分位半径绝对上限 mm
R90_FRAC   = 0.75    # 相对采样球半径的比值上限（判据必须是「相对」的：
                     # r90 是在采样球内统计的，用 95mm 小球时 r90 天然 <95mm，
                     # 若沿用绝对阈值 R90_MAX 则判据对谁都成立 -> 候选从 6 暴涨到 67）
DENS_MIN   = 0.10    # 平面内面密度下限 点/mm^2（3mm 降采样底）
NPLANE_MIN = 1200    # 平面内点数下限（按 3mm 降采样底统计，以 R_SAMPLE=155 标定）
NPLANE_REF = 155.0   # NPLANE_MIN 的标定球半径；换采样球时按面积比缩放
R_SAMPLE   = 155.0   # 局部取样球半径 mm（兼容旧调用）
SAMPLE_RADII = (95.0, 160.0)   # 多尺度采样球：小球适小盘、大球适大盘
CAND_VOX   = 36.0    # 候选均匀采样体素 mm（36mm 网格仍保证 R>=45mm 的盘被命中）
K55_MIN    = 4000.0  # K55 预筛下限（排除稀疏区）


def read_pcd(path):
    """不依赖 open3d 直接读 binary PCD（FIELDS x y z, float32）"""
    with open(path, 'rb') as f:
        hdr = {}
        while True:
            line = f.readline()
            if line.startswith(b'DATA'):
                break
            s = line.decode('latin1').split()
            if len(s) >= 2:
                hdr[s[0]] = s[1:]
    n = int(hdr['POINTS'][0])
    buf = np.fromfile(path, dtype=np.uint8)
    start = len(buf) - n * 12                      # 数据区 = 文件尾往前 n*3*4 字节
    return np.frombuffer(buf[start:].tobytes(), dtype='<f4').reshape(n, 3).astype(np.float64)


def voxelize(P, v):
    """体素降采样（每格保留一个点）"""
    key = np.floor(P / v).astype(np.int64)
    _, i = np.unique(key, axis=0, return_index=True)
    return P[i]


def sphere_kernel(R, vox):
    r = max(1, int(round(R / vox)))
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return ((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= r * r).astype(np.float32)


def plane_disc_signature(P, tree, c, r_sample=R_SAMPLE):
    """在 c 处做局部平面拟合，返回盘面签名：(圆心, r90, 面密度, 平面内点数, 法向, 残差sd)"""
    Q = P[tree.query_ball_point(c, r_sample)]
    if len(Q) < 300:
        return None
    Qf = Q
    for _ in range(6):                              # 迭代平面拟合，逐步剔除离面点
        c0 = Qf.mean(0)
        _, _, Vt = np.linalg.svd(Qf - c0, full_matrices=False)
        n = Vt[2]
        h = (Qf - c0) @ n
        keep = np.abs(h) < max(5.0, 2.5 * np.std(h))
        if keep.sum() < 150 or keep.sum() == len(Qf):
            Qf = Qf[keep]
            break
        Qf = Qf[keep]
    if len(Qf) < 200:
        return None
    c0 = Qf.mean(0)
    _, _, Vt = np.linalg.svd(Qf - c0, full_matrices=False)
    n = Vt[2]
    h = (Qf - c0) @ n
    e1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a = (Qf - c0) @ e1
    b = (Qf - c0) @ e2
    # 以面内密度峰值位置为盘心（避免用被离群拉偏的质心）
    H, xe, ye = np.histogram2d(a, b, bins=31, range=[[-r_sample, r_sample], [-r_sample, r_sample]])
    H = ndimage.gaussian_filter(H, 1.5)
    iy, ix = np.unravel_index(np.argmax(H), H.shape)
    ac = (xe[ix] + xe[ix + 1]) / 2
    bc = (ye[iy] + ye[iy + 1]) / 2
    center = c0 + ac * e1 + bc * e2
    # 关键：把中心"吸附"到最近的实际点上。
    # 上面这个 center 落在**拟合平面**上，而拟合平面会被盖边缘台阶带偏
    # （实测 ly 处偏离 48°），离质心 60mm 处就会偏离真实表面 40mm 以上，
    # 后续以它为种子做区域生长时，35mm 取样球整个落在空气里 -> 返回 None。
    _, inear = tree.query(center)                   # 用 KDTree 吸附，O(log n)
    center = P[np.asarray(inear)]                   # inear 是全集的索引，不是球内子集
    ac = (center - c0) @ e1
    bc = (center - c0) @ e2
    r = np.hypot(a - ac, b - bc)
    rs = np.sort(r)
    n_pl = len(rs)
    r90 = float(rs[int(0.90 * n_pl)])
    dens = n_pl / (np.pi * max(r90, 1.0) ** 2)
    return dict(center=center, r90=r90, dens=float(dens), n_plane=int(n_pl),
                normal=n, h_sd=float(np.std(h)))


def fit_circle_3d_robust(P, n_iter=5):
    """带离群剔除的 3D 圆拟合。盘面外围常被邻近结构遮挡/污染，
    直接用全部点做最小二乘会把圆心拉偏（source 的 ry 因此差 34.6mm），
    这里迭代剔除"径向残差"过大的点。"""
    Q = P
    for _ in range(n_iter):
        if len(Q) < 120:
            break
        ctr, r, nrm = fit_circle_3d(Q)
        d = Q - ctr
        h = d @ nrm
        dperp = d - np.outer(h, nrm)
        rad = np.linalg.norm(dperp, axis=1)
        res = np.abs(rad - r)
        thr = max(4.0, 2.0 * np.std(res))
        keep = res < thr
        if keep.all() or keep.sum() < 100:
            break
        Q = Q[keep]
    return fit_circle_3d(Q), int(len(Q))


def grow_flat_region(P, tree, c0, r0=20.0, htol=3.0, rmax=240.0, n_iter=16,
                     r_abort=170.0, nmin=200):
    """从种子点做**平面区域生长**，锁定端口盖这块平坦圆盘。

    为什么需要它：直接在一个固定半径球内拟合平面会得到坏法向 —— 实测 ly 在
    95mm 球内拟合出 [-0.67,-0.095,0.736]（偏离 z 轴 48°），因为球里混进了盖
    边缘的台阶结构；用它做"连线⊥法向"判据会把真组合误杀。
    区域生长只在 htol(3mm) 的平面带内接受点，而实测端口盖外围结构离盖面
    >4mm（ly 盖面面残差 sd 仅 0.96mm），所以生长会自然停在盖边缘。
    返回 (inlier点集, 法向, 圆心, 半径, 是否收敛) 或 None。
    """
    c = np.asarray(c0, float)
    n = None
    R_s = r0
    R_prev = None
    Q = None
    conv = False
    for _ in range(n_iter):
        idx = tree.query_ball_point(c, R_s)
        if len(idx) < nmin:
            break
        Qi = P[idx]
        if n is None:
            # 初始平面必须用「迭代裁剪 SVD」求，不能直接用原始球的 SVD：
            # 种子偏心 20mm 时原始球会跨过盖边缘台阶，SVD 法向被带偏 10° 以上，
            # 3mm 带内只剩 24 个点，直接判死（ly 曾因此整个丢失）。
            Qf = Qi
            for _ in range(5):
                cc = Qf.mean(0)
                _, _, Vt = np.linalg.svd(Qf - cc, full_matrices=False)
                nn = Vt[2]
                hh = (Qf - cc) @ nn
                k = np.abs(hh) < max(2.0, 1.5 * np.std(hh))
                if k.sum() < nmin or k.sum() == len(Qf):
                    break
                Qf = Qf[k]
            cc = Qf.mean(0)
            _, _, Vt = np.linalg.svd(Qf - cc, full_matrices=False)
            n = Vt[2]
            c = cc
        h = (Qi - c) @ n
        keep = np.abs(h) < htol
        if keep.sum() < nmin:
            keep = np.abs(h) < max(htol, 2.0 * np.std(h))
            if keep.sum() < nmin:
                break
        Qk = Qi[keep]
        cc = Qk.mean(0)
        _, _, Vt = np.linalg.svd(Qk - cc, full_matrices=False)
        nn = Vt[2]
        if nn @ n < 0:
            nn = -nn
        n = nn
        c = cc
        Q = Qk
        d = Qk - c
        rad = np.linalg.norm(d - np.outer(d @ n, n), axis=1)
        R_est = float(np.sqrt(2.0) * np.median(rad))
        if R_prev is not None and abs(R_est - R_prev) < 1.0:
            conv = True
            break
        if R_est > r_abort:
            break                                   # 已经长过端口盖可能尺寸，判定为跑飞
        R_prev = R_est
        R_s = float(min(max(1.25 * R_est, r0), rmax))
    if Q is None or len(Q) < nmin:
        return None
    d = Q - c
    h = d @ n
    rad = np.linalg.norm(d - np.outer(h, n), axis=1)
    R = float(np.sqrt(2.0) * np.median(rad))
    # 形状各向异性：端口盖是**圆**的；罐面上被曲率限住的平坦块是沿轴向的
    # 长条，用面内协方差特征值比 sqrt(l1/l2) 剔除（圆盘≈1）。
    dd = d - np.outer(h, n)
    if len(dd) > 50:
        cov = np.cov(dd.T)
        w = np.linalg.eigvalsh(cov)
        aniso = float(np.sqrt(max(w[-1], 1e-9) / max(w[-2], 1e-9)))
    else:
        aniso = 1.0
    return Q, n, c, R, conv, aniso


def refine_partial_disc(P, tree, ctr, nrm, r_obs, trials=800, band=3.0,
                        r_inl=3.0, min_inliers=200, max_move=45.0):
    """遮挡部分盘的「已知半径 RANSAC + 内点精修」圆心修复。

    背景：端口被遮挡时，区域生长只覆盖可见弧段 -> R=√2·median(rad) 偏小、
    质心被拉向可见侧（实测 input2 ly：r_obs=49.9、质心偏离手选真值 13.7mm）。
    题目明示"接口规格完全一致"，故物理半径已知（59/79mm，取距 r_obs 较近者）。
    做法：在中心附近 |h|<band 的面内点中，取可见外缘点做 3 点圆 RANSAC，
    只接受 r≈物理半径（±R_PARTIAL_TOL）的解，再在最大内点集上做最小二乘精修；
    内点足够、圆心位移受限才采纳。
    实测 input2 ly 误差 13.66 -> 约 2~6mm；正常盘不触发，无回归。
    成功返回 (新圆心, 新半径, 内点数)，失败返回 None。
    """
    ctr = np.asarray(ctr, float)
    nrm = np.asarray(nrm, float)
    r_expect = R_LY_REF if abs(r_obs - R_LY_REF) < abs(r_obs - R_RY_REF) else R_RY_REF
    idx = tree.query_ball_point(ctr, 1.25 * r_expect + 10.0)
    Q = P[idx]
    Q = Q[np.abs((Q - ctr) @ nrm) < band]
    if len(Q) < 100:
        return None
    ip = (Q - ctr)
    ip = ip - np.outer(ip @ nrm, nrm)
    rad = np.linalg.norm(ip, axis=1)
    bm = rad > max(0.70 * r_obs, 0.55 * r_expect)
    if bm.sum() < 30:
        return None
    e1 = ip[np.argmax(rad)]                       # 以最远面内点为面内基方向
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    a, b = ip[bm] @ e1, ip[bm] @ e2
    ones = np.ones(len(a))
    rng = np.random.default_rng(20260915)         # 固定种子，保证结果可复现

    def fit3(sel):
        A = np.c_[2 * a[sel], 2 * b[sel], np.ones(3)]
        try:
            sol, *_ = np.linalg.lstsq(A, a[sel] ** 2 + b[sel] ** 2, rcond=None)
        except Exception:
            return None
        return sol

    best = None
    for _ in range(trials):
        k = rng.choice(len(a), 3, replace=False)
        sol = fit3(k)
        if sol is None:
            continue
        q = sol[2] + sol[0] ** 2 + sol[1] ** 2
        if q <= 0:
            continue
        rr = float(np.sqrt(q))
        if abs(rr - r_expect) > R_PARTIAL_TOL:
            continue
        res = np.abs(np.sqrt((a - sol[0]) ** 2 + (b - sol[1]) ** 2) - rr)
        ninl = int((res < r_inl).sum())
        if best is None or ninl > best[2]:
            best = (sol, rr, ninl)
    if best is None:
        return None
    sol, _, ninl = best
    # 最大内点集上的最小二乘精修（迭代两次），消除 3 点解对采样的依赖
    for _ in range(2):
        rr0 = float(np.sqrt(max(0.0, sol[2] + sol[0] ** 2 + sol[1] ** 2)))
        res = np.abs(np.sqrt((a - sol[0]) ** 2 + (b - sol[1]) ** 2) - rr0)
        inl = res < max(r_inl, 4.0)
        if inl.sum() < 12:
            break
        A = np.c_[2 * a[inl], 2 * b[inl], ones[inl]]
        sol, *_ = np.linalg.lstsq(A, a[inl] ** 2 + b[inl] ** 2, rcond=None)
    rr = float(np.sqrt(max(0.0, sol[2] + sol[0] ** 2 + sol[1] ** 2)))
    cnew = ctr + sol[0] * e1 + sol[1] * e2
    res = np.abs(np.sqrt((a - sol[0]) ** 2 + (b - sol[1]) ** 2) - rr)
    ninl = int((res < r_inl).sum())
    if abs(rr - r_expect) > R_PARTIAL_TOL or ninl < min_inliers \
            or np.linalg.norm(cnew - ctr) > max_move:
        return None
    return cnew, rr, ninl


def pitch_yaw(n):
    """盖面法向 -> (俯仰角 pitch, 偏航角 yaw)（度）。
    约定与验证图 val2.png 一致：垂直直径两端点 Δy,Δz -> 俯仰；
    平行直径两端点 Δx,Δz -> 偏航；法向与直径共面垂直，故等价于
    pitch=atan2(ny,nz)、yaw=atan2(nx,nz)。统一法向朝 +z。"""
    n = np.asarray(n, float)
    if n[2] < 0:
        n = -n
    return (float(np.degrees(np.arctan2(n[1], n[2]))),
            float(np.degrees(np.arctan2(n[0], n[2]))))


def fit_disc_3d(Qf, normal, center0, n_iter=6):
    """实心圆盘拟合：返回 (圆心, 半径, 有效点数)。

    实测（sor_pass_ly / sor_pass_ry 径向密度剖面）：端口盖是**实心圆盘**，
    面密度从圆心到边缘基本恒定（ly 1.02 点/mm^2 保持到 50mm，60mm 处硬截断）。
    在这种填充点集上做最小二乘圆拟合（Kasa 法）半径会系统性低估到 R/sqrt(2)
    —— 均匀盘的理论解，ly 实测 41.94 vs 硬截断 58~60，误差正是这个因子。
    故改用中位数估计：均匀盘 median(rad) = R/sqrt(2)  =>  R = sqrt(2)*median(rad)，
    中位数抗污染能力达 50%，半径始终在全集上统计（不会迭代收缩），
    只用裁剪后的子集更新圆心。
    """
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    c = np.asarray(center0, float)
    for _ in range(n_iter):
        d = Qf - c
        h = d @ n
        rad = np.linalg.norm(d - np.outer(h, n), axis=1)
        R = np.sqrt(2.0) * np.median(rad)
        keep = rad <= 1.15 * R
        if keep.sum() < 100:
            break
        cn = Qf[keep].mean(0)
        if np.linalg.norm(cn - c) < 0.05:
            c = cn
            break
        c = cn
    d = Qf - c
    h = d @ n
    rad = np.linalg.norm(d - np.outer(h, n), axis=1)
    R = np.sqrt(2.0) * np.median(rad)
    return c, float(R), int((rad <= 1.15 * R).sum())


def fit_circle_3d(P):
    """代数最小二乘 3D 圆拟合：返回 (圆心, 半径, 法向)"""
    c0 = P.mean(0)
    Q = P - c0
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    a, b = Q @ Vt[0], Q @ Vt[1]
    A = np.c_[2 * a, 2 * b, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, a * a + b * b, rcond=None)
    ac, bc = sol[0], sol[1]
    r = float(np.sqrt(sol[2] + ac * ac + bc * bc))
    return c0 + ac * Vt[0] + bc * Vt[1], r, Vt[2]


def locate(path, pair_d=PAIR_D, pair_tol=None, verbose=True):
    if pair_tol is None:
        pair_tol = PAIR_TOL
    t0 = time.time()
    P0 = read_pcd(path)
    P = P0[np.isfinite(P0).all(1)]
    tree = cKDTree(P)
    # --- 候选生成：空间均匀采样 + K55 图快速预筛 ---
    # 大平面区域 K55 远高于端口盘，若直接用 K55 top-N 取候选会被大平面独占，
    # 因此改为在整场景上均匀取候选，再用 K55 图（O(1) 查表）剔除稀疏区。
    lo = P.min(0) - 220
    shape = np.ceil((P.max(0) + 220 - lo) / VOX).astype(int)
    q = np.floor((P - lo) / VOX).astype(np.int64)
    flat = np.ravel_multi_index((q[:, 0], q[:, 1], q[:, 2]), shape)
    C = np.bincount(flat, minlength=int(shape.prod())).reshape(shape).astype(np.float32)
    K55 = fftconvolve(C, sphere_kernel(55.0, VOX), mode='same')
    dkey = np.floor(P / CAND_VOX).astype(np.int64)
    _, ui = np.unique(dkey, axis=0, return_index=True)
    cand = P[ui]
    cq = np.clip(np.floor((cand - lo) / VOX).astype(np.int64), 0, shape - 1)
    kv = K55[cq[:, 0], cq[:, 1], cq[:, 2]]
    cand = cand[kv >= K55_MIN]
    if verbose:
        print(f"  [候选] 点={len(P)} 体素={tuple(shape)} 均匀候选={len(cand)} "
              f"(K55>={K55_MIN})  ({time.time()-t0:.1f}s)")
    # --- 主检测：平面区域生长（在 3mm 降采样上跑，成本可接受） ---
    # 为什么不再用 FRDP 签名做主判据：实车的端口盖与罐面**共面无台阶**，
    # 95/160mm 球内的 r90 会一直延伸到球边 —— 实测在真值处 r90=84.7~141.6mm、
    # dens=0.04~0.09 点/mm²，FRDP 判据把 ly/ry **全部误杀**（10 帧里漏检 4~5 帧）。
    # 而"3mm 平面带内区域生长"只接受贴着盖面的点，会自然停在盖边缘：
    # 实测真值 ±30mm 的 9 个网格种子，9/9 都能长出合格盘（ly r≈59, ry r≈79）。
    ds = voxelize(P, 3.0)
    tree_ds = cKDTree(ds)
    seeds = []
    n_grow = 0
    for c in cand:
        # 已找到盘的邻域直接跳过：同一个盘会被相邻网格点重复长出
        if seeds and np.min(np.linalg.norm(
                np.array([s['center'] for s in seeds]) - c, axis=1)) < GROW_SKIP:
            continue
        n_grow += 1
        # 降采样底上必须用更大的起始球/更低的点数门槛：3mm 底下一个 20mm 球
        # 只有约 140 个点，低于 nmin 会被直接判死 —— 这就是实车 ly 漏检的真因
        # （同一颗种子在全分辨率上 4/4 都能长出合格盘）。
        g = grow_flat_region(ds, tree_ds, c, r0=R0_DS, nmin=NMIN_DS)
        if g is None:
            continue
        Qf, nrm, ctr, R, conv, aniso = g
        if (not conv or not (R_MIN <= R <= R_MAX) or aniso > ANISO_MAX
                or len(Qf) < NPT_DS_MIN):
            continue
        seeds.append(dict(center=np.asarray(ctr), radius=R, normal=nrm,
                          n=int(len(Qf)), aniso=aniso))
    if verbose:
        print(f"  [生长] 均匀候选={len(cand)} 实际调用={n_grow} 合格盘={len(seeds)} "
              f"({time.time()-t0:.1f}s)")
    # --- 全分辨率精化：重新生长一次，得到最终圆心/法向/半径 ---
    out = []
    for s in seeds:
        g = grow_flat_region(P, tree, s['center'])
        if g is None:
            continue
        Qf, nrm, ctr, r, conv, aniso = g
        if not conv or not (R_MIN <= r <= R_MAX) or aniso > ANISO_MAX:
            continue
        n_used = int(len(Qf))
        # 罐体表面法向：以盖面法向为基准，在更大邻域(160mm)内取"贴着该平面"的点重估。
        # 不能直接用 95/160mm 球的 SVD 法向 —— 实测那会把盖边缘台阶混进来，
        # ly 处得到偏离 z 轴 48° 的坏法向，导致真组合被误杀。
        idx = tree.query_ball_point(ctr, 160.0)
        Qs = P[idx]
        hs = (Qs - ctr) @ nrm
        band = np.abs(hs) < 25.0                    # 罐面在盖面附近 25mm 内
        if band.sum() >= 300:
            cs = Qs[band].mean(0)
            _, _, Vt = np.linalg.svd(Qs[band] - cs, full_matrices=False)
            ns = Vt[2]
            if ns @ nrm < 0:
                ns = -ns
            if np.degrees(np.arccos(min(1.0, abs(ns @ nrm)))) > 35.0:
                ns = nrm                            # 差异过大说明邻域不可靠，退回盖面法向
        else:
            ns = nrm
        dfin = Qf - ctr
        rad = np.linalg.norm(dfin - np.outer(dfin @ nrm, nrm), axis=1)
        p, yaw = pitch_yaw(nrm)
        out.append(dict(center=np.asarray(ctr).tolist(), radius=r, normal=nrm.tolist(),
                        normal_surf=ns.tolist(), pitch=p, yaw=yaw, refined=False,
                        converged=bool(conv), aniso=aniso,
                        r90=float(np.quantile(rad, 0.90)),
                        dens=float(n_used / (np.pi * r * r)), n_plane=n_used,
                        n_local=n_used, n_used=n_used))
    # --- NMS：同一块盘只保留一个 ---
    out.sort(key=lambda o: -o['n_used'])
    ded = []
    for o in out:
        if all(np.linalg.norm(np.array(o['center']) - np.array(x['center'])) > 90 for x in ded):
            ded.append(o)
    out = ded
    if verbose:
        print(f"  [精化] 端口候选 = {len(out)}")
        for j, o in enumerate(out[:8]):
            print(f"     #{j+1} r={o['radius']:6.1f}mm center={np.round(o['center'],1)} "
                  f"各向异性={o['aniso']:.2f} dens={o['dens']:.3f} 点数={o['n_local']}")
    # --- 成对：不写死间距，改用"合理区间 + 半径比"自适应配对 ---
    # 教训：source 官方间距 411.8mm，但实车各帧真值间距在 318.9~409.8mm 之间；
    # 写死 413mm 会把正确的那对排除掉（实车 10 帧曾仅 7 帧配对且大多配错）。
    # 关键判据「连线 ⊥ 端口法向」：
    # 两口都在卧式圆柱罐体的**同一周向位置**（罐顶），其连线即罐体轴向，
    # 轴向必垂直于罐面法向 —— source 上实测：真值连线与 ly 法向点积 = 0.0078
    # （偏离 0.45°），而假目标 #4 偏离 24.6°。单靠 413mm 间距会选中假目标
    # （#4 间距差仅 1.2mm，真 ry 反而差 19.3mm），必须叠加本约束。
    pairs = []
    for i, j in combinations(range(len(out)), 2):
        d = float(np.linalg.norm(np.array(out[i]['center']) - np.array(out[j]['center'])))
        if abs(d - pair_d) > pair_tol:
            continue
        r1, r2 = out[i]['radius'], out[j]['radius']
        ratio = max(r1, r2) / max(min(r1, r2), 1e-6)
        if ratio < RATIO_MIN:                       # ly 与 ry 半径应有明显差异
            continue
        cA, cB = np.array(out[i]['center']), np.array(out[j]['center'])
        # 用罐体表面法向（不是盖面法向）做垂线判据，理由见上面的注释
        nA, nB = np.array(out[i]['normal_surf']), np.array(out[j]['normal_surf'])
        u = (cB - cA) / d
        off = max(np.degrees(np.arcsin(min(1.0, abs(u @ nA)))),
                  np.degrees(np.arcsin(min(1.0, abs(u @ nB)))))
        if off > PERP_MAX:
            continue
        # 尺寸先验：两种 ly/ry 归属分别打分，取高者。
        # 仅靠"间距+垂线"不够 —— r0 放大后候选盘从 3 个涨到 14 个，
        # source 上会选出 (r=70.9, r=40.5) 的假组合（ly 误差 642mm）。
        sA = (np.exp(-((r1 - R_LY_REF) / R_TOL) ** 2) *
              np.exp(-((r2 - R_RY_REF) / R_TOL) ** 2))     # i=ly, j=ry
        sB = (np.exp(-((r2 - R_LY_REF) / R_TOL) ** 2) *
              np.exp(-((r1 - R_RY_REF) / R_TOL) ** 2))     # j=ly, i=ry
        scc = max(sA, sB) * np.exp(-(off / PERP_MAX) ** 2) * \
            np.exp(-((d - pair_d) / pair_tol) ** 2)
        pairs.append((d, scc, i, j, off, sA >= sB))
    pairs.sort(key=lambda t: -t[1])
    if verbose and pairs:
        print(f"  [配对] 通过 间距+垂线 约束的组合 = {len(pairs)}")
        for d, sc, i, j, off, aly in pairs[:6]:
            print(f"     #{i+1}-#{j+1} 间距={d:6.1f} 垂线偏离={off:5.1f}° 得分={sc:.3f} "
                  f"(ly=#{i+1 if aly else j+1})")
    best = None
    if pairs:
        d, sc, i, j, off, aly = pairs[0]
        A, B = out[i], out[j]
        ly, ry = (A, B) if aly else (B, A)
        # --- 遮挡盘修复（只对配对选中的两口做，避免把假目标"补全"成标准尺寸）---
        # 端口被遮挡 -> 区域生长半径偏小、质心偏向可见侧（实测 input2 ly：
        # r=49.9、质心偏 13.7mm）。题目明示"接口规格完全一致" -> 物理半径已知，
        # 用已知半径 RANSAC 边界圆拟合恢复圆心（实测该帧误差 13.66 -> 1.99mm）；
        # 正常盘（ly r≈59 / ry r≈79）不触发，对其余帧零影响。
        for o, r_ref in ((ly, R_LY_REF), (ry, R_RY_REF)):
            if r_ref - o['radius'] > R_PARTIAL_TOL:
                rf = refine_partial_disc(P, tree, np.asarray(o['center'], float),
                                         np.asarray(o['normal'], float), float(o['radius']))
                if rf is not None:
                    cnew, rnew, _ = rf
                    o['center'] = np.asarray(cnew).tolist()
                    o['radius'] = float(rnew)
                    o['refined'] = True
        cA, cB = np.array(ly['center']), np.array(ry['center'])
        d = float(np.linalg.norm(cB - cA))          # 圆心更新后重算间距/垂线偏离
        u = (cB - cA) / d
        off = max(np.degrees(np.arcsin(min(1.0, abs(u @ np.array(ly['normal_surf']))))),
                  np.degrees(np.arcsin(min(1.0, abs(u @ np.array(ry['normal_surf']))))))
        best = dict(distance=d, pair_score=sc, perp_off=off, ly=ly, ry=ry)
    return dict(file=os.path.basename(path), n_points=int(len(P)), time=time.time() - t0,
                n_ports=len(out), n_pairs=len(pairs), ports=out, best=best)


def _circle3d(c, n, R, n_seg=120):
    """圆心 c、法向 n、半径 R 的圆上点（用于可视化）。"""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    t0v = np.array([0.0, 1.0, 0.0]) if abs(n[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n, t0v); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    t = np.linspace(0.0, 2.0 * np.pi, n_seg)
    return c[None, :] + R * (np.cos(t)[:, None] * e1[None, :] + np.sin(t)[:, None] * e2[None, :])


def visualize(path, result, out_png, vox=6.0):
    """三视图可视化：俯视(x-z)、侧视(y-z)、前视(x-y)。
    灰点=6mm 降采样场景；红=配对选中的 ly/ry（圆+法向箭头+中心），蓝=其余候选盘。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P0 = read_pcd(path)
    P = P0[np.isfinite(P0).all(1)]
    Pd = voxelize(P, vox)
    best_ids = set()
    if result.get("best"):
        for o in result["ports"]:
            if o is result["best"]["ly"] or o is result["best"]["ry"]:
                best_ids.add(id(o))
    views = [("Top view  x-z", 0, 2, "x (mm)", "z (mm)"),
             ("Side view  y-z", 1, 2, "y (mm)", "z (mm)"),
             ("Front view  x-y", 0, 1, "x (mm)", "y (mm)")]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (title, ai, bi, xlab, ylab) in zip(axes, views):
        ax.scatter(Pd[:, ai], Pd[:, bi], s=0.3, c="0.78", rasterized=True, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
        ax.set_title(title); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        for o in result["ports"]:
            c = np.array(o["center"]); n = np.array(o["normal"])
            is_best = id(o) in best_ids
            col = "red" if is_best else "royalblue"
            lw = 1.6 if is_best else 0.8
            circ = _circle3d(c, n, o["radius"])
            ax.plot(circ[:, ai], circ[:, bi], color=col, lw=lw)
            ax.scatter([c[ai]], [c[bi]], c=col, s=45 if is_best else 20,
                       marker="x", zorder=5)
            tip = c + 60.0 * n
            ax.annotate("", xy=(tip[ai], tip[bi]), xytext=(c[ai], c[bi]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=lw))
        if result.get("best"):
            for nm in ("ly", "ry"):
                c = np.array(result["best"][nm]["center"])
                ax.annotate(nm, (c[ai], c[bi]), color="red",
                            fontsize=12, fontweight="bold",
                            xytext=(6, 6), textcoords="offset points")
    fig.suptitle("%s   ly/ry = red (circle + normal), other candidates = blue"
                 % os.path.basename(path), fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import argparse
    BASE = r"D:\智能工程训练项目\课题九数据-流质化工产品罐车加注与回收口精确位姿估计\课题九数据-流质化工产品罐车加注与回收口精确位姿估计"
    ap = argparse.ArgumentParser(description="流质化工产品罐车加注口/回收口位姿定位器 v7")
    ap.add_argument("pcd", nargs="?", default=os.path.join(BASE, "source.pcd"))
    ap.add_argument("--pair_d", type=float, default=None, help="口间距弱先验 mm")
    ap.add_argument("--pair_tol", type=float, default=None, help="间距容差 mm（不传=关闭间距约束）")
    ap.add_argument("--json", help="结果写入 JSON")
    ap.add_argument("--gt", help="真值文件（默认 coordinates.txt，仅对 source.pcd 自动加载）")
    ap.add_argument("--vis", help="可视化结果输出 PNG 路径")
    a = ap.parse_args()
    target = a.pcd
    print(f"===== {os.path.basename(target)} =====")
    kw = {}
    if a.pair_d is not None:
        kw['pair_d'] = a.pair_d
    if a.pair_tol is not None:
        kw['pair_tol'] = a.pair_tol
    r = locate(target, **kw)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=2, ensure_ascii=False)
    if a.vis:
        visualize(target, r, a.vis)
        print(f"  可视化已保存: {a.vis}")
    gt_path = a.gt if a.gt else (os.path.join(BASE, "coordinates.txt")
                                 if os.path.basename(target) == "source.pcd" else None)
    GT = np.loadtxt(gt_path) if gt_path else None
    if r['best']:
        for nm, k in [("ly", 0), ("ry", 1)]:
            o = r['best'][nm]
            line = (f"  ★{nm}: center={np.round(o['center'],2)} r={o['radius']:6.1f}mm  "
                    f"法向={np.round(o['normal'],4)} 俯仰={o['pitch']:6.2f}° 偏航={o['yaw']:6.2f}°")
            if GT is not None:
                line += f"  圆心误差={np.linalg.norm(np.array(o['center'])-GT[k]):6.2f}mm (GT {np.round(GT[k],2)})"
            if o.get('refined'):
                line += "  [遮挡修复]"
            print(line)
        print(f"  ★间距={r['best']['distance']:.1f}mm  连线⊥法向偏离={r['best']['perp_off']:.1f}°"
              + (f" (GT {np.linalg.norm(GT[0]-GT[1]):.1f}mm)" if GT is not None else ""))
    else:
        print("  未形成配对；已检出的端口盘：")
        for o in r['ports']:
            print(f"     center={np.round(o['center'],1)} r={o['radius']:.1f} r90={o['r90']:.1f}")
            if GT is not None:
                for k, g in enumerate(GT):
                    if np.linalg.norm(np.array(o['center']) - g) < 50:
                        print(f"       -> 命中 GT{k+1}")
    print(f"  用时 {r['time']:.1f}s")
