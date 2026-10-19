# -*- coding: utf-8 -*-
"""纯 numpy/scipy 的统计核心：不依赖 statsmodels，可单独测试。"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize_scalar

import config as C


# ============================================================ 参数（敏感性分析可覆盖）
def params(**overrides) -> dict:
    keys = ["T1_PASS_AVAIL", "T1_MIN_HOSP_PCT", "T1_COND_AVAIL", "RELIABILITY_TARGET",
            "T3_CEILING_PCT", "T3_ZERO_PCT", "T3_MIN_EVENTS", "T3_PHI", "POOLING_HORIZON_Q"]
    p = {k: getattr(C, k) for k in keys}
    p["T2_POINT_ESTIMATE_RULE"] = False
    for k, v in overrides.items():
        if k not in p:
            raise KeyError(f"未知参数 {k}")
        p[k] = v
    return p


# ============================================================ Test 2：REML 随机截距 ICC
def _group_stats(gid, y):
    _, inv = np.unique(gid, return_inverse=True)
    ni = np.bincount(inv).astype(float)
    ybar = np.bincount(inv, weights=y) / ni
    ssw = float(((y - ybar[inv]) ** 2).sum())
    return ni, ybar, ssw


def icc_reml(gid, y) -> float:
    """单因素随机截距模型的 REML ICC。

    -2 logL_REML 对残差方差解析剖面化后只剩方差比 lambda = s2b/s2e，
    在 log(lambda) ∈ [-20, 20] 上一维搜索。最优解落在下界时返回 0。
    """
    y = np.asarray(y, float)
    ni, ybar, ssw = _group_stats(gid, y)
    a, N = len(ni), len(y)
    if a < 2 or N <= a:
        return float("nan")

    def neg2l(loglam):
        lam = math.exp(loglam)
        c = ni / (1.0 + ni * lam)
        sc = c.sum()
        mu = (c * ybar).sum() / sc
        tot = ssw + (c * (ybar - mu) ** 2).sum()
        if tot <= 0:
            return 1e18
        return (N - 1) * math.log(tot) + np.log1p(ni * lam).sum() + math.log(sc)

    r = minimize_scalar(neg2l, bounds=(-20.0, 20.0), method="bounded", options={"xatol": 1e-8})
    if r.x <= -19.9:
        return 0.0
    if r.x >= 19.9:
        return 1.0
    lam = math.exp(r.x)
    return lam / (1.0 + lam)


def bootstrap_icc(gid, y, B: int, rng: np.random.Generator) -> np.ndarray:
    """按机构整体重抽（cluster bootstrap），每次用同一个 REML 估计量重拟合。"""
    y = np.asarray(y, float)
    hosp, inv = np.unique(gid, return_inverse=True)
    nh = len(hosp)
    order = np.argsort(inv, kind="stable")
    sizes = np.bincount(inv)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, nh, size=nh)
        rows = np.concatenate([order[starts[p]:starts[p] + sizes[p]] for p in pick])
        newg = np.repeat(np.arange(nh), sizes[pick])
        out[b] = icc_reml(newg, y[rows])
    return out[np.isfinite(out)]


def sb_quarters(icc: float, target: float) -> float:
    """Spearman–Brown：达到信度 target 所需的季度数（未取整）。"""
    if icc is None or not np.isfinite(icc) or icc <= 0:
        return math.inf
    if icc >= 1:
        return 1.0
    return target * (1 - icc) / (icc * (1 - target))


def quarters_needed(icc, lo, hi, target):
    """返回 (点估计, 置信下限, 置信上限)，均向上取整；ICC 下限 <=0 时上限为 inf（不可识别）。"""
    def ce(v):
        return math.inf if not np.isfinite(v) else max(1, math.ceil(v - 1e-9))
    return ce(sb_quarters(icc, target)), ce(sb_quarters(hi, target)), ce(sb_quarters(lo, target))


# ============================================================ Test 3：过度离散
def spiegelhalter_phi(o, n, winsor=C.PHI_WINSOR) -> float:
    """机构层面合并计数的过度离散因子（Spiegelhalter 2005，z 值 10%/90% 缩尾）。"""
    o, n = np.asarray(o, float), np.asarray(n, float)
    keep = n > 0
    o, n = o[keep], n[keep]
    if len(o) < 3 or o.sum() <= 0:
        return float("nan")
    lam = o.sum() / n.sum()
    z = (o / n - lam) / np.sqrt(lam / n)
    lo, hi = np.quantile(z, winsor)
    return max(float((np.clip(z, lo, hi) ** 2).mean()), 1.0)


# ============================================================ 补充：容量校正信度
def dl_tau2(y, v) -> float:
    """DerSimonian–Laird 机构间方差（矩估计，按 1/v 加权），截断于 0。"""
    y, v = np.asarray(y, float), np.asarray(v, float)
    w = 1.0 / v
    ybar = (w * y).sum() / w.sum()
    q = (w * (y - ybar) ** 2).sum()
    denom = w.sum() - (w ** 2).sum() / w.sum()
    return max(0.0, (q - (len(y) - 1)) / denom) if denom > 0 else 0.0


def volume_reliability(o, n, kind: str):
    """每家医院合并估计的信度 r_i = tau2 / (tau2 + v_i)。

    event：v_i = lambda / n_i（泊松）；process：v_i = p(1-p) / n_i（二项）。
    v1.6 用未加权矩估计，容易被分母极小、率极端的医院主导；这里改为 DL 加权估计。
    """
    o, n = np.asarray(o, float), np.asarray(n, float)
    keep = n > 0
    o, n = o[keep], n[keep]
    if len(o) < C.MIN_HOSP or o.sum() <= 0:
        return None
    pooled = o.sum() / n.sum()
    if kind == "event":
        v = pooled / n
    else:
        if pooled >= 1:
            return None
        v = pooled * (1 - pooled) / n
    tau2 = dl_tau2(o / n, v)
    return tau2 / (tau2 + v)


# ============================================================ Test 1：精度阈值
def precision_threshold(pooled: float, kind: str) -> float:
    """event：在合并率下期望事件数 >= 1.96^2 所需分母；process：95% CI 半宽 <= 5 个百分点所需分母。"""
    if not np.isfinite(pooled) or pooled <= 0:
        return math.inf
    if kind == "event":
        return C.PRECISION_EXPECTED_EVENTS / pooled
    return (1.96 ** 2) * pooled * (1 - pooled) / C.PRECISION_HALF_WIDTH ** 2


# ============================================================ 判定规则
def test1_verdict(avail_pct, hosp_pct, p) -> str:
    if avail_pct < p["T1_COND_AVAIL"] or hosp_pct < p["T1_MIN_HOSP_PCT"]:
        return "Fail"
    if avail_pct < p["T1_PASS_AVAIL"]:
        return "Conditional"
    return "Pass"


def test2_verdict(icc, lo, p) -> str:
    if icc is None or not np.isfinite(icc):
        return "Not estimable"
    r = p["RELIABILITY_TARGET"]
    if p["T2_POINT_ESTIMATE_RULE"]:
        return "Pass" if icc >= r else "Fail"
    if np.isfinite(lo) and lo >= r:
        return "Pass"
    if icc >= r:
        return "Provisional"
    return "Fail"


def test3_verdict(zero_pct, ceiling_pct, events, phi, p):
    reasons = []
    if np.isfinite(ceiling_pct) and ceiling_pct > p["T3_CEILING_PCT"]:
        reasons.append("ceiling effect")
    if np.isfinite(zero_pct) and zero_pct > p["T3_ZERO_PCT"]:
        reasons.append("excess zeros")
    if events < p["T3_MIN_EVENTS"]:
        reasons.append("sparse events")
    if np.isfinite(phi) and phi > p["T3_PHI"]:
        reasons.append("extreme overdispersion")
    if not reasons:
        return "Pass", reasons
    if "ceiling effect" in reasons or ("sparse events" in reasons and "excess zeros" in reasons):
        return "Fail", reasons
    return "Conditional", reasons


DEFINITE = "Definitely usable for single-quarter comparison"
PROVISIONAL = "Provisionally usable for single-quarter comparison"
POOLING = "Usable only with pooling and/or adjustment"
NOT_USABLE = "Not usable as currently specified"


def overall_verdict(t1, t2, t3, t3_reasons, q_hi, p):
    """综合判定。

    v1.6 代码（p9b）有两处与正文规则不符：
      1) Test 2 = Pass 而 Test 3 = Conditional 时落到最后的 else，被判为 Not usable；
      2) Test 2 = Provisional 时无论所需季度上限多少都判为 pooling。
    这里按正文规则重写，并把“需要做什么”单独列出。
    """
    actions = []
    if t1 == "Fail" or t3 == "Fail" or t2 == "Not estimable":
        why = []
        if t1 == "Fail":
            why.append("Test 1 failed")
        if t2 == "Not estimable":
            why.append("reliability not estimable")
        if t3 == "Fail":
            why.append("Test 3 failed (" + ", ".join(t3_reasons) + ")")
        return NOT_USABLE, "; ".join(why)
    if t1 == "Pass" and t3 == "Pass" and t2 == "Pass":
        return DEFINITE, ""
    if t1 == "Pass" and t3 == "Pass" and t2 == "Provisional":
        return PROVISIONAL, "confirm reliability as further quarters accrue"
    if not (np.isfinite(q_hi) and q_hi <= p["POOLING_HORIZON_Q"]):
        qs = "not identified" if not np.isfinite(q_hi) else f"{q_hi}"
        return NOT_USABLE, f"quarters needed for R={p['RELIABILITY_TARGET']:.2f} (upper limit {qs}) exceed {p['POOLING_HORIZON_Q']}"
    if t2 != "Pass":
        actions.append(f"pool at least {q_hi} quarters")
    if "extreme overdispersion" in t3_reasons:
        actions.append("overdispersion-adjusted control limits")
    if "excess zeros" in t3_reasons or "sparse events" in t3_reasons:
        actions.append("pooled counts rather than quarterly rates")
    if t1 == "Conditional":
        actions.append("restrict to hospitals with analysable data and report availability")
    return POOLING, "; ".join(actions)


# ============================================================ 机构比较：EB 收缩与漏斗图
def eb_gamma_poisson(o, n):
    """Gamma–Poisson 经验贝叶斯收缩（矩估计，与 v1.6 相同）。返回 (收缩率, 合并率)。"""
    o, n = np.asarray(o, float), np.asarray(n, float)
    lam = o.sum() / n.sum()
    tau2 = max(np.var(o / n, ddof=1) - np.mean(lam / n), 1e-12)
    a, b = lam ** 2 / tau2, lam / tau2
    return (o + a) / (n + b), lam


def funnel_counts(o, n, phi):
    """未校正与过度离散校正后，95% 与 99.8% 控制限之外的机构数。"""
    o, n = np.asarray(o, float), np.asarray(n, float)
    lam = o.sum() / n.sum()
    z = (o / n - lam) / np.sqrt(lam / n)
    za = z / math.sqrt(phi)
    return {"n": int(len(o)),
            "outside95_unadj": int((np.abs(z) > 1.96).sum()),
            "outside998_unadj": int((np.abs(z) > 3.09).sum()),
            "outside95_adj": int((np.abs(za) > 1.96).sum()),
            "outside998_adj": int((np.abs(za) > 3.09).sum())}


def bh_fdr(pvals):
    """Benjamini–Hochberg q 值。"""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q
