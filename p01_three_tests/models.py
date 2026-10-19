# -*- coding: utf-8 -*-
"""示范分析与 REML 交叉核对（需要 statsmodels）。

与 v1.6 的差异：
- v1.6 的“负二项 GEE”用 sm.families.NegativeBinomial(alpha=1.0)，alpha 固定为 1、没有估计，
  它只是一种工作方差函数；加上 GEE 的稳健标准误，与 Poisson GEE 的差别主要来自加权方式。
- 现在主模型为 Poisson GEE（交换相关，稳健 SE）；敏感性模型为估计 alpha 的 NB2 极大似然，
  按机构聚类稳健 SE（与 robust_reanalysis.py 的做法一致）。
- 季度改为分类变量 C(quarter)，吸收各季度上报波次的整体变化。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

import config as CFG          # 不能叫 C：statsmodels 的公式在调用方命名空间里解析，
import core                   # 若这里用 C，公式中的 C(quarter) 会被解析成 config 模块
from data import IND

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    HAVE_SM = True
except ImportError:   # pragma: no cover
    HAVE_SM = False

STAFF_TERMS = ["z_staff_b", "z_staff_w"]


def _prep(main, key, device):
    d = IND[key]
    den = CFG.MODULES[d["module"]]["den"]
    cols = ["ID", "季度", d["num"], den, "tertiary", "staff_per100", "log_beds"] + ([device] if device else [])
    x = main.loc[main[f"ok_{key}"], cols].dropna().copy()
    x = x.rename(columns={d["num"]: "count", den: "den", "季度": "quarter"})
    if device:
        x = x.rename(columns={device: "dev"})
    terms = ["tertiary"]
    for src, nm in [("staff_per100", "staff"), ("log_beds", "beds")] + ([("dev", "dev")] if device else []):
        gm = x.groupby("ID")[src].transform("mean")
        for suf, v in [("_b", gm), ("_w", x[src] - gm)]:
            sd = v.std()
            x[f"z_{nm}{suf}"] = (v - v.mean()) / (sd if sd > 0 else 1.0)
            terms.append(f"z_{nm}{suf}")
    x["log_den"] = np.log(x["den"])
    return x.sort_values("ID"), terms


def _table(res, outcome, model, x, converged, alpha=np.nan):
    ci = res.conf_int()
    params = res.params.drop(labels=["alpha"], errors="ignore")
    slopes = params.drop(labels=["Intercept"], errors="ignore")
    slopes = slopes[[t for t in slopes.index if not t.startswith("C(quarter)")]]
    unstable = (not converged) or bool((slopes.abs() > CFG.DEMO_MAX_ABS_COEF).any()) or not np.all(np.isfinite(slopes))
    rows = []
    for t in slopes.index:
        rows.append({"Outcome": outcome, "Model": model, "Term": t, "IRR": float(np.exp(params[t])),
                     "CI lower": float(np.exp(ci.loc[t, 0])), "CI upper": float(np.exp(ci.loc[t, 1])),
                     "P": float(res.pvalues[t]), "Observations": len(x), "Hospitals": x["ID"].nunique(),
                     "Events": int(x["count"].sum()), "Converged": converged, "Stable": not unstable,
                     "NB alpha": alpha})
    return rows


def demonstration_models(main) -> pd.DataFrame:
    if not HAVE_SM:
        raise RuntimeError("示范分析需要 statsmodels")
    out = []
    for spec in CFG.DEMO_OUTCOMES:
        key, device = spec["key"], spec["device"]
        x, terms = _prep(main, key, device)
        rhs = " + ".join(terms) + " + C(quarter)"
        en = IND[key]["en"]
        # 主模型：Poisson GEE
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                res = smf.gee(f"count ~ {rhs}", groups="ID", data=x, family=sm.families.Poisson(),
                              offset=x["log_den"], cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=300)
            conv = not any(issubclass(i.category, ConvergenceWarning) for i in w)
            out += _table(res, en, "Poisson GEE (primary)", x, conv)
        except Exception as e:
            out.append({"Outcome": en, "Model": "Poisson GEE (primary)", "Term": f"fit failed: {e}",
                        "Observations": len(x), "Hospitals": x["ID"].nunique(), "Stable": False})
        # 敏感性：NB2 极大似然，估计 alpha，机构聚类稳健 SE
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                res = smf.negativebinomial(f"count ~ {rhs}", data=x, offset=x["log_den"]).fit(
                    disp=False, maxiter=500, cov_type="cluster", cov_kwds={"groups": pd.factorize(x["ID"])[0]})
            conv = bool(res.mle_retvals.get("converged", True)) and not any(
                issubclass(i.category, ConvergenceWarning) for i in w)
            out += _table(res, en, "NB2 MLE, cluster-robust SE (sensitivity)", x, conv,
                          alpha=float(res.params.get("alpha", np.nan)))
        except Exception as e:
            out.append({"Outcome": en, "Model": "NB2 MLE, cluster-robust SE (sensitivity)",
                        "Term": f"fit failed: {e}", "Observations": len(x), "Hospitals": x["ID"].nunique(),
                        "Stable": False})
    t = pd.DataFrame(out)
    t["FDR q"] = np.nan
    for (o, m), g in t.groupby(["Outcome", "Model"]):
        sel = g[g["Term"].isin(STAFF_TERMS) & g["Stable"].fillna(False).astype(bool)]
        if len(sel) == 2:
            t.loc[sel.index, "FDR q"] = core.bh_fdr(sel["P"].to_numpy())
    t["Reported"] = t["Stable"].fillna(False).astype(bool)
    return t


_MIXEDLM_OPTS = ({}, {"method": "bfgs"}, {"method": "cg"}, {"method": "powell"}, {"method": "nm"})


def mixedlm_crosscheck(main, rel) -> pd.DataFrame:
    """用 statsmodels MixedLM（REML）重算点估计，核对快速剖面 REML。

    注意：MixedLM.fit 不收敛时不会抛异常，只发警告并返回一个停在边界上的值。原实现
    “第一个不报错的拟合就采用”，在旧数据上无碍，但在模拟数据上 VAP 的默认优化器不收敛，
    返回 0.005 而真值 0.067，使交叉核对失败。现在按优化器逐个尝试，只接受 converged=True
    且随机效应方差未塌缩到 0 的拟合（真值为 0 的指标除外），并记录所用优化器。
    """
    from analysis import reliability_targets, _icc_data
    rows = []
    for col, key, en, tr in reliability_targets():
        gid, y = _icc_data(main, col, tr)
        fast = rel.set_index("key").at[key, "ICC"]
        v, opt, conv = np.nan, None, False
        if HAVE_SM and len(np.unique(gid)) >= CFG.MIN_HOSP:
            for kw in _MIXEDLM_OPTS:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        m = MixedLM(y, np.ones((len(y), 1)), groups=gid).fit(reml=True, **kw)
                    if not getattr(m, "converged", True):
                        continue
                    vb, vw = float(np.atleast_2d(m.cov_re)[0, 0]), float(m.scale)
                    collapsed = vb <= 1e-12 and np.isfinite(fast) and fast > 1e-6
                    if collapsed:      # 随机效应方差塌缩到 0：优化器停在边界，不是真值
                        continue
                    v = min(max(vb / (vb + vw), 0.0), 1.0)
                    opt, conv = kw.get("method", "lbfgs (default)"), True
                    break
                except Exception:
                    continue
        rows.append({"key": key, "Indicator": en, "ICC fast REML": fast, "ICC statsmodels MixedLM": v,
                     "Absolute difference": abs(fast - v) if np.isfinite(fast) and np.isfinite(v) else np.nan,
                     "MixedLM optimizer": opt, "MixedLM converged": conv})
    return pd.DataFrame(rows)
