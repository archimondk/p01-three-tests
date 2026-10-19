# -*- coding: utf-8 -*-
"""后果分析：不先做三项检验就直接排名，会出现什么偏差（PROTOCOL.md 第 10A 节）。

A 部分只用上报数据：
  A1 排名不确定性：经验贝叶斯后验抽样得到每家医院名次的 95% 区间；
  A2 粗率“最好/最差 10%”与过度离散校正漏斗图的一致性，以及最好 10% 中零事件医院的比例；
  A3 “朴素排名”：把分母为 0 或缺失的记录按率 = 0 处理（v1.6 清洗前数据的样子），
      统计最好 10% 里有多少医院其实没有任何可分析数据。
B 部分需要质控中心提供实际使用的表扬/督查名单（可选输入）：
  逐条判断名单上的结论是否得到数据支持。
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats

import config as C
import core
from data import IND


def _den(key):
    return C.MODULES[IND[key]["module"]]["den"]


def period_quarters(main, period: str) -> list[str]:
    qs = sorted(main["季度"].unique())
    if period == "latest":
        return [qs[-1]]
    if period == "pooled" or period in ("all", "", None):
        return qs
    if len(period) == 4 and period.isdigit():            # 年度，如 2025
        return [q for q in qs if q.startswith(period)]
    return [period] if period in qs else []


def hospital_counts(main, key, quarters):
    d = IND[key]
    s = main[main["季度"].isin(quarters) & main[f"ok_{key}"]]
    g = s.groupby("ID").agg(o=(d["num"], "sum"), n=(_den(key), "sum"))
    return g[g["n"] > 0]


# ============================================================ 经验贝叶斯先验
def prior_params(o, n, kind):
    """event：Gamma(a, b)（矩估计，与 core.eb_gamma_poisson 相同）；process：Beta(α, β)（矩估计）。"""
    o, n = np.asarray(o, float), np.asarray(n, float)
    p = o.sum() / n.sum()
    r = o / n
    if kind == "event":
        tau2 = max(np.var(r, ddof=1) - np.mean(p / n), 1e-12)
        return p ** 2 / tau2, p / tau2
    tau2 = max(np.var(r, ddof=1) - np.mean(p * (1 - p) / n), 1e-12)
    ab = min(max(p * (1 - p) / tau2 - 1, 1e-3), 1e7)
    return p * ab, (1 - p) * ab


def rank_intervals(g, key, rng, draws=C.CONSEQ_POSTERIOR_DRAWS):
    """返回每家医院的粗率、收缩率与名次 95% 区间（名次 1 = 最好）。"""
    d = IND[key]
    o, n = g["o"].to_numpy(float), g["n"].to_numpy(float)
    a, b = prior_params(o, n, d["kind"])
    if d["kind"] == "event":
        theta = rng.gamma(o + a, 1.0 / (n + b), size=(draws, len(o)))
        shr = (o + a) / (n + b)
    else:
        theta = rng.beta(o + a, n - o + b, size=(draws, len(o)))
        shr = (o + a) / (n + a + b)
    sign = 1.0 if C.BETTER[key] == "lower" else -1.0
    ranks = stats.rankdata(sign * theta, axis=1)
    lo, med, hi = np.percentile(ranks, [2.5, 50, 97.5], axis=0)
    crude = o / n
    crank = stats.rankdata(sign * crude)
    return pd.DataFrame({"ID": g.index, "events": o, "denominator": n, "crude": crude * d["mult"],
                         "shrunken": shr * d["mult"], "crude_rank": crank,
                         "rank_median": med, "rank_lo": lo, "rank_hi": hi})


# ============================================================ A 部分
def _funnel_flags(g, key):
    """过度离散校正后在 95% 控制限之外（-1 好于预期，+1 差于预期，0 限内）；只用于感染结局。"""
    o, n = g["o"].to_numpy(float), g["n"].to_numpy(float)
    lam = o.sum() / n.sum()
    phi = core.spiegelhalter_phi(o, n)
    if not np.isfinite(phi):
        return np.zeros(len(o), int), phi
    z = (o / n - lam) / np.sqrt(lam / n) / math.sqrt(phi)
    return np.where(z > 1.96, 1, np.where(z < -1.96, -1, 0)), phi


def naive_rates(main, key, quarters):
    """朴素做法：分母为 0 或缺失记为率 0；分母低于阈值也照算。按医院合并。"""
    d = IND[key]
    den = _den(key)
    s = main[main["季度"].isin(quarters)].copy()
    num = s[d["num"]].fillna(0)
    dd = s[den].fillna(0)
    s["_o"], s["_n"] = num.where(dd > 0, 0), dd.clip(lower=0)
    g = s.groupby("ID").agg(o=("_o", "sum"), n=("_n", "sum"), any_ok=(f"ok_{key}", "any"))
    g["rate"] = np.where(g["n"] > 0, g["o"] / g["n"], 0.0)
    return g


def part_a(main, rng):
    rows, intervals = [], []
    idx_mean = main.groupby("ID")["avail_index"].mean()
    for period in C.CONSEQ_PERIODS:
        qs = period_quarters(main, period)
        plabel = qs[0] if period == "latest" else f"{qs[0]}–{qs[-1]}"
        for d in C.INDICATORS:
            key = d["key"]
            g = hospital_counts(main, key, qs)
            if len(g) < C.MIN_HOSP or g["o"].sum() <= 0:
                rows.append({"Period": period, "Quarters": plabel, "key": key, "Indicator": d["en"],
                             "Hospitals ranked": len(g), "Note": "too few hospitals or no events"})
                continue
            ri = rank_intervals(g, key, rng)
            N = len(ri)
            width = (ri["rank_hi"] - ri["rank_lo"]) / N
            k = max(1, math.ceil(C.CONSEQ_DECILE * N))
            best = ri.nsmallest(k, "crude_rank")
            worst = ri.nlargest(k, "crude_rank")
            conf_top = ri["rank_hi"] <= C.CONSEQ_QUARTILE * N
            conf_bot = ri["rank_lo"] >= (1 - C.CONSEQ_QUARTILE) * N
            row = {"Period": period, "Quarters": plabel, "key": key, "Indicator": d["en"], "Hospitals ranked": N,
                   "Median rank-interval width (% of hospitals)": float(np.median(width) * 100),
                   f"Rank interval >= {C.CONSEQ_WIDE_INTERVAL:.0%} of ranks (%)": float((width >= C.CONSEQ_WIDE_INTERVAL).mean() * 100),
                   "Confidently in best quartile (n)": int(conf_top.sum()),
                   "Confidently in worst quartile (n)": int(conf_bot.sum()),
                   "Crude best decile (n)": k,
                   "Crude best decile with zero events (%)": float((best["events"] == 0).mean() * 100) if d["kind"] == "event" else np.nan,
                   "Crude best decile confidently in best quartile (%)": float(conf_top[best.index].mean() * 100),
                   "Crude worst decile confidently in worst quartile (%)": float(conf_bot[worst.index].mean() * 100),
                   "Availability index, crude best decile (median)": float(idx_mean.reindex(best["ID"]).median()),
                   "Availability index, all ranked (median)": float(idx_mean.reindex(ri["ID"]).median())}
            if d["kind"] == "event":
                flags, phi = _funnel_flags(g, key)
                ri["funnel_flag"] = flags
                bad_dir = 1
                row["Overdispersion factor"] = phi
                row["Crude worst decile outside adjusted 95% limit (%)"] = float((ri.loc[worst.index, "funnel_flag"] == bad_dir).mean() * 100)
                row["Adjusted funnel outliers, worse than expected (n)"] = int((flags == 1).sum())
                row["Adjusted funnel outliers, better than expected (n)"] = int((flags == -1).sum())
            # 朴素排名
            nv = naive_rates(main, key, qs)
            sign = 1.0 if C.BETTER[key] == "lower" else -1.0
            nv["naive_rank"] = stats.rankdata(sign * nv["rate"])
            kn = max(1, math.ceil(C.CONSEQ_DECILE * len(nv)))
            nbest = nv.nsmallest(kn, "naive_rank")
            row["Naive ranking: hospitals ranked"] = len(nv)
            row["Naive best decile without any analysable data (n)"] = int((~nbest["any_ok"]).sum())
            row["Naive best decile without any analysable data (%)"] = float((~nbest["any_ok"]).mean() * 100)
            rows.append(row)
            ri.insert(0, "key", key); ri.insert(1, "Period", period)
            intervals.append(ri)
    return pd.DataFrame(rows), pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame()


# ============================================================ B 部分
def read_custodian_flags(path) -> pd.DataFrame:
    f = pd.read_excel(path) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path)
    missing = set(C.CUSTODIAN_FLAG_COLUMNS) - set(f.columns)
    if missing:
        raise ValueError(f"质控名单缺少字段 {sorted(missing)}；需要 {C.CUSTODIAN_FLAG_COLUMNS}")
    f["ID"] = f["ID"].astype(str).str.strip()
    f["period"] = f["period"].astype(str).str.strip()
    f["indicator"] = f["indicator"].astype(str).str.strip()
    f["flag"] = f["flag"].astype(str).str.strip().str.lower()
    bad = ~f["flag"].isin(["good", "bad"])
    if bad.any():
        raise ValueError("flag 只能是 good 或 bad")
    return f


def part_b(main, flags: pd.DataFrame, three: pd.DataFrame, rng) -> pd.DataFrame:
    """对质控名单上的每条结论，判断数据是否支持。

    指标型结论“得到支持”须同时满足：该医院在该期有可分析数据；该指标总评不是 Not usable；
    医院名次的 95% 区间整体落在相应的前/后 25%。综合评分型结论只报告可得性。
    """
    verdict = three.set_index("key")["Overall"]
    cache, out = {}, []
    for _, f in flags.iterrows():
        qs = period_quarters(main, f["period"])
        row = {"ID": f["ID"], "period": f["period"], "indicator": f["indicator"], "flag": f["flag"],
               "quarters matched": len(qs)}
        hosp = main[(main["ID"] == f["ID"]) & main["季度"].isin(qs)]
        row["hospital reported in period"] = len(hosp) > 0
        row["availability index (mean)"] = float(hosp["avail_index"].mean()) if len(hosp) else np.nan
        if f["indicator"] in IND:
            key = f["indicator"]
            ck = (key, tuple(qs))
            if ck not in cache:
                g = hospital_counts(main, key, qs)
                cache[ck] = rank_intervals(g, key, rng).set_index("ID") if len(g) >= C.MIN_HOSP and g["o"].sum() > 0 else None
            ri = cache[ck]
            analysable = ri is not None and f["ID"] in ri.index
            row["indicator verdict"] = verdict.get(key, "")
            row["analysable"] = analysable
            if analysable:
                N = len(ri); r = ri.loc[f["ID"]]
                conf = (r["rank_hi"] <= C.CONSEQ_QUARTILE * N) if f["flag"] == "good" else (r["rank_lo"] >= (1 - C.CONSEQ_QUARTILE) * N)
                row["rank 95% interval"] = f"{r['rank_lo']:.0f}–{r['rank_hi']:.0f} of {N}"
                row["confidently in matching quartile"] = bool(conf)
                row["supported by data"] = bool(conf and verdict.get(key) != core.NOT_USABLE)
            else:
                row["supported by data"] = False
            row["reason if not supported"] = "" if row.get("supported by data") else (
                "no analysable data" if not analysable else
                "indicator not usable as specified" if verdict.get(key) == core.NOT_USABLE else
                "rank interval not confined to matching quartile")
        else:
            n_ok = int(hosp[[f"ok_{d['key']}" for d in C.INDICATORS]].any(axis=0).sum()) if len(hosp) else 0
            row["indicators with analysable data (of 11)"] = n_ok
            row["supported by data"] = np.nan
            row["reason if not supported"] = "composite score: availability reported only"
        out.append(row)
    return pd.DataFrame(out)
