# -*- coding: utf-8 -*-
"""三重检验及描述性分析（不依赖 statsmodels）。"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats

import config as C
import core
from data import IND, code_status, reporting_patterns, classify_pattern

INDEX_TARGETS = [("avail_index", "Analytic availability index (reference)"),
                 ("avail_index_core", "Availability index, core modules (sensitivity)"),
                 ("avail_index_v16", "Availability index, v1.6 ten-module definition (sensitivity)")]


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _den(key):
    return C.MODULES[IND[key]["module"]]["den"]


# ============================================================ Test 1
def test1_table(main: pd.DataFrame, p=None) -> pd.DataFrame:
    p = p or core.params()
    nh_all = main["ID"].nunique()
    rows = []
    for d in C.INDICATORS:
        ok = main[f"ok_{d['key']}"]
        st = main[f"RS_{d['module']}"]
        avail = ok.mean() * 100
        nh = main.loc[ok, "ID"].nunique()
        rows.append({"key": d["key"], "Indicator": d["en"],
                     "Analysable (n)": int(ok.sum()), "Analysable (%)": avail,
                     "Below threshold (n)": int((st == 1).sum()),
                     "Zero or missing denominator (n)": int((st == 0).sum()),
                     "Numerator invalid, denominator adequate (n)": int(((st == 2) & ~ok).sum()),
                     "Hospitals with analysable data": nh, "Hospitals (%)": nh / nh_all * 100,
                     "Test 1": core.test1_verdict(avail, nh / nh_all * 100, p)})
    return pd.DataFrame(rows)


def precision_table(main: pd.DataFrame) -> pd.DataFrame:
    """精度阈值压力检验。合并率按分母 >0 且分子非缺失的记录计算（与 v1.6 p7 相同）。"""
    rows = []
    for d in C.INDICATORS:
        den = _den(d["key"])
        s = main[(main[den] > 0) & main[d["num"]].notna()]
        pooled = s[d["num"]].sum() / s[den].sum() if len(s) else np.nan
        thr = core.precision_threshold(pooled, d["kind"])
        prag = C.MODULES[d["module"]]["thr"]
        meets = main[den] >= thr
        rows.append({"key": d["key"], "Indicator": d["en"], "Kind": d["kind"],
                     "Pooled value": pooled * d["mult"], "Threshold used": prag,
                     "Availability at threshold used (%)": main[f"ok_{d['key']}"].mean() * 100,
                     "Precision-based threshold": thr,
                     "Availability at precision threshold (%)": meets.mean() * 100,
                     "Hospitals meeting precision threshold": int(main.loc[meets, "ID"].nunique()),
                     "Median denominator": float(main[den].median()),
                     "P90 denominator": float(main[den].quantile(.9)),
                     "Precision / used": thr / prag})
    return pd.DataFrame(rows)


# ============================================================ Test 2
def _icc_data(main, col, transform):
    x = main[["ID", col]].dropna()
    x = x[x.groupby("ID")["ID"].transform("size") >= C.MIN_OBS_PER_HOSP]
    y = np.log1p(x[col].to_numpy(float)) if transform == "log1p" else x[col].to_numpy(float)
    return x["ID"].to_numpy(), y


def reliability_targets():
    t = [(f"rate_{d['key']}", d["key"], d["en"], d["transform"]) for d in C.INDICATORS]
    t += [(col, col, en, C.INDEX_TRANSFORM) for col, en in INDEX_TARGETS]
    return t


def reliability_table(main: pd.DataFrame, B: int, seed: int = C.SEED, progress=None) -> pd.DataFrame:
    rows = []
    for i, (col, key, en, tr) in enumerate(reliability_targets()):
        gid, y = _icc_data(main, col, tr)
        nh = len(np.unique(gid))
        row = {"key": key, "Indicator": en, "Transform": tr, "Hospitals": nh, "Observations": len(y),
               "ICC": np.nan, "CI lower": np.nan, "CI upper": np.nan, "Bootstrap resamples used": 0,
               "Skew (untransformed)": float(pd.Series(main[col]).dropna().skew())}
        if nh >= C.MIN_HOSP:
            rng = np.random.default_rng([seed, i])     # 每个指标独立、可复现的随机流
            row["ICC"] = core.icc_reml(gid, y)
            boot = core.bootstrap_icc(gid, y, B, rng)
            row["CI lower"], row["CI upper"] = np.percentile(boot, [2.5, 97.5])
            row["Bootstrap resamples used"] = len(boot)
        q = core.quarters_needed(row["ICC"], row["CI lower"], row["CI upper"], C.RELIABILITY_TARGET)
        row["Quarters for R=0.70"], row["Quarters CI lower"], row["Quarters CI upper"] = q
        rows.append(row)
        if progress:
            progress(en, row)
    return pd.DataFrame(rows)


# ============================================================ Test 3
def test3_table(main: pd.DataFrame, p=None) -> pd.DataFrame:
    p = p or core.params()
    rows = []
    for d in C.INDICATORS:
        den = _den(d["key"])
        s = main[main[f"ok_{d['key']}"]]
        events = int(s[d["num"]].sum())
        zero = (s[d["num"]] == 0).mean() * 100 if len(s) else np.nan
        ceiling = (s[f"rate_{d['key']}"] >= 99.99).mean() * 100 if d["kind"] == "process" and len(s) else np.nan
        phi = np.nan
        if d["kind"] == "event" and len(s):
            g = s.groupby("ID").agg(o=(d["num"], "sum"), n=(den, "sum"))
            phi = core.spiegelhalter_phi(g["o"], g["n"])
        v, reasons = core.test3_verdict(zero, ceiling, events, phi, p)
        rows.append({"key": d["key"], "Indicator": d["en"], "Events": events, "Zero numerator (%)": zero,
                     "Ceiling at 100% (%)": ceiling, "Overdispersion factor": phi,
                     "Test 3": v, "Test 3 reasons": ", ".join(reasons)})
    return pd.DataFrame(rows)


# ============================================================ 综合
def three_tests(t1, rel, t3, p=None) -> pd.DataFrame:
    p = p or core.params()
    r = rel.set_index("key")
    a, c = t1.set_index("key"), t3.set_index("key")
    rows = []
    for d in C.INDICATORS:
        k = d["key"]
        icc, lo, hi = r.at[k, "ICC"], r.at[k, "CI lower"], r.at[k, "CI upper"]
        t2 = core.test2_verdict(icc, lo, p)
        q = core.quarters_needed(icc, lo, hi, p["RELIABILITY_TARGET"])
        reasons = [x for x in c.at[k, "Test 3 reasons"].split(", ") if x]
        ov, action = core.overall_verdict(a.at[k, "Test 1"], t2, c.at[k, "Test 3"], reasons, q[2], p)
        rows.append({"key": k, "Indicator": d["en"], "Analysable (%)": a.at[k, "Analysable (%)"],
                     "Test 1": a.at[k, "Test 1"], "ICC": icc, "CI lower": lo, "CI upper": hi,
                     "Quarters for R": q[0], "Quarters CI lower": q[1], "Quarters CI upper": q[2],
                     "Test 2": t2, "Test 3": c.at[k, "Test 3"], "Test 3 reasons": c.at[k, "Test 3 reasons"],
                     "Overall": ov, "Required action / reason": action})
    return pd.DataFrame(rows)


def verdict_sensitivity(main, rel, t1_base, t3_base) -> pd.DataFrame:
    """判定界值敏感性：沿用同一套自助区间，只改判定参数。"""
    base = three_tests(t1_base, rel, t3_base).set_index("key")
    out = []
    for name, ov in [("Base case", {})] + C.VERDICT_SENSITIVITY:
        p = core.params(**ov)
        t1 = test1_table(main, p)
        t3 = test3_table(main, p)
        res = three_tests(t1, rel, t3, p).set_index("key")
        for k in res.index:
            out.append({"Scenario": name, "key": k, "Indicator": res.at[k, "Indicator"],
                        "Test 1": res.at[k, "Test 1"], "Test 2": res.at[k, "Test 2"],
                        "Test 3": res.at[k, "Test 3"], "Overall": res.at[k, "Overall"],
                        "Changed from base": res.at[k, "Overall"] != base.at[k, "Overall"]})
    return pd.DataFrame(out)


# ============================================================ 补充：容量校正信度
def volume_reliability_table(main, rel) -> pd.DataFrame:
    rows = []
    r = rel.set_index("key")
    for d in C.INDICATORS:
        s = main[main[f"ok_{d['key']}"]]
        g = s.groupby("ID").agg(o=(d["num"], "sum"), n=(_den(d["key"]), "sum"))
        ri = core.volume_reliability(g["o"], g["n"], d["kind"])
        rows.append({"key": d["key"], "Indicator": d["en"], "Hospitals": len(g),
                     "ICC (single quarter)": r.at[d["key"], "ICC"],
                     "Volume-adjusted reliability, median": np.nan if ri is None else float(np.median(ri)),
                     "IQR lower": np.nan if ri is None else float(np.percentile(ri, 25)),
                     "IQR upper": np.nan if ri is None else float(np.percentile(ri, 75)),
                     "Hospitals >=0.70 (%)": np.nan if ri is None else float((ri >= .7).mean() * 100)})
    t = pd.DataFrame(rows)
    ok = t.dropna(subset=["ICC (single quarter)", "Volume-adjusted reliability, median"])
    rho = stats.spearmanr(ok["ICC (single quarter)"], ok["Volume-adjusted reliability, median"])[0] if len(ok) > 3 else np.nan
    t.attrs["spearman_rho"] = float(rho)
    t.attrs["spearman_n"] = len(ok)
    return t


# ============================================================ 网络覆盖与上报流向
def coverage_table(df_all, main, coded_n, coded_source) -> pd.DataFrame:
    fw = C.CUSTODIAN["framework_n"]
    quarters = sorted(df_all["季度"].unique())
    allq = (df_all.groupby("ID")["季度"].nunique() == len(quarters)).sum()
    mq = sorted(main["季度"].unique())
    allmain = (main.groupby("ID")["季度"].nunique() == len(mq)).sum()
    rows = [("Institutions within the provincial quality-control framework", fw, C.CUSTODIAN["framework_source"]),
            ("Assigned a reporting code on the platform", coded_n, coded_source),
            (f"Contributed >=1 quarterly return, {quarters[0]}–{quarters[-1]}", df_all["ID"].nunique(), "Extracted returns"),
            (f"Included in the main analysis ({mq[0]}–{mq[-1]})", main["ID"].nunique(), "Main analysis"),
            (f"Reported in all {len(mq)} main-analysis quarters", int(allmain), "Balanced panel (main quarters)"),
            (f"Reported in all {len(quarters)} extracted quarters", int(allq), "All extracted quarters")]
    t = pd.DataFrame(rows, columns=["Stage", "n", "Source"])
    t["% of framework"] = t["n"] / fw * 100
    t["% of coded institutions"] = t["n"] / coded_n * 100 if coded_n else np.nan
    t.loc[0, "% of coded institutions"] = np.nan   # 框架总数不是已编码机构的子集
    return t


def flow_tables(df_all, quarters):
    piv = reporting_patterns(df_all, quarters)
    flow, prev, seen = [], set(), set()
    last_q = df_all.groupby("ID")["季度"].max()
    for i, q in enumerate(quarters):
        now = set(piv.index[piv[q] == 1])
        flow.append({"Quarter": q, "Reporting": len(now), "First appearance": len(now - seen),
                     "Returning after a gap": len((now & seen) - prev),
                     "Ceased since previous quarter": len(prev - now) if i else 0,
                     "Of which permanently (no later return)": int(sum(last_q[h] == quarters[i - 1] for h in prev - now)) if i else 0})
        prev, seen = now, seen | now
    patterns = piv["pattern"].value_counts().rename_axis("Pattern").reset_index(name="Hospitals")
    return pd.DataFrame(flow), patterns


def attrition_table(main, main_quarters) -> pd.DataFrame:
    """Table S1：各上报模式组在首次上报季度的特征（组和 n 均由数据计算）。"""
    piv = reporting_patterns(main, main_quarters)
    grp = piv["pattern"].map(classify_pattern)
    first = main.sort_values("季度").groupby("ID").first()
    first["Group"] = grp
    var = [("实际开放床位数", "Open beds", 0), (C.MODULES["overall"]["den"], "Patient-days per quarter", 0),
           (C.STRUCT_COLS["staff"], "Full-time IPC staff", 0), ("staff_per100", "IPC staff per 100 beds", 3),
           ("avail_index", "Analytic availability index", 2), ("rate_hh", "Hand hygiene compliance (%)", 1),
           ("rate_overall_hai", "HAI incidence density (per 1000 patient-days)", 3)]
    groups = [g for g in ["Continuous", "Ceased permanently", "Entered late", "Entered late and ceased",
                          "Intermittent (gap then return)"] if (first["Group"] == g).any()]
    rows = []
    for col, en, dp in var:
        r = {"Variable": en}
        for g in groups:
            v = first.loc[first["Group"] == g, col].dropna()
            r[f"{g} (n={int((first['Group'] == g).sum())})"] = (
                f"{v.median():.{dp}f} ({v.quantile(.25):.{dp}f}–{v.quantile(.75):.{dp}f})" if len(v) else "—")
        a = first.loc[first["Group"] == "Continuous", col].dropna()
        b = first.loc[first["Group"] == "Ceased permanently", col].dropna()
        r["P (continuous vs ceased)"] = _fmt_p(stats.mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else "—"
        rows.append(r)
    t = pd.DataFrame(rows)
    lvl = first.groupby("Group")["tertiary"].agg(["sum", "count"])
    t.attrs["tertiary_pct"] = {g: float(lvl.at[g, "sum"] / lvl.at[g, "count"] * 100) for g in lvl.index}
    return t


def flagged_quarter_diagnostic(df_all, main, flagged_q, main_quarters) -> pd.DataFrame:
    """对每个不完整季度：上报与未上报机构在主分析期内的上报连续性、级别和规模。"""
    ids_main = set(main["ID"])
    rep = set(df_all.loc[df_all["季度"] == flagged_q, "ID"]) & ids_main
    non = ids_main - rep
    nq = main.groupby("ID")["季度"].nunique()
    lvl = main.sort_values("季度").groupby("ID")["tertiary"].last()
    beds = main.groupby("ID")["实际开放床位数"].median()
    full = len(main_quarters)

    def fisher(flag):
        a = sum(flag[h] for h in rep); b = len(rep) - a
        c = sum(flag[h] for h in non); d = len(non) - c
        return stats.fisher_exact([[a, b], [c, d]])[1]

    allq = (nq == full)
    rows = [{"Characteristic": "Institutions", "Reported": len(rep), "Did not report": len(non), "P": "—"},
            {"Characteristic": f"Reported in all {full} main quarters (%)",
             "Reported": np.mean([allq[h] for h in rep]) * 100, "Did not report": np.mean([allq[h] for h in non]) * 100,
             "P": _fmt_p(fisher(allq))},
            {"Characteristic": "Tertiary level (%)",
             "Reported": np.mean([lvl[h] for h in rep]) * 100, "Did not report": np.mean([lvl[h] for h in non]) * 100,
             "P": _fmt_p(fisher(lvl.astype(bool)))},
            {"Characteristic": "Median open beds",
             "Reported": float(np.nanmedian([beds[h] for h in rep])), "Did not report": float(np.nanmedian([beds[h] for h in non])),
             "P": _fmt_p(stats.mannwhitneyu([beds[h] for h in rep if np.isfinite(beds[h])],
                                            [beds[h] for h in non if np.isfinite(beds[h])]).pvalue)}]
    t = pd.DataFrame(rows)
    t.insert(0, "Quarter", flagged_q)
    return t


# ============================================================ Table 2：机构层面
TABLE2_VARS = [("实际开放床位数", "Open beds", 0), (C.MODULES["overall"]["den"], "Patient-days per quarter", 0),
               (C.STRUCT_COLS["staff"], "Full-time IPC staff", 1), ("staff_per100", "IPC staff per 100 beds", 3),
               ("beds_per_staff", "Beds per IPC staff member", 0), ("pdays_per_bed", "Patient-days per bed", 1),
               ("class1_per100pd", "Class I incision procedures per 100 patient-days", 2),
               ("util_central_line", "Central line utilisation ratio", 4),
               ("util_ventilator", "Ventilator utilisation ratio", 4),
               ("util_catheter", "Urinary catheter utilisation ratio", 4),
               ("avail_index", "Analytic availability index", 2)] + \
              [(f"rate_{d['key']}", f"{d['en']} ({d['unit']})", 3 if d["kind"] == "event" else 1)
               for d in C.INDICATORS if d["key"] not in ("neonatal",)]


def table2_hospital_level(main) -> pd.DataFrame:
    """v1.6 在 674 个医院-季度上做 Mann–Whitney，把同院重复观测当独立样本，P 值偏小。
    这里先取每家医院主分析期内的中位数，再在医院层面比较。"""
    lvl = main.sort_values("季度").groupby("ID")["tertiary"].last()
    rows = []
    for col, en, dp in TABLE2_VARS:
        h = main.groupby("ID")[col].median()
        a, b = h[lvl == 1].dropna(), h[lvl == 0].dropna()

        def q(v):
            return f"{v.median():.{dp}f} ({v.quantile(.25):.{dp}f}–{v.quantile(.75):.{dp}f})" if len(v) else "—"
        p = stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
        rows.append({"Variable": en, "All": q(h.dropna()), "n hospitals": int(h.notna().sum()),
                     "Tertiary": q(a), "n tertiary": len(a), "Secondary": q(b), "n secondary": len(b), "P": _fmt_p(p)})
    return pd.DataFrame(rows)


# ============================================================ 机构比较
def benchmark_and_funnel(main):
    bench, funnel, eb = [], [], []
    for d in C.INDICATORS:
        s = main[main[f"ok_{d['key']}"]]
        den = _den(d["key"])
        for lab, sub in [("All", s), ("Tertiary", s[s["tertiary"] == 1]), ("Secondary", s[s["tertiary"] == 0])]:
            if not len(sub):
                continue
            g = sub.groupby("ID").agg(o=(d["num"], "sum"), n=(den, "sum"))
            hr = g["o"] / g["n"] * d["mult"]
            bench.append({"key": d["key"], "Indicator": d["en"], "Stratum": lab, "Hospitals": len(g),
                          "Records": len(sub), "Events": int(g["o"].sum()),
                          "Pooled rate": g["o"].sum() / g["n"].sum() * d["mult"],
                          "P10": hr.quantile(.1), "P25": hr.quantile(.25), "Median": hr.median(),
                          "P75": hr.quantile(.75), "P90": hr.quantile(.9)})
    for k in C.FUNNEL_INDICATORS:
        d = IND[k]
        s = main[main[f"ok_{k}"]]
        g = s.groupby("ID").agg(o=(d["num"], "sum"), n=(_den(k), "sum"))
        g = g[g["n"] > 0]
        phi = core.spiegelhalter_phi(g["o"], g["n"])
        fc = core.funnel_counts(g["o"], g["n"], phi)
        fc.update({"key": k, "Indicator": d["en"], "Overdispersion factor": phi})
        funnel.append(fc)
        shr, lam = core.eb_gamma_poisson(g["o"], g["n"])
        e = g.assign(crude=g["o"] / g["n"] * d["mult"], shrunken=shr * d["mult"], key=k).reset_index()
        e["movement"] = e["shrunken"] - e["crude"]
        eb.append(e)
    return pd.DataFrame(bench), pd.DataFrame(funnel), pd.concat(eb, ignore_index=True)


# ============================================================ 敏感性
def threshold_sweep(clean_main) -> pd.DataFrame:
    """所有务实阈值同乘系数后重算：可分析比例、ICC 点估计、合并率。"""
    rows = []
    for s in C.THRESHOLD_SCALES:
        m = code_status(clean_main, scale=s)
        for col, key, en, tr in reliability_targets():
            if key in IND:
                ok = m[f"ok_{key}"]
                d = IND[key]
                sub = m[ok]
                pooled = sub[d["num"]].sum() / sub[_den(key)].sum() * d["mult"] if len(sub) else np.nan
                avail = ok.mean() * 100
            else:
                pooled, avail = np.nan, np.nan
            gid, y = _icc_data(m, col, tr)
            icc = core.icc_reml(gid, y) if len(np.unique(gid)) >= C.MIN_HOSP else np.nan
            rows.append({"Scale": s, "key": key, "Indicator": en, "Analysable (%)": avail,
                         "ICC": icc, "Pooled rate": pooled})
    return pd.DataFrame(rows)


def panel_sensitivity(main, df_all_coded, main_quarters) -> pd.DataFrame:
    nq = main.groupby("ID")["季度"].nunique()
    balanced = main[main["ID"].isin(nq[nq == len(main_quarters)].index)]
    sets = [("Main analysis", main), ("Balanced panel (all main quarters)", balanced),
            ("Including flagged quarters", df_all_coded)]
    rows = []
    for name, d_ in sets:
        for d in C.INDICATORS:
            sub = d_[d_[f"ok_{d['key']}"]]
            gid, y = _icc_data(d_, f"rate_{d['key']}", d["transform"])
            rows.append({"Analysis set": name, "key": d["key"], "Indicator": d["en"],
                         "Hospitals": d_["ID"].nunique(), "Records": len(d_),
                         "Pooled rate": sub[d["num"]].sum() / sub[_den(d["key"])].sum() * d["mult"] if len(sub) else np.nan,
                         "ICC": core.icc_reml(gid, y) if len(np.unique(gid)) >= C.MIN_HOSP else np.nan})
    return pd.DataFrame(rows)
