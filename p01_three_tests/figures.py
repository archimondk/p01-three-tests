# -*- coding: utf-8 -*-
"""投稿图。术语按 v1.4 以后的定稿：analytic availability、zero or missing denominator（reason indeterminate），
不再出现 v1.6 旧图里残留的 “Not performed”“Surveillance adequacy index”。"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
import core
from data import IND

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
GREEN, LGREEN, AMBER, RED, BLUE, GREY = "#2e7d5b", "#7bab8e", "#e0a63a", "#c0563f", "#2c5480", "#777777"
SHORT = {"overall_hai": "Overall HAI", "mdro": "MDRO", "cauti": "CAUTI", "clabsi": "CLABSI", "vap": "VAP",
         "ssi": "SSI (class I)", "hd": "Haemodialysis", "neonatal": "Neonatal HAI", "hh": "Hand hygiene",
         "sap": "SAP (class I)", "micro": "Microbiological testing",
         "avail_index": "Analytic availability index", "avail_index_core": "Availability index (core)",
         "avail_index_v16": "Availability index (v1.6 definition)"}


def _save(fig, out: Path, stem: str):
    for ext in ("png", "tif"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
    plt.close(fig)


def fig1_availability(main, out):
    rows = []
    for mk, m in C.MODULES.items():
        s = main[f"RS_{mk}"]
        rows.append((m["en"], (s == 2).mean() * 100, (s == 1).mean() * 100, (s == 0).mean() * 100,
                     main.loc[s == 2, "ID"].nunique()))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows]); c = np.array([r[3] for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(y, a, color=GREEN, label="Analysable")
    ax.barh(y, b, left=a, color=AMBER, label="Denominator present but below threshold")
    ax.barh(y, c, left=a + b, color=RED, label="Zero or missing denominator, reason indeterminate")
    ax.set_yticks(y, [f"{r[0]} ({r[4]} hospitals)" for r in rows])
    ax.set_xlim(0, 100); ax.set_xlabel(f"Hospital–quarter records (%), n = {len(main)}")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.45, -.13), ncol=1, fontsize=7.5)
    _save(fig, out, "Figure1_analytic_availability")


def fig2_verdicts(three, out):
    rank = {core.DEFINITE: 0, core.PROVISIONAL: 1, core.POOLING: 2, core.NOT_USABLE: 3}
    order = list(three.assign(_r=three["Overall"].map(rank), _i=three["ICC"].fillna(-1))
                 .sort_values(["_r", "_i"], ascending=[True, False])["key"])
    t = three.set_index("key")
    score = {"Pass": 3, "Provisional": 2, "Conditional": 1}
    col = {3: GREEN, 2: LGREEN, 1: AMBER, 0: RED}
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    n = len(order)
    for i, k in enumerate(order):
        for j, c in enumerate(["Test 1", "Test 2", "Test 3"]):
            v = t.at[k, c]; s = score.get(v, 0)
            ax.add_patch(plt.Rectangle((j, n - 1 - i), .94, .9, color=col[s]))
            ax.text(j + .47, n - 1 - i + .45, "Fail" if s == 0 and v != "Not estimable" else v,
                    ha="center", va="center", color="white", weight="bold", fontsize=7.3)
        ax.text(3.05, n - 1 - i + .45, t.at[k, "Overall"].replace(" for single-quarter comparison", "")
                .replace(" as currently specified", ""), va="center", fontsize=7.3)
    ax.set_xlim(-.05, 4.6); ax.set_ylim(-.15, n)
    ax.set_xticks([.47, 1.47, 2.47], ["Test 1\nAnalytic availability", "Test 2\nReliability", "Test 3\nStatistical comparability"])
    ax.set_yticks([n - 1 - i + .45 for i in range(n)], [SHORT[k] for k in order])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    _save(fig, out, "Figure2_three_tests")


def fig3_icc(rel, out):
    r = rel[rel["ICC"].notna() & ~rel["key"].isin(["avail_index_core", "avail_index_v16"])].sort_values("ICC")
    y = np.arange(len(r))
    tgt = C.RELIABILITY_TARGET
    cols = np.where(r["CI lower"] >= tgt, GREEN, np.where(r["ICC"] >= tgt, LGREEN, np.where(r["ICC"] >= .3, AMBER, RED)))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hlines(y, r["CI lower"], r["CI upper"], color=cols, lw=2.2)
    ax.plot(r["ICC"], y, "o", color="#25405c", ms=5)
    ax.axvline(tgt, ls="--", c="#555", lw=.9); ax.axvline(.3, ls=":", c="#888", lw=.9)
    ax.set_yticks(y, [SHORT[k] for k in r["key"]]); ax.set_xlim(0, 1)
    B = int(r["Bootstrap resamples used"].max())
    ax.set_xlabel(f"Intraclass correlation coefficient (95% CI, {B:,} cluster bootstrap resamples, REML)")
    _save(fig, out, "Figure3_reliability")


def fig4_funnels(main, out):
    ks = C.FUNNEL_INDICATORS
    fig, axes = plt.subplots(1, len(ks), figsize=(3.5 * len(ks), 3.4))
    for ax, k in zip(np.atleast_1d(axes), ks):
        d = IND[k]
        s = main[main[f"ok_{k}"]]
        g = s.groupby("ID").agg(o=(d["num"], "sum"), n=(C.MODULES[d["module"]]["den"], "sum"))
        g = g[g["n"] > 0]
        lam = g["o"].sum() / g["n"].sum(); phi = core.spiegelhalter_phi(g["o"], g["n"])
        rate = g["o"] / g["n"] * d["mult"]
        z = (g["o"] / g["n"] - lam) / np.sqrt(lam / g["n"]) / math.sqrt(phi)
        out998 = z.abs() > 3.09
        nn = np.geomspace(g["n"].min(), g["n"].max(), 300); se = np.sqrt(phi * lam / nn)
        ax.scatter(g.loc[~out998, "n"], rate[~out998], s=14, alpha=.6, color=BLUE, lw=0)
        ax.scatter(g.loc[out998, "n"], rate[out998], s=22, color=RED, lw=0)
        ax.plot(nn, np.full_like(nn, lam * d["mult"]), color="#333", lw=1)
        for zq, ls in [(1.96, "--"), (3.09, ":")]:
            ax.plot(nn, np.maximum(lam - zq * se, 0) * d["mult"], ls, color=AMBER if zq < 3 else RED, lw=.9)
            ax.plot(nn, (lam + zq * se) * d["mult"], ls, color=AMBER if zq < 3 else RED, lw=.9)
        ax.set_xscale("log"); ax.set_ylim(0, np.nanpercentile(rate, 98) * 1.7 + .05)
        ax.set_xlabel("Denominator, pooled (log scale)"); ax.set_ylabel(d["unit"])
        ax.set_title(f"{SHORT[k]}  (φ = {phi:.1f}, {len(g)} hospitals)", fontsize=8.5)
    fig.tight_layout()
    _save(fig, out, "Figure4_funnel_plots")


def figS1_flow(flow, qtab, out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(flow))
    ax.plot(x, flow["Reporting"], "o-", color=BLUE, lw=1.8)
    for i, r in flow.iterrows():
        ax.annotate(str(r["Reporting"]), (i, r["Reporting"]), xytext=(0, 8), textcoords="offset points", ha="center")
        if i:
            ax.annotate(f"−{r['Ceased since previous quarter']} / +{r['First appearance'] + r['Returning after a gap']}",
                        (i - .5, (flow.at[i - 1, "Reporting"] + r["Reporting"]) / 2), ha="center", fontsize=7, color=GREY)
    for i, q in enumerate(flow["Quarter"]):
        if not qtab.set_index("quarter").at[q, "main_analysis"]:
            ax.axvspan(i - .5, i + .5, color="#dddddd", alpha=.6, zorder=0)
    ax.set_xticks(x, flow["Quarter"]); ax.set_ylabel("Hospitals reporting")
    ax.set_ylim(0, flow["Reporting"].max() * 1.2)
    _save(fig, out, "FigureS1_reporting_flow")


def figS2_spearman_brown(rel, out):
    r = rel[rel["ICC"].notna() & rel["key"].isin(IND.keys())].sort_values("ICC", ascending=False)
    ks = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for (_, row), c in zip(r.iterrows(), plt.cm.viridis(np.linspace(.05, .85, len(r)))):
        icc = row["ICC"]
        ax.plot(ks, ks * icc / (1 + (ks - 1) * icc), color=c, lw=1.4, label=SHORT[row["key"]])
    ax.axhline(C.RELIABILITY_TARGET, ls="--", c="#555", lw=.9); ax.axhline(.8, ls=":", c="#888", lw=.9)
    ax.axvline(4, ls=":", c=RED, lw=1)
    ax.set_xlim(1, 20); ax.set_ylim(0, 1); ax.set_xlabel("Quarters pooled"); ax.set_ylabel("Reliability")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _save(fig, out, "FigureS2_spearman_brown")


def figS3_sweep(sweep, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, g in sweep.groupby("key"):
        if key in ("avail_index_core", "avail_index_v16"):
            continue
        g = g.sort_values("Scale")
        axes[0].plot(g["Scale"], g["ICC"], marker="o", ms=2.5, lw=1.1, label=SHORT[key])
        if key in IND:
            axes[1].plot(g["Scale"], g["Analysable (%)"], marker="o", ms=2.5, lw=1.1, label=SHORT[key])
    for ax, yl in zip(axes, ["ICC", "Analysable records (%)"]):
        ax.set_xscale("log"); ax.set_xticks([.25, .5, 1, 2, 4], ["0.25×", "0.5×", "1×", "2×", "4×"])
        ax.axvline(1, ls=":", c=RED); ax.set_xlabel("Denominator threshold (multiple of base case)"); ax.set_ylabel(yl)
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1, .5))
    fig.tight_layout()
    _save(fig, out, "FigureS3_threshold_sensitivity")


def figS4_rank_intervals(ri, out, key="overall_hai", period="latest"):
    """毛虫图：各医院名次的 95% 后验区间；粗率排名“最好 10%”的医院标红。"""
    d = ri[(ri["key"] == key) & (ri["Period"] == period)].sort_values("rank_median").reset_index(drop=True)
    if not len(d):
        return
    N = len(d)
    k = max(1, math.ceil(C.CONSEQ_DECILE * N))
    best = d["crude_rank"] <= d["crude_rank"].nsmallest(k).max()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(1, N + 1)
    ax.vlines(x[~best], d.loc[~best, "rank_lo"], d.loc[~best, "rank_hi"], color=GREY, lw=.9, alpha=.7)
    ax.vlines(x[best], d.loc[best, "rank_lo"], d.loc[best, "rank_hi"], color=RED, lw=1.3)
    ax.plot(x, d["rank_median"], "o", ms=2.2, color="#25405c")
    for q in (C.CONSEQ_QUARTILE * N, (1 - C.CONSEQ_QUARTILE) * N):
        ax.axhline(q, ls=":", c="#888", lw=.9)
    ax.set_xlabel("Hospitals ordered by posterior median rank")
    ax.set_ylabel("Rank (1 = best), 95% interval")
    ax.set_xlim(0, N + 1); ax.set_ylim(0, N + 1)
    ax.text(.01, .97, f"{SHORT[key]}, {period}: red = best decile by crude rate", transform=ax.transAxes,
            va="top", fontsize=7.5, color=RED)
    _save(fig, out, f"FigureS4_rank_intervals_{key}")
