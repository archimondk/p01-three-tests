# -*- coding: utf-8 -*-
"""冻结流程入口。

    python run.py --raw data/raw                    # 正式运行（B=5000）
    python run.py --raw data/raw --B 200 --skip-models   # 快速试跑

所有正文和表格里的数字都应取自本次运行输出的 numbers.json 与 tables/，
不要再从不同脚本、不同批次的结果里分别抄数字（v1.6 正文与表格不一致即由此而来）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import core
import data as D
import analysis as A
import figures as F

HERE = Path(__file__).resolve().parent


def _git():
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=HERE,
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _ci(icc, lo, hi):
    if icc is None or not np.isfinite(icc):
        return "—"
    return f"{icc:.3f} ({lo:.2f}–{hi:.2f})"


def _q(v):
    return "not identified" if not np.isfinite(v) else int(v)


def build_numbers(ctx) -> dict:
    three, rel, prec, t3, funnel = ctx["three"], ctx["rel"], ctx["prec"], ctx["t3"], ctx["funnel"]
    rel_i, prec_i, t3_i = rel.set_index("key"), prec.set_index("key"), t3.set_index("key")
    fun_i = funnel.set_index("key")
    ind = {}
    for _, r in three.iterrows():
        k = r["key"]
        ind[k] = {
            "indicator": r["Indicator"], "analysable_pct": round(r["Analysable (%)"], 1), "test1": r["Test 1"],
            "icc": _jsonable(r["ICC"]), "icc_ci_lower": _jsonable(r["CI lower"]), "icc_ci_upper": _jsonable(r["CI upper"]),
            "icc_text": _ci(r["ICC"], r["CI lower"], r["CI upper"]),
            "quarters_for_R": _q(r["Quarters for R"]),
            "quarters_ci": [_q(r["Quarters CI lower"]), _q(r["Quarters CI upper"])],
            "test2": r["Test 2"], "test3": r["Test 3"], "test3_reasons": r["Test 3 reasons"],
            "overall": r["Overall"], "action": r["Required action / reason"],
            "events": int(t3_i.at[k, "Events"]), "zero_numerator_pct": _jsonable(round(t3_i.at[k, "Zero numerator (%)"], 1)),
            "ceiling_pct": _jsonable(t3_i.at[k, "Ceiling at 100% (%)"]),
            "overdispersion": _jsonable(t3_i.at[k, "Overdispersion factor"]),
            "precision_threshold": _jsonable(prec_i.at[k, "Precision-based threshold"]),
            "availability_at_precision_pct": round(prec_i.at[k, "Availability at precision threshold (%)"], 1),
            "hospitals_meeting_precision": int(prec_i.at[k, "Hospitals meeting precision threshold"]),
            "median_denominator": _jsonable(prec_i.at[k, "Median denominator"]),
            "hospitals_in_icc": int(rel_i.at[k, "Hospitals"]),
        }
        if k in fun_i.index:
            ind[k]["funnel"] = {c: _jsonable(fun_i.at[k, c]) for c in fun_i.columns if c != "Indicator"}
    idx = {k: {"icc_text": _ci(rel_i.at[k, "ICC"], rel_i.at[k, "CI lower"], rel_i.at[k, "CI upper"]),
               "icc": _jsonable(rel_i.at[k, "ICC"])} for k, _ in A.INDEX_TARGETS}
    counts = three["Overall"].value_counts().to_dict()
    conseq = {}
    for _, r in ctx["conseq"].iterrows():
        conseq.setdefault(r["Period"], {})[r["key"]] = {c: _jsonable(v) for c, v in r.items()
                                                       if c not in ("Period", "key") and not (isinstance(v, float) and not np.isfinite(v))}
    cust = None
    if "custodian" in ctx and len(ctx["custodian"]):
        cu = ctx["custodian"]
        cu_ind = cu[cu["indicator"] != "composite"]
        cust = {"flags": len(cu), "indicator_flags": len(cu_ind),
                "supported": int(cu_ind["supported by data"].fillna(False).astype(bool).sum()),
                "reasons": {str(k): int(v) for k, v in cu_ind["reason if not supported"].value_counts().items()}}
    return {
        "consequence": conseq,
        "custodian_flags": cust,
        "freeze_version": C.FREEZE_VERSION,
        "coverage": ctx["coverage"][["Stage", "n", "% of framework", "% of coded institutions"]].round(1).to_dict("records"),
        "quarters": ctx["qtab"].to_dict("records"),
        "main_quarters": ctx["main_quarters"],
        "records": {"raw_rows": ctx["load_log"]["raw_rows"], "exact_duplicates_removed": ctx["load_log"]["exact_duplicates_removed"],
                    "records_after_dedup": ctx["load_log"]["records"], "main_records": len(ctx["main"]),
                    "main_hospitals": int(ctx["main"]["ID"].nunique()),
                    "main_tertiary_hospitals": int(ctx["main"].sort_values("季度").groupby("ID")["tertiary"].last().sum())},
        "cleaning": ctx["clean_counts"],
        "indicators": ind,
        "availability_index": idx,
        "verdict_counts": counts,
        "volume_reliability_spearman": {"rho": _jsonable(ctx["vol"].attrs["spearman_rho"]), "n": ctx["vol"].attrs["spearman_n"]},
        "bootstrap": {"B": ctx["B"], "seed": C.SEED},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=C.PATHS["raw_dir"])
    ap.add_argument("--out", default=C.PATHS["out_dir"])
    ap.add_argument("--code-register", default=C.PATHS["code_register"])
    ap.add_argument("--custodian-flags", default=None,
                    help="质控中心实际使用的表扬/督查名单（CSV/XLSX，字段见 config.CUSTODIAN_FLAG_COLUMNS）")
    ap.add_argument("--B", type=int, default=C.B_BOOT)
    ap.add_argument("--skip-models", action="store_true", help="跳过需要 statsmodels 的示范分析与交叉核对")
    ap.add_argument("--skip-figures", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out) / f"run_{stamp}"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    say = lambda *m: print(f"[{time.time() - t0:7.1f}s]", *m, flush=True)

    raw, load_log = D.load_raw(a.raw)
    say(f"读入 {load_log['raw_rows']} 行，去完全重复 {load_log['exact_duplicates_removed']} 行，{load_log['hospitals']} 家机构")
    cleaned, changes, clean_counts = D.clean(raw)
    df_all = D.code_status(cleaned)
    qtab = D.quarter_table(df_all)
    main_quarters = list(qtab.loc[qtab["main_analysis"], "quarter"])
    flagged = list(qtab.loc[~qtab["main_analysis"], "quarter"])
    main = df_all[df_all["季度"].isin(main_quarters)].copy()
    say(f"主分析季度 {main_quarters}；标记为不完整：{flagged or '无'}；主分析 {len(main)} 条 / {main['ID'].nunique()} 家")

    coded_n, coded_src = C.CUSTODIAN["coded_n"], C.CUSTODIAN["coded_source"]
    if a.code_register:
        reg = pd.read_excel(a.code_register) if str(a.code_register).endswith(("xlsx", "xls")) else pd.read_csv(a.code_register)
        coded_n, coded_src = int(reg.iloc[:, 1].astype(str).str.strip().nunique()), f"Code register: {Path(a.code_register).name}"

    ctx = {"load_log": load_log, "clean_counts": clean_counts, "qtab": qtab, "main_quarters": main_quarters,
           "main": main, "B": a.B}
    ctx["coverage"] = A.coverage_table(df_all, main, coded_n, coded_src)
    quarters = sorted(df_all["季度"].unique())
    ctx["flow"], ctx["patterns"] = A.flow_tables(df_all, quarters)
    ctx["attrition"] = A.attrition_table(main, main_quarters)
    ctx["flagged_diag"] = pd.concat([A.flagged_quarter_diagnostic(df_all, main, q, main_quarters) for q in flagged],
                                    ignore_index=True) if flagged else pd.DataFrame()

    ctx["t1"] = A.test1_table(main)
    ctx["prec"] = A.precision_table(main)
    say(f"Test 2 自助重抽 B={a.B} …")
    ctx["rel"] = A.reliability_table(main, a.B, progress=lambda en, r: say(
        f"  {en}: ICC {r['ICC']:.3f} ({r['CI lower']:.2f}–{r['CI upper']:.2f})" if np.isfinite(r["ICC"]) else f"  {en}: 不可估计"))
    ctx["t3"] = A.test3_table(main)
    ctx["three"] = A.three_tests(ctx["t1"], ctx["rel"], ctx["t3"])
    ctx["verdict_sens"] = A.verdict_sensitivity(main, ctx["rel"], ctx["t1"], ctx["t3"])
    ctx["vol"] = A.volume_reliability_table(main, ctx["rel"])
    ctx["table2"] = A.table2_hospital_level(main)
    ctx["bench"], ctx["funnel"], ctx["eb"] = A.benchmark_and_funnel(main)
    say("后果分析 …")
    import consequence as Q
    crng = np.random.default_rng([C.SEED, 9001])
    ctx["conseq"], ctx["rank_intervals"] = Q.part_a(main, crng)
    if a.custodian_flags:
        ctx["custodian"] = Q.part_b(main, Q.read_custodian_flags(a.custodian_flags), ctx["three"], crng)
    say("阈值扫描与样本集敏感性 …")
    ctx["sweep"] = A.threshold_sweep(main)
    ctx["panel"] = A.panel_sensitivity(main, df_all, main_quarters)

    if not a.skip_models:
        import models as M
        say("示范分析（Poisson GEE + NB2）与 MixedLM 交叉核对 …")
        ctx["demo"] = M.demonstration_models(main)
        ctx["crosscheck"] = M.mixedlm_crosscheck(main, ctx["rel"])

    if not a.skip_figures:
        fo = out / "figures"
        F.fig1_availability(main, fo); F.fig2_verdicts(ctx["three"], fo); F.fig3_icc(ctx["rel"], fo)
        F.fig4_funnels(main, fo); F.figS1_flow(ctx["flow"], qtab, fo)
        F.figS2_spearman_brown(ctx["rel"], fo); F.figS3_sweep(ctx["sweep"], fo)
        if len(ctx["rank_intervals"]):
            F.figS4_rank_intervals(ctx["rank_intervals"], fo)

    tables = {
        "T1_network_coverage": ctx["coverage"], "T2_hospital_characteristics": ctx["table2"],
        "T3_precision_stress_test": ctx["prec"], "T4_three_tests": ctx["three"],
        "T4a_test1_availability": ctx["t1"], "T4b_test2_reliability": ctx["rel"], "T4c_test3_comparability": ctx["t3"],
        "TS1_attrition": ctx["attrition"], "TS2_flagged_quarters": ctx["flagged_diag"],
        "TS3_volume_adjusted_reliability": ctx["vol"], "TS4_verdict_sensitivity": ctx["verdict_sens"],
        "TS5_threshold_sweep": ctx["sweep"], "TS6_analysis_set_sensitivity": ctx["panel"],
        "TS7_benchmarks": ctx["bench"], "TS8_funnel_counts": ctx["funnel"], "TS9_eb_shrinkage": ctx["eb"],
        "TS11_consequence_summary": ctx["conseq"], "TS12_rank_intervals": ctx["rank_intervals"],
        "X_quarters": qtab, "X_reporting_flow": ctx["flow"], "X_reporting_patterns": ctx["patterns"],
        "X_cleaning_changes": changes,
    }
    if "custodian" in ctx:
        tables["TS13_custodian_flags"] = ctx["custodian"]
    if "demo" in ctx:
        tables["T5_demonstration_models"] = ctx["demo"]
        tables["TS10_reml_crosscheck"] = ctx["crosscheck"]
    with pd.ExcelWriter(out / "results.xlsx", engine="openpyxl") as w:
        for name, t in tables.items():
            t.to_csv(out / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
            t.to_excel(w, sheet_name=name[:31], index=False)

    numbers = build_numbers(ctx)
    (out / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")

    import scipy, matplotlib
    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    try:
        import statsmodels
        versions["statsmodels"] = statsmodels.__version__
    except ImportError:
        versions["statsmodels"] = None
    manifest = {
        "freeze_version": C.FREEZE_VERSION, "git": _git(), "started": stamp, "seconds": round(time.time() - t0, 1),
        "arguments": vars(a), "inputs": load_log["inputs"], "code_sha256": {
            p.name: D.sha256(p) for p in sorted(HERE.glob("*.py"))} | {"PROTOCOL.md": D.sha256(HERE / "PROTOCOL.md")
                                                                        if (HERE / "PROTOCOL.md").exists() else None},
        "versions": versions, "skipped": {"models": a.skip_models, "figures": a.skip_figures},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"完成：{out}")
    print(ctx["three"][["Indicator", "Test 1", "Test 2", "Test 3", "Overall"]].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
