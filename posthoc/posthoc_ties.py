# -*- coding: utf-8 -*-
"""事后分析（非预设，不改冻结代码）：后果分析中的并列处理与“无可分析数据”的机制拆分。

起因：外部审阅指出
  (1) 冻结代码用 nsmallest(k) 取粗率“最好 10%”，并列时按机构编号取前 k 家；
  (2) 朴素排名中“无可分析数据”的医院混合了缺失/零分母（记为率 0）、仅有低于阈值的小分母、
      分母足够但分子缺失三种情况，机制不同。

本脚本：
  A. 粗率排名（读取冻结运行输出的 TS12_rank_intervals.csv，排名区间与冻结运行完全一致）：
     报告与第 k 名并列的整组医院，以及随机打破并列时“最好 10%”各项比例的期望值。
  B. 朴素排名：按上面三种机制给每家医院归类，报告并列感知的最好层级构成与期望值。

只输出汇总表，不含机构编号，可以分享。

用法（在冻结仓库外运行，路径都用参数传入）：
  python -I posthoc_ties.py --repo <p01_three_tests 目录> --raw <原始数据目录> \
      --run <run_20261020_070811 目录> --out <输出目录>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MECH = ["analysable", "no denominator (blank or zero)", "below-threshold denominator only",
        "adequate denominator, numerator missing"]


def tie_aware(rank: pd.Series, k: int):
    """返回 (严格优于第 k 名的索引, 与第 k 名并列的索引)。rank 越小越好。"""
    cut = rank.sort_values().iloc[k - 1]
    return rank.index[rank < cut], rank.index[rank == cut]


def expected_share(flag: pd.Series, strict, bnd, k: int) -> float:
    """随机打破并列时，最好 k 家中 flag 为真的期望比例（%）。"""
    n_from_bnd = k - len(strict)
    e = flag.loc[strict].sum() + n_from_bnd * flag.loc[bnd].mean()
    return float(e / k * 100)


def mechanism(main, key, quarters, C, IND):
    d = IND[key]
    mk = d["module"]
    s = main[main["季度"].isin(quarters)]
    rs = s.groupby("ID")[f"RS_{mk}"]
    any_ok = s.groupby("ID")[f"ok_{key}"].any()
    any_rs2 = rs.apply(lambda x: (x == 2).any())
    any_rs1 = rs.apply(lambda x: (x == 1).any())
    out = pd.Series(MECH[1], index=any_ok.index)
    out[any_rs1] = MECH[2]
    out[any_rs2 & ~any_ok] = MECH[3]
    out[any_ok] = MECH[0]
    return out


def main_(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(a.repo).resolve()))
    import config as C          # noqa: E402
    import data as D            # noqa: E402
    from consequence import naive_rates, period_quarters  # noqa: E402
    from data import IND        # noqa: E402

    raw, _ = D.load_raw(a.raw)
    cleaned, _, _ = D.clean(raw)
    df_all = D.code_status(cleaned)
    qtab = D.quarter_table(df_all)
    mq = list(qtab.loc[qtab["main_analysis"], "quarter"])
    main = df_all[df_all["季度"].isin(mq)].copy()

    run = Path(a.run)
    nums = json.loads((run / "numbers.json").read_text(encoding="utf-8"))
    assert len(main) == nums["records"]["main_records"], "主分析记录数与冻结运行不一致，停止"
    ts12 = pd.read_csv(run / "tables" / "TS12_rank_intervals.csv")
    ts11 = pd.read_csv(run / "tables" / "TS11_consequence_summary.csv")
    idx_mean = main.groupby("ID")["avail_index"].mean()

    rows = []
    for period in C.CONSEQ_PERIODS:
        qs = period_quarters(main, period)
        for dd in C.INDICATORS:
            key = dd["key"]
            g = ts12[(ts12["Period"] == period) & (ts12["key"] == key)].set_index("ID")
            frozen = ts11[(ts11["Period"] == period) & (ts11["key"] == key)].iloc[0]
            row = {"Period": period, "key": key, "Indicator": dd["en"], "Hospitals ranked": len(g)}
            if len(g):
                N = len(g)
                k = max(1, math.ceil(C.CONSEQ_DECILE * N))
                strict, bnd = tie_aware(g["crude_rank"], k)
                conf = g["rank_hi"] <= C.CONSEQ_QUARTILE * N
                zero = g["events"] == 0
                tier = strict.union(bnd)
                row.update({
                    "Crude: decile size k": k,
                    "Crude: strictly better than k-th (n)": len(strict),
                    "Crude: tied at k-th (n)": len(bnd),
                    "Crude: best tier incl. ties (n)": len(tier),
                    "Crude: best tier with zero events (%)": float(zero.loc[tier].mean() * 100),
                    "Crude: best tier confidently in best quartile (n)": int(conf.loc[tier].sum()),
                    "Crude: expected % confidently best, random tie-break": expected_share(conf, strict, bnd, k),
                    "Crude: frozen output % confidently best": frozen["Crude best decile confidently in best quartile (%)"],
                    "Crude: best tier availability index (median)": float(idx_mean.reindex(tier).median()),
                    "Crude: all ranked availability index (median)": float(idx_mean.reindex(g.index).median()),
                    "Crude: frozen output best-decile availability (median)": frozen["Availability index, crude best decile (median)"],
                })
            nv = naive_rates(main, key, qs)
            sign = 1.0 if C.BETTER[key] == "lower" else -1.0
            nv["naive_rank"] = pd.Series(stats.rankdata(sign * nv["rate"]), index=nv.index)
            nv["mech"] = mechanism(main, key, qs, C, IND).reindex(nv.index)
            kn = max(1, math.ceil(C.CONSEQ_DECILE * len(nv)))
            strict, bnd = tie_aware(nv["naive_rank"], kn)
            tier = strict.union(bnd)
            row.update({"Naive: hospitals": len(nv), "Naive: decile size k": kn,
                        "Naive: strictly better than k-th (n)": len(strict), "Naive: tied at k-th (n)": len(bnd),
                        "Naive: best tier incl. ties (n)": len(tier),
                        "Naive: frozen output, no analysable data (n)":
                            frozen.get("Naive best decile without any analysable data (n)", np.nan)})
            for m in MECH:
                flag = nv["mech"] == m
                row[f"Naive tier: {m} (n)"] = int(flag.loc[tier].sum())
                row[f"Naive decile expected: {m} (n)"] = round(expected_share(flag, strict, bnd, kn) * kn / 100, 2)
            rows.append(row)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "PH1_ties_and_mechanisms.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(max_cols=8))
    print(f"\n已写出 {out / 'PH1_ties_and_mechanisms.csv'}")


if __name__ == "__main__":
    main_()
