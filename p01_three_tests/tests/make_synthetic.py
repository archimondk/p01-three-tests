# -*- coding: utf-8 -*-
"""生成与平台导出格式相同的模拟季度申报表（GBK 编码），只用于测试流程，数值无任何实际意义。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]


def _fname(q):
    return f"t_{q[2:4]}_S_{q[-1]}.csv"


def make_raw(out_dir, n_hosp=150, quarters=QUARTERS, partial_last=True, seed=1,
             exact_dups=2, conflicting_dup=False):
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    H = pd.DataFrame({"ID": [f"H{i:04d}" for i in range(1, n_hosp + 1)]})
    H["tert"] = rng.random(n_hosp) < 0.67
    H["级"] = np.where(H["tert"], "三级", "二级")
    H["等"] = rng.choice(["甲等", "乙等", "未定等"], n_hosp, p=[.5, .35, .15])
    H["beds"] = np.where(H["tert"], rng.lognormal(np.log(520), .35, n_hosp), rng.lognormal(np.log(200), .4, n_hosp)).round()
    H["staff"] = np.maximum(1, np.round(H["beds"] / rng.uniform(120, 260, n_hosp)))
    H["hai"] = rng.gamma(4, 0.35 / 4, n_hosp) * np.where(H["tert"], 1.3, 0.6) / 1000
    H["mdro"] = H["hai"] * rng.uniform(.1, .3, n_hosp)
    H["hh"] = np.clip(rng.normal(.84, .07, n_hosp), .4, .99)
    H["sap"] = np.clip(rng.normal(.32, .12, n_hosp), .02, .95)
    H["micro"] = np.clip(rng.normal(.6, .2, n_hosp), .05, 1)
    H["has_icu"] = rng.random(n_hosp) < np.where(H["tert"], .8, .3)
    H["has_hd"] = rng.random(n_hosp) < np.where(H["tert"], .75, .35)
    H["has_neo"] = rng.random(n_hosp) < np.where(H["tert"], .15, .02)
    # 上报模式：连续 / 2025Q2 后停报 / 2025Q3 进入 / 间断
    pat = rng.choice(["cont", "stop", "late", "gap"], n_hosp, p=[.80, .12, .05, .03])
    pat = np.where(~H["tert"] & (rng.random(n_hosp) < .15), "stop", pat)
    rows = []
    last = quarters[-1]
    for _, h in H.assign(pat=pat).iterrows():
        for qi, q in enumerate(quarters):
            if h["pat"] == "stop" and qi >= 2: continue
            if h["pat"] == "late" and qi < 2: continue
            if h["pat"] == "gap" and qi in (2, 3): continue
            if partial_last and q == last and rng.random() > .35: continue
            pd_ = h["beds"] * rng.uniform(78, 92)
            r = {"ID": h["ID"], "级": h["级"], "等": h["等"], "同期感控专职人员数": h["staff"],
                 "实际开放床位数": h["beds"], "感控专职人员床位比": f"1:{int(h['beds'] / h['staff'])}",
                 "同期住院天数": round(pd_)}
            r["住院患者累计住院天数"] = r["同期住院天数"]
            r["感染例次数"] = rng.poisson(pd_ * h["hai"])
            r["多重耐药菌感染新发例次数"] = rng.poisson(pd_ * h["mdro"])
            opp = rng.integers(100, 2500); r["应执行手卫生时机数"] = opp
            r["手卫生实际执行时机数"] = rng.binomial(opp, h["hh"])
            comb = rng.integers(0, 60) if rng.random() > .3 else 0
            r["联合使用抗菌药物总人数"] = comb
            r["送检住院患者人数"] = comb if rng.random() < .25 else rng.binomial(comb, h["micro"])
            c1 = rng.poisson(pd_ * 0.009); r["1类切口手术总例次数"] = c1; r["1类切口手术总例次数1"] = c1
            r["预防使用抗菌药物1类切口例次数"] = rng.binomial(c1, h["sap"])
            r["1类切口手术部位感染例次数"] = rng.binomial(c1, 0.0015)
            cl = rng.poisson(pd_ * 0.012) if h["has_icu"] else 0
            ve = rng.poisson(pd_ * 0.004) if h["has_icu"] else 0
            ca = rng.poisson(pd_ * 0.035)
            r["中央血管导管累计使用天数"] = cl; r["血流感染新发病例例次数"] = rng.poisson(cl * 0.00018)
            r["有创呼吸机累计使用天数"] = ve; r["呼吸机相关肺炎新发病例例次数"] = rng.poisson(ve * 0.0016)
            r["导尿管累计使用天数"] = ca; r["尿路感染新发病例例次数"] = rng.poisson(ca * 0.0007 * rng.gamma(2, .5))
            hd = rng.integers(20, 200) if h["has_hd"] else 0
            r["血液透析总人数"] = hd; r["血液透析相关感染例次数"] = rng.binomial(hd, 0.002) if hd else 0
            neo = rng.integers(150, 1500) if h["has_neo"] else 0
            r["新生儿住院天数"] = neo; r["新生儿感染例次数"] = rng.poisson(neo * 0.0006)
            # 平台自带的率字段（流程不使用，只保留格式）
            r["感染日例次率(‰)"] = r["感染例次数"] / r["同期住院天数"] * 1000
            r["_q"] = q
            rows.append(r)
    df = pd.DataFrame(rows)
    # 注入异常：一条手卫生分子 > 分母，一条感控人员为 0，一个分母为空（blank）
    df.loc[3, "手卫生实际执行时机数"] = df.loc[3, "应执行手卫生时机数"] + 5
    df.loc[7, "同期感控专职人员数"] = 0
    df.loc[11, "血液透析总人数"] = np.nan
    for q in quarters:
        part = df[df["_q"] == q].drop(columns="_q")
        if exact_dups and q == quarters[0]:
            part = pd.concat([part, part.iloc[:exact_dups]], ignore_index=True)
        if conflicting_dup and q == quarters[0]:
            x = part.iloc[[0]].copy(); x["感染例次数"] = x["感染例次数"] + 1
            part = pd.concat([part, x], ignore_index=True)
        part = part.rename(columns={"ID": "机构代码"})
        part.to_csv(out / _fname(q), index=False, encoding="gbk")
    return out


if __name__ == "__main__":
    import sys
    print(make_raw(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"))
