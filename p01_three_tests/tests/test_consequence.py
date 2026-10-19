# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import analysis as A  # noqa: E402
import consequence as Q  # noqa: E402
import core  # noqa: E402
import data as D  # noqa: E402
from make_synthetic import make_raw  # noqa: E402


def test_rank_intervals_separate_clearly_different_hospitals():
    rng = np.random.default_rng(0)
    n = np.full(30, 1e6)
    true = np.linspace(1e-4, 1e-3, 30)
    g = pd.DataFrame({"o": rng.poisson(n * true), "n": n}, index=[f"H{i}" for i in range(30)])
    ri = Q.rank_intervals(g, "overall_hai", np.random.default_rng(1), draws=500)
    assert ri.loc[0, "rank_hi"] <= 3                      # 真实最低率的医院可信地排在最前
    assert (ri["rank_hi"] - ri["rank_lo"]).median() <= 4
    # 方向：手卫生越高越好，同样的数据排名应反过来
    gp = pd.DataFrame({"o": (n * np.linspace(.5, .95, 30)).round(), "n": n}, index=g.index)
    rh = Q.rank_intervals(gp, "hh", np.random.default_rng(1), draws=300)
    assert rh.loc[29, "rank_median"] == 1


def test_rank_intervals_wide_when_events_are_sparse():
    rng = np.random.default_rng(2)
    n = rng.integers(100, 400, 60).astype(float)
    g = pd.DataFrame({"o": rng.poisson(n * 0.002), "n": n}, index=[f"H{i}" for i in range(60)])
    ri = Q.rank_intervals(g, "clabsi", np.random.default_rng(3), draws=500)
    assert ((ri["rank_hi"] - ri["rank_lo"]) / 60).median() > 0.5


def test_prior_params_process():
    a, b = Q.prior_params(np.array([80, 85, 90, 70]), np.array([100, 100, 100, 100]), "process")
    assert 0.75 < a / (a + b) < 0.85


@pytest.fixture(scope="module")
def coded(tmp_path_factory):
    raw = make_raw(tmp_path_factory.mktemp("raw"), n_hosp=120, seed=21)
    df, _ = D.load_raw(raw)
    cl, _, _ = D.clean(df)
    st = D.code_status(cl)
    return st[st["季度"] != "2026Q3"].copy()


def test_naive_ranking_flags_hospitals_without_analysable_data(coded):
    main = coded.copy()
    ids = sorted(main["ID"].unique())[:5]
    main.loc[main["ID"].isin(ids), "中央血管导管累计使用天数"] = 0     # 这 5 家完全没有 CLABSI 分母
    main = D.code_status(main)
    nv = Q.naive_rates(main, "clabsi", sorted(main["季度"].unique()))
    assert set(ids) <= set(nv.index[nv["rate"] == 0])
    assert not nv.loc[ids, "any_ok"].any()
    summary, _ = Q.part_a(main, np.random.default_rng(4))
    row = summary[(summary["key"] == "clabsi") & (summary["Period"] == "pooled")].iloc[0]
    assert row["Naive best decile without any analysable data (n)"] >= 1


def test_part_b_supports_only_well_founded_flags(coded):
    main = coded
    rel = A.reliability_table(main, 30)
    three = A.three_tests(A.test1_table(main), rel, A.test3_table(main))
    qs = sorted(main["季度"].unique())
    g = Q.hospital_counts(main, "hh", qs)
    ri = Q.rank_intervals(g, "hh", np.random.default_rng(5)).sort_values("rank_median")
    best, mid = ri.iloc[0]["ID"], ri.iloc[len(ri) // 2]["ID"]
    flags = pd.DataFrame([
        {"ID": best, "period": "pooled", "indicator": "hh", "flag": "good"},
        {"ID": mid, "period": "pooled", "indicator": "hh", "flag": "good"},
        {"ID": "H9999", "period": "pooled", "indicator": "hh", "flag": "bad"},
        {"ID": best, "period": "2025", "indicator": "composite", "flag": "good"},
    ])
    out = Q.part_b(main, flags, three, np.random.default_rng(6))
    hh_verdict = three.set_index("key").at["hh", "Overall"]
    assert bool(out.loc[0, "supported by data"]) == (hh_verdict != core.NOT_USABLE)
    assert out.loc[1, "supported by data"] == False  # noqa: E712  中位医院不应被判为可信的“最好”
    assert out.loc[2, "reason if not supported"] == "no analysable data"
    assert out.loc[3, "indicators with analysable data (of 11)"] >= 1


def test_read_custodian_flags_validates(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame([{"ID": "H1", "period": "2025Q1", "indicator": "hh", "flag": "excellent"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="good 或 bad"):
        Q.read_custodian_flags(p)
