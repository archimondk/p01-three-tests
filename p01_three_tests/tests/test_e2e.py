# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import core  # noqa: E402
import run  # noqa: E402
from make_synthetic import make_raw  # noqa: E402

try:
    import statsmodels  # noqa: F401
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    raw = make_raw(tmp_path_factory.mktemp("raw"), n_hosp=150, seed=11)
    out = tmp_path_factory.mktemp("out")
    extra = [] if HAVE_SM else ["--skip-models"]
    r1 = run.main(["--raw", str(raw), "--out", str(out / "a"), "--B", "60"] + extra)
    r2 = run.main(["--raw", str(raw), "--out", str(out / "b"), "--B", "60", "--skip-figures", "--skip-models"])
    return r1, r2


def test_outputs_exist(runs):
    r1, _ = runs
    for f in ["numbers.json", "manifest.json", "results.xlsx", "tables/T4_three_tests.csv",
              "tables/TS11_consequence_summary.csv", "figures/FigureS4_rank_intervals_overall_hai.png",
              "figures/Figure2_three_tests.png", "figures/Figure3_reliability.tif"]:
        assert (r1 / f).exists(), f


def test_verdicts_valid_and_numbers_consistent(runs):
    r1, _ = runs
    nums = json.loads((r1 / "numbers.json").read_text(encoding="utf-8"))
    t4 = pd.read_csv(r1 / "tables/T4_three_tests.csv")
    valid = {core.DEFINITE, core.PROVISIONAL, core.POOLING, core.NOT_USABLE}
    assert set(t4["Overall"]) <= valid and len(t4) == 11
    for _, r in t4.iterrows():   # numbers.json 与表格同源
        assert nums["indicators"][r["key"]]["overall"] == r["Overall"]
    assert nums["main_quarters"][-1] == "2026Q2"          # 2026Q3 被识别为不完整


def test_bootstrap_reproducible_across_runs(runs):
    r1, r2 = runs
    a = pd.read_csv(r1 / "tables/T4b_test2_reliability.csv")
    b = pd.read_csv(r2 / "tables/T4b_test2_reliability.csv")
    pd.testing.assert_series_equal(a["CI lower"], b["CI lower"])


@pytest.mark.skipif(not HAVE_SM, reason="statsmodels 未安装")
def test_models_and_crosscheck(runs):
    r1, _ = runs
    x = pd.read_csv(r1 / "tables/TS10_reml_crosscheck.csv").dropna(subset=["Absolute difference"])
    assert (x["Absolute difference"] < 1e-3).all()
    demo = pd.read_csv(r1 / "tables/T5_demonstration_models.csv")
    assert {"Poisson GEE (primary)", "NB2 MLE, cluster-robust SE (sensitivity)"} <= set(demo["Model"])
    # 示范模型必须真的拟合出来：公式解析失败时 run.py 会把异常写成 “fit failed”，
    # 而只看 Model 列是看不出来的（v1.6 的 C(quarter) 与 config 模块同名即由此漏过）。
    assert not demo["Term"].astype(str).str.startswith("fit failed").any(), demo["Term"].head().tolist()
    assert demo["Converged"].fillna(False).astype(bool).all()
    assert demo["IRR"].notna().all()
