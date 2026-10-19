# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import config as C  # noqa: E402
import data as D  # noqa: E402
from make_synthetic import make_raw  # noqa: E402


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    return make_raw(tmp_path_factory.mktemp("raw"), n_hosp=60, seed=2)


def test_exact_duplicates_removed(raw_dir):
    df, log = D.load_raw(raw_dir)
    assert log["exact_duplicates_removed"] == 2
    assert not df.duplicated(["ID", "季度"]).any()


def test_conflicting_duplicate_stops(tmp_path):
    make_raw(tmp_path, n_hosp=30, seed=3, conflicting_dup=True)
    with pytest.raises(D.DataError, match="内容不同"):
        D.load_raw(tmp_path)


def test_missing_column_stops(tmp_path):
    make_raw(tmp_path, n_hosp=30, seed=4)
    f = sorted(tmp_path.glob("*.csv"))[0]
    pd.read_csv(f, encoding="gbk").drop(columns=["导尿管累计使用天数"]).to_csv(f, index=False, encoding="gbk")
    with pytest.raises(D.DataError, match="缺少必需字段"):
        D.load_raw(tmp_path)


def test_cleaning_and_status(raw_dir):
    df, _ = D.load_raw(raw_dir)
    cl, changes, counts = D.clean(df)
    assert counts.get("C03", 0) >= 1 and counts.get("C01", 0) >= 1
    st = D.code_status(cl)
    hh_bad = changes[changes["rule"] == "C03"].iloc[0]
    row = st[(st["ID"] == hh_bad["ID"]) & (st["季度"] == hh_bad["季度"])].iloc[0]
    assert row["RS_hh"] == 0 and not row["ok_hh"]
    assert set(st["RS_neonatal"]) <= {0, 1, 2}
    assert {"blank", "zero"} <= set(st["RSdetail_hd"])
    # 率由分子/分母重算，只在可分析记录上有值
    m = st[st["ok_overall_hai"]]
    assert np.allclose(m["rate_overall_hai"], m["感染例次数"] / m["同期住院天数"] * 1000)
    assert st.loc[~st["ok_overall_hai"], "rate_overall_hai"].isna().all()
    # overall 与 mdro 状态恒相同 -> 主指数不重复计入
    assert (st["RS_overall"] == st["RS_mdro"]).all()
    assert "mdro" not in C.INDEX_MODULES


def test_incomplete_quarter_flagged(raw_dir):
    df, _ = D.load_raw(raw_dir)
    q = D.quarter_table(df)
    assert list(q.loc[~q["main_analysis"], "quarter"]) == ["2026Q3"]


@pytest.mark.parametrize("pattern,group", [("11111", "Continuous"), ("11000", "Ceased permanently"),
                                           ("00111", "Entered late"), ("11011", "Intermittent (gap then return)"),
                                           ("01100", "Entered late and ceased")])
def test_classify_pattern(pattern, group):
    assert D.classify_pattern(pattern) == group


def test_revision_audit(tmp_path):
    import shutil
    import revision_audit as R
    old = make_raw(tmp_path / "old", n_hosp=40, seed=5, quarters=["2025Q1", "2025Q2"], partial_last=False, exact_dups=0)
    new = tmp_path / "new"; shutil.copytree(old, new)
    f = new / "t_25_S_1.csv"
    d = pd.read_csv(f, encoding="gbk")
    d.loc[0, "感染例次数"] = d.loc[0, "感染例次数"] + 3          # 一条记录被修改
    d = d.drop(index=5)                                           # 一条记录在新提取中消失
    extra = d.iloc[[1]].assign(机构代码="H9999")                  # 一条补报记录
    pd.concat([d, extra]).to_csv(f, index=False, encoding="gbk")
    make_raw(tmp_path / "q3", n_hosp=40, seed=5, quarters=["2025Q3"], partial_last=False, exact_dups=0)
    shutil.copy(tmp_path / "q3" / "t_25_S_3.csv", new)           # 新提取多一个季度，不参与比对
    s = R.main(["--old", str(old), "--new", str(new), "--out", str(tmp_path / "audit")])
    assert s["overlap_quarters"] == ["2025Q1", "2025Q2"]
    assert s["records_with_any_change"] == 1
    assert s["records_only_in_old"] == 1 and s["records_added_in_new"] == 1
