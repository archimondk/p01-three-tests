# -*- coding: utf-8 -*-
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402

P = core.params()


def anova_icc_balanced(gid, y):
    u, inv = np.unique(gid, return_inverse=True)
    a, k = len(u), len(y) // len(u)
    gm = np.bincount(inv, weights=y) / k
    msb = k * ((gm - y.mean()) ** 2).sum() / (a - 1)
    msw = ((y - gm[inv]) ** 2).sum() / (len(y) - a)
    s2b = (msb - msw) / k
    return s2b / (s2b + msw)


def test_reml_equals_anova_in_balanced_design():
    rng = np.random.default_rng(0)
    for icc_true in (0.2, 0.5, 0.85):
        a, k = 120, 5
        b = rng.normal(0, math.sqrt(icc_true), a)
        gid = np.repeat(np.arange(a), k)
        y = b[gid] + rng.normal(0, math.sqrt(1 - icc_true), a * k)
        assert core.icc_reml(gid, y) == pytest.approx(anova_icc_balanced(gid, y), abs=1e-6)


def test_reml_boundary_returns_zero():
    gid = np.repeat(np.arange(30), 4)
    y = np.tile([1.0, 2.0, 3.0, 4.0], 30)          # 各组均值完全相同
    assert core.icc_reml(gid, y) == 0.0


def test_bootstrap_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(1)
    gid = np.repeat(np.arange(80), 5)
    y = rng.normal(0, 1, 80)[gid] + rng.normal(0, 1, 400)
    b1 = core.bootstrap_icc(gid, y, 300, np.random.default_rng([7, 0]))
    b2 = core.bootstrap_icc(gid, y, 300, np.random.default_rng([7, 0]))
    assert np.array_equal(b1, b2)
    lo, hi = np.percentile(b1, [2.5, 97.5])
    assert lo < core.icc_reml(gid, y) < hi


def test_spearman_brown_matches_v16_table4():
    # v1.6 Table 4：CLABSI 0.197 -> 10 季度；SSI 0.187 (0.08–0.31) -> 11 (6–29)；CAUTI 0.366 (0.25–0.48) -> 5 (3–7)
    assert core.quarters_needed(0.197, 0.0, 0.46, .7)[0] == 10
    assert core.quarters_needed(0.187, 0.08, 0.31, .7)[:2] == (11, 6)   # 上限依赖未取整的下限
    assert core.quarters_needed(0.366, 0.25, 0.48, .7) == (5, 3, 7)
    assert core.quarters_needed(0.197, 0.0, 0.46, .7)[2] == math.inf


V16 = [  # 指标, Test1, ICC, 下限, 上限, Test3, Test3 原因, v1.6 总评
    ("HAI", "Pass", .786, .64, .87, "Conditional", ["extreme overdispersion"], core.POOLING),
    ("MDRO", "Pass", .771, .35, .90, "Conditional", ["extreme overdispersion"], core.POOLING),
    ("CAUTI", "Pass", .366, .25, .48, "Pass", [], core.POOLING),
    ("CLABSI", "Conditional", .197, .00, .46, "Fail", ["excess zeros", "sparse events"], core.NOT_USABLE),
    ("VAP", "Conditional", .403, .24, .54, "Pass", [], core.POOLING),
    ("SSI", "Pass", .187, .08, .31, "Pass", [], core.NOT_USABLE),
    ("HD", "Conditional", .570, .29, .79, "Conditional", ["excess zeros"], core.POOLING),
    ("Neonatal", "Fail", float("nan"), float("nan"), float("nan"), "Fail", ["excess zeros", "sparse events"], core.NOT_USABLE),
    ("HH", "Pass", .799, .66, .89, "Pass", [], core.PROVISIONAL),
    ("SAP", "Pass", .750, .64, .83, "Pass", [], core.PROVISIONAL),
    ("Micro", "Fail", .908, .81, .97, "Fail", ["ceiling effect"], core.NOT_USABLE),
]


@pytest.mark.parametrize("name,t1,icc,lo,hi,t3,reasons,expected", V16)
def test_new_rules_reproduce_v16_verdicts(name, t1, icc, lo, hi, t3, reasons, expected):
    t2 = core.test2_verdict(icc, lo, P)
    qhi = core.quarters_needed(icc, lo, hi, .7)[2]
    assert core.overall_verdict(t1, t2, t3, reasons, qhi, P)[0] == expected


def test_v16_bug_pass_with_conditional_test3_is_not_unusable():
    # v1.6 p9b：T1 Pass + T2 Pass + T3 Conditional 会落到 else 判为 Not usable
    ov, action = core.overall_verdict("Pass", "Pass", "Conditional", ["extreme overdispersion"], 1, P)
    assert ov == core.POOLING and "overdispersion" in action


def test_provisional_with_unidentified_pooling_is_not_usable():
    # v1.6 p9b：Provisional 一律判 pooling；按正文规则所需季度上限 >12 时应为 Not usable
    ov, _ = core.overall_verdict("Conditional", "Provisional", "Pass", [], math.inf, P)
    assert ov == core.NOT_USABLE


def test_test2_three_levels_and_point_rule():
    assert core.test2_verdict(.908, .81, P) == "Pass"
    assert core.test2_verdict(.799, .66, P) == "Provisional"
    assert core.test2_verdict(.69, .5, P) == "Fail"
    assert core.test2_verdict(float("nan"), float("nan"), P) == "Not estimable"
    assert core.test2_verdict(.799, .66, core.params(T2_POINT_ESTIMATE_RULE=True)) == "Pass"


def test_test3_rules():
    assert core.test3_verdict(30, 26.6, 5000, float("nan"), P)[0] == "Fail"            # ceiling
    assert core.test3_verdict(81.5, float("nan"), 109, 1.5, P)[0] == "Fail"            # sparse + zeros
    assert core.test3_verdict(75.8, float("nan"), 300, 2, P) == ("Conditional", ["excess zeros"])
    assert core.test3_verdict(10, float("nan"), 12765, 20.3, P) == ("Conditional", ["extreme overdispersion"])
    assert core.test3_verdict(40, float("nan"), 919, 2.4, P) == ("Pass", [])


def test_test1_rules():
    assert core.test1_verdict(98.8, 98, P) == "Pass"
    assert core.test1_verdict(62.6, 66, P) == "Conditional"
    assert core.test1_verdict(30.1, 34, P) == "Fail"
    assert core.test1_verdict(85, 40, P) == "Fail"


def test_precision_threshold_reproduces_v16_table3():
    # v1.6 Table 3 总 HAI 9 239：对应 1.96^2 / 0.0004156，而不是正文所写的 4 例
    pooled = 0.416 / 1000                                   # 正文报告的合并率
    assert core.precision_threshold(pooled, "event") == pytest.approx(9239, abs=15)
    assert 4 / pooled == pytest.approx(9615, abs=1)         # 若真按“4 例”计算应为 9 615
    assert core.precision_threshold(0.79, "process") == pytest.approx(1.96 ** 2 * .79 * .21 / .0025)


def test_phi_near_one_for_poisson_data():
    rng = np.random.default_rng(3)
    n = rng.integers(5_000, 80_000, 400).astype(float)
    o = rng.poisson(n * 0.0004)
    assert core.spiegelhalter_phi(o, n) == pytest.approx(1.0, abs=0.35)


def test_dl_tau2():
    assert core.dl_tau2(np.ones(30), np.full(30, .1)) == 0.0
    rng = np.random.default_rng(4)
    v = rng.uniform(.01, .05, 4000)
    y = rng.normal(0, math.sqrt(.04), 4000) + rng.normal(0, np.sqrt(v))
    assert core.dl_tau2(y, v) == pytest.approx(.04, rel=.1)


def test_bh_fdr_matches_v16_table5():
    q = core.bh_fdr([0.291, 0.002])
    assert q[0] == pytest.approx(0.291) and q[1] == pytest.approx(0.004)
