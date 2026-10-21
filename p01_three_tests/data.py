# -*- coding: utf-8 -*-
"""读入、校验、清洗、上报状态编码。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

IND = {d["key"]: d for d in C.INDICATORS}
PROPORTION_KEYS = [d["key"] for d in C.INDICATORS if d["mult"] == 100]   # 分子不得超过分母


class DataError(RuntimeError):
    """数据不满足冻结流程的前提，必须人工处理后再运行。"""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quarter_of(fname: str):
    m = re.search(C.RAW_FILE_PATTERN, fname)
    if not m:
        return None
    return f"20{m.group(1)}Q{m.group(2)}"


def required_columns() -> list[str]:
    cols = {m["den"] for m in C.MODULES.values()}
    cols |= {d["num"] for d in C.INDICATORS}
    cols |= {C.STRUCT_COLS["staff"], C.STRUCT_COLS["beds"], C.STRUCT_COLS["level"], C.STRUCT_COLS["grade"]}
    return sorted(cols)


def _read_csv(path: Path) -> pd.DataFrame:
    last = None
    for enc in C.RAW_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except UnicodeDecodeError as e:
            last = e
    raise DataError(f"{path.name} 无法用 {C.RAW_ENCODINGS} 解码：{last}")


def load_raw(raw_dir) -> tuple[pd.DataFrame, dict]:
    """读入全部季度申报表。

    与 v1.6 的差异：完全相同的重复行删除；同一机构同一季度存在内容不同的多行时报错停止，
    不再像 audit.py 那样 drop_duplicates(['ID','季度'], keep='first') 静默保留第一行。
    """
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.iterdir() if quarter_of(p.name))
    if not files:
        raise DataError(f"{raw_dir} 下没有符合 {C.RAW_FILE_PATTERN} 的文件")
    frames, inputs = [], []
    for p in files:
        d = _read_csv(p)
        d = d.rename(columns={d.columns[0]: "ID"}).rename(columns=C.COLUMN_ALIASES)
        d.insert(1, "季度", quarter_of(p.name))
        miss = sorted(set(required_columns()) - set(d.columns))
        if miss:   # 逐个文件检查：某一季度缺字段时，合并后会被 NaN 填平而无法发现
            raise DataError(f"{p.name} 缺少必需字段 {miss}；若是字段改名，请在 config.COLUMN_ALIASES 中映射")
        frames.append(d)
        inputs.append({"file": p.name, "quarter": quarter_of(p.name), "rows": len(d), "sha256": sha256(p)})
    df = pd.concat(frames, ignore_index=True)
    log = {"inputs": inputs, "raw_rows": len(df)}

    missing = sorted(set(required_columns()) - set(df.columns))
    if missing:
        raise DataError(f"缺少必需字段 {missing}；若是字段改名，请在 config.COLUMN_ALIASES 中映射")

    df["ID"] = df["ID"].astype(str).str.strip()
    bad_id = ~df["ID"].str.match(C.ID_PATTERN, na=False)
    log["non_id_rows_dropped"] = int(bad_id.sum())
    df = df[~bad_id].copy()

    exact = df.duplicated(keep="first")
    log["exact_duplicates_removed"] = int(exact.sum())
    log["exact_duplicates_detail"] = df.loc[exact, ["ID", "季度"]].to_dict("records")
    df = df[~exact].copy()
    conflict = df.duplicated(["ID", "季度"], keep=False)
    if conflict.any():
        raise DataError("同一机构同一季度存在内容不同的多行，需与质控中心核实后再运行：\n"
                        + df.loc[conflict, ["ID", "季度"]].drop_duplicates().to_string(index=False))

    text_cols = {"ID", "季度", C.STRUCT_COLS["level"], C.STRUCT_COLS["grade"]}
    for c in df.columns:
        if c not in text_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in (C.STRUCT_COLS["level"], C.STRUCT_COLS["grade"]):
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    log["records"] = len(df)
    log["hospitals"] = int(df["ID"].nunique())
    return df.reset_index(drop=True), log


# ============================================================ 清洗
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """返回 (清洗后数据, 置缺失明细, 规则计数)。记录从不删除，数值从不插补。"""
    df = df.copy()
    changes, counts = [], {}

    def blank(mask, cols, code, desc):
        mask = mask.fillna(False) if hasattr(mask, "fillna") else mask
        idx = df.index[mask]
        for i in idx:
            changes.append({"ID": df.at[i, "ID"], "季度": df.at[i, "季度"], "rule": code, "description": desc,
                            "original": "; ".join(f"{c}={df.at[i, c]}" for c in cols)})
        df.loc[idx, cols] = np.nan
        counts[code] = counts.get(code, 0) + len(idx)

    # 冗余字段：只核对，不静默删除
    red = {}
    for a, b in C.REDUNDANT_PAIRS:
        if a in df.columns:
            diff = ~((df[a] == df[b]) | (df[a].isna() & df[b].isna()))
            red[f"{a} vs {b}"] = int(diff.sum())
    counts["redundant_field_disagreements"] = red

    # 负值没有意义
    num_cols = sorted({m["den"] for m in C.MODULES.values()} | {d["num"] for d in C.INDICATORS}
                      | {C.STRUCT_COLS["staff"], C.STRUCT_COLS["beds"]})
    for c in num_cols:
        blank(df[c] < 0, [c], "C00", f"{c} 为负值")

    staff, beds, pdays = C.STRUCT_COLS["staff"], C.STRUCT_COLS["beds"], C.MODULES["overall"]["den"]
    blank(df[staff].isna() | (df[staff] <= 0), [staff], "C01", "感控专职人员数为 0 或缺失")
    blank(df[beds].isna() | (df[beds] <= 0), [beds], "C02", "实际开放床位数为 0 或缺失")
    blank((df[pdays] > 0) & (df[pdays] < C.MIN_PATIENT_DAYS), [pdays], "C04",
          f"同期住院天数 <{C.MIN_PATIENT_DAYS}，分母异常")

    # 合理性上限：每百床感控专职人员数。旧数据中有记录把床位数抄进了人员数字段
    # （人员数与床位数相等），而 C01 只拦 <=0。一条这样的记录足以把医院间标准差
    # 抬一个量级，示范分析的"每标准差 IRR"随之失去意义，因此按上限置缺失。
    staff_rate = df[staff] / df[beds] * 100
    blank(pd.Series(staff_rate > C.STAFF_PER100_CEILING, index=df.index), [staff], "C01b",
          f"每百床感控专职人员数 > {C.STAFF_PER100_CEILING}，疑人员数与床位数录错")

    # 比例型指标：分子 > 分母 逻辑上不可能
    for k in PROPORTION_KEYS:
        d = IND[k]
        den = C.MODULES[d["module"]]["den"]
        bad = df[d["num"]] > df[den]
        if k == "hh":   # 保留 v1.6 规则：手卫生逻辑错误时整个模块不可用
            blank(bad, [d["num"], den], "C03", "手卫生分子 > 分母，模块置缺失")
        else:
            blank(bad, [d["num"]], "C07", f"{d['en']} 分子 > 分母")

    # 合理性上限
    for k, ceil in C.PLAUSIBILITY_CEILING.items():
        d = IND[k]
        den = C.MODULES[d["module"]]["den"]
        rate = np.where(df[den] > 0, df[d["num"]] / df[den] * d["mult"], np.nan)
        blank(pd.Series(rate > ceil, index=df.index), [d["num"]], "C08", f"{d['en']} > {ceil}，疑填报错误")

    return df, pd.DataFrame(changes), counts


# ============================================================ 上报状态与派生变量
def code_status(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """三态编码：2 可分析；1 分母低于阈值；0 分母为 0 或缺失（原因不可判定）。

    指标层面另加一条：分子缺失（含清洗置缺失）时该指标不可分析。
    率一律由分子/分母重算，不使用平台计算的率字段。
    """
    df = df.copy()
    for mk, m in C.MODULES.items():
        den = df[m["den"]]
        thr = m["thr"] * scale
        df[f"RS_{mk}"] = np.where(den.isna() | (den <= 0), 0, np.where(den < thr, 1, 2))
        df[f"RSdetail_{mk}"] = np.select(
            [den.isna(), den <= 0, den < thr], ["blank", "zero", "below threshold"], "analysable")
    for d in C.INDICATORS:
        m = C.MODULES[d["module"]]
        ok = (df[f"RS_{d['module']}"] == 2) & df[d["num"]].notna()
        df[f"ok_{d['key']}"] = ok
        df[f"rate_{d['key']}"] = np.where(ok, df[d["num"]] / df[m["den"]] * d["mult"], np.nan)

    def index(mods):
        return (df[[f"RS_{m}" for m in mods]] == 2).mean(axis=1)

    df["avail_index"] = index(C.INDEX_MODULES)
    df["avail_index_core"] = index(C.CORE_INDEX_MODULES)
    df["avail_index_v16"] = index(C.V16_INDEX_MODULES)

    staff, beds, pdays = C.STRUCT_COLS["staff"], C.STRUCT_COLS["beds"], C.MODULES["overall"]["den"]
    df["staff_per100"] = np.where(df[beds] > 0, df[staff] / df[beds] * 100, np.nan)
    df["beds_per_staff"] = np.where(df[staff] > 0, df[beds] / df[staff], np.nan)
    df["log_beds"] = np.log(df[beds].where(df[beds] > 0))
    df["pdays_per_bed"] = np.where(df[beds] > 0, df[pdays] / df[beds], np.nan)
    df["class1_per100pd"] = np.where(df[pdays] > 0, df[C.MODULES["class1"]["den"]] / df[pdays] * 100, np.nan)
    for mk, name in [("clabsi", "util_central_line"), ("vap", "util_ventilator"), ("cauti", "util_catheter")]:
        u = np.where(df[pdays] > 0, df[C.MODULES[mk]["den"]] / df[pdays], np.nan)
        df[name] = np.where(df[f"RS_{mk}"] == 0, np.nan, u)   # 未开展时器械日 0 不代表零使用
    df["tertiary"] = (df[C.STRUCT_COLS["level"]] == "三级").astype(int)
    return df


# ============================================================ 季度与主分析集
def quarter_table(df: pd.DataFrame) -> pd.DataFrame:
    """各季度上报机构数，并按预设规则标记不完整季度。"""
    qs = sorted(df["季度"].unique())
    n = df.groupby("季度")["ID"].nunique().reindex(qs)
    rows = []
    for q in qs:
        others = n.drop(q)
        ref = float(others.median()) if len(others) else float(n[q])
        ratio = n[q] / ref if ref > 0 else np.nan
        flagged = bool(ratio < C.INCOMPLETE_QUARTER_RATIO)
        reason = f"reporting {n[q]} < {C.INCOMPLETE_QUARTER_RATIO:.0%} of median {ref:.0f}" if flagged else ""
        if q in C.CUSTODIAN["force_exclude_quarters"]:
            flagged, reason = True, "excluded on custodian confirmation"
        if q in C.CUSTODIAN["force_include_quarters"]:
            flagged, reason = False, "included on custodian confirmation"
        rows.append({"quarter": q, "hospitals_reporting": int(n[q]), "ratio_to_median_of_others": round(ratio, 3),
                     "main_analysis": not flagged, "note": reason})
    return pd.DataFrame(rows)


def reporting_patterns(df: pd.DataFrame, quarters: list[str]) -> pd.DataFrame:
    piv = pd.crosstab(df["ID"], df["季度"]).reindex(columns=quarters, fill_value=0).clip(upper=1)
    piv["pattern"] = piv[quarters].astype(str).agg("".join, axis=1)
    return piv


def classify_pattern(pattern: str) -> str:
    """按主分析季度的上报模式分组（不再像 p5 那样把组名和 n 写死）。"""
    s = pattern
    first, last = s.find("1"), s.rfind("1")
    gap = "0" in s[first:last + 1]
    if gap:
        return "Intermittent (gap then return)"
    if first == 0 and last == len(s) - 1:
        return "Continuous"
    if first == 0:
        return "Ceased permanently"
    if last == len(s) - 1:
        return "Entered late"
    return "Entered late and ceased"
