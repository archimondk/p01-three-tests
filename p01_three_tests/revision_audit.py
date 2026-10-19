# -*- coding: utf-8 -*-
"""新旧两次提取的重叠季度比对（预设的描述性补充分析）。

    python revision_audit.py --old 旧提取目录 --new 新提取目录 --out outputs/revision_audit

回答三个问题：
1. 旧季度里有多少记录是后来补报的（新提取有、旧提取无）；
2. 有多少记录在新提取中消失（旧提取有、新提取无）；
3. 两次都有的记录，哪些字段的数值被修改过、改了多少。
不改动任何数据，也不影响主分析。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data as D


def compare(old: pd.DataFrame, new: pd.DataFrame, rtol=1e-9):
    overlap_q = sorted(set(old["季度"]) & set(new["季度"]))
    old, new = old[old["季度"].isin(overlap_q)], new[new["季度"].isin(overlap_q)]
    key = ["ID", "季度"]
    m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    status = m["_merge"].map({"left_only": "only in old extract", "right_only": "added in new extract",
                              "both": "in both"})
    by_q = pd.crosstab(m["季度"], status).reset_index()
    both = m[m["_merge"] == "both"]
    fields = [c for c in D.required_columns() if c not in ("ID", "季度")]
    rows, detail = [], []
    for c in fields:
        a, b = both[f"{c}_old"], both[f"{c}_new"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            changed = pd.Series(~np.isclose(a.astype(float), b.astype(float), rtol=rtol, equal_nan=True), index=a.index)
        else:
            changed = ~((a == b) | (a.isna() & b.isna()))
        rows.append({"Field": c, "Records compared": len(both), "Changed": int(changed.sum()),
                     "Changed (%)": changed.mean() * 100 if len(both) else np.nan,
                     "Blank to value": int((a.isna() & b.notna()).sum()),
                     "Value to blank": int((a.notna() & b.isna()).sum())})
        for i in both.index[changed]:
            detail.append({"ID": both.at[i, "ID"], "季度": both.at[i, "季度"], "Field": c,
                           "Old": both.at[i, f"{c}_old"], "New": both.at[i, f"{c}_new"]})
    any_change = pd.DataFrame(detail).groupby(["ID", "季度"]).size() if detail else pd.Series(dtype=int)
    summary = {"overlap_quarters": overlap_q, "records_in_both": len(both),
               "records_with_any_change": int(len(any_change)),
               "records_only_in_old": int((m["_merge"] == "left_only").sum()),
               "records_added_in_new": int((m["_merge"] == "right_only").sum())}
    return summary, by_q, pd.DataFrame(rows), pd.DataFrame(detail)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="outputs/revision_audit")
    a = ap.parse_args(argv)
    old, _ = D.load_raw(a.old)
    new, _ = D.load_raw(a.new)
    summary, by_q, fields, detail = compare(old, new)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "revision_audit.xlsx", engine="openpyxl") as w:
        pd.DataFrame([summary]).astype(str).to_excel(w, sheet_name="summary", index=False)
        by_q.to_excel(w, sheet_name="records_by_quarter", index=False)
        fields.to_excel(w, sheet_name="changes_by_field", index=False)
        detail.to_excel(w, sheet_name="change_detail", index=False)
    print(summary)
    print(fields[fields["Changed"] > 0].to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
