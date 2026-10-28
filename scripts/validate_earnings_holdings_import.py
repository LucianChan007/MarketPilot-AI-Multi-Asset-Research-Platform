from __future__ import annotations
"""Offline validation for point-in-time earnings and ETF holdings CSV imports.

This utility never writes to the production database. It produces a machine-readable
validation report and a human-readable summary so bad or incomplete data cannot be
silently promoted into a research run.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.earnings_features import validate_earnings_frame, REQUIRED_COLUMNS

HOLDINGS_REQUIRED = ["ticker", "weight", "available_at", "source_url"]


def validate_holdings(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    missing = [c for c in HOLDINGS_REQUIRED if c not in frame.columns]
    if missing:
        return frame.iloc[0:0].copy(), ["missing_required_columns:" + ",".join(missing)], {"input_rows": len(frame), "accepted_rows": 0}
    h = frame.copy()
    issues: list[str] = []
    h["ticker"] = h["ticker"].fillna("").astype(str).str.strip().str.upper()
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    h["available_at"] = pd.to_datetime(h["available_at"], utc=True, errors="coerce")
    h["source_url"] = h["source_url"].fillna("").astype(str).str.strip()
    if "retrieved_at" in h:
        h["retrieved_at"] = pd.to_datetime(h["retrieved_at"], utc=True, errors="coerce")
    valid = h["ticker"].ne("") & h["weight"].notna() & h["weight"].gt(0) & h["available_at"].notna() & h["source_url"].ne("")
    if (~valid).any():
        issues.append(f"dropped_invalid_rows:{int((~valid).sum())}")
    accepted = h.loc[valid].copy()
    duplicate_key = ["ticker", "available_at"] + (["snapshot_id"] if "snapshot_id" in accepted.columns else [])
    dup_count = int(accepted.duplicated(duplicate_key, keep="last").sum()) if not accepted.empty else 0
    if dup_count:
        issues.append(f"duplicate_holdings_rows:{dup_count}")
        accepted = accepted.drop_duplicates(duplicate_key, keep="last")
    accepted["snapshot_date_utc"] = accepted["available_at"].dt.strftime("%Y-%m-%d")
    snapshots = []
    group_key = "snapshot_id" if "snapshot_id" in accepted.columns and accepted["snapshot_id"].fillna("").astype(str).str.strip().ne("").any() else "snapshot_date_utc"
    for key, group in accepted.groupby(group_key, dropna=False):
        raw_sum = float(group["weight"].sum())
        normalized_sum = raw_sum / 100.0 if raw_sum > 1.5 else raw_sum
        snapshots.append({"snapshot": str(key), "rows": int(len(group)), "unique_tickers": int(group.ticker.nunique()), "raw_weight_sum": raw_sum, "interpreted_weight_sum": normalized_sum, "weight_sum_warning": not (0.85 <= normalized_sum <= 1.15)})
    return accepted.reset_index(drop=True), issues, {"input_rows": len(frame), "accepted_rows": len(accepted), "unique_tickers": int(accepted.ticker.nunique()) if not accepted.empty else 0, "snapshot_grouping": group_key, "snapshots": snapshots}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earnings", type=Path, default=ROOT / "data" / "earnings_point_in_time.csv")
    parser.add_argument("--holdings", type=Path, default=ROOT / "data" / "holdings_point_in_time.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "research" / "earnings_holdings_import_validation.json")
    args = parser.parse_args()
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "gate": "validation_only_not_imported", "inputs": {}, "findings": []}
    if args.earnings.exists():
        raw = pd.read_csv(args.earnings)
        clean, issues = validate_earnings_frame(raw)
        report["inputs"]["earnings"] = {"path": str(args.earnings), "input_rows": len(raw), "accepted_rows": len(clean), "issues": issues, "required_columns": REQUIRED_COLUMNS}
        if len(clean) == 0: report["findings"].append("财报文件没有可接受记录；检查字段、可用时点、来源链接及时间顺序。")
    else:
        report["inputs"]["earnings"] = {"path": str(args.earnings), "status": "not_found"}
        report["findings"].append("未提供财报 CSV；模板数据不能当成真实历史数据使用。")
    if args.holdings.exists():
        raw = pd.read_csv(args.holdings)
        clean, issues, summary = validate_holdings(raw)
        report["inputs"]["holdings"] = {"path": str(args.holdings), **summary, "issues": issues, "required_columns": HOLDINGS_REQUIRED}
        if any(s.get("weight_sum_warning") for s in summary.get("snapshots", [])):
            report["findings"].append("部分持仓快照的权重合计不在 85% 至 115% 范围内；确认数据是否只包含部分持仓或单位是否为百分比。")
        if len(clean) == 0: report["findings"].append("持仓文件没有可接受记录；检查字段、权重、可用时点及来源链接。")
    else:
        report["inputs"]["holdings"] = {"path": str(args.holdings), "status": "not_found"}
        report["findings"].append("未提供历史持仓快照；当前持仓不能替代历史权重。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已写入：{args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
