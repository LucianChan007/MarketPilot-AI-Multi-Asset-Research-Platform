from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(left: pd.Series, right: pd.Series) -> dict[str, float | int | None]:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) == 0:
        return {
            "n": 0,
            "mean_abs_diff": None,
            "median_abs_diff": None,
            "p95_abs_diff": None,
            "max_abs_diff": None,
            "median_abs_pct_diff": None,
            "p95_abs_pct_diff": None,
        }
    abs_diff = np.abs(a - b)
    positive = np.abs(a) > 1e-12
    abs_pct = abs_diff[positive] / np.abs(a[positive]) * 100.0
    return {
        "n": int(len(a)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "median_abs_diff": float(np.median(abs_diff)),
        "p95_abs_diff": float(np.quantile(abs_diff, 0.95)),
        "max_abs_diff": float(np.max(abs_diff)),
        "median_abs_pct_diff": float(np.median(abs_pct)) if len(abs_pct) else None,
        "p95_abs_pct_diff": float(np.quantile(abs_pct, 0.95)) if len(abs_pct) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="核对 QQQM 模型的 Tiingo 与 SQLite 数据源一致性")
    parser.add_argument("--csv", default="data/QQQM_tiingo_history.csv", help="Tiingo 历史 CSV 路径")
    parser.add_argument("--db", default="data/market_data.db", help="SQLite 行情数据库路径")
    parser.add_argument("--ticker", default="QQQM", help="SQLite ticker，默认 QQQM")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.next_day_model import (  # noqa: E402
        FEATURE_COLUMNS,
        build_feature_frame,
        load_history,
        load_recent_history_from_sqlite,
    )

    csv_path = (root / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    db_path = (root / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
    if not csv_path.exists():
        raise FileNotFoundError(f"Tiingo CSV 不存在：{csv_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite 数据库不存在：{db_path}")

    tiingo = load_history(csv_path)
    sqlite_history = load_recent_history_from_sqlite(db_path, ticker=args.ticker, limit=5000)

    tiingo = tiingo.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    sqlite_history = sqlite_history.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    common_dates = sorted(set(tiingo["date"]) & set(sqlite_history["date"]))
    tiingo_only = sorted(set(tiingo["date"]) - set(sqlite_history["date"]))
    sqlite_only = sorted(set(sqlite_history["date"]) - set(tiingo["date"]))

    print("========== QQQM 数据源一致性核对 ==========")
    print(f"Tiingo CSV：{csv_path}")
    print(f"SQLite DB：{db_path}")
    print(f"Tiingo 行数/区间：{len(tiingo)}，{tiingo['date'].min()} 至 {tiingo['date'].max()}")
    print(f"SQLite 行数/区间：{len(sqlite_history)}，{sqlite_history['date'].min()} 至 {sqlite_history['date'].max()}")
    print(f"重叠交易日：{len(common_dates)}，{common_dates[0] if common_dates else '无'} 至 {common_dates[-1] if common_dates else '无'}")
    print(f"仅 Tiingo 日期数：{len(tiingo_only)}；仅 SQLite 日期数：{len(sqlite_only)}")

    output_dir = root / "reports" / "forecast"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_columns = ["close", "open", "high", "low", "volume"]
    base_left = tiingo[["date", *base_columns]].rename(
        columns={column: f"{column}_tiingo" for column in base_columns}
    )
    base_right = sqlite_history[["date", *base_columns]].rename(
        columns={column: f"{column}_sqlite" for column in base_columns}
    )
    base_compare = base_left.merge(base_right, on="date", how="inner").sort_values("date")

    base_rows: list[dict[str, object]] = []
    for column in base_columns:
        result = _metrics(base_compare[f"{column}_tiingo"], base_compare[f"{column}_sqlite"])
        base_rows.append({"field": column, **result})
    base_metrics = pd.DataFrame(base_rows)

    tiingo_features = build_feature_frame(tiingo)
    sqlite_features = build_feature_frame(sqlite_history)
    feature_columns = [column for column in FEATURE_COLUMNS if column in tiingo_features.columns and column in sqlite_features.columns]
    feature_left = tiingo_features[["date", "target_date", "target_return", "target_up", *feature_columns]].copy()
    feature_left = feature_left.rename(columns={
        "target_date": "target_date_tiingo",
        "target_return": "target_return_tiingo",
        "target_up": "target_up_tiingo",
        **{column: f"{column}_tiingo" for column in feature_columns},
    })
    feature_right = sqlite_features[["date", "target_date", "target_return", "target_up", *feature_columns]].copy()
    feature_right = feature_right.rename(columns={
        "target_date": "target_date_sqlite",
        "target_return": "target_return_sqlite",
        "target_up": "target_up_sqlite",
        **{column: f"{column}_sqlite" for column in feature_columns},
    })
    feature_compare = feature_left.merge(feature_right, on="date", how="inner").sort_values("date")

    feature_rows: list[dict[str, object]] = []
    for column in feature_columns:
        result = _metrics(feature_compare[f"{column}_tiingo"], feature_compare[f"{column}_sqlite"])
        feature_rows.append({"feature": column, **result})
    feature_metrics = pd.DataFrame(feature_rows)

    same_target_date = feature_compare["target_date_tiingo"].astype(str).eq(feature_compare["target_date_sqlite"].astype(str))
    label_mask = (
        same_target_date
        & feature_compare["target_up_tiingo"].notna()
        & feature_compare["target_up_sqlite"].notna()
    )
    label_n = int(label_mask.sum())
    label_agreement = None
    if label_n:
        label_agreement = float(
            (feature_compare.loc[label_mask, "target_up_tiingo"].astype(int)
             == feature_compare.loc[label_mask, "target_up_sqlite"].astype(int)).mean()
        )

    print("\n基础行情字段差异（相对 Tiingo；不是自动判定对错）：")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(base_metrics.to_string(index=False, float_format=lambda value: f"{value:.8g}"))

    print("\n技术特征差异（跨源计算同一日期特征；不是自动判定对错）：")
    if not feature_metrics.empty:
        shown = feature_metrics.sort_values("p95_abs_diff", ascending=False, na_position="last")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(shown.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    print("\n下一交易日标签核对：")
    print(f"目标日期一致且标签有效的重叠样本：{label_n}")
    print(f"上涨/非上涨标签一致率：{label_agreement:.4%}" if label_agreement is not None else "标签一致率：无法计算")

    base_path = output_dir / f"QQQM_source_base_comparison_{stamp}.csv"
    feature_path = output_dir / f"QQQM_source_feature_comparison_{stamp}.csv"
    summary_path = output_dir / f"QQQM_source_alignment_{stamp}.json"
    base_compare.to_csv(base_path, index=False)
    feature_compare.to_csv(feature_path, index=False)

    summary = {
        "ticker": args.ticker.upper(),
        "tiingo_csv": str(csv_path),
        "sqlite_db": str(db_path),
        "tiingo_rows": int(len(tiingo)),
        "tiingo_date_min": str(tiingo["date"].min()),
        "tiingo_date_max": str(tiingo["date"].max()),
        "sqlite_rows": int(len(sqlite_history)),
        "sqlite_date_min": str(sqlite_history["date"].min()),
        "sqlite_date_max": str(sqlite_history["date"].max()),
        "common_date_count": int(len(common_dates)),
        "common_date_min": common_dates[0] if common_dates else None,
        "common_date_max": common_dates[-1] if common_dates else None,
        "tiingo_only_date_count": int(len(tiingo_only)),
        "sqlite_only_date_count": int(len(sqlite_only)),
        "base_field_metrics": base_rows,
        "feature_metrics": feature_rows,
        "label_comparable_rows": label_n,
        "label_agreement_rate": label_agreement,
        "notes": [
            "两种数据源的调整价格可能因供应商公司行动调整口径不同而存在差异。",
            "特征差异较大不自动证明某个源错误，但当前训练/实时特征混源前应解释差异。",
            "这个脚本只审计数据，不修改模型、不发起网络请求、不写入 SQLite。",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n输出文件：")
    print(f"基础行情逐日对照：{base_path}")
    print(f"技术特征逐日对照：{feature_path}")
    print(f"汇总 JSON：{summary_path}")
    print("提示：不要只凭某个单一差异阈值判断供应商数据错误；重点看重叠日期、日期对齐和特征差异规模。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"诊断失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
