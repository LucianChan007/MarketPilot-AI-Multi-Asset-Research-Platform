from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
from app.config import REPORTS_DIR
from app.db import load_history


def main() -> int:
    parser = argparse.ArgumentParser(description="将 QQQM 单日预测与目标交易日真实 OHLC 对照，供模型迭代复盘")
    parser.add_argument("--target-date", required=True, help="预测记录对应日期，例如 2026-10-19")
    args = parser.parse_args()
    forecast_path = REPORTS_DIR / f"QQQM_forecast_{args.target_date.replace('-', '')}.json"
    if not forecast_path.exists():
        print(f"[失败] 找不到预测快照：{forecast_path}\n请确保预测生成时的 JSON 文件仍在 reports 目录。", file=sys.stderr)
        return 1
    try:
        payload = json.loads(forecast_path.read_text(encoding="utf-8"))
        forecast = payload["intraday_forecast"]
        history = load_history("QQQM").copy().sort_index()
        index = pd.DatetimeIndex(pd.to_datetime(history.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        history.index = index.normalize()
        key = pd.Timestamp(args.target_date)
        if key not in history.index:
            print(f"[尚无真实行情] 数据库里还没有 {args.target_date} 的交易日记录。先同步 QQQM 行情，再运行本脚本。", file=sys.stderr)
            return 2
        actual = history.loc[key]
        if isinstance(actual, pd.DataFrame):
            actual = actual.iloc[-1]
        required = ["open", "high", "low", "close"]
        missing = [name for name in required if name not in actual.index or pd.isna(actual[name])]
        if missing:
            print(f"[失败] 真实行情缺少 OHLC 字段：{', '.join(missing)}。请换用完整 OHLC 数据源。", file=sys.stderr)
            return 1
        result = {"target_date": args.target_date, "ticker": "QQQM", "forecast": {}, "actual": {}, "error": {},
                  "created_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat()}
        mapping = {"open": "estimated_open", "high": "estimated_high_reference", "low": "estimated_low_reference", "close": "estimated_close"}
        for actual_field, forecast_field in mapping.items():
            pred = float(forecast[forecast_field])
            obs = float(actual[actual_field])
            result["forecast"][actual_field] = round(pred, 4)
            result["actual"][actual_field] = round(obs, 4)
            result["error"][actual_field] = {"usd": round(pred-obs, 4), "abs_usd": round(abs(pred-obs), 4),
                                               "pct_of_actual": round((pred/obs-1)*100, 4) if obs else None}
        prev_close = float(forecast["previous_close"])
        result["forecast"]["direction_vs_previous_close"] = "up" if float(forecast["estimated_close"]) >= prev_close else "down"
        result["actual"]["direction_vs_previous_close"] = "up" if float(actual["close"]) >= prev_close else "down"
        result["direction_correct"] = result["forecast"]["direction_vs_previous_close"] == result["actual"]["direction_vs_previous_close"]
        out = REPORTS_DIR / f"QQQM_intraday_evaluation_{args.target_date.replace('-', '')}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        dataset = REPORTS_DIR / "QQQM_intraday_evaluation_history.csv"
        row = {"target_date": args.target_date, "previous_close": prev_close,
               "predicted_open": result["forecast"]["open"], "actual_open": result["actual"]["open"],
               "predicted_high_reference": result["forecast"]["high"], "actual_high": result["actual"]["high"],
               "predicted_low_reference": result["forecast"]["low"], "actual_low": result["actual"]["low"],
               "predicted_close": result["forecast"]["close"], "actual_close": result["actual"]["close"],
               "open_abs_error_usd": result["error"]["open"]["abs_usd"], "close_abs_error_usd": result["error"]["close"]["abs_usd"],
               "direction_correct": result["direction_correct"]}
        df = pd.DataFrame([row])
        if dataset.exists():
            old = pd.read_csv(dataset)
            old = old[old["target_date"].astype(str) != args.target_date]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(dataset, index=False, encoding="utf-8-sig")
        print(f"[复盘完成] {args.target_date}")
        for field in required:
            print(f"{field}: 预测 {result['forecast'][field]:.2f} | 实际 {result['actual'][field]:.2f} | 绝对误差 ${result['error'][field]['abs_usd']:.2f}")
        print(f"收盘方向判断：{'正确' if result['direction_correct'] else '错误'}")
        print(f"复盘结果：{out}")
        print(f"累计评估数据集：{dataset}")
        return 0
    except Exception as exc:
        print(f"[复盘失败] {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
