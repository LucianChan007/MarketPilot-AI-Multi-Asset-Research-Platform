from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.forecast_training_data import export_training_dataset


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="审计已验证的实时预测快照，并导出不含新闻/宏观特征的市场特征训练集。"
    )
    parser.add_argument("--ticker", default="QQQM")
    parser.add_argument("--db", default=str(root / "data" / "market_data.db"))
    parser.add_argument("--output-dir", default=str(root / "reports" / "model_lab"))
    args = parser.parse_args()

    result = export_training_dataset(db_path=args.db, output_dir=args.output_dir, ticker=args.ticker)
    print("========== 预测特征训练集就绪度审计 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n提示：本脚本只导出符合条件的数据；不会训练模型、自动晋级模型或修改生产预测器。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
