from pathlib import Path
import json
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_DIR = (
    BASE_DIR
    / "reports"
    / "forecast"
)


def find_latest_forecast_file():
    """
    自动寻找最新生成的收益预测文件
    """

    files = list(
        FORECAST_DIR.glob(
            "QQQM_return_forecast_latest_*.json"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"没有找到预测文件目录: {FORECAST_DIR}"
        )

    latest = max(
        files,
        key=lambda x: x.stat().st_mtime
    )

    return latest



def load_latest_forecast():

    path = find_latest_forecast_file()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data



def build_forecast_summary():

    data = load_latest_forecast()


    result = {

        "预测日期":
            data.get(
                "as_of_date",
                "未知"
            ),

        "当前价格":
            data.get(
                "latest_adjusted_close"
            ),

        "模型预测":

            data.get(
                "point_forecasts",
                {}
            ),

        "价格区间":

            data.get(
                "historical_return_bands",
                {}
            ),

        "免责声明":

            "模型预测仅用于研究分析，不构成投资建议。"

    }


    return result



if __name__ == "__main__":

    summary = build_forecast_summary()

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )