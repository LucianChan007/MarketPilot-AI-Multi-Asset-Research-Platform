from app.forecast_service import load_latest_forecast


def build_price_scenario():

    data = load_latest_forecast()

    current = data["latest_adjusted_close"]

    band = data["historical_return_bands"][
        "last_252_sessions"
    ]


    return {

        "current_price": current,

        "low_price":
            band["p10_price_scenario"],

        "middle_price":
            band["p50_price_scenario"],

        "high_price":
            band["p90_price_scenario"]

    }