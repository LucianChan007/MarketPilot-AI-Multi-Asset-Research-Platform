import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(ROOT)
)


from app.forecast_service import (
    build_forecast_summary
)


result = build_forecast_summary()

print(result)