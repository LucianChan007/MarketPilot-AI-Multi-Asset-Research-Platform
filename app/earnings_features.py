"""Point-in-time earnings feature utilities.

Only records whose public release timestamp is <= prediction cutoff are eligible.
Missing consensus is NaN, never silently imputed as zero. QQQM is an ETF, so records
must identify the underlying constituent ticker and should be weighted by a separate,
point-in-time holdings file before aggregating to the ETF level.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "ticker", "fiscal_period", "actual_release_timestamp", "available_at",
    "eps_actual", "eps_estimate", "revenue_actual", "revenue_estimate",
    "guidance_change_pct", "source_name", "source_url", "retrieved_at",
]
NUMERIC_COLUMNS = ["eps_actual", "eps_estimate", "revenue_actual", "revenue_estimate", "guidance_change_pct"]


def validate_earnings_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize and validate vendor/export rows; returns accepted rows and reasons."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        return frame.iloc[0:0].copy(), ["missing_required_columns:" + ",".join(missing)]
    df = frame.copy()
    issues: list[str] = []
    for col in ("actual_release_timestamp", "available_at", "retrieved_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["source_url"] = df["source_url"].fillna("").astype(str).str.strip()
    before = len(df)
    df = df[df.ticker.ne("") & df.actual_release_timestamp.notna() & df.available_at.notna()]
    if len(df) != before: issues.append(f"dropped_missing_ticker_or_timestamps:{before-len(df)}")
    # Availability must never predate the stated actual release; when it does, reject the row.
    bad = df.available_at < df.actual_release_timestamp
    if bad.any():
        issues.append(f"dropped_available_before_release:{int(bad.sum())}")
        df = df.loc[~bad]
    no_source = df.source_url.eq("")
    if no_source.any():
        issues.append(f"rows_missing_source_url:{int(no_source.sum())}")
        df = df.loc[~no_source]
    df = df.drop_duplicates(subset=["ticker", "fiscal_period", "actual_release_timestamp", "source_url"], keep="last")
    return df.reset_index(drop=True), issues


def features_asof(earnings: pd.DataFrame, cutoff) -> dict[str, float]:
    """Build only the latest eligible constituent-level event summary as of cutoff.

    This does not claim to be QQQM-weighted. Caller must apply point-in-time ETF
    holdings weights for portfolio aggregation. Features are NaN when no valid
    observation exists, allowing the experiment to gate coverage honestly.
    """
    if earnings.empty:
        return {"earnings_eps_surprise_pct": float("nan"), "earnings_revenue_surprise_pct": float("nan"), "earnings_guidance_change_pct": float("nan"), "earnings_event_count_20d": 0.0}
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    df, _ = validate_earnings_frame(earnings)
    df = df[df.available_at <= cutoff].sort_values("available_at")
    if df.empty:
        return {"earnings_eps_surprise_pct": float("nan"), "earnings_revenue_surprise_pct": float("nan"), "earnings_guidance_change_pct": float("nan"), "earnings_event_count_20d": 0.0}
    latest = df.iloc[-1]
    def surprise(actual, estimate):
        if pd.isna(actual) or pd.isna(estimate) or estimate == 0: return float("nan")
        return float((actual-estimate)/abs(estimate)*100.0)
    return {
        "earnings_eps_surprise_pct": surprise(latest.eps_actual, latest.eps_estimate),
        "earnings_revenue_surprise_pct": surprise(latest.revenue_actual, latest.revenue_estimate),
        "earnings_guidance_change_pct": float(latest.guidance_change_pct) if pd.notna(latest.guidance_change_pct) else float("nan"),
        "earnings_event_count_20d": float(((df.available_at >= cutoff-pd.Timedelta(days=20)) & (df.available_at <= cutoff)).sum()),
    }


def load_earnings_csv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.exists(): return pd.DataFrame(columns=REQUIRED_COLUMNS), [f"file_not_found:{path}"]
    return validate_earnings_frame(pd.read_csv(path))


def aggregate_constituent_features_asof(
    earnings: pd.DataFrame,
    holdings: pd.DataFrame,
    cutoff,
    min_weight_coverage: float = 0.70,
) -> dict[str, float]:
    """Aggregate constituent earnings using holdings known at cutoff.

    Holdings columns: ticker, weight, available_at, source_url; an optional
    snapshot_id is strongly recommended when a vendor provides one. If present,
    the latest eligible snapshot_id is selected. Otherwise, records are grouped
    by UTC availability date and the latest eligible row per ticker is selected.
    Weights may be decimals or percentages. The aggregate is withheld when the
    weighted coverage is below threshold, avoiding a misleading sparse estimate.
    """
    empty = {
        "earnings_eps_surprise_pct": float("nan"),
        "earnings_revenue_surprise_pct": float("nan"),
        "earnings_guidance_change_pct": float("nan"),
        "earnings_weight_coverage": 0.0,
        "earnings_constituent_count": 0.0,
    }
    if holdings is None or holdings.empty or earnings is None or earnings.empty:
        return empty
    required = {"ticker", "weight", "available_at", "source_url"}
    if not required.issubset(holdings.columns):
        return empty
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    h = holdings.copy()
    h["ticker"] = h.ticker.astype(str).str.upper().str.strip()
    h["weight"] = pd.to_numeric(h.weight, errors="coerce")
    h["available_at"] = pd.to_datetime(h.available_at, utc=True, errors="coerce")
    h["source_url"] = h.source_url.fillna("").astype(str).str.strip()
    h = h[h.available_at.le(cutoff) & h.source_url.ne("") & h.ticker.ne("") & h.weight.gt(0)]
    if h.empty:
        return empty
    # Select one point-in-time holdings snapshot, not a future/current portfolio.
    # Prefer vendor snapshot_id. Without it, infer a snapshot from source URL and
    # a 30-minute timestamp cluster. This tolerates per-row download timestamps
    # while avoiding mixing morning and afternoon holdings exports from one day.
    if "snapshot_id" in h.columns:
        h["snapshot_id"] = h["snapshot_id"].fillna("").astype(str).str.strip()
    identified = h[h.get("snapshot_id", pd.Series("", index=h.index)).ne("")].copy()
    if not identified.empty:
        latest_snapshot_row = identified.sort_values("available_at").iloc[-1]
        latest_snapshot = latest_snapshot_row["snapshot_id"]
        h = identified[identified["snapshot_id"].eq(latest_snapshot)]
    else:
        h = h.sort_values("available_at").copy()
        h["_source_key"] = h["source_url"].astype(str)
        # Timestamp clusters break when adjacent records are over 30 minutes apart.
        gap = h.groupby("_source_key")["available_at"].diff().gt(pd.Timedelta(minutes=30))
        h["_cluster"] = gap.fillna(False).groupby(h["_source_key"]).cumsum()
        h["_cluster_key"] = h["_source_key"] + "|" + h["_cluster"].astype(str)
        latest_cluster = h.sort_values("available_at").iloc[-1]["_cluster_key"]
        h = h[h["_cluster_key"].eq(latest_cluster)]
    h = h.sort_values("available_at").drop_duplicates("ticker", keep="last")
    total_weight = float(h.weight.sum())
    if total_weight <= 0:
        return empty
    if total_weight > 1.5:
        h["weight"] = h.weight / 100.0
    h["weight"] = h.weight / h.weight.sum()
    e, _ = validate_earnings_frame(earnings)
    e = e[e.available_at.le(cutoff)].sort_values("available_at").drop_duplicates("ticker", keep="last")
    joined = h.merge(e, on="ticker", how="inner", suffixes=("_holding", "_earnings"))
    if joined.empty:
        return empty
    joined["eps_surprise"] = (joined.eps_actual - joined.eps_estimate) / joined.eps_estimate.abs().replace(0, float("nan")) * 100
    joined["revenue_surprise"] = (joined.revenue_actual - joined.revenue_estimate) / joined.revenue_estimate.abs().replace(0, float("nan")) * 100
    joined["guidance"] = pd.to_numeric(joined.guidance_change_pct, errors="coerce")
    valid_any = joined[["eps_surprise", "revenue_surprise", "guidance"]].notna().any(axis=1)
    coverage = float(joined.loc[valid_any, "weight"].sum())
    result = dict(empty)
    result["earnings_weight_coverage"] = coverage
    result["earnings_constituent_count"] = float(valid_any.sum())
    if coverage < min_weight_coverage:
        return result
    for source, target in [("eps_surprise", "earnings_eps_surprise_pct"), ("revenue_surprise", "earnings_revenue_surprise_pct"), ("guidance", "earnings_guidance_change_pct")]:
        valid = joined[source].notna()
        denom = float(joined.loc[valid, "weight"].sum())
        if denom > 0:
            result[target] = float((joined.loc[valid, source] * joined.loc[valid, "weight"]).sum() / denom)
    return result
