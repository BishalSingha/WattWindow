"""WattWindow: data loading, live fetching, forecasting and window planning."""
import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

TZ = "Europe/Helsinki"
VAT = 1.255      # Finnish VAT on electricity (25.5%)
HORIZON = 48     # forecasting horizon of the trained model, in hours


# ---------------------------------------------------------------------------
# Loading the saved assets exported from the notebook
# ---------------------------------------------------------------------------
def read_tz_csv(path):
    """Timestamp-indexed CSV, returned in Helsinki time."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    return df.sort_index()


def load_assets(folder="."):
    """Model, feature order, full history, test-period results and metrics."""
    model = joblib.load(f"{folder}/price_model.joblib")
    feature_cols = joblib.load(f"{folder}/feature_cols.joblib")
    history = read_tz_csv(f"{folder}/history_full.csv")
    test_results = read_tz_csv(f"{folder}/test_results.csv")
    with open(f"{folder}/metrics.json") as f:
        metrics = json.load(f)
    return model, feature_cols, history, test_results, metrics


# ---------------------------------------------------------------------------
# Fetching live data (ENTSO-E prices, FMI weather)
# ---------------------------------------------------------------------------
def fetch_live_prices(token, days=21):
    """Hourly day-ahead prices for Finland in EUR/MWh, including hours already published for tomorrow."""
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=token)
    now = pd.Timestamp.now(tz=TZ).floor("h")
    start = now - pd.Timedelta(days=days)
    end = now + pd.Timedelta(days=2)
    prices = client.query_day_ahead_prices("FI", start=start, end=end)
    prices = prices.resample("1h").mean().dropna()
    prices.index = prices.index.tz_convert(TZ)
    return prices


def fetch_live_weather(days=21):
    """Hourly temperature and wind speed observations for Helsinki."""
    from fmiopendata.wfs import download_stored_query

    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    rows = []
    chunk_start = start
    while chunk_start < end:
        # Requesting one week at a time, since FMI caps a request at 168 hours
        chunk_end = min(chunk_start + timedelta(hours=167, minutes=59), end)
        obs = download_stored_query(
            "fmi::observations::weather::multipointcoverage",
            args=[
                "place=Helsinki",
                f"starttime={chunk_start.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                f"endtime={chunk_end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                "parameters=temperature,windspeedms",
            ],
        )
        for timestamp, stations in obs.data.items():
            for station_name, params in stations.items():
                row = {"time": timestamp, "station": station_name}
                for param_name, param_val in params.items():
                    row[param_name] = param_val["value"]
                rows.append(row)
        chunk_start = chunk_end + timedelta(seconds=1)

    if not rows:
        raise RuntimeError("FMI returned no observations")
    df = pd.DataFrame(rows).set_index("time").resample("1h").mean(numeric_only=True)
    df.index = df.index.tz_localize("UTC").tz_convert(TZ)
    return df[["Air temperature", "Wind speed"]]


def fallback_inputs(history, days=21):
    """Last weeks of the saved history, used when live data is unavailable."""
    recent = history.tail(24 * days)
    return recent["price"], recent[["Air temperature", "Wind speed"]]


# ---------------------------------------------------------------------------
# Building the features exactly as in the notebook
# ---------------------------------------------------------------------------
def compute_features(hist):
    """Feature table for an hourly, Helsinki-time DataFrame with price, Air temperature and Wind speed."""
    d = hist.sort_index().copy()

    # Lagged prices and weekly rolling average
    d["price_lag_24h"] = d["price"].shift(24)
    d["price_lag_168h"] = d["price"].shift(168)
    d["price_lag_336h"] = d["price"].shift(336)
    d["price_rolling_168h_mean"] = d["price"].rolling(168).mean()

    # Lagged weather
    d["temp_lag_24h"] = d["Air temperature"].shift(24)
    d["windspeed_lag_24h"] = d["Wind speed"].shift(24)

    # Price momentum
    d["price_diff_1h"] = d["price"].diff(1)
    d["price_diff_24h"] = d["price"].diff(24)

    # Heating-degree transform of temperature, lagged
    heating_degree = (17 - d["Air temperature"]).clip(lower=0)
    d["heating_degree_lag_24h"] = heating_degree.shift(24)

    # Position of the price within its own recent range
    rolling_min = d["price"].rolling(168).min()
    rolling_max = d["price"].rolling(168).max()
    d["price_relative_position"] = (d["price"] - rolling_min) / (rolling_max - rolling_min)

    # Calendar features
    d["hour"] = d.index.hour
    d["day_of_week"] = d.index.dayofweek
    d["month"] = d.index.month
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)

    try:
        import holidays

        fi_holidays = holidays.Finland()
        d["is_holiday"] = d.index.to_series().dt.date.isin(fi_holidays).astype(int)
        tomorrow = (d.index + pd.Timedelta(days=1)).to_series().dt.date
        d["day_before_holiday"] = tomorrow.isin(fi_holidays).astype(int).values
    except ImportError:
        d["is_holiday"] = 0
        d["day_before_holiday"] = 0

    return d.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Building the timeline: published prices, then the forecast beyond them
# ---------------------------------------------------------------------------
def build_timeline(prices, weather, model, feature_cols):
    """Published prices followed by 48 forecast hours. Returns (timeline, last_known).

    The model predicts the price 48 hours after a feature row. Applying it to the
    last 48 feature rows therefore covers exactly the 48 hours after the last
    published price, so no forecast ever overlaps an exact, published hour.
    """
    prices = prices.dropna()
    last_known = prices.index.max()
    grid = pd.date_range(prices.index.min(), last_known, freq="h")

    hist = pd.DataFrame(index=grid)
    hist["price"] = prices.reindex(grid).interpolate(limit=3)
    wx = weather.reindex(grid)
    # Carrying the last observed weather forward, since only 24h-lagged weather enters the features
    hist["Air temperature"] = wx["Air temperature"].ffill()
    hist["Wind speed"] = wx["Wind speed"].ffill()

    feats = compute_features(hist)
    basis = feats.iloc[-HORIZON:].dropna(subset=feature_cols)
    if basis.empty:
        raise ValueError("Not enough recent history to build the features")

    predictions = model.predict(basis[feature_cols])
    forecast = pd.DataFrame({
        "time": basis.index + pd.Timedelta(hours=HORIZON),
        "price": predictions,
        "source": "Forecast",
    })
    published = pd.DataFrame({"time": prices.index, "price": prices.values, "source": "Published"})
    timeline = pd.concat([published, forecast], ignore_index=True)
    timeline = timeline.sort_values("time").reset_index(drop=True)
    return timeline, last_known


# ---------------------------------------------------------------------------
# Planning the cheapest window for an appliance
# ---------------------------------------------------------------------------
def plan_windows(timeline, start_from, hours, horizon_hours, power_kw, vat=False):
    """Every possible start time for a run of `hours` hours within the look-ahead, or None."""
    end_at = start_from + pd.Timedelta(hours=horizon_hours)
    tl = timeline[(timeline["time"] >= start_from) & (timeline["time"] < end_at)].reset_index(drop=True)
    if len(tl) < hours:
        return None

    price_c = tl["price"].to_numpy() / 10 * (VAT if vat else 1.0)   # EUR/MWh -> c/kWh
    is_forecast = (tl["source"] == "Forecast").to_numpy()
    times = tl["time"]

    rows = []
    for j in range(len(tl) - hours + 1):
        span = times.iloc[j + hours - 1] - times.iloc[j]
        if span != pd.Timedelta(hours=hours - 1):
            continue   # skipping windows that straddle a gap in the data
        rows.append({
            "j": j,
            "start": times.iloc[j],
            "end": times.iloc[j + hours - 1] + pd.Timedelta(hours=1),
            "mean_c": float(price_c[j:j + hours].mean()),
            "uses_forecast": bool(is_forecast[j:j + hours].any()),
        })
    if not rows:
        return None

    windows = pd.DataFrame(rows)
    windows["cost_eur"] = power_kw * hours * windows["mean_c"] / 100
    return windows, tl


def summarize(windows, hours, power_kw):
    """Best window, comparison with an average and the worst time, and the top three separate options."""
    best = windows.loc[windows["mean_c"].idxmin()]
    average_c = float(windows["mean_c"].mean())
    worst_c = float(windows["mean_c"].max())
    energy_kwh = power_kw * hours

    picked = []
    for idx, row in windows.sort_values("mean_c").iterrows():
        if all(abs(row["j"] - windows.loc[p, "j"]) >= hours for p in picked):
            picked.append(idx)
        if len(picked) == 3:
            break

    return {
        "best": best,
        "average_c": average_c,
        "worst_c": worst_c,
        "energy_kwh": energy_kwh,
        "save_vs_avg_c": average_c - float(best["mean_c"]),
        "save_vs_avg_eur": energy_kwh * (average_c - float(best["mean_c"])) / 100,
        "save_vs_worst_eur": energy_kwh * (worst_c - float(best["mean_c"])) / 100,
        "top": windows.loc[picked],
    }
