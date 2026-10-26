# ⚡ WattWindow

**Find the cheapest time to run your appliances on a Finnish spot-price electricity contract.**

Pick what you want to run (washing machine, EV charging, sauna heater, ...), and WattWindow shows the cheapest window over the next few days, what it will cost, and how much you save compared with running it at an average time.

**🔗 Live app:** https://wattwindow.streamlit.app/

*Course project for Introduction to Data Science, University of Helsinki, 2026.*

---

## What it does

- **Plans your appliance.** Choose an appliance, a duration and a look-ahead, and get the best start time, an estimated cost and your top three options.
- **Shows exact vs. forecast prices.** Hours already published by the day-ahead market are exact. For hours that are not published yet, a machine learning model forecasts the price up to 48 hours beyond the last published hour. The two are clearly marked in the planner chart.
- **Explains Finnish price patterns.** Interactive charts of price by weekday and hour, price over time, price vs. wind power, and monthly volatility.
- **Is honest about reliability.** A dedicated tab shows forecast vs. actual prices on unseen data, compares all models tested, and states the limitations.

## How it compares with existing tools

Most price apps show prices that are already published. A few retailers publish multi-day forecasts as charts or articles. WattWindow turns a forecast into a concrete "when should I run this?" answer for your appliance, and tests whether following that answer would actually have saved money.

## Key results

Evaluated on about 200 days of held-out data (Feb to Sep 2026), strictly chronological split, prices before VAT.

| Measure | Result |
|---|---|
| Average forecast error (48 h ahead) | **2.8 c/kWh** (27.6 EUR/MWh) |
| Error of a "same hour last week" baseline | 3.4 c/kWh |
| Error reduction vs. baseline | **19%** |
| Cost of a 3-hour run using our advice vs. a random time | **about 1.1 c/kWh cheaper** |
| Days the chosen window beat the daily average | **77%** (199 days tested) |
| Best possible saving with perfect hindsight | about 2.1 c/kWh |

**What the numbers do not show:** a simple "same hour last week" rule picked windows almost as well as our model (1.13 vs. 1.11 c/kWh saved). Most of the practical value comes from the weekly price pattern. The test period was also unusually calm; sudden cold snaps and price spikes are where the forecast is weakest.

### Models compared

| Model | Average error (c/kWh) |
|---|---|
| Same hour last week (baseline) | 3.42 |
| Gradient boosting | 2.84 |
| Gradient boosting (tuned) | 2.88 |
| LSTM (price history only) | 2.88 |
| LSTM (+ weather, calendar) | 2.79 |
| **Gradient boosting (expanded features)** | **2.76** |

Hyperparameter tuning made almost no difference, and neither did a more complex architecture. Adding better features helped most.

## How it works

1. **Collect open data.** Day-ahead prices, weather and wind power (see below).
2. **Clean and align.** Everything is put on one hourly grid. Resolution changes (Fingrid datasets, and the move of Finnish day-ahead prices from hourly to 15-minute in October 2025) and a timezone mismatch are handled explicitly.
3. **Engineer features.** Lagged prices, weekly rolling average, price momentum, a heating-degree transform of temperature, calendar and holiday flags, and the price's position within its recent range.
4. **Train a model.** Gradient boosting predicts the price 48 hours ahead. Applying it to the last 48 feature rows covers exactly the 48 hours after the last published price, so a forecast never overlaps an exact hour.
5. **Plan the window.** Every possible start time is scored, and the cheapest is shown.

Fingrid's own wind and demand forecasts were left out of the final model: they correlated 0.98 with actual values, which suggests they were archived close to delivery time and would leak future information.

## Data sources

| Source | Used for | Access |
|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | Day-ahead spot prices, Finland | Free API token |
| [Finnish Meteorological Institute open data](https://en.ilmatieteenlaitos.fi/open-data) | Temperature and wind speed, Helsinki | No key |
| [Fingrid open data](https://data.fingrid.fi) | Wind power output (exploration and charts) | Free API key |

No personal data is collected.

## Repository structure

*Adjust this to match your repo.*

```
WattWindow/
├── app.py                          # Streamlit dashboard (UI)
├── engine.py                       # Data loading, live fetching, features, forecast, planner
├── requirements.txt
├── price_model.joblib              # Trained model
├── feature_cols.joblib             # Exact feature order used in training
├── history_full.csv                # Hourly history, used for charts and as offline fallback
├── test_results.csv                # Forecast vs. actual on the test period
├── metrics.json                    # Numbers shown in the reliability tab
└── WattWindow.ipynb   # Full analysis notebook
```

## Run it locally

```bash
# Installing the dependencies
pip install -r requirements.txt

# Adding your ENTSO-E token (optional: without it the app falls back to the saved history)
mkdir -p .streamlit
echo 'ENTSOE_API_KEY = "your-token-here"' > .streamlit/secrets.toml

# Starting the app
streamlit run app.py
```

With a valid token the app fetches live prices and weather (cached for 30 minutes). Without one, or if a fetch fails, it shows a clear warning and uses the saved history instead.

**Never commit `.streamlit/secrets.toml`.** Add it to `.gitignore`.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`.
3. Under **Advanced settings → Secrets**, add `ENTSOE_API_KEY = "your-token-here"`.
4. If the model fails to load, make sure the versions in `requirements.txt` match the training environment (the notebook ran on Python 3.13), and pick a matching Python version in the advanced settings.

`requirements.txt`:

```
streamlit
plotly
joblib
holidays
entsoe-py
fmiopendata
scikit-learn==1.6.1
numpy==2.1.3
pandas==2.2.3
```

## Reproduce the analysis

The full pipeline (data collection, preprocessing, EDA, models, backtest and export for the dashboard) is in `WattWindow.ipynb`.

1. Open it in Google Colab.
2. Add `FINGRID_API_KEY` and `ENTSOE_API_KEY` under Colab Secrets and enable notebook access.
3. Run all cells. The Fingrid pulls and the LSTM training take the longest, so expect roughly 20 to 40 minutes.

## Limitations

- Forecast hours are estimates, not guarantees. Typical error is about 2.8 c/kWh.
- Evaluated on one calm period. Performance during volatile periods, such as the January 2024 cold snap, is likely worse.
- Weather beyond the latest observation is assumed unchanged, which only affects the 24-hour-lagged weather features.
- Appliance power values are typical averages. Savings are small for small appliances (cents per run) and larger for EV charging and heating.
- Estimates only. This is not financial advice.

## Team

Bishal Singha, Shounak and Aakash. Introduction to Data Science, University of Helsinki, 2026.

## Acknowledgements

Price data from ENTSO-E, weather from the Finnish Meteorological Institute and wind data from Fingrid, all used under their open data terms.
