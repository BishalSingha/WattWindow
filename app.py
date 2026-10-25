"""Cheapest Hours: electricity price forecast and appliance planner for Finnish households."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine

st.set_page_config(page_title="WattWindow", page_icon="⚡", layout="wide")

GREEN, AMBER, RED, GREY, NAVY, GOLD = "#2e9e5b", "#e8a317", "#d64545", "#c4c9d1", "#14213d", "#f2a900"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Typical run length (hours) and average power (kW); adjustable by the user
APPLIANCES = {
    "Washing machine": (2, 0.5),
    "Dishwasher": (2, 0.6),
    "Tumble dryer": (2, 1.3),
    "Water heater": (3, 2.0),
    "EV charging": (4, 7.4),
    "Sauna heater": (2, 6.0),
    "Custom": (3, 2.0),
}


# ---------------------------------------------------------------------------
# Loading data (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_assets():
    return engine.load_assets()


@st.cache_data(ttl=1800, show_spinner="Fetching the latest prices and weather ...")
def get_live():
    try:
        token = st.secrets["ENTSOE_API_KEY"]
        prices = engine.fetch_live_prices(token)
        weather = engine.fetch_live_weather()
        return prices, weather, None
    except Exception as err:
        return None, None, f"{type(err).__name__}: {err}"[:300]


model, feature_cols, history, test_results, metrics = get_assets()
final_mae = metrics["mae_cents"]["Gradient boosting (expanded features)"]

with st.sidebar:
    st.header("Settings")
    vat = st.checkbox("Show prices including VAT (25.5%)", value=False)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Prices are in cents per kilowatt-hour (c/kWh).")

# Trying live data first, then falling back to the saved history
prices, weather, live_error = get_live()
live_mode = live_error is None
if live_mode:
    try:
        timeline, last_known = engine.build_timeline(prices, weather, model, feature_cols)
    except Exception as err:
        live_mode = False
        live_error = f"{type(err).__name__}: {err}"[:300]
if not live_mode:
    prices, weather = engine.fallback_inputs(history)
    timeline, last_known = engine.build_timeline(prices, weather, model, feature_cols)

if live_mode:
    now_h = pd.Timestamp.now(tz=engine.TZ).floor("h")
else:
    now_h = last_known
plan_start = now_h + pd.Timedelta(hours=1)
vat_factor = engine.VAT if vat else 1.0
unit = "c/kWh incl. VAT" if vat else "c/kWh"


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def price_chart(tl_all, view_start, plan_end, best):
    tl = tl_all[(tl_all["time"] >= view_start) & (tl_all["time"] < plan_end)].copy()
    tl["c"] = tl["price"] / 10 * vat_factor
    tl["x"] = tl["time"].dt.tz_localize(None)
    future = tl[tl["time"] >= plan_start]
    low, high = future["c"].quantile([0.33, 0.66])

    def colour(v):
        return GREEN if v <= low else (RED if v >= high else AMBER)

    fig = go.Figure()
    groups = [
        ("Earlier", tl[tl["time"] < plan_start], 1.0, False),
        ("Published price", future[future["source"] == "Published"], 1.0, True),
        ("Forecast", future[future["source"] == "Forecast"], 0.55, True),
    ]
    for name, g, opacity, coloured in groups:
        if g.empty:
            continue
        fig.add_trace(go.Bar(
            x=g["x"], y=g["c"], name=name, opacity=opacity, offset=0, width=3600000 * 0.9,
            marker_color=[colour(v) for v in g["c"]] if coloured else GREY,
            hovertemplate="%{x|%a %d %b %H:%M}<br>%{y:.1f} " + unit + "<extra>" + name + "</extra>",
        ))

    # Highlighting the best window
    fig.add_shape(
        type="rect", xref="x", yref="paper", y0=0, y1=1,
        x0=best["start"].tz_localize(None), x1=best["end"].tz_localize(None),
        fillcolor="rgba(20,33,61,0.10)", line=dict(color=NAVY, width=2),
    )
    # Marking where published prices end and the forecast begins
    boundary = (last_known + pd.Timedelta(hours=1)).tz_localize(None)
    if tl["x"].min() < boundary < tl["x"].max():
        fig.add_shape(type="line", xref="x", yref="paper", x0=boundary, x1=boundary, y0=0, y1=1,
                      line=dict(color="#555555", dash="dash", width=1))
        fig.add_annotation(x=boundary, y=1.05, yref="paper", text="Forecast starts", showarrow=False,
                           xanchor="left")
    fig.update_layout(
        barmode="overlay", height=430, bargap=0, margin=dict(t=40, b=10, l=10, r=10),
        yaxis_title=f"Price ({unit})", xaxis_tickformat="%a %d %b<br>%H:%M",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚡ WattWindow")
st.caption("When is electricity cheapest in Finland over the next few days? "
           "Choose what you want to run and we find the best time.")

if live_mode:
    st.success(f"Live data. Prices are published up to {last_known:%a %d %b, %H:%M}; "
               "hours after that are our forecast.")
else:
    st.warning(f"Live data is unavailable, so this page shows saved data that ends {last_known:%d %b %Y}. "
               f"The forecast below is for that date, not for today. ({live_error})")

tab_plan, tab_patterns, tab_accuracy, tab_about = st.tabs(
    ["Plan my appliance", "Price patterns", "How reliable is it?", "About"])


# ---------------------------------------------------------------------------
# Tab 1: planner
# ---------------------------------------------------------------------------
with tab_plan:
    published = timeline[timeline["source"] == "Published"]
    latest = published[published["time"] <= now_h].iloc[-1]
    next24 = timeline[(timeline["time"] >= plan_start) & (timeline["time"] < plan_start + pd.Timedelta(hours=24))]
    cheapest = next24.loc[next24["price"].idxmin()]
    priciest = next24.loc[next24["price"].idxmax()]

    t1, t2, t3 = st.columns(3)
    t1.metric("Latest known price", f"{latest['price'] / 10 * vat_factor:.1f} {unit}",
              help=f"Hour starting {latest['time']:%a %H:%M}")
    t2.metric("Cheapest hour, next 24 h", f"{cheapest['time']:%a %H:%M}",
              f"{cheapest['price'] / 10 * vat_factor:.1f} {unit}", delta_color="off")
    t3.metric("Most expensive hour, next 24 h", f"{priciest['time']:%a %H:%M}",
              f"{priciest['price'] / 10 * vat_factor:.1f} {unit}", delta_color="off")

    st.divider()
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.4])
    appliance = c1.selectbox("What do you want to run?", list(APPLIANCES))
    default_hours, default_kw = APPLIANCES[appliance]
    hours = c2.slider("For how many hours?", 1, 8, default_hours, key=f"hours_{appliance}")
    power_kw = c3.number_input("Average power (kW)", 0.1, 22.0, float(default_kw), step=0.1,
                               key=f"kw_{appliance}", help="Typical values; adjust to your own appliance.")
    look = c4.radio("Look ahead", ["24 h", "48 h", "As far as possible"], index=1, horizontal=True)
    horizon_hours = {"24 h": 24, "48 h": 48, "As far as possible": 24 * 14}[look]

    result = engine.plan_windows(timeline, plan_start, hours, horizon_hours, power_kw, vat)
    if result is None:
        st.error("Not enough price data for that duration. Try a shorter run or a longer look-ahead.")
    else:
        windows, horizon_tl = result
        summary = engine.summarize(windows, hours, power_kw)
        best = summary["best"]

        st.markdown(f"### Best time to run it: **{best['start']:%a %d %b, %H:%M} to {best['end']:%H:%M}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average price in that window", f"{best['mean_c']:.1f} {unit}")
        m2.metric("Estimated cost", f"€{best['cost_eur']:.2f}")
        m3.metric("Saving vs. an average time", f"€{summary['save_vs_avg_eur']:.2f}",
                  f"{summary['save_vs_avg_c']:.1f} c/kWh cheaper")
        m4.metric("Saving vs. the worst time", f"€{summary['save_vs_worst_eur']:.2f}")

        if best["uses_forecast"]:
            st.info(f"This window includes forecast hours, so its price is an estimate "
                    f"(typical error about {final_mae:.1f} c/kWh). Published hours are exact.")
        else:
            st.success("This window uses published prices only, so its price is exact.")

        view_start = plan_start - pd.Timedelta(hours=12)
        plan_end = horizon_tl["time"].max() + pd.Timedelta(hours=1)
        st.plotly_chart(price_chart(timeline, view_start, plan_end, best), use_container_width=True)
        st.caption("Green is cheap, amber is medium and red is expensive, relative to the period shown. "
                   "Faded bars are forecasts. The outlined box is the best window.")

        top = summary["top"]
        table = pd.DataFrame({
            "Start": top["start"].dt.strftime("%a %d %b %H:%M"),
            "End": top["end"].dt.strftime("%H:%M"),
            f"Average price ({unit})": top["mean_c"].round(1),
            "Estimated cost (€)": top["cost_eur"].round(2),
            "Based on": np.where(top["uses_forecast"], "Includes forecast", "Published prices"),
        })
        st.markdown("**Your best three options**")
        st.dataframe(table, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: price patterns
# ---------------------------------------------------------------------------
with tab_patterns:
    st.subheader("What drives Finnish electricity prices?")
    st.caption(f"Based on hourly prices from {history.index.min():%b %Y} to {history.index.max():%b %Y}.")

    season = st.radio("Season", ["All year", "Winter (Nov to Mar)", "Summer (Jun to Aug)"], horizontal=True)
    h = (history["price"] / 10).to_frame("p")
    if season.startswith("Winter"):
        h = h[h.index.month.isin([11, 12, 1, 2, 3])]
    elif season.startswith("Summer"):
        h = h[h.index.month.isin([6, 7, 8])]
    grid = h.groupby([h.index.dayofweek, h.index.hour])["p"].mean().unstack()

    heat = go.Figure(go.Heatmap(
        z=grid.values, x=list(grid.columns), y=[WEEKDAYS[i] for i in grid.index],
        colorscale="YlOrRd", colorbar=dict(title="c/kWh"),
        hovertemplate="%{y} %{x}:00<br>%{z:.1f} c/kWh<extra></extra>",
    ))
    heat.update_layout(height=340, margin=dict(t=30, b=10), xaxis_title="Hour of day",
                       title="Average price by weekday and hour (before VAT)")
    heat.update_yaxes(autorange="reversed")
    st.plotly_chart(heat, use_container_width=True)
    st.caption("Prices are usually lowest at night and highest in the morning and evening peaks. "
               "This daily rhythm is the main thing the planner exploits.")

    left, right = st.columns(2)
    with left:
        p = history["price"] / 10
        line = go.Figure()
        line.add_trace(go.Scatter(x=p.index.tz_localize(None), y=p.values, name="Hourly price",
                                  line=dict(width=0.6, color=GREY)))
        line.add_trace(go.Scatter(x=p.index.tz_localize(None),
                                  y=p.rolling(24 * 30, min_periods=24 * 7).mean(), name="30-day average",
                                  line=dict(width=2.5, color=NAVY)))
        line.update_layout(height=340, margin=dict(t=40, b=10), yaxis_title="c/kWh (before VAT)",
                           title="Price over time", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(line, use_container_width=True)
        st.caption("The January 2024 cold snap is the large spike. Prices have been calmer since.")
    with right:
        if "wind_actual" in history.columns:
            d = pd.DataFrame({"wind": history["wind_actual"], "p": history["price"] / 10}).dropna()
            d["bin"] = pd.cut(d["wind"], bins=10)
            g = d.groupby("bin", observed=True)["p"]
            mid = [iv.mid for iv in g.median().index]
            wind = go.Figure()
            wind.add_trace(go.Scatter(x=mid, y=g.quantile(0.75).values, line=dict(width=0), showlegend=False))
            wind.add_trace(go.Scatter(x=mid, y=g.quantile(0.25).values, fill="tonexty", line=dict(width=0),
                                      fillcolor="rgba(242,169,0,0.25)", name="Middle 50% of hours"))
            wind.add_trace(go.Scatter(x=mid, y=g.median().values, name="Median price", mode="lines+markers",
                                      line=dict(color=NAVY)))
            wind.update_layout(height=340, margin=dict(t=40, b=10), xaxis_title="Wind power output (MW)",
                               yaxis_title="c/kWh (before VAT)", title="Price vs wind power",
                               legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(wind, use_container_width=True)
            st.caption("When Finland's wind farms produce a lot, electricity tends to be cheaper.")

    vol = (history["price"].resample("MS").std() / 10)
    bars = go.Figure(go.Bar(x=vol.index.tz_localize(None), y=vol.values, marker_color=NAVY))
    bars.update_layout(height=280, margin=dict(t=40, b=10), yaxis_title="c/kWh",
                       title="How much prices swing within each month (standard deviation)")
    st.plotly_chart(bars, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: reliability
# ---------------------------------------------------------------------------
with tab_accuracy:
    st.subheader("Can you trust the forecast?")
    wd = metrics["within_day"]
    tr = test_results

    st.markdown(
        f"We tested the forecast on **{wd['days']} days it had never seen**. "
        f"Following the advice was about **{wd['model']:.1f} c/kWh cheaper than a random time of day**, "
        f"and the chosen window beat the day's average price on **{wd['beat_avg_share']:.0%} of days**. "
        f"Its price guesses were off by about **{final_mae:.1f} c/kWh** on average, "
        f"**{metrics['improvement_vs_baseline_pct']:.0f}% less** than simply assuming 'same hour last week'.")

    a1, a2 = st.columns(2)
    with a1:
        bt = go.Figure(go.Bar(
            x=["Simple rule<br>(same hour last week)", "Cheapest Hours", "Perfect hindsight"],
            y=[wd["naive"], wd["model"], wd["best"]], marker_color=["#8b95a3", GOLD, NAVY],
            text=[f"{v:.1f}" for v in (wd["naive"], wd["model"], wd["best"])], textposition="outside"))
        bt.update_layout(height=340, margin=dict(t=50, b=10), yaxis_title="c/kWh cheaper than a random time",
                         title="Does following the advice pay off?")
        st.plotly_chart(bt, use_container_width=True)
    with a2:
        mae = pd.Series(metrics["mae_cents"]).sort_values()
        colours = [GOLD if v == mae.min() else "#4c78a8" for v in mae.values]
        cmp_fig = go.Figure(go.Bar(x=mae.values, y=mae.index, orientation="h", marker_color=colours,
                                   text=[f"{v:.2f}" for v in mae.values], textposition="outside"))
        cmp_fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=40), xaxis_title="Average error (c/kWh)",
                              title="Forecast error by model (lower is better)")
        cmp_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(cmp_fig, use_container_width=True)

    st.info(
        f"**Honest limits.** The simple 'same hour last week' rule picked windows almost as well as our "
        f"forecast ({wd['naive']:.2f} vs {wd['model']:.2f} c/kWh saved): most of the value comes from the weekly "
        f"price pattern. Our test period ({tr.index.min():%b %Y} to {tr.index.max():%b %Y}) was also unusually calm, "
        f"so sudden cold snaps and price spikes are where the forecast is weakest.")

    st.markdown("#### Forecast vs. actual price")
    default_end = tr.index.max().date()
    default_start = (tr.index.max() - pd.Timedelta(days=14)).date()
    sel = st.date_input("Period to look at", value=(default_start, default_end),
                        min_value=tr.index.min().date(), max_value=default_end)
    start_d, end_d = sel if isinstance(sel, (tuple, list)) and len(sel) == 2 else (default_start, default_end)
    mask = (tr.index.date >= start_d) & (tr.index.date <= end_d)
    view = tr[mask]
    xs = view.index.tz_localize(None)
    cmp_line = go.Figure()
    cmp_line.add_trace(go.Scatter(x=xs, y=view["actual"] / 10, name="Actual price", line=dict(color=NAVY)))
    cmp_line.add_trace(go.Scatter(x=xs, y=view["predicted"] / 10, name="Our forecast (48 h ahead)",
                                  line=dict(color=GOLD)))
    if st.checkbox("Also show the simple rule (same hour last week)"):
        cmp_line.add_trace(go.Scatter(x=xs, y=view["naive"] / 10, name="Simple rule",
                                      line=dict(color="#8b95a3", dash="dot")))
    cmp_line.update_layout(height=360, margin=dict(t=20, b=10), yaxis_title="c/kWh (before VAT)",
                           legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(cmp_line, use_container_width=True)

    by_month = (tr["actual"] - tr["predicted"]).abs().groupby(tr.index.strftime("%Y-%m")).mean() / 10
    month_fig = go.Figure(go.Bar(x=by_month.index, y=by_month.values, marker_color="#4c78a8"))
    month_fig.update_layout(height=260, margin=dict(t=40, b=10), yaxis_title="c/kWh",
                            title="Average forecast error by month of the test period")
    st.plotly_chart(month_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: about
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("What is Cheapest Hours?")
    st.markdown(
        "If you have a spot-price electricity contract in Finland, your price changes every hour. "
        "Cheapest Hours shows you the best time to run something flexible, like laundry, EV charging "
        "or the water heater, so you pay less for the same energy.")

    st.subheader("How it works")
    st.markdown(
        "1. **Collect open data.** Day-ahead prices (ENTSO-E), weather (Finnish Meteorological Institute) "
        "and wind power (Fingrid).\n"
        "2. **Learn the patterns.** A model learns how hour, weekday, season, recent prices and weather "
        "affect the price.\n"
        "3. **Forecast what is not published yet.** Tomorrow's prices are published around lunchtime. "
        "Hours after that are forecast up to 48 hours further ahead.\n"
        "4. **Find your window.** We test every possible start time and show the cheapest one.")

    st.subheader("Good to know")
    st.markdown(
        "- Published prices are exact; forecast hours are estimates, not guarantees.\n"
        "- Savings are usually modest for small appliances and larger for EV charging and heating.\n"
        "- Sudden cold snaps and price spikes are hard to predict.\n"
        "- Power draw values are typical averages; adjust them to your own appliance.\n"
        "- Prices are shown before VAT unless you tick the box in the sidebar.")

    st.subheader("Data and privacy")
    st.markdown("We use only public grid, market and weather data. We collect no personal data.")

    st.subheader("Data status")
    st.markdown(
        f"- Mode: **{'live' if live_mode else 'saved data (live fetch unavailable)'}**\n"
        f"- Prices published up to: **{last_known:%a %d %b %Y, %H:%M}**\n"
        f"- Training history: **{history.index.min():%b %Y} to {history.index.max():%b %Y}**")

    st.caption("Built by Bishal, Shounak and Aakash for Introduction to Data Science, University of Helsinki, 2026.")
