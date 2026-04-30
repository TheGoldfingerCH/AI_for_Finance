import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats as scipy_stats
from datetime import date

API_URL = "http://127.0.0.1:8000/global_predict"

st.set_page_config(layout="wide")
st.title("Bitcoin — ML Signal")

# ── SIDE PANEL ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    date_pivot = st.date_input("Training cutoff date", value=date(2024, 1, 1), min_value=date(2023, 11, 4), max_value=date.today())
    date_pivot = date_pivot.strftime("%Y-%m-%d")

    st.divider()
    st.subheader("Strategies")

    show_signal = st.checkbox("ML Signal (threshold 0.5)", value=True)
    show_flex = st.checkbox("Custom signal (adjustable threshold)", value=True)
    show_long_only = st.checkbox("Long only", value=True)
    show_random = st.checkbox("Random", value=False)

    st.divider()
    st.subheader("Buy signal probability threshold")
    threshold = st.slider("Threshold", min_value=0.5, max_value=0.75, value=0.5, step=0.005)

    horizon = 5

    st.divider()
    st.subheader("Model")
    model_name = st.selectbox(
    "Model",
    options=["xgb", "rnn", "linear"],
    format_func=lambda x: {"xgb": "XGBoost", "rnn": "RNN", "linear": "Linear Regression"}[x]
    )

    load = st.button("Load data", type="primary")

# ── DATA LOADING ─────────────────────────────────────────────────
if load:
    response = requests.get(API_URL, params={"date_pivot": date_pivot, "model_name": model_name})
    data = response.json()
    df = pd.DataFrame(data["df_for_streamlit"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    st.session_state["df"] = df

# ── CHARTS ───────────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]

    def simulate_portfolio(df, mode, horizon, threshold=0.5):
        n = len(df)

        decisions = []
        for i in range(0, n, horizon):
            row = df.iloc[i]
            end_idx = min(i + horizon, n - 1)
            price_end = df.iloc[end_idx]["Close"]

            if mode == "long_only":
                invest = True
            elif mode == "signal":
                invest = float(row["probability"]) >= 0.5
            elif mode == "flex signal":
                invest = float(row["probability"]) >= threshold
            elif mode == "random":
                import random
                random.seed(i)
                invest = random.random() > 0.5

            for j in range(i, min(i + horizon, n)):
                decisions.append(invest)

        decisions = decisions[:n]

        portfolio = 100.0
        values = [100.0]
        for i in range(1, n):
            daily_ret = (df.iloc[i]["Close"] - df.iloc[i-1]["Close"]) / df.iloc[i-1]["Close"]
            if decisions[i]:
                portfolio *= (1 + daily_ret)
            values.append(portfolio)

        return values

    fig = go.Figure()

    btc_base100 = df["Close"] / df["Close"].iloc[0] * 100
    fig.add_trace(go.Scatter(
        x=df["Date"], y=btc_base100,
        name="BTC Price (base 100)",
        line=dict(color="#378ADD", width=1.5, dash="dot"),
        opacity=0.5
    ))

    strat_colors = {
        "signal": "#EF9F27",
        "long_only": "#5B8DEF",
        "flex signal": "#1D9E75",
        "random": "#888780"
    }
    strat_labels = {
        "signal": "ML Signal",
        "long_only": "Long only",
        "flex signal": "Custom signal",
        "random": "Random"
    }
    active = {
        "signal": show_signal,
        "flex signal": show_flex,
        "long_only": show_long_only,
        "random": show_random
    }

    for mode, show in active.items():
        if show:
            vals = simulate_portfolio(df, mode, horizon, threshold)
            fig.add_trace(go.Scatter(
                x=df["Date"], y=vals,
                name=strat_labels[mode],
                line=dict(color=strat_colors[mode], width=2)
            ))

    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        template="plotly_dark",
        title="Strategy performance (base 100)",
        height=500,
        xaxis_title="Date",
        yaxis_title="Value (base 100)",
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig, use_container_width=True, key="fig_strategies")

    # Scatter plot
    st.subheader("Signal probability vs actual price movement")

    df_scatter = df.copy()
    df_scatter["real_return"] = df_scatter["Close"].shift(-horizon) / df_scatter["Close"] - 1
    df_scatter = df_scatter.dropna(subset=["real_return"])

    x_all = df_scatter["probability"].values
    y_all = df_scatter["real_return"].values * 100

    mask_above = x_all >= threshold
    mask_below = x_all < threshold

    def ols_slope_intercept(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        n = x.size
        if n < 1:
            return np.nan, np.nan
        if n == 1:
            return 0.0, float(y[0])
        if np.ptp(x) == 0.0 or np.std(x) == 0.0:
            return 0.0, float(np.mean(y))
        slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
        return float(slope), float(intercept)

    slope_all, intercept_all = ols_slope_intercept(x_all, y_all)
    slope_above, intercept_above = ols_slope_intercept(x_all[mask_above], y_all[mask_above])
    slope_below, intercept_below = ols_slope_intercept(x_all[mask_below], y_all[mask_below])

    if x_all.size and x_all.min() == x_all.max():
        x_line = np.linspace(0, 1, 200)
    else:
        x_line = np.linspace(x_all.min(), x_all.max(), 200)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=x_all[mask_above], y=y_all[mask_above],
        mode="markers",
        marker=dict(color="#1D9E75", size=3, opacity=0.6),
        text=df_scatter[mask_above]["Date"].dt.strftime("%Y-%m-%d"),
        hovertemplate="Date: %{text}<br>Prob: %{x:.2f}<br>Return: %{y:.1f}%<extra></extra>",
        name="Invested"
    ))

    fig2.add_trace(go.Scatter(
        x=x_all[mask_below], y=y_all[mask_below],
        mode="markers",
        marker=dict(color="#E24B4A", size=3, opacity=0.6),
        text=df_scatter[mask_below]["Date"].dt.strftime("%Y-%m-%d"),
        hovertemplate="Date: %{text}<br>Prob: %{x:.2f}<br>Return: %{y:.1f}%<extra></extra>",
        name="Cash"
    ))

    if np.isfinite(slope_all) and np.isfinite(intercept_all):
        fig2.add_trace(go.Scatter(
            x=x_line, y=slope_all * x_line + intercept_all,
            mode="lines", line=dict(color="#378ADD", width=1.5, dash="dot"),
            name="General regression"
        ))

    if np.isfinite(slope_above) and np.isfinite(intercept_above):
        fig2.add_trace(go.Scatter(
            x=x_line, y=slope_above * x_line + intercept_above,
            mode="lines", line=dict(color="#1D9E75", width=2),
            name=f"Invested regression (≥ {threshold})"
        ))

    if np.isfinite(slope_below) and np.isfinite(intercept_below):
        fig2.add_trace(go.Scatter(
            x=x_line, y=slope_below * x_line + intercept_below,
            mode="lines", line=dict(color="#E24B4A", width=2),
            name=f"Cash regression (< {threshold})"
        ))

    fig2.add_vline(x=threshold, line_dash="dash", line_color="#1D9E75", opacity=0.7,
                   annotation_text=f"Threshold ({threshold})", annotation_position="top right")
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)

    fig2.update_layout(
        template="plotly_dark",
        height=500,
        xaxis=dict(title="Signal probability", domain=[0, 0.78]),
        xaxis2=dict(domain=[0.82, 1]),
        yaxis=dict(title="Actual movement at t+horizon (%)"),
        yaxis2=dict(title="Actual return (%)", anchor="x2"),
        legend=dict(orientation="h", y=-0.2),
        boxmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True, key="fig_scatter")

    # Summary
    st.subheader("Summary")
    n_buy = int((df["prediction"] == 1).sum())
    n_total = len(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Buy signals", f"{n_buy}/{n_total}")
    col2.metric("Average probability", f"{df['probability'].mean():.2%}")
    col3.metric("Latest prediction", "BUY" if df['prediction'].iloc[-1] == 1 else "CASH")

    st.divider()

    with st.expander("Raw data (last 20 rows)"):


        st.dataframe(
            df[["Date", "Close", "prediction", "probability"]].tail(20),
            use_container_width=True
        )


    # ── PERFORMANCE STATS ─────────────────────────────────────
    st.subheader("Performance statistics")

    TRADING_DAYS_PER_YEAR = 365

    def compute_stats(values, label):
        s = pd.Series(values)
        daily_ret = s.pct_change().dropna()
        n = len(daily_ret)
        if n == 0:
            return {}

        wealth = (1 + daily_ret).prod()
        ann_ret = wealth ** (TRADING_DAYS_PER_YEAR / n) - 1
        ann_vol = daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_ret.std() > 0 else np.nan
        w = s / s.iloc[0]
        dd = w / w.cummax() - 1
        mdd = dd.min()
        skew = scipy_stats.skew(daily_ret, bias=False)
        kurt = scipy_stats.kurtosis(daily_ret, fisher=True, bias=False)
        total_ret = (s.iloc[-1] / s.iloc[0] - 1)

        return {
            "Strategy": label,
            "Total return": f"{total_ret:.1%}",
            "Ann. return": f"{ann_ret:.1%}",
            "Ann. volatility": f"{ann_vol:.1%}",
            "Sharpe": f"{sharpe:.2f}",
            "Max drawdown": f"{mdd:.1%}",
            "Skewness": f"{skew:.2f}",
            "Kurtosis": f"{kurt:.2f}",
        }

    stats_rows = []
    strat_vals = {
        "BTC Price": btc_base100.tolist(),
    }
    if show_signal:
        strat_vals["Signal ML"] = simulate_portfolio(df, "signal", horizon, threshold)
        strat_vals["Signal personnalisé"] = simulate_portfolio(df, "flex signal", horizon, threshold)
    # if show_long_only:
    #     strat_vals["Long only"] = simulate_portfolio(df, "long_only", horizon, threshold)
    # if show_random:
    #     strat_vals["Random"] = simulate_portfolio(df, "random", horizon, threshold)

    for label, vals in strat_vals.items():
        stats_rows.append(compute_stats(vals, label))

    stats_df = pd.DataFrame(stats_rows).set_index("Strategy")
    st.dataframe(stats_df, use_container_width=True)

    # Accuracy on 3 populations
    df_eval = df.copy()
    df_eval["real_up"] = (df_eval["Close"].shift(-horizon) > df_eval["Close"]).astype(int)
    df_eval = df_eval.dropna(subset=["real_up"])

    mask_inv = df_eval["probability"] >= threshold
    mask_cash = df_eval["probability"] < threshold

    acc_all = (df_eval["prediction"] == df_eval["real_up"]).mean()
    acc_inv = (df_eval[mask_inv]["prediction"] == df_eval[mask_inv]["real_up"]).mean()
    acc_cash = (df_eval[mask_cash]["prediction"] == df_eval[mask_cash]["real_up"]).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall accuracy", f"{acc_all:.1%}")
    col2.metric(f"Invested accuracy (≥{threshold})", f"{acc_inv:.1%}")
    col3.metric(f"Cash accuracy (<{threshold})", f"{acc_cash:.1%}")

    # ── CALIBRATION ──────────────────────────────────
    st.subheader("Model calibration")
    df_eval["proba_bucket"] = pd.cut(df_eval["probability"], bins=10)
    calib = df_eval.groupby("proba_bucket", observed=True)["real_up"].agg(["mean", "count"])
    calib["mid"] = [b.mid for b in calib.index]

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=calib["mid"], y=calib["mean"],
        mode="markers+lines",
        marker=dict(size=calib["count"]/calib["count"].max()*20 + 4, color="#EF9F27"),
        name="Actual upside rate per bucket"
    ))
    fig_cal.add_trace(go.Scatter(
        x=[0.4, 0.7], y=[0.4, 0.7],
        mode="lines", line=dict(dash="dot", color="gray", width=1),
        name="Perfect calibration"
    ))
    fig_cal.add_vline(x=threshold, line_dash="dash", line_color="#1D9E75", opacity=0.7)
    fig_cal.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.4)
    fig_cal.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Predicted probability",
        yaxis_title="Actual upside rate",
        xaxis=dict(range=[0.4, 0.7]),
        yaxis=dict(range=[0.3, 0.7])
    )
    st.plotly_chart(fig_cal, use_container_width=True, key="fig_calibration")

    # ── ROLLING TEST ──────────────────────────────────
    st.subheader("Rolling test — 1-month strategy window")

    window = 30
    results_rolling = []
    n = len(df)

    for start_idx in range(0, n - window, horizon):
        end_idx = min(start_idx + window, n - 1)
        df_window = df.iloc[start_idx:end_idx].reset_index(drop=True)
        if len(df_window) < horizon + 1:
            continue

        val_signal = simulate_portfolio(df_window, "flex signal", horizon, threshold)
        val_long = simulate_portfolio(df_window, "long_only", horizon, threshold)

        results_rolling.append({
            "start": df_window["Date"].iloc[0],
            "signal_return": val_signal[-1] - 100,
            "long_return": val_long[-1] - 100,
        })

    df_roll = pd.DataFrame(results_rolling)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=df_roll["start"], y=df_roll["signal_return"],
        mode="lines+markers", marker=dict(size=4),
        line=dict(color="#1D9E75", width=1.5),
        name="Custom signal"
    ))
    fig_roll.add_trace(go.Scatter(
        x=df_roll["start"], y=df_roll["long_return"],
        mode="lines+markers", marker=dict(size=4),
        line=dict(color="#5B8DEF", width=1.5),
        name="Long only"
    ))
    fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_roll.add_hline(y=df_roll["signal_return"].mean(),
                       line_dash="dot", line_color="#1D9E75", opacity=0.6,
                       annotation_text=f"Avg signal {df_roll['signal_return'].mean():.1f}%",
                       annotation_position="right")
    fig_roll.add_hline(y=df_roll["long_return"].mean(),
                       line_dash="dot", line_color="#5B8DEF", opacity=0.6,
                       annotation_text=f"Avg long only {df_roll['long_return'].mean():.1f}%",
                       annotation_position="right")
    fig_roll.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Window start date",
        yaxis_title="1-month return (%)",
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig_roll, use_container_width=True, key="fig_rolling")

    pct_beat = (df_roll["signal_return"] > df_roll["long_return"]).mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Windows tested", len(df_roll))
    col2.metric("Signal beats Long only", f"{pct_beat:.0%}")
    col3.metric("Avg signal return", f"{df_roll['signal_return'].mean():.1f}%")
    col4.metric("Avg long only return", f"{df_roll['long_return'].mean():.1f}%")

# ── CALIBRATION ──────────────────────────────────
    st.subheader("Calibration du modèle")
    df_eval["proba_bucket"] = pd.cut(df_eval["probability"], bins=10)
    calib = df_eval.groupby("proba_bucket", observed=True)["real_up"].agg(["mean", "count"])
    calib["mid"] = [b.mid for b in calib.index]

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=calib["mid"], y=calib["mean"],
        mode="markers+lines",
        marker=dict(size=calib["count"]/calib["count"].max()*20 + 4, color="#EF9F27"),
        name="Taux de hausse réel par bucket"
    ))
    fig_cal.add_trace(go.Scatter(
        x=[0.4, 0.7], y=[0.4, 0.7],
        mode="lines", line=dict(dash="dot", color="gray", width=1),
        name="Calibration parfaite"
    ))
    fig_cal.add_vline(x=threshold, line_dash="dash", line_color="#1D9E75", opacity=0.7)
    fig_cal.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.4)
    fig_cal.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Probabilité prédite",
        yaxis_title="Taux de hausse réel",
        xaxis=dict(range=[0.4, 0.7]),
        yaxis=dict(range=[0.3, 0.7])
    )
    st.plotly_chart(fig_cal, use_container_width=True, key="fig_calibration")

    # ── ROLLING TEST ──────────────────────────────────
    st.subheader("Rolling test — stratégie sur fenêtre 1 mois")

    window = 30
    results_rolling = []
    n = len(df)

    for start_idx in range(0, n - window, horizon):
        end_idx = min(start_idx + window, n - 1)
        df_window = df.iloc[start_idx:end_idx].reset_index(drop=True)
        if len(df_window) < horizon + 1:
            continue

        val_signal = simulate_portfolio(df_window, "flex signal", horizon, threshold)
        val_long = simulate_portfolio(df_window, "long_only", horizon, threshold)

        results_rolling.append({
            "start": df_window["Date"].iloc[0],
            "signal_return": val_signal[-1] - 100,
            "long_return": val_long[-1] - 100,
        })

    df_roll = pd.DataFrame(results_rolling)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=df_roll["start"], y=df_roll["signal_return"],
        mode="lines+markers", marker=dict(size=4),
        line=dict(color="#1D9E75", width=1.5),
        name="Signal personnalisé"
    ))
    fig_roll.add_trace(go.Scatter(
        x=df_roll["start"], y=df_roll["long_return"],
        mode="lines+markers", marker=dict(size=4),
        line=dict(color="#5B8DEF", width=1.5),
        name="Long only"
    ))
    fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig_roll.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Date de début de la fenêtre",
        yaxis_title="Rendement sur 1 mois (%)",
        legend=dict(orientation="h", y=-0.25)
    )

    fig_roll.add_hline(y=df_roll["signal_return"].mean(),
                       line_dash="dot", line_color="#1D9E75", opacity=0.6,
                       annotation_text=f"Moy. signal {df_roll['signal_return'].mean():.1f}%",
                       annotation_position="right")
    fig_roll.add_hline(y=df_roll["long_return"].mean(),
                       line_dash="dot", line_color="#5B8DEF", opacity=0.6,
                       annotation_text=f"Moy. long only {df_roll['long_return'].mean():.1f}%",
                       annotation_position="right")
    st.plotly_chart(fig_roll, use_container_width=True, key="fig_rolling")

    pct_beat = (df_roll["signal_return"] > df_roll["long_return"]).mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fenêtres testées", len(df_roll))
    col2.metric("Signal bat Long only", f"{pct_beat:.0%}")
    col3.metric("Rendement moyen signal", f"{df_roll['signal_return'].mean():.1f}%")
    col4.metric("Rendement moyen long only", f"{df_roll['long_return'].mean():.1f}%")

# ── PERFORMANCES MENSUELLES ──────────────────────────────────
    st.subheader("Performances mensuelles — Signal ML")

    signal_vals = simulate_portfolio(df, "flex signal", horizon, threshold)
    perf_df = pd.DataFrame({
        "Date": df["Date"].values,
        "value": signal_vals
    }).set_index("Date")
    perf_df.index = pd.DatetimeIndex(perf_df.index)

    month_end = perf_df.resample("ME").last()
    monthly_ret = month_end["value"].pct_change().dropna()
    monthly_ret.index = monthly_ret.index.to_period("M")

    pivot = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "ret": monthly_ret.values
    }).pivot(index="year", columns="month", values="ret")

    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    annual = monthly_ret.groupby(monthly_ret.index.year).apply(lambda x: (1+x).prod()-1)
    pivot["Year"] = annual.values
    pivot = (pivot * 100).round(2)

    st.dataframe(
        pivot.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-15, vmax=15),
        use_container_width=True
    )
