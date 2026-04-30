# ==============================================
# 5) Fonctions de features techniques (pipeline)
# ==============================================

import pandas as pd

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calcul simple du RSI (Relative Strength Index)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Moyennes mobiles des gains/pertes
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-12)  # epsilon pour stabilité numérique
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_technical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme un DataFrame OHLCV brut en features techniques.

    Important:
    - on travaille uniquement avec informations présentes au temps t
    - pas d'information future utilisée dans les features
    """
    data = X.copy()

    # S'assure que l'ordre temporel est correct
    data = data.sort_index()

    # MAs de base
    data["ma10"] = data["Close"].rolling(10).mean()
    data["ma30"] = data["Close"].rolling(30).mean()
    data["ma50"] = data["Close"].rolling(50).mean()
    data["ma200"] = data["Close"].rolling(200).mean()

    # 1) pente MA200 (variation relative sur 5 jours)
    data["slope_ma200_5d"] = data["ma200"].pct_change(5)

    # 2) MA50
    data["ma50_level"] = data["ma50"]

    # 3) distance spot / MA30
    data["dist_spot_ma30"] = (data["Close"] / (data["ma30"] + 1e-12)) - 1

    # 4) distance spot / MA50
    data["dist_spot_ma50"] = (data["Close"] / (data["ma50"] + 1e-12)) - 1

    # 5) crossover MA10 / MA200
    data["ma10_200_crossover"] = (data["ma10"] / (data["ma200"] + 1e-12)) - 1

    # 6) RSI
    data["rsi14"] = compute_rsi(data["Close"], window=14)

    # 7) volatility ratio 30 / 200
    daily_ret = data["Close"].pct_change()
    vol30 = daily_ret.rolling(30).std()
    vol200 = daily_ret.rolling(200).std()
    data["vol_ratio_30_200"] = vol30 / (vol200 + 1e-12)

    # Features additionnelles
    data["ret_1d"] = daily_ret
    data["mom_5d"] = data["Close"].pct_change(5)
    data["mom_10d"] = data["Close"].pct_change(10)
    data["mom_20d"] = data["Close"].pct_change(20)
    data["daily_range"] = (data["High"] - data["Low"]) / (data["Close"] + 1e-12)
    data["dist_spot_ma200"] = (data["Close"] / (data["ma200"] + 1e-12)) - 1
    data["slope_ma50_5d"] = data["ma50"].pct_change(5)
    data["volume_change_1d"] = data["Volume"].pct_change(1)
    data["rolling_vol_20"] = daily_ret.rolling(20).std()

    # Colonnes finales utilisées par le modèle
    feature_cols = [
        "slope_ma200_5d",
        "ma50_level",
        "dist_spot_ma30",
        "dist_spot_ma50",
        "ma10_200_crossover",
        "rsi14",
        "vol_ratio_30_200",
        "ret_1d",
        "mom_5d",
        "mom_10d",
        "mom_20d",
        "daily_range",
        "dist_spot_ma200",
        "slope_ma50_5d",
        "volume_change_1d",
        "rolling_vol_20",
    ]

    return data[feature_cols]
