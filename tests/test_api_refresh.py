import shutil
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.api import fast
from app.ml_logic.data import get_financial_data


client = TestClient(fast.app)


def test_predict_fetches_and_appends_missing_market_rows(monkeypatch):
    cache_csv = Path("data_folder/cache/BTC-USD.csv")
    backup_csv = cache_csv.with_suffix(".csv.test_backup")

    if not cache_csv.exists():
        get_financial_data(tickers=["BTC-USD"], period_years=1, force_refresh=True)

    if backup_csv.exists():
        backup_csv.unlink()
    shutil.copy2(cache_csv, backup_csv)

    original_df = pd.read_csv(cache_csv)
    assert len(original_df) > 3
    original_rows = len(original_df)
    truncated_df = original_df.iloc[:-3].copy()
    truncated_df.to_csv(cache_csv, index=False)

    monkeypatch.setattr(
        fast,
        "my_prediction_function",
        lambda open_price, high, low, close, volume: [1],
    )

    try:
        response = client.get(
            "/predict",
            params={
                "open_price": 10.0,
                "high": 12.0,
                "low": 9.5,
                "close": 11.0,
                "volume": 1000,
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": 1}

        refreshed_df = pd.read_csv(cache_csv)
        assert len(refreshed_df) >= original_rows
        assert refreshed_df["Date"].duplicated().sum() == 0
    finally:
        shutil.move(str(backup_csv), str(cache_csv))


def test_global_predict_returns_predictions(monkeypatch):
    sample_df = pd.DataFrame(
        [
            {
                "Date": "2026-04-01",
                "Open": 10.0,
                "High": 12.0,
                "Low": 9.0,
                "Close": 11.0,
                "Volume": 1000,
            },
            {
                "Date": "2026-04-02",
                "Open": 11.0,
                "High": 13.0,
                "Low": 10.0,
                "Close": 12.0,
                "Volume": 1100,
            },
        ]
    )

    monkeypatch.setattr(fast, "ensure_market_data_up_to_date", lambda tickers=None: None)
    monkeypatch.setattr(fast.pd, "read_csv", lambda _: sample_df.copy())
    monkeypatch.setattr(
        fast,
        "global_prediction_function",
        lambda df, model_name: ([1] * len(df), [0.8] * len(df)),
    )

    response = client.get(
        "/global_predict",
        params={"model_name": "xgb", "date_pivot": "2026-04-01"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "xgb"
    assert payload["date_pivot"] == "2026-04-01"
    assert len(payload["df_for_streamlit"]) == 2
    assert payload["df_for_streamlit"][0]["prediction"] == 1
