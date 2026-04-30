from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from fastapi import FastAPI, HTTPException


app = FastAPI()
from app.ai_for_finance import global_prediction_function, my_prediction_function, xgboost_prediction_function
from app.ml_logic.features import build_technical_features
from app.ml_logic.data import ensure_market_data_up_to_date


def _refresh_data_for_inference():
    """
    Reusable market data refresh step for inference endpoints.
    """
    try:
        ensure_market_data_up_to_date(tickers=["BTC-USD"])
    except Exception as error:
        print(f"Market data refresh failed: {error}")

# Cross-Origin Middleware Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
    open_price:float,
    high:float,
    low:float,
    close:float,
    volume:int
):
    _refresh_data_for_inference()
    prediction = my_prediction_function(open_price, high, low, close, volume)
    return {"prediction": int(prediction[0])}

@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }



@app.get("/predict_xgboost")
def xg_boost_predict( date_pivot :str = '2023-11-04', optional_user_date:str = '2026-04-01'):
    _refresh_data_for_inference()

     # Charger le CSV
    df = pd.read_csv("data_folder/cache/BTC-USD.csv")


    # Convertir la 1re colonne en datetime
    #df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

    # String de date à partir duquel tu veux filtrer
    date_str = date_pivot
    # Convertir ce string en datetime
    date_cut = pd.to_datetime(date_str)
    # Garder toutes les lignes à partir de cette date
    sub_df = df[df["Date"] >= date_cut]


    prediction_xgb, probability_xgb = xgboost_prediction_function(sub_df)

    result_df = sub_df.copy()
    result_df["prediction"] = prediction_xgb
    result_df["probability"] = probability_xgb


    #return {"prediction": int(prediction_xgb[-1]), "probability": float(probability_xgb[-1])}

    return {'df_for_streamlit': result_df.to_dict(orient='records')}





@app.get("/global_predict")
def predict_model(
    model_name: str = "xgb",
    date_pivot: str = "2023-11-04",
    optional_user_date: str = "2026-04-01",
):
    _refresh_data_for_inference()

    try:
        # Charger le CSV
        df = pd.read_csv("data_folder/cache/BTC-USD.csv")

        # Convertir la colonne Date en datetime
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        # Convertir la date pivot en datetime
        date_cut = pd.to_datetime(date_pivot)

        # Garder toutes les lignes à partir de cette date
        sub_df = df[df["Date"] >= date_cut].copy()

        # Appel de la fonction globale
        prediction, probability = global_prediction_function(sub_df, model_name)

        # Ajouter les résultats au DataFrame
        result_df = sub_df.copy()
        result_df["prediction"] = prediction
        result_df["probability"] = probability

        return {
            "model_name": model_name,
            "date_pivot": date_pivot,
            "df_for_streamlit": result_df.to_dict(orient="records"),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
