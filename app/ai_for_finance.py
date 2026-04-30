import os
import pickle
import pandas as pd


import __main__
from app.ml_logic.features import build_technical_features
from app.ml_logic.rnn_inference import rnn_predict_from_artifacts


ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

def my_prediction_function(
    open_price,
    high,
    low,
    close,
    volume
):
    model_path = os.path.join(ROOT_PATH, 'models', 'linear_regression_1.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        prediction = model.predict([[open_price, high, low, close, volume]])

    return prediction



def xgboost_prediction_function(df:pd.DataFrame):
    model_xgb_path = os.path.join(ROOT_PATH, 'models', 'xgb_pipeline_0426.pkl')
    #model_xgb_path = os.path.join(ROOT_PATH, 'models', 'average_linear_pipeline.pkl')
    __main__.build_technical_features = build_technical_features

    with open(model_xgb_path, 'rb') as file:
        model_xgb = pickle.load(file)
        prediction_xgb = model_xgb.predict(df)
        probability_xgb = model_xgb.predict_proba(df)[:, 1]  # Probabilité de la classe positive

    return prediction_xgb, probability_xgb


def rnn_prediction_function(df: pd.DataFrame):
    __main__.build_technical_features = build_technical_features
    prediction_rnn, probability_rnn = rnn_predict_from_artifacts(df, ROOT_PATH)
    return prediction_rnn, probability_rnn

def linear_prediction_function(df:pd.DataFrame):
    model_linear_path = os.path.join(ROOT_PATH, 'models', 'average_linear_pipeline.pkl')

    __main__.build_technical_features = build_technical_features

    with open(model_linear_path, 'rb') as file:
        model_linear = pickle.load(file)
        prediction_linear = model_linear.predict(df)
        probability_linear = model_linear.predict_proba(df)[:, 1]  # Probabilité de la classe positive

    return prediction_linear, probability_linear


def global_prediction_function(df: pd.DataFrame, model_name: str):

    model_functions = {
        "xgb": xgboost_prediction_function,
        "rnn": rnn_prediction_function,
        "linear": linear_prediction_function
    }

    if model_name not in model_functions:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return model_functions[model_name](df)
