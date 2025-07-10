import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
import os

#load assets
@st.cache_data
def load_assets():
    base_path = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_path, "df_v7.csv"), parse_dates=["Date"])
    model = load_model(os.path.join(base_path, "lstm_model.h5"))
    scaler = joblib.load(os.path.join(base_path, "lstm_scaler.pkl"))
    return df, model, scaler

df, model, scaler = load_assets()

# Set up
st.title("📈 Uber Stock Price Predictor")
st.write("This app predicts the next day's closing stock price using a trained LSTM model.")

# Show latest data
st.subheader("Latest Available Data")
st.dataframe(df.tail(5))

# Feature Columns
feature_cols = ['Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_7d_ma', 'Close_7d_std', 'High_Volume_Flag']
target_col = 'Close'

# Prediction Function
def predict_next_day(df, model, scaler):
    latest_input = df[feature_cols + [target_col]].iloc[-1:]
    scaled_input = scaler.transform(latest_input)
    X_input = scaled_input[:, :-1].reshape((1, 1, len(feature_cols)))
    pred_scaled = model.predict(X_input).flatten()[0]
    merged_input = np.hstack((scaled_input[:, :-1], [[pred_scaled]]))
    predicted_close = scaler.inverse_transform(merged_input)[0, -1]
    return round(predicted_close, 2)

# Prediction
if st.button("Predict Next Day Close Price"):
    predicted_price = predict_next_day(df, model, scaler)
    st.success(f"Predicted Close Price: ${predicted_price}")