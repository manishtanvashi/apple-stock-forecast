import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import timedelta

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Stock Forecast", layout="wide")

col1, col2 = st.columns([1, 6])

with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
        width=70
    )

with col2:
    st.title("Apple Stock Price Forecast")

st.markdown("Random Forest Model | Technical Indicators | 30-Day Forecast")


# ------------------ LOAD MODEL ------------------
model = joblib.load("random_forest_model.pkl")

# ------------------ SIDEBAR ------------------
st.sidebar.header("User Input")

stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
forecast_days = st.sidebar.slider("Forecast Days", 5, 60, 30)


predict_button = st.sidebar.button("🚀 Generate Forecast")

# ------------------ FEATURE LIST ------------------
features = [
    'Close',
    'Volume',
    'SP500_Return',
    'data_diff',
    'Close_Lag1',
    'Close_Lag2',
    'Close_Lag3',
    'MA10',
    'MA20',
    'EMA10',
    'Volatility_20',
    'RSI'
]

# ------------------ MAIN LOGIC ------------------

if predict_button:

    st.info("Fetching Data...")

    # Download stock data
    from datetime import datetime

    df = yf.download(stock_symbol, start="2012-01-01", end="2019-12-26")

    df.columns = df.columns.get_level_values(0)

    # Download S&P500
    sp_raw = yf.download("^GSPC", start="2012-01-01", end="2019-12-26")
    sp_raw.columns = sp_raw.columns.get_level_values(0)
    sp = sp_raw[['Close']]
    sp.columns = ['SP500_Close']

    # Merge
    df = df.merge(sp, left_index=True, right_index=True, how='inner')

    # Create Features
    df['SP500_Return'] = df['SP500_Close'].pct_change()
    df['data_diff'] = df['Close'].diff()
    df['Target_Close'] = df['Close'].shift(-1)

    # Lag Features
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)

    # Moving Averages
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Volatility_20'] = df['Close'].rolling(20).std()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df = df.dropna()

    # ------------------ FORECASTING ------------------

    temp_df = df.copy()
    future_predictions = []

    for i in range(forecast_days):

        last_features = temp_df[features].iloc[-1:]
        next_price = model.predict(last_features)[0]
        future_predictions.append(next_price)

        new_row = temp_df.iloc[-1:].copy()
        new_row['Close'] = next_price

        new_row['Close_Lag1'] = next_price
        new_row['Close_Lag2'] = temp_df['Close'].iloc[-1]
        new_row['Close_Lag3'] = temp_df['Close'].iloc[-2]

        temp_df = pd.concat([temp_df, new_row])

        # Recalculate indicators
        temp_df['MA10'] = temp_df['Close'].rolling(10).mean()
        temp_df['MA20'] = temp_df['Close'].rolling(20).mean()
        temp_df['EMA10'] = temp_df['Close'].ewm(span=10, adjust=False).mean()
        temp_df['Volatility_20'] = temp_df['Close'].rolling(20).std()

        delta = temp_df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        temp_df['RSI'] = 100 - (100 / (1 + rs))

    # Create future dates
    last_date = df.index[-1]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq='B'
    )

    forecast_series = pd.Series(future_predictions, index=future_dates)


    # ------------------ PLOT ------------------

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Price'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_series.index,
        y=forecast_series.values,
        mode='lines',
        name='Forecast',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title=f"{stock_symbol} Stock Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------ DOWNLOAD BUTTON ------------------

    forecast_df = pd.DataFrame({
        "Date": forecast_series.index,
        "Predicted_Close": forecast_series.values
    })

    st.download_button(
        label="📥 Download Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name="forecast.csv",
        mime="text/csv"
    )

    st.success("Forecast Generated Successfully!")
