import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go
import pandas as pd

st.title("Simulación de Precios de Acciones con GBM")

# Inputs del usuario
symbol = st.text_input("Símbolo de la acción", "AAPL")
start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2022-01-01"))

df = yf.download(symbol, start=start_date)

if df.empty or 'Close' not in df.columns:
    st.error(f"No se pudieron obtener datos válidos para el símbolo '{symbol}'. Verifica el símbolo.")
    st.write(f"DataFrame vacío: {df.empty}, Columnas: {list(df.columns)}")
    st.stop()

df.dropna(inplace=True)

# Fixed dropna to avoid KeyError

if df.empty:
    st.error("No hay datos suficientes después de limpiar NaN.")
    st.stop()

if df.empty:
    st.error("No se pudieron obtener datos para el símbolo especificado.")
    st.stop()

# Add technical indicators
try:
    close_series = df['Close'].squeeze()  # Ensure 1D
    df['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
    df['EMA_12'] = ta.trend.ema_indicator(close_series, window=12)
    df['RSI'] = ta.momentum.rsi(close_series, window=14)
    df['BB_upper'] = ta.volatility.bollinger_hband(close_series, window=20)
    df['BB_lower'] = ta.volatility.bollinger_lband(close_series, window=20)
except Exception as e:
    print(f"Error adding indicators: {e}")
    pass


prices = df['Close']
st.write(f"Length of df: {len(df)}, Length of prices: {len(prices)}")

import numpy as np

returns = np.log(prices / prices.shift(1)).dropna()
mu = returns.mean()
sigma = returns.std()

def gbm(S0, mu, sigma, T, N):
    dt = float(T) / float(N)
    W = np.random.normal(0, np.sqrt(dt), int(N))
    W = np.cumsum(W)
    t = np.linspace(0, float(T), int(N))
    S = float(S0) * np.exp((float(mu) - 0.5 * float(sigma)**2) * t + float(sigma) * W)
    return np.array(S)
paths = [gbm(prices.iloc[-1], mu, sigma, 1, 252) for _ in range(1000)]


upper = np.percentile(paths, 95, axis=0)
lower = np.percentile(paths, 5, axis=0)
mean_path = np.mean(paths, axis=0)

st.write("Simulación completada exitosamente")

# Crear gráfico con Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(mean_path)), y=mean_path, mode='lines', name='Camino Medio'))
fig.add_trace(go.Scatter(x=np.arange(len(upper)), y=upper, mode='lines', name='Percentil 95', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=np.arange(len(lower)), y=lower, mode='lines', name='Percentil 5', line=dict(dash='dash')))
fig.update_layout(title=f"Simulación GBM para {symbol}", xaxis_title="Días", yaxis_title="Precio")
st.plotly_chart(fig)

