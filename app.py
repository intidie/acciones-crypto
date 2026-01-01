import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go

symbol = "LULU"
df = yf.download(symbol, start="2022-01-01")
df.dropna(inplace=True)

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
paths = [gbm(float(prices.iloc[-1]), float(mu), float(sigma), 1, 252) for _ in range(1000)]


upper = np.percentile(paths, 95, axis=0)
lower = np.percentile(paths, 5, axis=0)
mean_path = np.mean(paths, axis=0)


print("Simulation completed successfully")

