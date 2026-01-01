import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import ccxt

symbol = "LULU"
df = yf.download(symbol, start="2022-01-01")
df.dropna(inplace=True)

# Add technical indicators
try:
    df = ta.add_trend_ta(df, high="High", low="Low", close="Close")
except:
    pass
try:
    df = ta.add_momentum_ta(df, high="High", low="Low", close="Close")
except:
    pass
try:
    df = ta.add_volatility_ta(df, high="High", low="Low", close="Close")
except:
    pass
try:
    df = ta.add_others_ta(df, close="Close")
except:
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

