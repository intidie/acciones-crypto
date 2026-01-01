import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Análisis Fundamental y Técnico", layout="wide")
st.title("Análisis Técnico y Fundamental de Acciones")

# --- Sidebar para selección de activo y tipo de análisis ---
with st.sidebar:
    symbol = st.text_input("Símbolo de la acción (ej. AAPL)", "AAPL")
    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2022-01-01"))

# --- Descargar datos históricos ---
try:
    df = yf.download(symbol, start=start_date)
    if df.empty or 'Close' not in df.columns:
        st.error(f"No se encontraron datos para '{symbol}'. Verifica el símbolo.")
        st.stop()
except Exception as e:
    st.error(f"Error al descargar datos: {e}")
    st.stop()

df.dropna(inplace=True)
if df.empty:
    st.error("No hay suficientes datos después de limpiar NaN.")
    st.stop()

# --- Calcular indicadores técnicos ---
close_series = df['Close']
df['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
df['EMA_12'] = ta.trend.ema_indicator(close_series, window=12)
df['RSI'] = ta.momentum.rsi(close_series, window=14)
df['BB_upper'] = ta.volatility.bollinger_hband(close_series, window=20)
df['BB_lower'] = ta.volatility.bollinger_lband(close_series, window=20)

# --- Soportes y resistencias (clásico) ---
def detect_support_resistance(data, window=5):
    # Encontrar mínimos y máximos locales
    lows = data['Low'].rolling(window=window*2+1, center=True).min() == data['Low']
    highs = data['High'].rolling(window=window*2+1, center=True).max() == data['High']
    supports = data[lows]['Low']
    resistances = data[highs]['High']
    return supports, resistances

supports, resistances = detect_support_resistance(df)

# --- Gráfico técnico con soportes y resistencias ---
fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Precio'
))
fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange'), name='SMA 20'))
fig_price.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], line=dict(color='purple'), name='EMA 12'))
fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='gray', dash='dot'), name='Bollinger Superior'))
fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='gray', dash='dot'), name='Bollinger Inferior'))

# Añadir soportes y resistencias
for s in supports[-10:]:  # Últimos 10 soportes
    fig_price.add_hline(y=s, line=dict(color='green', dash='dash'), opacity=0.5)
for r in resistances[-10:]:  # Últimas 10 resistencias
    fig_price.add_hline(y=r, line=dict(color='red', dash='dash'), opacity=0.5)

fig_price.update_layout(title=f"Análisis Técnico de {symbol}", xaxis_title="Fecha", yaxis_title="Precio")
st.plotly_chart(fig_price, use_container_width=True)

# --- Análisis Fundamental ---
st.subheader("Análisis Fundamental")
try:
    ticker = yf.Ticker(symbol)
    info = ticker.info
except Exception as e:
    st.warning("No se pudo obtener información fundamental.")
    info = {}

# Ratios fundamentales comunes
fundamental_ratios = {
    "P/E Ratio": info.get("trailingPE"),
    "Forward P/E": info.get("forwardPE"),
    "PEG Ratio": info.get("pegRatio"),
    "Price to Book": info.get("priceToBook"),
    "ROE": info.get("returnOnEquity"),
    "ROIC": info.get("returnOnAssets"),  # Nota: Yahoo no da ROIC directo; se puede aproximar
    "Debt to Equity": info.get("debtToEquity"),
    "EBITDA Margins": info.get("ebitdaMargins"),
    "Profit Margins": info.get("profitMargins"),
    "Dividend Yield": info.get("dividendYield"),
    "Market Cap": info.get("marketCap"),
    "Enterprise Value": info.get("enterpriseValue"),
}

# Convertir a DataFrame
df_ratios = pd.DataFrame.from_dict(fundamental_ratios, orient='index', columns=["Valor"])
df_ratios = df_ratios.dropna()

# Buscador interactivo
selected_ratio = st.selectbox("Selecciona un ratio fundamental", options=df_ratios.index)
st.metric(label=selected_ratio, value=df_ratios.loc[selected_ratio, "Valor"])

# Mostrar tabla completa (opcional)
with st.expander("Ver todos los ratios disponibles"):
    st.dataframe(df_ratios)

# --- Notas finales ---
st.info("Nota: Los ratios fundamentales provienen de Yahoo Finance. Algunos pueden no estar disponibles para ciertos activos o ser aproximados.")
