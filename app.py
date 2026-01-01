import streamlit as st
import yfinance as yf
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

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Calcular indicadores técnicos ---
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['RSI'] = calculate_rsi(df['Close'], period=14)
rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['BB_upper'] = rolling_mean + 2 * rolling_std
df['BB_lower'] = rolling_mean - 2 * rolling_std

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
for s in supports.tail(10).values:  # Últimos 10 soportes
    fig_price.add_hline(y=s, line=dict(color='green', dash='dash'), opacity=0.5)
for r in resistances.tail(10).values:  # Últimas 10 resistencias
    fig_price.add_hline(y=r, line=dict(color='red', dash='dash'), opacity=0.5)

fig_price.update_layout(title=f"Análisis Técnico de {symbol}", xaxis_title="Fecha", yaxis_title="Precio")
st.plotly_chart(fig_price, width='stretch')

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



def calcular_precios_justos(symbol):
    from yahooquery import Ticker
    t = Ticker(symbol)
    
    # Datos fundamentales
    try:
        info = t.summary_detail[symbol]
        fin_data = t.financial_data.get(symbol, {})
        key_stats = t.key_stats.get(symbol, {})
        income = t.income_statement(frequency='a', trailing=False)
        cashflow = t.cash_flow(frequency='a', trailing=False)
    except Exception as e:
        st.warning(f"Error al obtener datos para {symbol}: {e}")
        return None, None, None

    shares_out = float(info.get('sharesOutstanding', 0))
    if not shares_out or shares_out <= 0:
        return None, None, None

    # --- 1. DCF simplificado (5 años, crecimiento constante) ---
    fcf = None
    try:
        fcf_series = cashflow[cashflow['type'] == 'Free Cash Flow']['FreeCashFlow'].dropna()
        if len(fcf_series) > 0:
            fcf = float(fcf_series.iloc[0])  # Último FCF anual
    except:
        pass

    fair_value_dcf = None
    if fcf and fcf > 0:
        try:
            # Supuestos razonables (ajustables)
            growth_rate = 0.03  # 3% perpetuo
            discount_rate = 0.08  # 8% WACC estimado

            # Proyección de 5 años
            fcf_proj = [fcf * (1 + growth_rate)**i for i in range(1, 6)]
            # Valor terminal (Gordon)
            terminal_value = fcf_proj[-1] * (1 + growth_rate) / (discount_rate - growth_rate)
            # VPN
            pv_fcf = sum(fcf_proj[i] / (1 + discount_rate)**(i+1) for i in range(5))
            pv_terminal = terminal_value / (1 + discount_rate)**5
            equity_value = pv_fcf + pv_terminal
            fair_value_dcf = equity_value / shares_out
        except:
            fair_value_dcf = None

    # --- 2. DDM (Dividend Discount Model) ---
    dividend = float(info.get('dividendRate', 0))
    fair_value_ddm = None
    if dividend and dividend > 0:
        try:
            payout_growth = 0.02  # 2% crecimiento de dividendos
            required_return = 0.08
            next_div = dividend * (1 + payout_growth)
            fair_value_ddm = next_div / (required_return - payout_growth)
        except:
            fair_value_ddm = None

    # --- 3. P/E relativo (usar promedio histórico de P/E) ---
    trailing_pe = float(key_stats.get('trailingPE', 0))
    eps = float(info.get('epsTrailingTwelveMonths', 0))
    fair_value_pe = None
    if trailing_pe and eps:
        # Suponemos que el "precio justo" es cuando la acción vuelve a su P/E histórico promedio
        # Aquí usamos el P/E actual como proxy (ideal: promedio 5 años, pero no siempre disponible)
        try:
            fair_value_pe = trailing_pe * eps
        except:
            fair_value_pe = None

    return fair_value_dcf, fair_value_ddm, fair_value_pe

st.subheader("Precio Justo Estimado (Valor Intrínseco)")

# Instalar yahooquery si no lo tienes: pip install yahooquery
try:
    fair_dcf, fair_ddm, fair_pe = calcular_precios_justos(symbol)
    
    # Mostrar en tarjetas
    cols = st.columns(3)
    with cols[0]:
        if fair_dcf:
            st.metric("DCF Estimado", f"${fair_dcf:,.2f}")
        else:
            st.metric("DCF Estimado", "N/A")
    with cols[1]:
        if fair_ddm:
            st.metric("DDM (Dividendos)", f"${fair_ddm:,.2f}")
        else:
            st.metric("DDM", "N/A")
    with cols[2]:
        if fair_pe:
            st.metric("P/E Relativo", f"${fair_pe:,.2f}")
        else:
            st.metric("P/E Relativo", "N/A")

    # Precio actual
    current_price = df['Close'].iloc[-1]
    st.metric("Precio Actual", f"${current_price:,.2f}")

    # Gráfico comparativo
    fair_values = {
        "Actual": current_price,
        "DCF": fair_dcf,
        "DDM": fair_ddm,
        "P/E Relativo": fair_pe
    }
    fair_df = pd.DataFrame({
        "Modelo": list(fair_values.keys()),
        "Precio Justo": list(fair_values.values())
    }).dropna()

    if not fair_df.empty:
        fig_fair = go.Figure()
        fig_fair.add_trace(go.Bar(
            x=fair_df["Modelo"],
            y=fair_df["Precio Justo"],
            marker_color=['blue', 'green', 'orange', 'red'][:len(fair_df)]
        ))
        fig_fair.update_layout(title="Comparación: Precio Actual vs. Precio Justo", yaxis_title="USD")
        st.plotly_chart(fig_fair, width='stretch')

except Exception as e:
    st.error(f"No se pudo calcular el precio justo: {e}")


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.title("Simulación de Precios de Acciones con GBM")

# Inputs del usuario
symbol = st.text_input("Símbolo de la acción", "AAPL")
start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2022-01-01"))

# Descargar datos
df = yf.download(symbol, start=start_date)
if df.empty or 'Close' not in df.columns:
    st.error(f"No se pudieron obtener datos válidos para el símbolo '{symbol}'. Verifica el símbolo.")
    st.stop()

df.dropna(inplace=True)
if df.empty:
    st.error("No hay datos suficientes después de limpiar NaN.")
    st.stop()

# Añadir indicadores técnicos
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['RSI'] = calculate_rsi(df['Close'], period=14)
rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['BB_upper'] = rolling_mean + 2 * rolling_std
df['BB_lower'] = rolling_mean - 2 * rolling_std

# === Detección de soportes y resistencias ===
def detect_support_resistance(data, window=5):
    lows = data['Low'].rolling(window=window*2+1, center=True).min() == data['Low']
    highs = data['High'].rolling(window=window*2+1, center=True).max() == data['High']
    supports = data[lows]['Low']
    resistances = data[highs]['High']
    return supports, resistances

supports, resistances = detect_support_resistance(df, window=5)

# === Mostrar resumen de soportes ===
st.subheader("🔍 Análisis Técnico: Soportes Clave")
if not supports.empty:
    # Ordenar por fecha descendente y tomar los últimos 3
    recent_supports = supports.tail(3)
    support_levels = sorted([round(level, 2) for level in recent_supports.values])
    
    st.write("**Niveles de soporte recientes (ordenados):**")
    for i, level in enumerate(support_levels, 1):
        st.write(f"- Soporte {i}: **${level}**")
    
    # Promedio de soportes como referencia
    avg_support = np.mean(support_levels)
    current_price = df['Close'].iloc[-1]
    st.metric("Precio actual", f"${current_price:.2f}")
    st.metric("Soporte promedio reciente", f"${avg_support:.2f}")
    
    if current_price < avg_support * 1.02:
        st.warning("⚠️ El precio está cerca o por debajo del soporte promedio.")
    else:
        st.success("✅ El precio está por encima del soporte promedio reciente.")
else:
    st.info("No se detectaron niveles claros de soporte con los parámetros actuales.")

# === Separador visual ===
st.markdown("---")