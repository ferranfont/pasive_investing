

# ============================
# Adaptive Allocation Streamlit App
# ============================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
import base64

# Configurar fechas y descarga de datos
st.title("Adaptive Allocation Portfolio")
today = pd.Timestamp.today()
rebalance_date = today.replace(day=1)
formated_rebalanced_date = rebalance_date.strftime('%d-%m-%Y')
start = '2006-01-01'
end = (today + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
tickers = ['SPY', 'IAU', 'TLT']

st.write(f"Hoy es: {today.strftime('%d/%m/%Y')}")

st.markdown("""
**Descripción del Modelo**

Los resultados del modelo de asignación táctica de activos desde enero de 2006 hasta enero de 2024 se basan en un modelo de asignación adaptativa que mantiene los 2 activos con mejor rendimiento. Los activos seleccionados se ponderan utilizando volatilidad inversa, en función de la ventana de volatilidad. El modelo utiliza una única ventana de rendimiento de 9 meses calendario. Los ajustes de riesgo basados en volatilidad se realizan con la volatilidad diaria durante los últimos 20 días de mercado. Las operaciones del modelo se ejecutan utilizando el precio de cierre del último día del mes, basándose en señales generadas a fin de mes. El período analizado está limitado por la disponibilidad de datos para el iShares Gold Trust (IAU) [Feb 2005 – Ene 2024].
""")

st.markdown("""
**Activos incluidos en el Modelo de Asignación Adaptativa**

| Ticker | Nombre del ETF                         |
|--------|----------------------------------------|
| TLT    | iShares 20+ Year Treasury Bond ETF     |
| IAU    | iShares Gold Trust                     |
| SPY    | SPDR S&P 500 ETF Trust                 |
""")

data = yf.download(tickers, start=start, end=end)
if data.empty or 'Close' not in data:
    st.error("No se han podido descargar datos desde Yahoo Finance.")
    st.stop()
close_prices = data['Close']

# Agregar cierre de mes si falta
eom_march = pd.Timestamp(f"{today.year}-03-31")
if today.month == 4 and eom_march not in close_prices.index:
    last_valid = close_prices.loc[:eom_march].iloc[-1]
    close_prices.loc[eom_march] = last_valid
    close_prices = close_prices.sort_index()

monthly_prices = close_prices.resample('M').last()
daily_returns = close_prices.pct_change()

# Parámetros del modelo
performance_window = st.number_input("Meses para evaluar rendimiento (performance window):", min_value=1, max_value=36, value=9, step=1, key="performance_window")
vol_window = st.number_input("Días para evaluar volatilidad (volatility window):", min_value=5, max_value=60, value=20, step=1, key="vol_window")
capital = st.number_input("Introduce el capital que deseas invertir ($):", min_value=1000, value=10000, step=500)

# Cálculos del modelo
portfolio_returns = []
benchmark_returns = []
weights_history = []

for i in range(performance_window, len(monthly_prices) - 1):
    date = monthly_prices.index[i]
    past_date = monthly_prices.index[i - performance_window]
    next_date = monthly_prices.index[i + 1]

    performance = monthly_prices.loc[date] / monthly_prices.loc[past_date] - 1
    top_assets = performance.nlargest(2).index.tolist()

    current_day = close_prices.index[close_prices.index.get_indexer([date], method='pad')[0]]
    vol_slice = daily_returns[top_assets].loc[:current_day].tail(vol_window)
    vol = vol_slice.std()
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    weights_history.append((date, weights.to_dict()))

    returns = (monthly_prices.loc[next_date, top_assets] / monthly_prices.loc[date, top_assets] - 1)
    port_return = np.dot(returns.values, weights.values)
    portfolio_returns.append((next_date, port_return))

    spy_return = (monthly_prices.loc[next_date, 'SPY'] / monthly_prices.loc[date, 'SPY']) - 1
    benchmark_returns.append((next_date, spy_return))

returns_df = pd.DataFrame(portfolio_returns, columns=['Date', 'Portfolio_Return']).set_index('Date')
benchmark_df = pd.DataFrame(benchmark_returns, columns=['Date', 'SPY_Return']).set_index('Date')
returns_combined = returns_df.join(benchmark_df)
returns_combined['Cumulative_Portfolio'] = capital * (1 + returns_combined['Portfolio_Return']).cumprod()
returns_combined['Cumulative_SPY'] = capital * (1 + returns_combined['SPY_Return']).cumprod()

# Rebalanceo actual
rebalance_history = []
for date, weights in weights_history:
    price_date = close_prices.index.asof(date)
    prices = close_prices.loc[price_date]
    for asset, weight in weights.items():
        allocation = capital * weight
        price = prices[asset]
        shares = int(allocation // price)
        rebalance_history.append({
            "Fecha": date.date(),
            "Ticker": asset,
            "Peso (%)": round(weight * 100, 2),
            "Asignación ($)": round(allocation, 2),
            "Precio actual ($)": round(price, 2),
            "Acciones a comprar": shares
        })
rebalance_df = pd.DataFrame(rebalance_history)

# Mostrar último rebalanceo
if not rebalance_df.empty:
    latest_date = rebalance_df["Fecha"].max()
    latest_df = rebalance_df[rebalance_df["Fecha"] == latest_date].set_index("Ticker")
    st.markdown(f"### 🗓️ Rebalanceo sugerido a fecha {formated_rebalanced_date}")
    st.dataframe(latest_df)

# Gráfico de crecimiento
st.markdown("### 📈 Evolución del Portafolio vs SPY")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(returns_combined.index, returns_combined['Cumulative_Portfolio'], label='Adaptive Allocation', color='blue')
ax.plot(returns_combined.index, returns_combined['Cumulative_SPY'], label='SPY Benchmark', linestyle='--', color='green', linewidth=1, alpha=0.7)
ax.set_title('Crecimiento del Portafolio')
ax.set_ylabel('Valor del Portafolio ($)')
ax.legend()
ax.grid(True, axis='y')
st.pyplot(fig)

# Métricas
st.markdown("### 📊 Métricas de Rendimiento")

def annualized_return(series):
    total_return = series.iloc[-1] / series.iloc[0] - 1
    num_years = (series.index[-1] - series.index[0]).days / 365.25
    return (1 + total_return) ** (1 / num_years) - 1

def annualized_volatility(returns):
    return returns.std() * np.sqrt(12)

def sharpe_ratio(returns, risk_free=0.0):
    excess = returns - risk_free / 12
    return (excess.mean() / excess.std()) * np.sqrt(12)

def sortino_ratio(returns, risk_free=0.0):
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(12)
    return (returns.mean() * 12 - risk_free) / downside_deviation if downside_deviation != 0 else np.nan

def max_drawdown(series):
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.min()

def best_worst_years(returns):
    annual = (1 + returns).resample('Y').prod() - 1
    return annual.max(), annual.min()

best_model, worst_model = best_worst_years(returns_df['Portfolio_Return'])
best_spy, worst_spy = best_worst_years(benchmark_df['SPY_Return'])

initial_capital = f"${capital:,.2f}"
final_capital_model = f"${returns_combined['Cumulative_Portfolio'].iloc[-1]:,.2f}"
final_capital_spy = f"${returns_combined['Cumulative_SPY'].iloc[-1]:,.2f}"

metrics_df = pd.DataFrame({
    "Modelo": [
        f"{annualized_return(returns_combined['Cumulative_Portfolio']):.2%}",
        f"{annualized_volatility(returns_df['Portfolio_Return']):.2%}",
        f"{sharpe_ratio(returns_df['Portfolio_Return']):.2f}",
        f"{sortino_ratio(returns_df['Portfolio_Return']):.2f}",
        f"{max_drawdown(returns_combined['Cumulative_Portfolio']):.2%}",
        f"{best_model:.2%}",
        f"{worst_model:.2%}",
        initial_capital,
        final_capital_model
    ],
    "Benchmark (SPY)": [
        f"{annualized_return(returns_combined['Cumulative_SPY']):.2%}",
        f"{annualized_volatility(benchmark_df['SPY_Return']):.2%}",
        f"{sharpe_ratio(benchmark_df['SPY_Return']):.2f}",
        f"{sortino_ratio(benchmark_df['SPY_Return']):.2f}",
        f"{max_drawdown(returns_combined['Cumulative_SPY']):.2%}",
        f"{best_spy:.2%}",
        f"{worst_spy:.2%}",
        initial_capital,
        final_capital_spy
    ]
}, index=[
    "Rend. Acum.", "Volatilidad", "Sharpe Ratio", "Sortino Ratio",
    "Drawdown Máximo", "Mejor Año", "Peor Año",
    "Capital Inicial", "Capital Final"
])

st.dataframe(metrics_df)

# Gráfico de Drawdown
st.markdown("### 📉 Drawdown del Modelo vs Benchmark")

def compute_drawdown(series):
    peak = series.cummax()
    return (series - peak) / peak

drawdown_model = compute_drawdown(returns_combined['Cumulative_Portfolio'])
drawdown_spy = compute_drawdown(returns_combined['Cumulative_SPY'])

fig_dd, ax_dd = plt.subplots(figsize=(12, 6))
drawdown_model.plot(ax=ax_dd, color='royalblue', linewidth=2, label='Adaptive Allocation')
drawdown_spy.plot(ax=ax_dd, color='green', alpha=0.6, linewidth=1, linestyle='--', label='SPY Benchmark')
ax_dd.set_title('Drawdown del Portafolio vs Benchmark')
ax_dd.set_ylabel('Drawdown')
ax_dd.set_xlabel('Fecha')
ax_dd.grid(True, axis='y')
ax_dd.legend()
st.pyplot(fig_dd)

# Return vs Benchmark por tramos
st.markdown("### 📉 Return del Modelo vs Benchmark por tramos")
df_temp = returns_combined[['Portfolio_Return', 'SPY_Return']].dropna().copy()
bins = np.arange(-0.10, 0.105, 0.015)
df_temp['SPY_Bin'] = pd.cut(df_temp['SPY_Return'], bins)
grouped = df_temp.groupby('SPY_Bin').agg({
    'SPY_Return': 'mean',
    'Portfolio_Return': 'mean'
}).dropna()

fig2, ax2 = plt.subplots(figsize=(12, 6))
bar_width = 0.35
x = np.arange(len(grouped))

ax2.bar(x - bar_width/2, grouped['Portfolio_Return'], width=bar_width, label='Modelo', color='cornflowerblue')
ax2.bar(x + bar_width/2, grouped['SPY_Return'], width=bar_width, label='SPY Benchmark', color='black')

ax2.set_title('Return vs. Benchmark por Tramos de SPY')
ax2.set_xlabel('Rango de retorno del benchmark (SPY)')
ax2.set_ylabel('Return medio en el tramo')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{b.left:.1%} to {b.right:.1%}" for b in grouped.index], rotation=45)
ax2.legend()
ax2.grid(True, axis='y')
st.pyplot(fig2)

# Botón de descarga del CSV
csv_bytes = rebalance_df.to_csv(index=False, sep=';').encode('utf-8')
b64 = base64.b64encode(csv_bytes).decode()
custom_button = f"""
    <a href="data:file/csv;base64,{b64}" download="historial_rebalanceo.csv" 
       style="display:inline-block;padding:12px 20px;background-color:#007BFF;color:white;
              font-weight:bold;text-decoration:none;border-radius:8px;margin-top:10px;">
       📥 Descargar histórico de rebalanceo (.csv)
    </a>
"""

# Mostrar el botón
st.markdown(custom_button, unsafe_allow_html=True)



