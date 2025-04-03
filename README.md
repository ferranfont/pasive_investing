readme_content = """
# Adaptive Allocation Portfolio App

Una aplicación en Streamlit que implementa un modelo de asignación táctica de activos basado en rendimiento y volatilidad.

## 🧠 Descripción

Esta app descarga precios de tres ETFs (`SPY`, `IAU`, `TLT`) desde Yahoo Finance, calcula los mejores activos según rendimiento de los últimos meses y los pondera por volatilidad inversa. La app incluye:

- 📅 Rebalanceo mensual sugerido
- 📈 Gráfico de crecimiento del portafolio vs SPY
- 📉 Gráfico de drawdown del modelo vs benchmark
- 📊 Análisis de rendimiento por tramos de retorno
- 🧮 Métricas completas de rendimiento y riesgo (Sharpe, Sortino, drawdown, etc.)
- 📥 Botón para descargar el rebalanceo histórico como CSV

## ▶️ Cómo ejecutarlo en Google Colab

1. Sube los archivos `app.py` y `requirements.txt` a tu entorno de Colab.
2. Ejecuta el siguiente bloque para instalar las dependencias y lanzar la app:

```bash
!pip install -r requirements.txt
!streamlit run app.py & npx localtunnel --port 8501
"""
