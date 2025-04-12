import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import snowflake.connector
from config import SNOWFLAKE_CONFIG
from gpt_helper import ask_gpt_about_forecast

# --- Helper functions ---
def fetch_sales_data(product_filter=None, region_filter=None):
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)

    query = """
        SELECT 
            DATE_TRUNC('hour', timestamp) AS ds,
            SUM(revenue) AS y
        FROM SALES_DATA
        WHERE revenue IS NOT NULL
    """

    if product_filter:
        query += f" AND product = '{product_filter}'"
    if region_filter:
        query += f" AND region = '{region_filter}'"

    query += """
        GROUP BY ds
        ORDER BY ds
    """

    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = ['ds', 'y']
    return df.dropna()

def forecast_sales(df, periods=12, what_if_percent=0):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)

    if what_if_percent != 0:
        factor = 1 + (what_if_percent / 100.0)
        forecast['yhat'] *= factor
        forecast['yhat_lower'] *= factor
        forecast['yhat_upper'] *= factor

    return model, forecast

# --- Streamlit UI ---
st.set_page_config(page_title="GenAI Sales Forecast", layout="centered")
st.title("📈 Sales Forecasting with GenAI")

# Sidebar filters
st.sidebar.header("🔍 Filter Your Forecast")
product_filter = st.sidebar.selectbox("Product", options=["All", "Laptop", "Tablet", "Phone"])
region_filter = st.sidebar.selectbox("Region", options=["All", "North", "South", "East", "West"])
what_if_percent = st.sidebar.slider("📊 Simulate Revenue Change (%)", -50, 50, 0)

# Convert 'All' to None
product_filter = None if product_filter == "All" else product_filter
region_filter = None if region_filter == "All" else region_filter

# Main Forecasting Button
if st.button("📊 Run Forecast"):
    df = fetch_sales_data(product_filter=product_filter, region_filter=region_filter)

    if len(df) < 2:
        st.warning("Not enough data to forecast. Please run the simulator a bit longer.")
    else:
        model, forecast = forecast_sales(df, what_if_percent=what_if_percent)

        # Forecast plot
        fig = model.plot(forecast)
        st.pyplot(fig)

        # --- AI Summary (Auto) ---
        st.subheader("💬 AI Summary")
        with st.spinner("Generating insight..."):
            summary = ask_gpt_about_forecast(forecast)
        st.success("Done!")
        st.markdown("**💡 GPT Summary:**")
        st.markdown(summary, unsafe_allow_html=True)

        # --- Download CSV ---
        st.subheader("⬇️ Export Forecast")
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
        st.download_button("Download Forecast as CSV", data=csv, file_name="forecast.csv", mime="text/csv")

        # --- GPT Chat (Interactive) ---
        st.markdown("---")
        st.subheader("🤖 Ask GPT About This Forecast")
        user_input = st.text_input("Type your question below:")

        if user_input:
            # Format forecast table as markdown
            last_rows = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5).copy()
            last_rows['ds'] = last_rows['ds'].dt.strftime("%Y-%m-%d %H:%M")
            markdown_table = last_rows.to_markdown(index=False)

            chat_prompt = f"""
You are a helpful business analyst.

Here is the forecast (with a {what_if_percent}% simulation):

```
{markdown_table}
```

User question: {user_input}

Please reply with bullet points outlining trends, risks, and suggestions.
"""

            with st.spinner("GPT is thinking..."):
                chat_response = ask_gpt_about_forecast(forecast, override_prompt=chat_prompt)
                st.markdown("**💬 GPT says:**")
                st.markdown(chat_response, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit, Prophet, Snowflake, and Hugging Face ✨")
