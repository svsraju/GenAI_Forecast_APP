import pandas as pd
from prophet import Prophet
import snowflake.connector
import matplotlib.pyplot as plt
from config import SNOWFLAKE_CONFIG
from gpt_helper import ask_gpt_about_forecast


def fetch_sales_data():
    # Connect to Snowflake and fetch clean data
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    query = """
        SELECT 
            DATE_TRUNC('minute', timestamp) AS ds,
            SUM(revenue) AS y
        FROM SALES_DATA
        WHERE revenue IS NOT NULL
        GROUP BY ds
        ORDER BY ds
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Ensure correct column names for Prophet
    df.columns = ['ds', 'y']
    return df

def forecast_sales(df, periods=14):
    # Drop missing values (extra safety)
    df = df.dropna()

    # Check if we have enough data
    if len(df) < 2:
        print("⚠️ Not enough data to run a forecast (need at least 2 data points).")
        print(df)
        return

    # Train the Prophet model
    model = Prophet()
    model.fit(df)

    # Forecast the next N periods (default 14 days)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Plot the forecast
    model.plot(forecast)
    plt.title("Hourly Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()
    # Let GPT analyze the forecast
    print("\n💬 Sending forecast to GPT for analysis...")
    summary = ask_gpt_about_forecast(forecast)
    print("\n📊 GPT Summary:")
    print(summary)


if __name__ == "__main__":
    print("🔍 Fetching data from Snowflake...")
    df = fetch_sales_data()

    if df.empty:
        print("⚠️ No data found in Snowflake.")
    else:
        print(f"✅ Data loaded: {len(df)} records")
        forecast_sales(df)
