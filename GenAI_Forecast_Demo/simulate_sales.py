import pandas as pd
import random
from faker import Faker
import schedule
import time
import uuid
import snowflake.connector
from config import SNOWFLAKE_CONFIG

fake = Faker()
PRODUCTS = ['Widget A', 'Widget B', 'Widget C']
REGIONS = ['North', 'South', 'East', 'West']

def generate_fake_sales_record():
    return {
        "id": str(uuid.uuid4()),
        "timestamp": pd.Timestamp.now(),
        "product": random.choice(PRODUCTS),
        "region": random.choice(REGIONS),
        "quantity": random.randint(1, 10),
        "revenue": round(random.uniform(20, 500), 2)
    }

def upload_to_snowflake(df):
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO SALES_DATA (id, timestamp, product, region, quantity, revenue)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            row['id'], str(row['timestamp']), row['product'],
            row['region'], row['quantity'], row['revenue']
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Uploaded {len(df)} records to Snowflake")



def job():
    data = [generate_fake_sales_record() for _ in range(5)]
    df = pd.DataFrame(data)
    upload_to_snowflake(df)

if __name__ == "__main__":
    print("🚀 Sales Data Simulator started. Streaming every 10 seconds...")
    schedule.every(10).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
