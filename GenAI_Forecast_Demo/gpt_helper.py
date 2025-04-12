import requests
import os
import pandas as pd

# Hugging Face model URL
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# ⚠️ Use a secure method to store this in production
HF_TOKEN = "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"  # Replace with your real token or use os.getenv()

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def ask_gpt_about_forecast(forecast_df, override_prompt=None):
    if override_prompt:
        prompt = override_prompt
    else:
        if forecast_df is None or forecast_df.empty:
            return "⚠️ No forecast data available to summarize."

        try:
            summary_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5).copy()
            summary_df['ds'] = pd.to_datetime(summary_df['ds']).dt.strftime("%Y-%m-%d %H:%M")
            table = summary_df.to_markdown(index=False)
        except Exception as e:
            return f"⚠️ Error preparing data: {e}"

        prompt = f"""
You are a helpful business analyst.

Here’s the forecasted sales data:

```
{table}
```

Please summarize the trends and risks.
"""

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        result = response.json()

        if isinstance(result, list):
            return result[0].get("generated_text", "⚠️ GPT returned no text.")
        elif "error" in result:
            return f"❌ HF API Error: {result['error']}"
        else:
            return str(result)

    except Exception as e:
        return f"❌ Error calling Hugging Face: {e}"
