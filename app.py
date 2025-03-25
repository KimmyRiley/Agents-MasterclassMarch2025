import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📈 AI Revenue Forecaster", page_icon="🔮", layout="wide")

st.title("🔮 AI Revenue Forecasting with Prophet")
st.markdown("Upload your Excel file with `Date` and `Revenue` columns to generate forecasts.")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Check required columns
        if not {"Date", "Revenue"}.issubset(df.columns):
            st.error("❌ Excel file must contain 'Date' and 'Revenue' columns.")
            st.stop()

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={"Date": "ds", "Revenue": "y"})

        # Forecasting with Prophet
        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=12, freq='M')  # Forecast 12 months ahead
        forecast = m.predict(future)

        st.subheader("📊 Historical & Forecasted Revenue")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        st.subheader("🧠 Forecast Components")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

        # 🧠 Generate FP&A Commentary
        st.subheader("🤖 AI-Generated Forecast Commentary")

        # Prepare data to send to Groq
        full_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24)
        data_for_ai = full_forecast.to_json(orient="records", date_format="iso")

        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
        You are the Head of FP&A at a SaaS company. Your task is to analyze the revenue forecast data and provide:
        - Key insights and trends.
        - Concerns or anomalies.
        - A CFO-ready summary using the Pyramid Principle.
        - Actionable recommendations to improve performance.

        Here is the forecast dataset in JSON:
        {data_for_ai}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )

        ai_commentary = response.choices[0].message.content
        st.markdown("### 📖 AI Commentary")
        st.write(ai_commentary)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
