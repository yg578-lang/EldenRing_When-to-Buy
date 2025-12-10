import pandas as pd
import numpy as np
import joblib
import streamlit as st

def load_data_and_models():
    df = pd.read_csv("data/daily_features.csv")
    df["date"] = pd.to_datetime(df["date"])

    rf_clf = joblib.load("data/elden_rf_classifier_advanced.joblib")
    rf_reg = joblib.load("data/elden_rf_regressor_30d.joblib")

    return df, rf_clf, rf_reg

def main():
    st.set_page_config(page_title="Elden Ring - Best Time to Buy", layout="wide")

    st.title("🎮 Elden Ring - Best Time to Buy On Steam")
    st.write(
        "This dashboard uses historical Steam price data and machine learning "
        "to estimate whether Elden Ring is likely to be **cheaper in the next 7 days**, "
        "and to forecast the **minimum price over the next 30 days**."
    )

    df, rf_clf, rf_reg = load_data_and_models()

    # Sidebar controls
    st.sidebar.header("Controls")

    latest = df.iloc[-1]

    use_latest = st.sidebar.checkbox("Use most recent data point", value=True)

    if use_latest:
        current_price = float(latest["discounted_price"])
        pct_off = float(latest["pct_off"])
        st.sidebar.write(f"Latest date in data: {latest['date'].date()}")
    else:
        #Letting user control the price and discount
        current_price = st.sidebar.number_input(
            "Current observed price ($)",
            min_value=0.0,
            value=float(latest["discounted_price"]),
            step=1.0,
        )
        pct_off = st.sidebar.slider(
            "Percent off full price (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=float(latest["pct_off"]),
            step=0.01,
        )

    template = latest.copy()

    template["discounted_price"] = current_price
    template["pct_off"] = pct_off

    feature_cols = [
        "discounted_price",
        "pct_off",
        "price_change",
        "rolling_avg_7",
        "rolling_avg_30",
        "rolling_std_7",
        "days_since_release",
        "day_of_week",
        "is_winter_sale_season",
        "is_summer_sale_season",
        "is_autumn_sale_season",
    ]

    X = np.array([[template[col] for col in feature_cols]])

    # Predictions
    prob_cheaper_7d = rf_clf.predict_proba(X)[0, 1]
    forecast_min_30d = rf_reg.predict(X)[0]

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current snapshot")
        st.metric("Current price", f"${current_price:,.2f}")
        st.metric("Percent off full price", f"{pct_off * 100:.1f}%")

    with col2:
        st.subheader("Model estimates")
        st.metric(
            "Probability of cheaper price in next 7 days",
            f"{prob_cheaper_7d * 100:.1f}%"
        )
        st.metric(
            "Predicted minimum price in next 30 days",
            f"${forecast_min_30d:,.2f}"
        )

    st.markdown("---")
    st.subheader("Price history")

    st.line_chart(
        df.set_index("date")[["discounted_price", "rolling_avg_30"]]
    )

    st.markdown(
        "Note: This project is based on historical SteamDB price data_"
    )

if __name__ == "__main__":
    main()