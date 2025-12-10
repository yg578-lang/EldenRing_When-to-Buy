import pandas as pd
import joblib
import numpy as np

def main():
    df = pd.read_csv("data/elden_features.csv")
    model = joblib.load("data/elden_rf_model.joblib")

    latest = df.iloc[-1]

    feature_cols = ["discounted_price", "price_change", "rolling_avg_price"]
    X = np.array([[latest[col] for col in feature_cols]])

    prob_cheaper = model.predict_proba(X)[0, 1]

    print("=== Elden Ring - Should you wait? ===")
    print(f"Date in data: {latest['date']}")
    print(f"Current price in data: ${latest['discounted_price']:.2f}")
    print(f"Model probability it will be cheaper in next 7 days: {prob_cheaper:.1%}")

    if prob_cheaper > 0.5:
        print("\nRecommendation: It's probably worth WAITING for a better price.")
    else:
        print("\nRecommendation: It's reasonable to BUY NOW based on past patterns.")

if __name__ == "__main__":
    main()
