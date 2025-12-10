import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

def main():
    # Raw data load
    df = pd.read_csv("data/elden_ring_prices.csv")

    expected_cols = {"date", "list_price", "discounted_price"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {expected_cols}. Found: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["discounted_price"] = df["discounted_price"].fillna(df["list_price"])

    df["prev_price"] = df["discounted_price"].shift(1)

    # Percent change 
    df["price_change"] = (df["discounted_price"] - df["prev_price"]) / df["prev_price"]

    df["price_change"] = df["price_change"].replace([np.inf, -np.inf], 0.0)
    df["price_change"] = df["price_change"].fillna(0.0)

    df["rolling_avg_price"] = df["discounted_price"].rolling(7, min_periods=1).mean()

    df["future_min_price_7d"] = (
        df["discounted_price"].shift(-1)
            .rolling(7, min_periods=1)
            .min()
    )

    df["future_cheaper_7d"] = df["future_min_price_7d"] < df["discounted_price"]

    df = df.dropna(subset=["future_min_price_7d"])


    df.to_csv("data/elden_features.csv", index=False)

    feature_cols = ["discounted_price", "price_change", "rolling_avg_price"]
    X = df[feature_cols]
    y = df["future_cheaper_7d"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    #Loading Models with Data for training and testing
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Model Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(model, "data/elden_rf_model.joblib")
    print("\nSaved model to data/elden_rf_model.joblib")
    print("Saved features to data/elden_features.csv")

if __name__ == "__main__":
    main()
