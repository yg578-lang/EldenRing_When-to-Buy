import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
    confusion_matrix
)
import joblib

def train_models():
    df = pd.read_csv("data/daily_features.csv")

    class_features = [
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
    X_class = df[class_features]
    y_class = df["future_cheaper_7d"].astype(int)

    X_reg = X_class.copy()
    y_reg = df["future_min_price_30d"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_class, y_class, test_size=0.2, shuffle=False
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, shuffle=False
    )

    # ----- Classification Models -----

    print("\n=== Classification: Predict cheaper price in next 7 days ===")

    # Logistic Regression
    logreg = LogisticRegression(max_iter=2000)
    logreg.fit(Xc_train, yc_train)
    y_pred_lr = logreg.predict(Xc_test)
    y_proba_lr = logreg.predict_proba(Xc_test)[:, 1]

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    rf_clf.fit(Xc_train, yc_train)
    y_pred_rf = rf_clf.predict(Xc_test)
    y_proba_rf = rf_clf.predict_proba(Xc_test)[:, 1]

    def print_class_metrics(name, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n{name}")
        print("-" * len(name))
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print("Confusion matrix [ [TN FP], [FN TP] ]:")
        print(cm)

    print_class_metrics("Logistic Regression", yc_test, y_pred_lr, y_proba_lr)
    print_class_metrics("Random Forest Classifier", yc_test, y_pred_rf, y_proba_rf)


    print("\n=== Regression: Predict 30-day minimum future price ===")

    rf_reg = RandomForestRegressor(
        n_estimators=400,
        random_state=42
    )
    rf_reg.fit(Xr_train, yr_train)
    y_pred_reg = rf_reg.predict(Xr_test)

    mae = mean_absolute_error(yr_test, y_pred_reg)
    mse = mean_squared_error(yr_test, y_pred_reg)
    rmse = mse ** 0.5

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    joblib.dump(logreg, "data/elden_logreg_model.joblib")
    joblib.dump(rf_clf, "data/elden_rf_classifier_advanced.joblib")
    joblib.dump(rf_reg, "data/elden_rf_regressor_30d.joblib")

    print("\nSaved models:")
    print("  data/elden_logreg_model.joblib")
    print("  data/elden_rf_classifier_advanced.joblib")
    print("  data/elden_rf_regressor_30d.joblib")

if __name__ == "__main__":
    train_models()
