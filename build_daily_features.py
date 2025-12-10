import pandas as pd
import numpy as np
from pathlib import Path

def build_daily_features():
    # Loading price data
    df = pd.read_csv("data/elden_ring_prices.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Sorting by the date on csv
    df = df.sort_values("date")

    # Max listed price and last discounted price per day, and aggregated to unique dates
    df = (
        df.groupby("date", as_index=False)
          .agg({
              "list_price": "max",
              "discounted_price": "last"
          })
          .sort_values("date")
    )

    full_index = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")

    df_daily = (
        df.set_index("date")
          .reindex(full_index)
          .rename_axis("date")
          .reset_index()
    )

    # Forward-filling the discounted price until we see a change in it
    df_daily["discounted_price"] = df_daily["discounted_price"].ffill()
    list_price = df["list_price"].max()
    df_daily["list_price"] = list_price

    df_daily["on_sale"] = df_daily["discounted_price"] < df_daily["list_price"]

    # Days since release
    release_date = df_daily["date"].min()
    df_daily["days_since_release"] = (df_daily["date"] - release_date).dt.days

    # Percent off full price
    df_daily["pct_off"] = 1.0 - (df_daily["discounted_price"] / df_daily["list_price"])

    # Price change
    df_daily["prev_price"] = df_daily["discounted_price"].shift(1)
    df_daily["price_change"] = (
        (df_daily["discounted_price"] - df_daily["prev_price"])
        / df_daily["prev_price"]
    )
    df_daily["price_change"] = df_daily["price_change"].replace([np.inf, -np.inf], 0.0)
    df_daily["price_change"] = df_daily["price_change"].fillna(0.0)

    # Rolling stats
    df_daily["rolling_avg_7"] = df_daily["discounted_price"].rolling(7, min_periods=1).mean()
    df_daily["rolling_avg_30"] = df_daily["discounted_price"].rolling(30, min_periods=1).mean()
    df_daily["rolling_std_7"] = (
        df_daily["discounted_price"].rolling(7, min_periods=1).std().fillna(0.0)
    )

    df_daily["day_of_week"] = df_daily["date"].dt.dayofweek

    month = df_daily["date"].dt.month
    df_daily["is_winter_sale_season"] = month.isin([12, 1]).astype(int)
    df_daily["is_summer_sale_season"] = month.isin([6, 7]).astype(int)
    df_daily["is_autumn_sale_season"] = month.isin([10, 11]).astype(int)

    # Classification label: is there a cheaper price in the next 7 days
    df_daily["future_min_price_7d"] = (
        df_daily["discounted_price"].shift(-1)
            .rolling(7, min_periods=1)
            .min()
    )
    df_daily["future_cheaper_7d"] = df_daily["future_min_price_7d"] < df_daily["discounted_price"]

    # Regression label: minimum price in next 30 days
    df_daily["future_min_price_30d"] = (
        df_daily["discounted_price"].shift(-1)
            .rolling(30, min_periods=1)
            .min()
    )

    df_daily = df_daily.dropna(subset=["future_min_price_7d", "future_min_price_30d"])

    out_path = Path("data/daily_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(out_path, index=False)
    print(f"Saved daily features to {out_path}")

if __name__ == "__main__":
    build_daily_features()
