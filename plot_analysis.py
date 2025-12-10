import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    df = pd.read_csv("data/daily_features.csv")
    df["date"] = pd.to_datetime(df["date"])

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)


    plt.figure()
    plt.plot(df["date"], df["discounted_price"])
    plt.title("Elden Ring Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(plots_dir / "price_over_time.png")
    plt.close()

    plt.figure()
    plt.plot(df["date"], df["pct_off"])
    plt.title("Percent Discount Over Time")
    plt.xlabel("Date")
    plt.ylabel("Percent off full price")
    plt.tight_layout()
    plt.savefig(plots_dir / "pct_off_over_time.png")
    plt.close()

    plt.figure()
    sns.histplot(df["price_change"], bins=50, kde=True)
    plt.title("Distribution of Daily Price Changes")
    plt.xlabel("Daily price change (fraction)")
    plt.tight_layout()
    plt.savefig(plots_dir / "price_change_histogram.png")
    plt.close()

    feature_cols = [
        "discounted_price",
        "pct_off",
        "price_change",
        "rolling_avg_7",
        "rolling_avg_30",
        "days_since_release",
        "day_of_week",
        "is_winter_sale_season",
        "is_summer_sale_season",
        "is_autumn_sale_season",
    ]
    corr = df[feature_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_correlation_heatmap.png")
    plt.close()

    print("Saved plots to 'plots/' folder")

if __name__ == "__main__":
    main()
