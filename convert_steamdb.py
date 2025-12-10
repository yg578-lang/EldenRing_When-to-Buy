import pandas as pd

def convert():
    df = pd.read_csv("data/steamdb_chart_1245620.csv")

    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "DateTime": "date",
        "Final price": "discounted_price"
    })

    # Converting date to YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"]).dt.date

    max_price = df["discounted_price"].max()
    df["list_price"] = max_price

    df = df[["date", "list_price", "discounted_price"]]

    df.to_csv("data/elden_ring_prices.csv", index=False)

    print("✅ Converted SteamDB CSV → data/elden_ring_prices.csv")

if __name__ == "__main__":
    convert()
