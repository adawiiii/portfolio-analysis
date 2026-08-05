import yfinance as yf
import pandas as pd

spx = yf.Ticker("^GSPC").history(
    start="2025-08-01",
    end="2026-08-04",  # end date is exclusive
    interval="1d",
    auto_adjust=False,
)

spx = spx.reset_index()
spx["Date"] = pd.to_datetime(spx["Date"]).dt.strftime("%Y-%m-%d")
spx.to_csv("spx_prices_2025-08-01_to_2026-08-03.csv", index=False)

print(spx.head())