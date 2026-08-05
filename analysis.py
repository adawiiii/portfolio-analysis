import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Plot Style
plt.style.use("dark_background")
plt.rcParams.update({
    "axes.facecolor": "#000000",
    "figure.facecolor": "#000000",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333333",
    "legend.facecolor": "#111111",
    "legend.edgecolor": "#222222",
    "font.size": 10
})

def nrelu(x: float) -> float:
    return (x - abs(x)) / 2

# Plot Setup
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(12, 8),
    sharex=True,
)

# Data Reading and Date Format Matching
portfolio = pd.read_csv("balance_history_2025-08-01_to_2026-08-03.csv")
portfolio['Date'] = pd.to_datetime(portfolio['Date'])

spx = pd.read_csv("spx_prices_2025-08-01_to_2026-08-03.csv")
spx['Date'] = pd.to_datetime(spx['Date'])

axes[0].plot(portfolio["Date"], portfolio["Close"])
axes[0].set_title("Portfolio Balance")
axes[0].set_ylabel("Balance")
axes[0].tick_params(axis="y", labelleft=False)
axes[0].grid(alpha=0.3)

axes[1].plot(spx["Date"], spx["Close"], color="purple")
axes[1].set_title("S&P 500")
axes[1].set_ylabel("SPX")
axes[1].grid(alpha=0.3)

portfolio["Percent Change"] = (
    portfolio["Close"] / portfolio["Close"].iloc[0] - 1
) * 100

spx["Percent Change"] = (
    spx["Close"] / spx["Close"].iloc[0] - 1
) * 100

axes[2].plot(
    portfolio["Date"],
    portfolio["Percent Change"],
    color="#90fa75",
    label="Portfolio",
)

axes[2].plot(
    spx["Date"],
    spx["Percent Change"],
    color="#ff4040",
    label="S&P 500",
)

axes[2].axhline(0, color="white", linewidth=0.8, alpha=0.5)
axes[2].set_title("Performance Against SPX")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Change (%)")
axes[2].legend()
axes[2].grid(alpha=0.3)

# Beta
portfolio["Daily Return"] = portfolio["Close"].pct_change()
spx["Daily Return"] = spx["Close"].pct_change()

portfolio_covariance = portfolio["Daily Return"].dropna().cov(spx["Daily Return"])
spx_variance = spx["Daily Return"].dropna().var()
portfolio_beta = portfolio_covariance/spx_variance
print(f"Beta: {portfolio_beta:.2f}")

# Sharpe
risk_free_annual = 0.04048 # At time of calculation (Aug. 5th, 2026, 00:00:00Z)
risk_free_daily = (1 + risk_free_annual) ** (1/252) - 1

portfolio["Daily Risk Adj Returns"] = portfolio["Daily Return"] - risk_free_daily
portfolio_sharpe_annualized = (portfolio["Daily Risk Adj Returns"].mean() / portfolio["Daily Risk Adj Returns"].std(ddof=1)) * np.sqrt(252)
print(f"Annualized Sharpe: {portfolio_sharpe_annualized:.2f}")

# Sortino
downside_deviation = np.sqrt(portfolio["Daily Risk Adj Returns"].apply(nrelu).pow(2).mean())
portfolio_sortino_annualized = (portfolio["Daily Risk Adj Returns"].mean() / downside_deviation) * np.sqrt(252)
print(f"Annualized Sortino: {portfolio_sortino_annualized:.2f}")

fig.tight_layout()
plt.show()
fig.savefig("figs/1y_benchmark_comp.png")