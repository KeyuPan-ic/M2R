import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, normaltest, jarque_bera, probplot

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# =====================================================
# 1. Universe and data
# =====================================================
tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "XOM", "V", "MA", "JNJ", "PG", "HD", "COST",
    "MRK", "ABBV", "CRM", "AMD", "NFLX", "PEP", "KO", "ADBE", "WMT", "BAC",
    "TMO", "CSCO", "MCD", "ABT", "ACN", "LIN", "ORCL", "DIS", "WFC", "INTC",
    "VZ", "CMCSA", "TXN", "DHR", "NKE", "PM", "UPS", "LOW", "QCOM", "IBM",
    "CAT", "GE", "AMGN", "HON", "UNP", "SPGI", "RTX", "NOW", "ISRG", "BKNG",
    "GS", "AXP", "MS", "ELV", "PFE", "PLD", "T", "BLK", "SCHW", "MDT",
    "DE", "SYK", "LMT", "TJX", "VRTX", "ADP", "MMC", "CI", "CB", "ADI",
    "REGN", "BSX", "PANW", "ETN", "MU", "GILD", "C", "LRCX", "KLAC", "ZTS",
    "FI", "SO", "MO", "EQIX", "DUK", "SHW", "ICE", "CL", "AMT", "WM",

    "MCO", "CME", "GD", "SNPS", "CDNS", "PH", "APH", "HCA", "ITW", "NOC",
    "USB", "TDG", "MMM", "PNC", "EOG", "MAR", "CTAS", "ORLY", "FCX", "APD",
    "AON", "MSI", "EMR", "ROP", "BDX", "AJG", "COF", "NSC", "PSX", "TGT",
    "FDX", "ECL", "NXPI", "HLT", "PCAR", "WELL", "GM", "AZO", "DXCM", "TRV",
    "AFL", "O", "ROST", "SPG", "MNST", "KMB", "D", "MPC", "SRE", "ALL",
    "MET", "OKE", "CCI", "AEP", "OXY", "MSCI", "PAYX", "DHI", "KMI", "TEL",
    "PSA", "F", "BK", "LULU", "JCI", "CPRT", "KDP", "NEM", "AMP", "GWW",
    "HUM", "AIG", "PRU", "STZ", "COR", "LEN", "IDXX", "FAST", "EXC", "YUM",
    "VLO", "KR", "CHTR", "GIS", "CTVA", "RSG", "KVUE", "FIS", "IQV", "HES",
    "CMG", "ODFL", "PEG", "EW", "IR", "PCG", "VRSK", "EA", "ACGL", "MLM"
]

benchmark = "SPY"

start = "2005-01-01"
end = "2024-12-31"

all_tickers = tickers + [benchmark]

prices = yf.download(
    all_tickers,
    start=start,
    end=end,
    auto_adjust=True,
    progress=True
)["Close"]

prices = prices.dropna(axis=1, thresh=int(0.9 * len(prices)))
prices = prices.ffill().dropna()

stock_prices = prices.drop(columns=[benchmark])
market_prices = prices[benchmark]

monthly_stock_prices = stock_prices.resample("ME").last()
monthly_market_prices = market_prices.resample("ME").last()

raw_returns = monthly_stock_prices.pct_change().dropna()
market_returns = monthly_market_prices.pct_change().dropna()

raw_returns, market_returns = raw_returns.align(market_returns, join="inner", axis=0)

# =====================================================
# 2. Construct market-adjusted alpha returns
# =====================================================
alpha_returns = raw_returns.sub(market_returns, axis=0)

tickers = list(alpha_returns.columns)
N = len(tickers)

print(f"Final number of stocks used: {N}")
print(f"Sample period: {alpha_returns.index[0].date()} to {alpha_returns.index[-1].date()}")


# =====================================================
# 3. Empirical motivation
# =====================================================
def average_pairwise_correlation(return_df):
    corr = return_df.corr()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return corr.where(mask).stack().mean()


def distribution_summary(return_df, name):
    pooled = return_df.values.flatten()
    pooled = pooled[~np.isnan(pooled)]

    return pd.Series({
        "Mean": np.mean(pooled),
        "Volatility": np.std(pooled),
        "Skewness": skew(pooled),
        "Excess Kurtosis": kurtosis(pooled),
        "Average Pairwise Correlation": average_pairwise_correlation(return_df),
        "Jarque-Bera p-value": jarque_bera(pooled).pvalue,
        "Normaltest p-value": normaltest(pooled).pvalue
    }, name=name)


distribution_results = pd.concat([
    distribution_summary(raw_returns, "Raw Returns"),
    distribution_summary(alpha_returns, "Market-Adjusted Alpha")
], axis=1).T

print("\nDistribution and Dependence Comparison:")
print(distribution_results.round(4))


# =====================================================
# 4. Visual comparison
# =====================================================
raw_pooled = raw_returns.values.flatten()
alpha_pooled = alpha_returns.values.flatten()

raw_pooled = raw_pooled[~np.isnan(raw_pooled)]
alpha_pooled = alpha_pooled[~np.isnan(alpha_pooled)]

plt.figure(figsize=(10, 5))
plt.hist(raw_pooled, bins=80, alpha=0.6, density=True, label="Raw Returns")
plt.hist(alpha_pooled, bins=80, alpha=0.6, density=True, label="Market-Adjusted Alpha")
plt.title("Distribution Comparison: Raw Returns vs Market-Adjusted Alpha")
plt.xlabel("Monthly Return")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
probplot(raw_pooled, dist="norm", plot=plt)
plt.title("QQ Plot: Raw Returns")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
probplot(alpha_pooled, dist="norm", plot=plt)
plt.title("QQ Plot: Market-Adjusted Alpha Returns")
plt.grid(True)
plt.tight_layout()
plt.show()


# =====================================================
# 5. James-Stein alpha estimator
# =====================================================
def james_stein_alpha(alpha_hat, cov, window, positive_part=False):
    n = len(alpha_hat)
    alpha_prior = alpha_hat.mean()

    sigma2 = np.mean(np.diag(cov)) / window
    dispersion = np.sum((alpha_hat - alpha_prior) ** 2)

    if dispersion <= 1e-12:
        return np.repeat(alpha_prior, n), 1.0

    B = ((n - 3) * sigma2) / dispersion

    if positive_part:
        B = np.clip(B, 0, 1)

    alpha_js = (1 - B) * alpha_hat + B * alpha_prior

    return alpha_js, B


# =====================================================
# 6. Portfolio construction from alpha forecasts
# =====================================================
def construct_alpha_portfolio(alpha, cov, ridge=1e-4, max_abs_weight=0.20):
    """
    Convert alpha forecasts into long-short portfolio weights.

    w ∝ Σ^{-1} alpha
    """
    n = len(alpha)
    cov_reg = cov + ridge * np.eye(n)

    try:
        raw_w = np.linalg.solve(cov_reg, alpha)
    except np.linalg.LinAlgError:
        raw_w = np.linalg.pinv(cov_reg) @ alpha

    if np.sum(np.abs(raw_w)) < 1e-8:
        w = np.repeat(1 / n, n)
    else:
        w = raw_w / np.sum(np.abs(raw_w))

    w = np.clip(w, -max_abs_weight, max_abs_weight)

    if np.sum(np.abs(w)) > 1e-8:
        w = w / np.sum(np.abs(w))

    return w


# =====================================================
# 7. Rolling alpha backtest
# =====================================================
window = 60

sample_alpha_pnl = []
js_alpha_pnl = []
js_plus_alpha_pnl = []

sample_alpha_weights = []
js_alpha_weights = []
js_plus_alpha_weights = []

js_B = []
js_plus_B = []

dates = []

for t in range(window, len(alpha_returns) - 1):
    train_alpha = alpha_returns.iloc[t - window:t]
    next_alpha = alpha_returns.iloc[t + 1]

    alpha_sample = train_alpha.mean().values
    cov_alpha = train_alpha.cov().values

    alpha_js, B_js = james_stein_alpha(
        alpha_sample,
        cov_alpha,
        window,
        positive_part=False
    )

    alpha_js_plus, B_js_plus = james_stein_alpha(
        alpha_sample,
        cov_alpha,
        window,
        positive_part=True
    )

    w_sample = construct_alpha_portfolio(alpha_sample, cov_alpha)
    w_js = construct_alpha_portfolio(alpha_js, cov_alpha)
    w_js_plus = construct_alpha_portfolio(alpha_js_plus, cov_alpha)

    sample_alpha_pnl.append(w_sample @ next_alpha.values)
    js_alpha_pnl.append(w_js @ next_alpha.values)
    js_plus_alpha_pnl.append(w_js_plus @ next_alpha.values)

    sample_alpha_weights.append(w_sample)
    js_alpha_weights.append(w_js)
    js_plus_alpha_weights.append(w_js_plus)

    js_B.append(B_js)
    js_plus_B.append(B_js_plus)

    dates.append(alpha_returns.index[t + 1])


sample_alpha_pnl = pd.Series(sample_alpha_pnl, index=dates, name="Sample Alpha")
js_alpha_pnl = pd.Series(js_alpha_pnl, index=dates, name="James-Stein Alpha")
js_plus_alpha_pnl = pd.Series(js_plus_alpha_pnl, index=dates, name="James-Stein+ Alpha")

sample_alpha_weights = pd.DataFrame(sample_alpha_weights, index=dates, columns=tickers)
js_alpha_weights = pd.DataFrame(js_alpha_weights, index=dates, columns=tickers)
js_plus_alpha_weights = pd.DataFrame(js_plus_alpha_weights, index=dates, columns=tickers)

js_B = pd.Series(js_B, index=dates, name="JS Shrinkage")
js_plus_B = pd.Series(js_plus_B, index=dates, name="JS+ Shrinkage")


# =====================================================
# 8. Performance analytics
# =====================================================
def annualized_performance(r):
    ann_return = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol
    return ann_return, ann_vol, sharpe


def turnover(weights):
    return weights.diff().abs().sum(axis=1)


def max_drawdown(r):
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return drawdown.min()


strategies = {
    "Sample Alpha": sample_alpha_pnl,
    "James-Stein Alpha": js_alpha_pnl,
    "James-Stein+ Alpha": js_plus_alpha_pnl
}

weights = {
    "Sample Alpha": sample_alpha_weights,
    "James-Stein Alpha": js_alpha_weights,
    "James-Stein+ Alpha": js_plus_alpha_weights
}

performance = []

for name, r in strategies.items():
    ann_return, ann_vol, sharpe = annualized_performance(r)

    performance.append([
        ann_return,
        ann_vol,
        sharpe,
        turnover(weights[name]).mean(),
        max_drawdown(r)
    ])

performance = pd.DataFrame(
    performance,
    index=strategies.keys(),
    columns=[
        "Annual Alpha Return",
        "Annual Alpha Volatility",
        "Alpha Sharpe",
        "Average Monthly Turnover",
        "Max Drawdown"
    ]
)

print("\nAlpha Strategy Performance:")
print(performance.round(4))

print("\nAverage Shrinkage Intensity:")
print("James-Stein Alpha :", round(js_B.mean(), 4))
print("James-Stein+ Alpha:", round(js_plus_B.mean(), 4))


# =====================================================
# 9. Strategy charts
# =====================================================
cum_alpha = pd.concat([
    (1 + sample_alpha_pnl).cumprod(),
    (1 + js_alpha_pnl).cumprod(),
    (1 + js_plus_alpha_pnl).cumprod()
], axis=1)

plt.figure(figsize=(10, 5))
plt.plot(cum_alpha.index, cum_alpha["Sample Alpha"], label="Sample Alpha")
plt.plot(cum_alpha.index, cum_alpha["James-Stein Alpha"], label="James-Stein Alpha")
plt.plot(cum_alpha.index, cum_alpha["James-Stein+ Alpha"], label="James-Stein+ Alpha")
plt.title("Cumulative Alpha Strategy Performance")
plt.xlabel("Date")
plt.ylabel("Growth of $1 Alpha PnL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


rolling_window = 36

plt.figure(figsize=(10, 5))

for name, r in strategies.items():
    rolling_sharpe = (
        r.rolling(rolling_window).mean()
        / r.rolling(rolling_window).std()
        * np.sqrt(12)
    )
    plt.plot(rolling_sharpe.index, rolling_sharpe, label=name)

plt.title("36-Month Rolling Alpha Sharpe")
plt.xlabel("Date")
plt.ylabel("Rolling Sharpe")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(turnover(sample_alpha_weights), label="Sample Alpha")
plt.plot(turnover(js_alpha_weights), label="James-Stein Alpha")
plt.plot(turnover(js_plus_alpha_weights), label="James-Stein+ Alpha")
plt.title("Monthly Alpha Portfolio Turnover")
plt.xlabel("Date")
plt.ylabel("Turnover")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(js_B.index, js_B, label="James-Stein Alpha")
plt.plot(js_plus_B.index, js_plus_B, label="James-Stein+ Alpha")
plt.title("Alpha Shrinkage Intensity")
plt.xlabel("Date")
plt.ylabel("Shrinkage Intensity B")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()