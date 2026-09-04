# -*- coding: utf-8 -*-
"""
Portfolio Optimization: Efficient Frontiers, Shrinkage Estimation & Robustness Testing
========================================================================================

An extended study of mean-variance portfolio construction on the Fama-French
17 Industry Portfolios (1980-2024), moving from the classical Markowitz
frontier through several layers of robustness checking:

    1. Classical mean-variance efficient frontier (with/without short-selling)
    2. Bayes-Stein shrinkage estimation (mean-only, and mean+covariance)
    3. Tangency portfolios, GMVPs, and the Capital Allocation Line
    4. Out-of-sample robustness check: rebuilding the frontier from individual
       stocks used as industry proxies, instead of the index portfolios
    5. Statistical robustness: testing whether the Sharpe ratio of a
       factor-based tangency portfolio is stable across two sub-periods
       (Jobson-Korkie and Ledoit-Wolf tests)
    6. Estimation-error robustness: comparing the Global Minimum Variance
       Portfolio under unconstrained, no-short-selling, and bootstrap-
       resampled estimation

Data sources (see README.md for download links):
    - 17_Industry_Portfolios.CSV        (Kenneth French Data Library)
    - F-F_Research_Data_Factors.CSV     (Kenneth French Data Library)
    - F-F_Research_Data_5_Factors_2x3.csv (Kenneth French Data Library)
    - Individual stock / mutual fund prices via yfinance (live data - see
      note on reproducibility in README.md)

Usage:
    python portfolio_analysis.py                  # core sections only
    python portfolio_analysis.py --with-live-data  # also runs yfinance sections
"""

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.optimize as sco

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INDUSTRY_FILE = os.path.join(DATA_DIR, "17_Industry_Portfolios.CSV")
FACTORS_FILE = os.path.join(DATA_DIR, "F-F_Research_Data_Factors.CSV")
FF5_FILE = os.path.join(DATA_DIR, "F-F_Research_Data_5_Factors_2x3.csv")

START_MONTH = dt.datetime(1980, 1, 1)
END_MONTH = dt.datetime(2024, 12, 1)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _savefig(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


def new_figure(figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.7, color="black")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5, color="gray")
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_industry_returns(path, start, end):
    data = pd.read_csv(path, header=11)
    cols = data.columns.to_list()
    cols[0] = "Month"
    data.columns = cols
    data.columns = [c.strip() for c in data.columns]
    data.set_index("Month", inplace=True)

    nan_rows = np.where(data.index.isna())[0]
    ret_df = data.iloc[: nan_rows[0]] if len(nan_rows) else data.copy()

    ret_df.index = pd.to_datetime(ret_df.index, format="%Y%m", errors="coerce")
    ret_df = ret_df[(ret_df.index >= start) & (ret_df.index <= end)]
    return ret_df.astype("float")


def load_risk_free_rate(path, start, end):
    rf = pd.read_csv(path, skiprows=3)
    rf = rf.iloc[: rf[rf["RF"].isna()].index[0]]
    rf.set_index(rf.columns[0], inplace=True)
    rf.index.name = ""
    rf = rf[["RF"]]
    rf.index = pd.to_datetime(rf.index, format="%Y%m")
    rf = rf[(rf.index >= start) & (rf.index <= end)]
    rf["RF"] = rf["RF"].astype(str).str.strip().astype("float")
    return rf


def load_ff5_factors(path, start, end):
    ff5 = pd.read_csv(path, skiprows=3)
    ff5 = ff5.iloc[: ff5[ff5["RF"].isna()].index[0]]
    ff5.set_index(ff5.columns[0], inplace=True)
    ff5.index.name = "Date"
    ff5.index = pd.to_datetime(ff5.index, format="%Y%m")
    ff5 = ff5.astype("float")
    ff5 = ff5[(ff5.index >= start) & (ff5.index <= end)]
    return ff5.drop(columns=["RF"])


# ---------------------------------------------------------------------------
# 2. Core mean-variance optimization
# ---------------------------------------------------------------------------
def port_risk_fn(w, cov_mat):
    return np.sqrt(w.T @ cov_mat @ w)


def min_risk(r_mat, mu_target, bs_mu=None, bs_cov=None, allow_short=True):
    asset_len = r_mat.shape[1]
    cov_mat = r_mat.cov().values if bs_cov is None else bs_cov
    mean_vec = r_mat.mean().values if bs_mu is None else bs_mu

    target_fn = lambda w: port_risk_fn(w, cov_mat)
    init_w = np.ones(asset_len) / asset_len

    cons = (
        {"type": "eq", "fun": lambda w: w.T @ mean_vec - mu_target},
        {"type": "eq", "fun": lambda w: w.T @ np.ones(asset_len) - 1},
    )
    bounds = None if allow_short else [(0, 1) for _ in range(asset_len)]

    result = minimize(target_fn, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    if not result["success"]:
        raise ValueError(f"Optimization failed: {result['message']}")

    w_opt = result["x"]
    return w_opt, port_risk_fn(w_opt, cov_mat)


def build_frontier(r_mat, mu_targets, bs_mu=None, bs_cov=None, allow_short=True):
    frontier = {"risk": [], "mean": [], "opt_w": []}
    for mu_target in mu_targets:
        try:
            opt_w, risk = min_risk(r_mat, mu_target, bs_mu=bs_mu, bs_cov=bs_cov, allow_short=allow_short)
            frontier["risk"].append(risk)
            frontier["opt_w"].append(opt_w)
        except ValueError as e:
            print(e)
            frontier["risk"].append(None)
            frontier["opt_w"].append(None)
        frontier["mean"].append(mu_target)
    return frontier


def sharpe_ratio(w, mu_vec, cov_mat, rf_rate):
    return w @ (mu_vec - rf_rate) / np.sqrt(w.T @ cov_mat @ w)


def get_tangency_port(mu_vec, cov_mat, rf_rate, allow_short=True):
    asset_len = len(mu_vec)
    target_fn = lambda w: -sharpe_ratio(w, mu_vec, cov_mat, rf_rate)
    init_w = np.ones(asset_len) / asset_len

    cons = ({"type": "eq", "fun": lambda w: w.T @ np.ones(asset_len) - 1},)
    bounds = None if allow_short else [(0, 1) for _ in range(asset_len)]

    result = minimize(target_fn, init_w, method="SLSQP", constraints=cons, bounds=bounds)
    if not result["success"]:
        raise ValueError(f"Optimization failed: {result['message']}")

    w_tp = result["x"]
    return port_risk_fn(w_tp, cov_mat), w_tp @ mu_vec, w_tp


# ---------------------------------------------------------------------------
# 3. Bayes-Stein shrinkage
# ---------------------------------------------------------------------------
def bayes_stein_shrinkage(frontier, return_df):
    """
    Shrink sample mean and covariance toward the Global Minimum Variance
    Portfolio, following Jorion (1986).
    """
    min_risk_index = int(np.argmin(frontier["risk"]))
    gmvp_mu = frontier["mean"][min_risk_index]

    K = return_df.shape[1]
    N = return_df.shape[0]

    mu = return_df.mean(axis=0).values
    cov = return_df.cov().values
    cov_inv = np.linalg.inv(cov)

    lam = (K + 2) * (N - 1) / ((N - K - 2) * ((mu - gmvp_mu).T @ cov_inv @ (mu - gmvp_mu)))
    bs_factor = lam / (N + lam)

    bs_mu = bs_factor * gmvp_mu + (1 - bs_factor) * mu
    bs_cov = cov * (1 + 1 / (N + lam)) + K * lam / (N * (N + 1 + lam) * cov_inv.sum())

    return bs_mu, bs_cov


# ---------------------------------------------------------------------------
# 4. Statistical tests for Sharpe ratio stability
# ---------------------------------------------------------------------------
def compute_tangency_weights_closed_form(excess_returns):
    """Closed-form (unconstrained) tangency weights: Sigma^-1 * mu, normalized to sum to 1."""
    mu = excess_returns.mean().values
    sigma = excess_returns.cov().values
    inv_sigma = np.linalg.inv(sigma)
    prelim = inv_sigma @ mu
    return prelim / np.sum(prelim)


def jobson_korkie_test(sr1, sr2, n):
    """Jobson-Korkie test statistic for the difference of two Sharpe ratios."""
    return np.sqrt(n) * (sr1 - sr2) / np.sqrt(2 + sr1**2 + sr2**2)


def newey_west_variance(series, lags=4):
    """Newey-West HAC variance estimate for the sample mean."""
    T = len(series)
    resid = series - series.mean()
    var_est = np.var(resid, ddof=1)
    for lag in range(1, lags + 1):
        gamma = np.dot(resid[lag:], resid[:-lag]) / (T - lag)
        var_est += 2 * (1 - lag / (lags + 1)) * gamma
    return var_est / T


def sharpe_ratio_variance(returns, lags=4):
    """Delta-method approximation of the variance of a Sharpe ratio estimate."""
    return newey_west_variance(returns, lags) / (returns.std(ddof=1) ** 2)


def ledoit_wolf_test(returns1, returns2, lags=4):
    """
    Ledoit-Wolf-style test for the difference of two Sharpe ratios, using
    Newey-West standard errors for robustness to autocorrelation.
    H0: SR1 = SR2.  Returns a statistic approximately standard normal under H0.
    """
    sr1 = returns1.mean() / returns1.std(ddof=1)
    sr2 = returns2.mean() / returns2.std(ddof=1)
    var_diff = sharpe_ratio_variance(returns1, lags) + sharpe_ratio_variance(returns2, lags)
    return (sr1 - sr2) / np.sqrt(var_diff)


# ---------------------------------------------------------------------------
# 5. GMVP robustness: unconstrained, constrained, bootstrap
# ---------------------------------------------------------------------------
def compute_gmvp_weights_closed_form(returns):
    """Unconstrained (short-selling allowed) closed-form GMVP weights."""
    sigma = returns.cov().values
    ones = np.ones(sigma.shape[0])
    inv_sigma = np.linalg.inv(sigma)
    return inv_sigma @ ones / (ones.T @ inv_sigma @ ones)


def compute_gmvp_constrained(returns):
    """No-short-selling GMVP via constrained optimization."""
    sigma = returns.cov().values
    n = sigma.shape[0]
    w0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n))
    res = sco.minimize(lambda w, S: w.T @ S @ w, w0, args=(sigma,), method="SLSQP",
                        bounds=bounds, constraints=cons)
    return res.x


def bootstrap_gmvp(returns, n_boot=500, random_state=42):
    """Average GMVP weights across bootstrap resamples of the return history."""
    rng = np.random.default_rng(random_state)
    n_obs = returns.shape[0]
    weights = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_obs, n_obs)
        sample = returns.iloc[idx]
        weights.append(compute_gmvp_weights_closed_form(sample))
    return np.mean(weights, axis=0)


# ---------------------------------------------------------------------------
# 6. Section 1 - Classical frontier, shrinkage, tangency, CAL
# ---------------------------------------------------------------------------
def run_core_analysis(ret_df, rf_rate):
    print("\n--- Section 1: Classical vs. Bayes-Stein Efficient Frontiers ---")
    mv_df = pd.DataFrame({"Mu": ret_df.mean(), "Sigma": ret_df.std()})
    mu_targets = np.arange(0.01 * mv_df.Mu.min(), 2 * mv_df.Mu.max(), step=0.01)

    mv_frontier = build_frontier(ret_df, mu_targets, allow_short=True)
    no_short_frontier = build_frontier(ret_df, mu_targets, allow_short=False)

    bs_mu, bs_cov = bayes_stein_shrinkage(mv_frontier, ret_df)
    bs_mean_frontier = build_frontier(ret_df, mu_targets, bs_mu=bs_mu)
    bs_frontier = build_frontier(ret_df, mu_targets, bs_mu=bs_mu, bs_cov=bs_cov)

    fig, ax = new_figure()
    plt.plot(mv_frontier["risk"], mv_frontier["mean"], "-b", label="Classical MV Frontier")
    plt.plot(no_short_frontier["risk"], no_short_frontier["mean"], "-m", label="Classical (No Short Selling)")
    plt.plot(bs_mean_frontier["risk"], bs_mean_frontier["mean"], "-r", label="BS Frontier (Mean only)")
    plt.plot(bs_frontier["risk"], bs_frontier["mean"], "--g", label="BS Frontier (Mean + Cov)")
    plt.plot(mv_df.Sigma, mv_df.Mu, ".k", label="Individual Industries")
    ax.set_title("Efficient Frontiers: Classical vs. Bayes-Stein Shrinkage", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sigma (%)"); ax.set_ylabel("E(R) (%)"); ax.legend()
    _savefig("01_frontiers_classical_vs_shrinkage.png")

    print("--- Section 1b: Tangency Portfolios and GMVPs ---")
    mu = ret_df.mean(axis=0).values
    cov = ret_df.cov().values

    classical_gmvp_idx = int(np.argmin(mv_frontier["risk"]))
    bs_mean_gmvp_idx = int(np.argmin(bs_mean_frontier["risk"]))
    bs_gmvp_idx = int(np.argmin(bs_frontier["risk"]))

    classical_tp = get_tangency_port(mu, cov, rf_rate)
    bs_mean_tp = get_tangency_port(bs_mu, cov, rf_rate)
    bs_tp = get_tangency_port(bs_mu, bs_cov, rf_rate)

    fig, ax = new_figure(figsize=(10, 7))
    plt.plot(mv_frontier["risk"], mv_frontier["mean"], "-b", label="Classical Frontier")
    plt.plot(mv_frontier["risk"][classical_gmvp_idx], mv_frontier["mean"][classical_gmvp_idx],
              "*b", label="Classical GMVP", markersize=12)
    plt.plot(classical_tp[0], classical_tp[1], "xb", label="Classical TP", markersize=9)

    plt.plot(bs_mean_frontier["risk"], bs_mean_frontier["mean"], "-r", label="BS Frontier (Mean only)")
    plt.plot(bs_mean_frontier["risk"][bs_mean_gmvp_idx], bs_mean_frontier["mean"][bs_mean_gmvp_idx],
              "*r", label="BS GMVP (Mean only)", markersize=12)
    plt.plot(bs_mean_tp[0], bs_mean_tp[1], "xr", label="BS TP (Mean only)", markersize=9)

    plt.plot(bs_frontier["risk"], bs_frontier["mean"], "--g", label="BS Frontier")
    plt.plot(bs_frontier["risk"][bs_gmvp_idx], bs_frontier["mean"][bs_gmvp_idx],
              "*g", label="BS GMVP", markersize=12)
    plt.plot(bs_tp[0], bs_tp[1], "xg", label="BS TP", markersize=9)

    plt.plot([0, classical_tp[0]], [rf_rate, classical_tp[1]], "-.k", label="CAL (Classical TP)")
    plt.plot(0, rf_rate, "^m", label="Risk-Free Rate", markersize=10)

    ax.set_title("GMVPs, Tangency Portfolios, and the Capital Allocation Line",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Sigma (%)"); ax.set_ylabel("E(R) (%)"); ax.legend(fontsize=8)
    _savefig("02_tangency_gmvp_cal.png")

    # Summary table: excess return, volatility, Sharpe ratio
    port_w = {
        "classical_tp": classical_tp[2],
        "bs_mean_tp": bs_mean_tp[2],
        "bs_tp": bs_tp[2],
        "classical_gmvp": mv_frontier["opt_w"][classical_gmvp_idx],
        "bs_mean_gmvp": bs_mean_frontier["opt_w"][bs_mean_gmvp_idx],
        "bs_gmvp": bs_frontier["opt_w"][bs_gmvp_idx],
    }
    mu_er = mu - rf_rate
    cov_er = (ret_df - rf_rate).cov().values

    port_stat = {}
    for name, w in port_w.items():
        mu_er_i = w.T @ mu_er
        std_er_i = np.sqrt(w.T @ cov_er @ w)
        port_stat[name] = {"Mean": mu_er_i, "Volatility": std_er_i, "Sharpe Ratio": mu_er_i / std_er_i}

    port_stat_df = pd.DataFrame(port_stat).T
    port_stat_df.index.name = "Portfolio (Excess Return basis)"
    print(port_stat_df.round(4))
    port_stat_df.to_csv(os.path.join(OUT_DIR, "portfolio_summary.csv"))

    return mv_frontier, bs_mu, bs_cov


# ---------------------------------------------------------------------------
# 7. Section 2 - Robustness: individual stocks as industry proxies
# ---------------------------------------------------------------------------
INDUSTRY_TICKERS = {
    "Food": "SYY", "Mines": "AEM", "Oil": "CVX", "Clths": "VFC", "Durbl": "WHR",
    "Chems": "DD", "Cnsum": "LLY", "Cnstr": "TPC", "Steel": "AA", "FabPr": "AVT",
    "Machn": "DE", "Cars": "F", "Trans": "MATX", "Utils": "DUK", "Rtail": "WMT",
    "Finan": "AIG", "Other": "DHR",
}


def run_stock_proxy_robustness_check(ret_df):
    """
    Rebuild the efficient frontier using individual stocks as proxies for
    each industry, and compare it against the frontier built from the
    official index portfolios over the same window. Requires yfinance and
    an internet connection; results will vary by run date (see README).
    """
    import yfinance as yf

    print("\n--- Section 2: Robustness Check - Individual Stocks as Industry Proxies ---")
    start_date = "1985-12-01"
    end_date = dt.datetime.today().strftime("%Y-%m-%d")

    prices = {}
    for industry, ticker in INDUSTRY_TICKERS.items():
        df = yf.download(ticker, start=start_date, end=end_date, interval="1mo", progress=False)
        if not df.empty:
            prices[industry] = df["Close"].dropna()

    price_df = pd.concat(prices, axis=1)
    stock_returns = price_df.pct_change().dropna() * 100
    stock_returns.index = stock_returns.index.to_period("M").to_timestamp()

    ret_df_aligned = ret_df[ret_df.index >= stock_returns.index.min()]

    mu_targets = np.arange(0.01 * ret_df_aligned.mean().min(), 2 * ret_df_aligned.mean().max(), step=0.01)
    index_frontier = build_frontier(ret_df_aligned, mu_targets)
    stock_frontier = build_frontier(stock_returns, mu_targets)

    fig, ax = new_figure()
    plt.plot(index_frontier["risk"], index_frontier["mean"], "-b", label="Industry Index Portfolios")
    plt.plot(stock_frontier["risk"], stock_frontier["mean"], "--r", label="Individual Stock Proxies")
    ax.set_title("Efficient Frontier: Index Portfolios vs. Individual Stock Proxies",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Sigma (%)"); ax.set_ylabel("E(R) (%)"); ax.legend()
    _savefig("03_stock_proxy_robustness.png")


# ---------------------------------------------------------------------------
# 8. Section 3 - Statistical robustness: Sharpe ratio stability over time
# ---------------------------------------------------------------------------
def run_sharpe_stability_test(ff5):
    print("\n--- Section 3: Is the Tangency Portfolio's Sharpe Ratio Stable Over Time? ---")
    ff5_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    period1 = ff5.loc[:"2002-12-31", ff5_cols]
    period2 = ff5.loc["2003-01-01":"2024-12-31", ff5_cols]

    weights = compute_tangency_weights_closed_form(period1)
    ret1 = period1.dot(weights)
    ret2 = period2.dot(weights)

    sr1 = ret1.mean() / ret1.std()
    sr2 = ret2.mean() / ret2.std()
    print(f"Tangency portfolio Sharpe ratio, Period 1 (pre-2003): {sr1:.4f}")
    print(f"Tangency portfolio Sharpe ratio, Period 2 (post-2003): {sr2:.4f}")

    jk_stat = jobson_korkie_test(sr1, sr2, len(ret1))
    lw_stat = ledoit_wolf_test(ret1, ret2, lags=4)
    print(f"Jobson-Korkie test statistic:  {jk_stat:.4f}")
    print(f"Ledoit-Wolf test statistic:    {lw_stat:.4f}")
    print("(|stat| > ~1.96 would suggest the Sharpe ratio differs significantly between periods)")

    return sr1, sr2, jk_stat, lw_stat


# ---------------------------------------------------------------------------
# 9. Section 4 - GMVP estimation-error robustness
# ---------------------------------------------------------------------------
def run_gmvp_robustness_check(ff5, mv_frontier_period1=None):
    print("\n--- Section 4: GMVP Robustness - Unconstrained vs. Constrained vs. Bootstrap ---")
    ff5_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    period1 = ff5.loc[:"2002-12-31", ff5_cols]
    period2 = ff5.loc["2003-01-01":"2024-12-31", ff5_cols]

    w_unconstrained = compute_gmvp_weights_closed_form(period1)
    w_constrained = compute_gmvp_constrained(period1)
    w_bootstrap = bootstrap_gmvp(period1, n_boot=500)

    results = {}
    for name, w in [("Unconstrained", w_unconstrained),
                     ("Constrained (No Short)", w_constrained),
                     ("Bootstrap-Averaged", w_bootstrap)]:
        ret1 = period1.dot(w)
        ret2 = period2.dot(w)
        results[name] = {
            "Sharpe (Period 1)": ret1.mean() / ret1.std(),
            "Sharpe (Period 2)": ret2.mean() / ret2.std(),
        }

    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    results_df.to_csv(os.path.join(OUT_DIR, "gmvp_robustness_summary.csv"))
    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-live-data", action="store_true",
                        help="Also run the yfinance-dependent stock proxy section (Section 2).")
    args = parser.parse_args()

    if not (os.path.exists(INDUSTRY_FILE) and os.path.exists(FACTORS_FILE) and os.path.exists(FF5_FILE)):
        raise FileNotFoundError(
            "Expected data files not found in ./data/. See README.md for download "
            "instructions from the Kenneth French Data Library."
        )

    ret_df = load_industry_returns(INDUSTRY_FILE, START_MONTH, END_MONTH)
    rf = load_risk_free_rate(FACTORS_FILE, START_MONTH, END_MONTH)
    rf_rate = rf.mean().item()
    ff5 = load_ff5_factors(FF5_FILE, START_MONTH, END_MONTH)

    run_core_analysis(ret_df, rf_rate)

    if args.with_live_data:
        try:
            run_stock_proxy_robustness_check(ret_df)
        except ImportError:
            print("\nSkipping Section 2: install yfinance to run the stock-proxy robustness check.")

    run_sharpe_stability_test(ff5)
    run_gmvp_robustness_check(ff5)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
