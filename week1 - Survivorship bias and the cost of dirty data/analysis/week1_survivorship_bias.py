"""
Week 1: Survivorship Bias and the Hidden Cost of Dirty Data
============================================================
Blog series: Applied Computational Finance & Portfolio Optimisation

This script demonstrates how backtesting on survivors-only data systematically
inflates performance metrics. We build two universes:
  1. Biased:    current S&P 500 constituents (everyone who "made it")
  2. Unbiased:  the actual ^GSPC index returns as ground truth

All data is from free sources (Wikipedia, yfinance, FRED via pandas-datareader).
Where historical constituent data is unavailable, we use a calibrated simulation
and flag it clearly. See SIMULATION_MODE flag printed at runtime.
"""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# -- Reproducibility ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -- Paths ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -- Global parameters ----------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
MIN_COVERAGE = 0.80          # drop tickers with < 80% non-NaN observations
ANNUAL_TRADING_DAYS = 252
SIMULATED_DELISTING_RATE = 0.03   # ~3% of universe delisted per year
DELISTING_RETURN_MEAN = -0.40     # mean terminal return for delisted stocks
DELISTING_RETURN_STD = 0.20


# ==============================================================================
# Step 1 -- Build the biased (survivors-only) dataset
# ==============================================================================

def sp500_current_tickers() -> list[str]:
    """Return the list of tickers currently in the S&P 500 (from Wikipedia)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia blocks default urllib User-Agent; supply a browser-like header.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(pd.io.common.StringIO(resp.text), attrs={"id": "constituents"})
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"  Fetched {len(tickers)} current S&P 500 tickers from Wikipedia.")
    return tickers


def raw_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers in one batched yfinance call.
    Tickers that fail silently produce NaN columns; we handle them in the next step.
    """
    print(f"  Downloading prices for {len(tickers)} tickers ({start} -> {end}) ...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # yfinance returns a MultiIndex when multiple tickers are requested
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    print(f"  Downloaded: {prices.shape[1]} tickers x {prices.shape[0]} days.")
    return prices


def survivors_only_prices(prices: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """
    Keep only tickers with >= min_coverage fraction of non-NaN rows.
    These are the "survivors" -- stocks that existed for almost the entire window.
    Dropping low-coverage tickers is exactly what practitioners do casually,
    and it's precisely where the bias enters.
    """
    coverage = prices.notna().mean()
    keep = coverage[coverage >= min_coverage].index
    dropped = prices.shape[1] - len(keep)
    print(f"  Coverage filter ({min_coverage:.0%}): kept {len(keep)}, dropped {dropped}.")
    return prices[keep].copy()


# ==============================================================================
# Step 2 -- Build the reference (unbiased) dataset
# ==============================================================================

def sp500_index_returns(start: str, end: str) -> pd.Series:
    """Fetch ^GSPC daily returns as the unbiased market ground truth."""
    gspc = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(gspc.columns, pd.MultiIndex):
        prices = gspc["Close"].squeeze()
    else:
        prices = gspc["Close"].squeeze()
    returns = prices.pct_change().dropna()
    print(f"  ^GSPC: {len(returns)} daily return observations.")
    return returns


def _fetch_sp500_constituent_history() -> pd.DataFrame | None:
    """
    Attempt to fetch historical S&P 500 constituent changes from the
    fja05680/sp500 GitHub repository. Returns a DataFrame with columns
    [date, ticker, action] or None if unreachable.
    """
    url = (
        "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(08-01-2023).csv"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        print("  Constituent history: fetched real historical changes from GitHub.")
        return df
    except Exception as exc:
        print(f"  Constituent history fetch failed ({exc}). Falling back to simulation.")
        return None


def simulated_point_in_time_returns(
    survivors_prices: pd.DataFrame,
    index_returns: pd.Series,
) -> pd.Series:
    """
    Simulate a point-in-time portfolio by injecting synthetic delistings
    into the survivors universe.

    For each calendar year, we randomly designate ~SIMULATED_DELISTING_RATE
    of the current universe as "delisted". Delisted stocks receive a terminal
    return drawn from a left-skewed (negative-mean) distribution -- mimicking
    the actual distribution of delisting returns documented in Shumway (1997)
    and Beaver et al. (2007). After the terminal return, those stocks leave
    the universe.

    NOTE: This is a calibrated approximation, not real constituent data.
          Results are clearly labelled SIMULATED throughout.
    """
    monthly_ret = survivors_prices.resample("ME").last().pct_change()
    universe = set(monthly_ret.columns)
    portfolio_returns: list[tuple[pd.Timestamp, float]] = []

    rng = np.random.default_rng(SEED)

    for period_end, row in monthly_ret.iterrows():
        active = list(universe)
        if not active:
            break

        # Apply delisting shocks at year-end
        if period_end.month == 12:
            n_delist = max(1, int(len(active) * SIMULATED_DELISTING_RATE))
            delistees = rng.choice(active, size=n_delist, replace=False).tolist()
            for ticker in delistees:
                # Terminal return: left-skewed draw
                shock = rng.normal(DELISTING_RETURN_MEAN, DELISTING_RETURN_STD)
                shock = max(shock, -0.95)   # floor at -95%
                # Override this month's return with the delisting shock
                if ticker in row.index:
                    row[ticker] = shock
                universe.discard(ticker)

        # Equal-weight portfolio return for this month
        valid = [t for t in active if t in row.index and pd.notna(row[t])]
        if valid:
            port_ret = row[valid].mean()
            portfolio_returns.append((period_end, port_ret))

    result = pd.Series(
        {ts: r for ts, r in portfolio_returns},
        name="pit_monthly",
    )
    result.index = pd.to_datetime(result.index)
    return result


# ==============================================================================
# Step 3 -- Construct both portfolios and compute metrics
# ==============================================================================

def equal_weight_monthly_returns(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio, rebalanced monthly. Forward-fills to handle gaps."""
    monthly = prices.resample("ME").last().ffill()
    returns = monthly.pct_change().dropna(how="all")
    # Drop months where fewer than 10 stocks have data (early burn-in)
    returns = returns[returns.notna().sum(axis=1) >= 10]
    port = returns.mean(axis=1)
    port.name = "biased_monthly"
    return port


def risk_free_rate_annual() -> float:
    """
    Fetch 3-month T-bill rate from FRED via yfinance (^IRX).
    Returns the mean annualised rate as a decimal over our sample period.
    Falls back to a reasonable long-run average if the fetch fails.
    """
    try:
        irx = yf.download("^IRX", start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if isinstance(irx.columns, pd.MultiIndex):
            irx = irx["Close"].squeeze()
        else:
            irx = irx["Close"].squeeze()
        rf_annual = irx.dropna().mean() / 100.0
        print(f"  Risk-free rate (^IRX mean): {rf_annual:.3%} p.a.")
        return rf_annual
    except Exception:
        fallback = 0.02
        print(f"  ^IRX fetch failed. Using fallback risk-free rate: {fallback:.1%}")
        return fallback


def annualised_return(monthly_returns: pd.Series) -> float:
    """Geometric annualised return from a monthly return series."""
    n_months = len(monthly_returns)
    cumulative = (1 + monthly_returns).prod()
    return cumulative ** (12 / n_months) - 1


def annualised_volatility(monthly_returns: pd.Series) -> float:
    """Annualised volatility from monthly returns."""
    return monthly_returns.std() * np.sqrt(12)


def sharpe_ratio(monthly_returns: pd.Series, rf_annual: float) -> float:
    """Annualised Sharpe ratio."""
    ret = annualised_return(monthly_returns)
    vol = annualised_volatility(monthly_returns)
    return (ret - rf_annual) / vol if vol > 0 else np.nan


def max_drawdown(monthly_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a negative fraction)."""
    cumulative = (1 + monthly_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def calmar_ratio(monthly_returns: pd.Series) -> float:
    """Calmar ratio = annualised return / abs(max drawdown)."""
    mdd = max_drawdown(monthly_returns)
    if mdd == 0:
        return np.nan
    return annualised_return(monthly_returns) / abs(mdd)


def portfolio_metrics(monthly_returns: pd.Series, rf_annual: float) -> dict:
    return {
        "Annual Return": annualised_return(monthly_returns),
        "Annual Volatility": annualised_volatility(monthly_returns),
        "Sharpe Ratio": sharpe_ratio(monthly_returns, rf_annual),
        "Max Drawdown": max_drawdown(monthly_returns),
        "Calmar Ratio": calmar_ratio(monthly_returns),
    }


# ==============================================================================
# Step 4 -- Quantify the bias
# ==============================================================================

def sharpe_inflation(biased_sharpe: float, true_sharpe: float) -> float:
    """Relative inflation of Sharpe ratio due to survivorship bias."""
    return (biased_sharpe - true_sharpe) / abs(true_sharpe)


def rolling_sharpe(monthly_returns: pd.Series, rf_annual: float, window_months: int = 36) -> pd.Series:
    """Rolling Sharpe ratio over a fixed-length window (default 3 years)."""
    rf_monthly = rf_annual / 12

    def _sharpe(x):
        excess = x - rf_monthly
        if excess.std() == 0:
            return np.nan
        return excess.mean() / excess.std() * np.sqrt(12)

    return monthly_returns.rolling(window_months).apply(_sharpe, raw=True)


def annual_return_decomposition(
    biased_monthly: pd.Series,
    unbiased_monthly: pd.Series,
) -> pd.DataFrame:
    """
    For each calendar year, decompose the Sharpe gap into two components:
      return_inflation:     how much of the gap is from higher returns
      vol_suppression:      how much is from lower volatility (expressed as
                            the additional Sharpe points if vol matched)
    Uses a Brinson-style attribution: hold one factor constant, vary the other.
    """
    rows = []
    # Align to common dates
    common = biased_monthly.index.intersection(unbiased_monthly.index)
    biased = biased_monthly.loc[common]
    unbiased = unbiased_monthly.loc[common]

    for year in sorted(biased.index.year.unique()):
        b = biased[biased.index.year == year]
        u = unbiased[unbiased.index.year == year]
        if len(b) < 6 or len(u) < 6:
            continue

        b_ret = annualised_return(b)
        u_ret = annualised_return(u)
        b_vol = annualised_volatility(b)
        u_vol = annualised_volatility(u)

        # Return component: what Sharpe gain comes from higher returns alone,
        # holding volatility fixed at the unbiased level?
        ret_contribution = (b_ret - u_ret) / u_vol if u_vol > 0 else 0

        # Vol component: what Sharpe gain comes from lower volatility alone,
        # holding returns fixed at the unbiased level?
        if b_vol > 0 and u_vol > 0:
            vol_contribution = u_ret * (1 / b_vol - 1 / u_vol)
        else:
            vol_contribution = 0

        rows.append({
            "Year": year,
            "Return Inflation": ret_contribution,
            "Vol Suppression": vol_contribution,
            "Total Gap": ret_contribution + vol_contribution,
        })

    return pd.DataFrame(rows).set_index("Year")


# ==============================================================================
# Step 5 -- Export data for site rendering
# ==============================================================================

def export_chart_data(
    biased_monthly: pd.Series,
    unbiased_monthly: pd.Series,
    metrics_biased: dict,
    metrics_unbiased: dict,
    decomp: pd.DataFrame,
    rf_annual: float,
    simulation_mode: bool,
) -> Path:
    """
    Write all chart data to a single JSON file consumed by the site renderer.
    Dates are ISO strings; floats are rounded to 6 dp to keep file size sane.
    """
    def _r(v):
        return round(float(v), 6) if v is not None and not np.isnan(v) else None

    common = biased_monthly.index.intersection(unbiased_monthly.index)
    b_cum = (1 + biased_monthly.loc[common]).cumprod()
    u_cum = (1 + unbiased_monthly.loc[common]).cumprod()

    b_roll = rolling_sharpe(biased_monthly.loc[common], rf_annual)
    u_roll = rolling_sharpe(unbiased_monthly.loc[common], rf_annual)

    dates_iso = [d.strftime("%Y-%m-%d") for d in common]

    payload = {
        "meta": {
            "simulation_mode": simulation_mode,
            "rf_annual": _r(rf_annual),
            "generated": pd.Timestamp.today().strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "cumulative_returns": {
            "dates": dates_iso,
            "biased": [_r(v) for v in b_cum.values],
            "reference": [_r(v) for v in u_cum.values],
        },
        "monthly_returns": {
            "dates": dates_iso,
            "biased": [_r(v) for v in biased_monthly.loc[common].values],
            "reference": [_r(v) for v in unbiased_monthly.loc[common].values],
        },
        "metrics": {
            "biased": {k: _r(v) for k, v in metrics_biased.items()},
            "reference": {k: _r(v) for k, v in metrics_unbiased.items()},
        },
        "rolling_sharpe": {
            "dates": dates_iso,
            "biased": [_r(v) for v in b_roll.values],
            "reference": [_r(v) for v in u_roll.values],
        },
        "decomposition": {
            "years": decomp.index.astype(str).tolist(),
            "return_inflation": [_r(v) for v in decomp["Return Inflation"].values],
            "vol_suppression": [_r(v) for v in decomp["Vol Suppression"].values],
            "total_gap": [_r(v) for v in decomp["Total Gap"].values],
        },
        "events": {
            "2011-08": "US downgrade",
            "2020-03": "COVID crash",
            "2022-01": "Rate hike cycle",
        },
    }

    out = DATA_DIR / "week1_chart_data.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved {out.name} ({out.stat().st_size // 1024} KB)")
    return out


# ==============================================================================
# Summary table
# ==============================================================================

def print_summary_table(
    metrics_biased: dict,
    metrics_unbiased: dict,
    simulation_mode: bool,
) -> None:
    label = "SIMULATED PIT" if simulation_mode else "^GSPC INDEX"
    sep = "-" * 58
    print(f"\n{sep}")
    print(f"  WEEK 1 -- SURVIVORSHIP BIAS SUMMARY")
    print(f"  Reference dataset: {label}")
    print(sep)
    fmt_map = {
        "Annual Return": "{:>12.2%}  {:>12.2%}  {:>10.2%}",
        "Annual Volatility": "{:>12.2%}  {:>12.2%}  {:>10.2%}",
        "Sharpe Ratio": "{:>12.3f}  {:>12.3f}  {:>10.1%}",
        "Max Drawdown": "{:>12.2%}  {:>12.2%}  {:>10.2%}",
        "Calmar Ratio": "{:>12.3f}  {:>12.3f}  {:>10.1%}",
    }
    header = f"  {'Metric':<22} {'Biased':>12}  {'Reference':>12}  {'Bias %':>10}"
    print(header)
    print(f"  {'-'*22} {'-'*12}  {'-'*12}  {'-'*10}")

    for metric, fmt in fmt_map.items():
        b = metrics_biased[metric]
        u = metrics_unbiased[metric]
        if u != 0:
            bias_pct = (b - u) / abs(u)
        else:
            bias_pct = float("nan")
        line = fmt.format(b, u, bias_pct)
        print(f"  {metric:<22}{line}")

    print(sep)
    inflation = sharpe_inflation(metrics_biased["Sharpe Ratio"], metrics_unbiased["Sharpe Ratio"])
    print(f"  Sharpe ratio inflation: {inflation:+.1%}")
    print(sep + "\n")


# ==============================================================================
# Main orchestration
# ==============================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("  Week 1: Survivorship Bias Analysis")
    print("=" * 60)

    # Step 1: Biased dataset
    print("\n[Step 1] Building survivors-only (biased) dataset ...")
    tickers = sp500_current_tickers()
    raw = raw_prices(tickers, START_DATE, END_DATE)
    survivor_prices = survivors_only_prices(raw, MIN_COVERAGE)

    biased_monthly = equal_weight_monthly_returns(survivor_prices)

    # Step 2: Reference dataset
    print("\n[Step 2] Building reference dataset ...")
    index_returns = sp500_index_returns(START_DATE, END_DATE)

    constituent_history = _fetch_sp500_constituent_history()
    simulation_mode = constituent_history is None

    if simulation_mode:
        print("  [!] SIMULATION MODE: point-in-time returns are synthetic.")
        print(f"      Delistings: {SIMULATED_DELISTING_RATE:.0%}/yr, "
              f"terminal return mean={DELISTING_RETURN_MEAN:.0%} sd={DELISTING_RETURN_STD:.0%}.")
        unbiased_monthly = simulated_point_in_time_returns(survivor_prices, index_returns)
    else:
        # Fall back to index returns as the cleanest unbiased reference
        # (constituent history processing left as a future extension)
        unbiased_monthly = index_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ).dropna()
        unbiased_monthly.name = "unbiased_monthly"
        print("  Using ^GSPC monthly returns as unbiased reference.")
        simulation_mode = False  # we have real data

    # Align series to common date range
    common_idx = biased_monthly.index.intersection(unbiased_monthly.index)
    biased_monthly = biased_monthly.loc[common_idx]
    unbiased_monthly = unbiased_monthly.loc[common_idx]
    print(f"  Common date range: {common_idx[0].date()} to {common_idx[-1].date()} "
          f"({len(common_idx)} months).")

    # Step 3: Portfolio metrics
    print("\n[Step 3] Computing portfolio metrics ...")
    rf = risk_free_rate_annual()
    m_biased = portfolio_metrics(biased_monthly, rf)
    m_unbiased = portfolio_metrics(unbiased_monthly, rf)

    # Step 4: Quantify bias
    print("\n[Step 4] Quantifying survivorship bias ...")
    inflation = sharpe_inflation(m_biased["Sharpe Ratio"], m_unbiased["Sharpe Ratio"])
    print(f"  Sharpe inflation: {inflation:+.1%}")
    decomp = annual_return_decomposition(biased_monthly, unbiased_monthly)

    # Step 5: Export data for site rendering
    print("\n[Step 5] Exporting chart data ...")
    export_chart_data(biased_monthly, unbiased_monthly, m_biased, m_unbiased,
                      decomp, rf, simulation_mode)

    # -- Summary ------------------------------------------------------------
    print_summary_table(m_biased, m_unbiased, simulation_mode)

    if simulation_mode:
        print("  NOTE: Unbiased portfolio uses simulated delistings.")
        print("  Real constituent history would sharpen these estimates.")
        print("  See the blog post for a discussion of the simulation assumptions.\n")


if __name__ == "__main__":
    main()

