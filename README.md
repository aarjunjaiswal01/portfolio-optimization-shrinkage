# Portfolio Optimization: Efficient Frontiers, Shrinkage & Robustness Testing

Independent project (MSc Quantitative Finance coursework, UCD Michael Smurfit
Graduate Business School) building on classical Markowitz portfolio theory and
testing how robust its conclusions are once you introduce estimation-error
correction, alternative data sources, and out-of-sample statistical testing.

Built on the Fama-French 17 Industry Portfolios and 5-Factor dataset, monthly,
1980–2024.

## What this project does

Classical mean-variance optimization builds an "optimal" portfolio directly
from sample estimates of expected return and covariance — but those sample
estimates are noisy, and the resulting portfolios are notoriously unstable
out-of-sample. This project works through that problem in four layers:

**1. The classical frontier and Bayes-Stein shrinkage**
Builds the mean-variance efficient frontier (with and without a
short-selling constraint), then applies Bayes-Stein shrinkage (Jorion, 1986)
to pull the noisy sample mean — and separately, the covariance matrix —
toward the Global Minimum Variance Portfolio. Compares the resulting
frontiers, GMVPs, and tangency (max-Sharpe) portfolios, and plots them
against the Capital Allocation Line.

**2. Robustness to the underlying assets: index portfolios vs. individual stocks**
Rebuilds the same frontier using individual stocks as real-world proxies for
each of the 17 industries (e.g. Chevron for Oil, Walmart for Retail),
and compares it to the frontier built from the official index portfolios,
to see how much the "textbook" result depends on using diversified index
data rather than tradeable individual securities.

**3. Statistical robustness: is the tangency portfolio's Sharpe ratio stable over time?**
Splits the Fama-French 5-Factor sample into a pre-2003 and post-2003 period,
computes the tangency portfolio's Sharpe ratio in each, and tests whether the
difference is statistically significant using two methods: the
**Jobson-Korkie test** and a **Ledoit-Wolf-style test with Newey-West
standard errors** (robust to autocorrelation in the return series).

**4. Estimation-error robustness: three ways to estimate the GMVP**
Compares the Global Minimum Variance Portfolio computed three ways —
closed-form unconstrained, constrained to disallow short-selling, and
bootstrap-resampled (averaging GMVP weights across 500 resampled histories)
— and checks how each holds up out-of-sample across the same two periods.

## Data sources

Available for free from
[Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):

- `17_Industry_Portfolios.CSV` — Average Value Weighted Returns, Monthly
- `F-F_Research_Data_Factors.CSV` — for the `RF` risk-free rate column
- `F-F_Research_Data_5_Factors_2x3.csv` — Fama-French 5-Factor model

Place all three in a `data/` folder in the project root:

```
portfolio-optimization/
├── data/
│   ├── 17_Industry_Portfolios.CSV
│   ├── F-F_Research_Data_Factors.CSV
│   └── F-F_Research_Data_5_Factors_2x3.csv
├── portfolio_analysis.py
└── README.md
```

## Running it

```bash
pip install pandas numpy scipy matplotlib
python portfolio_analysis.py
```

The individual-stock robustness section (Section 2) additionally requires
`yfinance` and downloads live market data, so it's off by default:

```bash
pip install yfinance
python portfolio_analysis.py --with-live-data
```

Outputs (plots and CSV summaries) are written to `./output/`.

## A note on reproducibility

Section 2 pulls current stock prices via `yfinance` rather than a fixed,
versioned dataset — so re-running it on a different date will produce
slightly different numbers than shown in any given run. This is expected and
intentional: the point of that section is to sanity-check the frontier
against real, tradeable securities rather than to produce a single
reproducible number. The core analysis (Sections 1, 3, and 4) uses the
versioned Fama-French CSVs and is fully reproducible.

## Key takeaways

- Shrinkage-adjusted frontiers pull toward more diversified, stable
  allocations than the classical frontier, which tends to produce extreme,
  concentrated tangency-portfolio weights driven by estimation noise in the
  sample mean.
- The frontier built from individual stocks sits meaningfully differently
  from the index-portfolio frontier — a reminder that "the efficient
  frontier" is only as good as the assets it's built from.
- Neither the Jobson-Korkie nor the Ledoit-Wolf test found strong evidence
  that the tangency portfolio's Sharpe ratio was stable across the pre- and
  post-2003 sub-periods — a caution against assuming an "optimal" portfolio
  estimated on historical data will keep performing the same way going
  forward.
- Of the three GMVP estimation approaches, the bootstrap-averaged and
  constrained versions were more stable out-of-sample than the unconstrained
  closed-form solution, consistent with the broader literature on
  estimation error in mean-variance optimization.

## Notes

- This was originally developed as coursework for a graduate Portfolio and
  Risk Management course. This version has been substantially restructured
  from the original notebook: rewritten around themes (frontier
  construction → shrinkage → robustness testing) rather than the original
  assignment's question sequence, trimmed to the sections with the clearest
  narrative, and refactored into reusable functions with relative file
  paths so it runs standalone.
- Academic, not investment, advice — this is an exercise in estimation-error
  methodology, not a live trading strategy.
