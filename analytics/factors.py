"""
factors.py
----------
Fama-French factor regressions, attribution and rolling exposures.

CRITICAL RETURN CONVENTION
--------------------------
French factor datasets are **simple percentage returns**, and so are the
regressions here: portfolio **simple** daily returns minus the French
daily RF, regressed on **simple** factor returns. Log returns are never
regressed against French factors. Conversions are explicit:

* ``data/factors.py`` returns raw **percent** frames (``0.42`` = 0.42%);
* :func:`to_decimal_returns` converts to decimals (``0.0042``);
* :func:`simple_returns_from_wealth` builds simple returns from a wealth
  path (``V/V.shift(1) − 1``), e.g. from the backtesting engine;
* the French **RF** series (same units, same dates) is the only risk-free
  rate used inside factor regressions — never the app-level input.

Methodology
-----------
* OLS coefficients via ``statsmodels`` (never hand-rolled inversion).
* Inference uses HAC/Newey-West robust covariance, default lag 5 trading
  days (documented at every use site); iid standard errors are not assumed
  reliable for daily returns.
* Daily alpha annualises by exact compounding ``(1+αd)^252 − 1``; the raw
  daily alpha is always retained. Beta is never annualised.
* Attribution is arithmetic (mean-based) and model-dependent: it sums to
  the fitted mean return, not to compounded wealth — documented wherever
  shown. Rolling windows are trailing only (no look-ahead).

All functions are pure: inputs are never mutated, alignment is by explicit
inner join on dates, and invalid input raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_FACTORS: dict[str, list[str]] = {
    "capm": ["Mkt-RF"],
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "ff5_mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}

MODEL_LABELS: dict[str, str] = {
    "capm": "CAPM",
    "ff3": "Fama-French 3-Factor",
    "ff5": "Fama-French 5-Factor",
    "ff5_mom": "Fama-French 5 + Momentum",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "capm": "Excess return on the broad US market only.",
    "ff3": "Market plus size (SMB) and value (HML).",
    "ff5": "Market, size, value, profitability (RMW) and investment (CMA).",
    "ff5_mom": "Five factors plus price momentum (Mom).",
}

FACTOR_LABELS: dict[str, str] = {
    "Mkt-RF": "Market",
    "SMB": "Size",
    "HML": "Value",
    "RMW": "Profitability",
    "CMA": "Investment",
    "Mom": "Momentum",
}

FACTOR_MEANINGS: dict[str, str] = {
    "Mkt-RF": "broad US market excess return",
    "SMB": "small minus large caps",
    "HML": "value minus growth",
    "RMW": "robust minus weak profitability",
    "CMA": "conservative minus aggressive investment",
    "Mom": "past winners minus past losers",
}

#: Default Newey-West lag in trading days (documented everywhere it matters).
DEFAULT_HAC_LAGS = 5
#: Hard floor for full-sample daily regressions.
MIN_FACTOR_OBS = 60
#: Below this, results carry an explicit small-sample warning.
WARN_FACTOR_OBS = 252
TRADING_DAYS = 252

_DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin=dict(l=60, r=30, t=60, b=60),
)


@dataclass
class FactorRegressionResult:
    """Structured outcome of one factor regression (simple-return space)."""

    model_name: str
    factor_names: list[str]
    alpha: float                    # daily intercept (decimal)
    alpha_annualised: float         # (1+αd)^252 − 1, exact compounding
    betas: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)  # HAC robust
    t_stats: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    alpha_se: float = float("nan")  # HAC robust SE of alpha
    alpha_t: float = float("nan")
    alpha_p: float = float("nan")
    r_squared: float = float("nan")
    adj_r_squared: float = float("nan")
    observations: int = 0
    residual_volatility: float = float("nan")  # annualised, ×√252
    durbin_watson: float = float("nan")
    condition_number: float = float("nan")
    hac_lags: int = DEFAULT_HAC_LAGS


# ---------------------------------------------------------------------------
# Return construction / conversion
# ---------------------------------------------------------------------------

def to_decimal_returns(percent_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert French percent units (``0.42`` = 0.42%) to decimals (``0.0042``).

    Explicit by design: forgetting this step inflates every factor return
    100×, so conversion lives in exactly one tested place. Infinite values
    are rejected; NaNs are preserved for the alignment step to drop.
    """
    if percent_returns is None or percent_returns.empty:
        raise ValueError("No factor data to convert.")
    out = pd.DataFrame(percent_returns, dtype=float) / 100.0
    if np.isinf(out.values).any():
        raise ValueError("Factor data contains infinite values.")
    return out


def simple_returns_from_wealth(values: pd.Series) -> pd.Series:
    """Daily **simple** returns from a wealth path: ``V/V.shift(1) − 1``.

    Used for factor work (French factors are simple returns). Never pass
    log returns here — and never use this series where log returns are
    expected elsewhere in the app.
    """
    v = pd.Series(values, dtype=float).dropna()
    if len(v) < 2:
        raise ValueError("Need at least 2 wealth observations for returns.")
    if ((v <= 0) | (~np.isfinite(v))).any():
        raise ValueError("Wealth path must contain only positive finite values.")
    return (v / v.shift(1) - 1).dropna().rename("portfolio")


def align_factor_returns(
    portfolio_simple: pd.Series, factors: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join portfolio simple returns with factor returns on dates.

    Only common dates survive — US holidays, missing Yahoo observations,
    factor gaps and IPO cutoffs can never shift the regression. Neither
    input is mutated. Raises ``ValueError`` when nothing overlaps.
    """
    if portfolio_simple is None or factors is None:
        raise ValueError("Portfolio and factor returns are both required.")
    y = pd.Series(portfolio_simple, dtype=float)
    X = pd.DataFrame(factors, dtype=float)
    if X.empty or X.shape[1] == 0:
        raise ValueError("No factor columns supplied.")
    joint = pd.concat([y.rename("portfolio"), X], axis=1, join="inner")
    joint = joint.dropna(how="any")
    if joint.empty:
        raise ValueError(
            "No overlapping dates between portfolio returns and factor data."
        )
    return joint["portfolio"].copy(), joint[X.columns].copy()


def check_min_observations(
    nobs: int,
    n_factors: int,
    floor: int = MIN_FACTOR_OBS,
    warn_level: int = WARN_FACTOR_OBS,
) -> bool:
    """Enforce minimum observations for a factor regression.

    Raises ``ValueError`` when ``nobs`` is below ``floor`` (default 60) or
    does not materially exceed the parameter count. Returns ``True`` when
    results should carry an explicit small-sample warning
    (``floor <= nobs < warn_level``, default 252).
    """
    nobs = int(nobs)
    if nobs <= int(n_factors) + 1:
        raise ValueError(
            f"Only {nobs} observations for {n_factors} factors — the model "
            "cannot be identified."
        )
    if nobs < floor:
        raise ValueError(
            f"Only {nobs} observations; at least {floor} are required for a "
            "stable factor regression."
        )
    return nobs < warn_level


# ---------------------------------------------------------------------------
# Regression engine (OLS + HAC via statsmodels)
# ---------------------------------------------------------------------------

def run_factor_regression(
    y_excess: pd.Series,
    X: pd.DataFrame,
    model_name: str,
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> FactorRegressionResult:
    """OLS regression of excess simple returns on simple factor returns.

    Coefficients are OLS; standard errors, t-stats and p-values use
    HAC/Newey-West robust covariance (default lag 5 trading days — daily
    returns are autocorrelated/heteroskedastic, so iid SEs are not trusted).
    Annualised alpha uses exact compounding ``(1+αd)^252 − 1``.
    """
    if model_name not in MODEL_FACTORS:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of "
            f"{sorted(MODEL_FACTORS)}."
        )
    y = pd.Series(y_excess, dtype=float).dropna()
    X = pd.DataFrame(X, dtype=float).dropna(how="any")
    common = y.index.intersection(X.index)
    y, X = y.loc[common], X.loc[common]
    if len(y) == 0:
        raise ValueError("No overlapping observations for factor regression.")
    if not np.all(np.isfinite(y.values)):
        raise ValueError("Excess returns contain non-finite values.")
    if not np.all(np.isfinite(X.values)):
        raise ValueError("Factor returns contain non-finite values.")
    if float(y.var(ddof=1)) == 0.0 and len(y) > 1:
        raise ValueError("Excess returns are constant — regression undefined.")
    for col in X.columns:
        if len(X) > 1 and float(X[col].var(ddof=1)) == 0.0:
            raise ValueError(f"Factor {col!r} is constant — regression undefined.")
    check_min_observations(len(y), X.shape[1])
    if int(hac_lags) < 0:
        raise ValueError(f"HAC lag must be >= 0, got {hac_lags!r}.")

    try:
        import statsmodels.api as sm
        from statsmodels.stats.stattools import durbin_watson
    except ImportError as e:
        raise ValueError(
            "Factor regressions need the 'statsmodels' package."
        ) from e

    Xc = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y.values, Xc.values).fit()
    robust = sm.OLS(y.values, Xc.values).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(hac_lags), "use_correction": True},
    )
    params = pd.Series(robust.params, index=Xc.columns)
    ses = pd.Series(robust.bse, index=Xc.columns)
    tvals = pd.Series(robust.tvalues, index=Xc.columns)
    pvals = pd.Series(robust.pvalues, index=Xc.columns)

    factor_names = list(X.columns)
    resid = np.asarray(ols.resid, dtype=float)
    resid_vol = float(pd.Series(resid).std(ddof=1) * np.sqrt(TRADING_DAYS)) \
        if len(resid) >= 2 else 0.0
    return FactorRegressionResult(
        model_name=model_name,
        factor_names=factor_names,
        alpha=float(params["const"]),
        alpha_annualised=float((1.0 + float(params["const"])) ** TRADING_DAYS - 1.0),
        betas={f: float(params[f]) for f in factor_names},
        std_errors={f: float(ses[f]) for f in factor_names},
        t_stats={f: float(tvals[f]) for f in factor_names},
        p_values={f: float(pvals[f]) for f in factor_names},
        alpha_se=float(ses["const"]),
        alpha_t=float(tvals["const"]),
        alpha_p=float(pvals["const"]),
        r_squared=float(ols.rsquared),
        adj_r_squared=float(ols.rsquared_adj),
        observations=int(ols.nobs),
        residual_volatility=resid_vol,
        durbin_watson=float(durbin_watson(resid)),
        condition_number=float(np.linalg.cond(np.asarray(Xc, dtype=float))),
        hac_lags=int(hac_lags),
    )


def _select_factors(factors: pd.DataFrame, model: str) -> pd.DataFrame:
    needed = MODEL_FACTORS[model]
    missing = [c for c in needed if c not in factors.columns]
    if missing:
        raise ValueError(
            f"Factor data is missing {missing} required for "
            f"{MODEL_LABELS[model]}."
        )
    return factors[needed].copy()


def run_capm(y_excess: pd.Series, factors: pd.DataFrame,
             hac_lags: int = DEFAULT_HAC_LAGS) -> FactorRegressionResult:
    """CAPM: ``Rp − Rf = α + βm(Mkt − Rf) + ε`` (all simple returns)."""
    return run_factor_regression(y_excess, _select_factors(factors, "capm"),
                                 "capm", hac_lags)


def run_ff3(y_excess: pd.Series, factors: pd.DataFrame,
            hac_lags: int = DEFAULT_HAC_LAGS) -> FactorRegressionResult:
    """Fama-French 3-factor regression (simple returns)."""
    return run_factor_regression(y_excess, _select_factors(factors, "ff3"),
                                 "ff3", hac_lags)


def run_ff5(y_excess: pd.Series, factors: pd.DataFrame,
            hac_lags: int = DEFAULT_HAC_LAGS) -> FactorRegressionResult:
    """Fama-French 5-factor regression (simple returns)."""
    return run_factor_regression(y_excess, _select_factors(factors, "ff5"),
                                 "ff5", hac_lags)


def run_ff5_momentum(y_excess: pd.Series, factors: pd.DataFrame,
                     hac_lags: int = DEFAULT_HAC_LAGS) -> FactorRegressionResult:
    """Fama-French 5-factor + momentum regression (simple returns)."""
    return run_factor_regression(y_excess, _select_factors(factors, "ff5_mom"),
                                 "ff5_mom", hac_lags)


def run_model(model: str, y_excess: pd.Series, factors: pd.DataFrame,
              hac_lags: int = DEFAULT_HAC_LAGS) -> FactorRegressionResult:
    """Dispatch to the requested model implementation."""
    runners = {"capm": run_capm, "ff3": run_ff3, "ff5": run_ff5,
               "ff5_mom": run_ff5_momentum}
    if model not in runners:
        raise ValueError(
            f"Unknown model {model!r}; expected one of {sorted(runners)}."
        )
    return runners[model](y_excess, factors, hac_lags)


# ---------------------------------------------------------------------------
# Rolling exposures (trailing windows, plain OLS — no look-ahead)
# ---------------------------------------------------------------------------

def rolling_factor_regression(
    y_excess: pd.Series,
    X: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Trailing-window OLS betas (and alpha) ending at each date.

    Each row uses only the ``window`` observations ending on that date
    (``min_periods=window`` — no partial windows, no centering, no future
    data). Plain OLS via least squares (identical coefficients to the
    full-sample engine); HAC inference is a full-sample concern and is not
    repeated per window.
    """
    y = pd.Series(y_excess, dtype=float).dropna()
    X = pd.DataFrame(X, dtype=float).dropna(how="any")
    common = y.index.intersection(X.index)
    y, X = y.loc[common], X.loc[common]
    window = int(window)
    if window < 2:
        raise ValueError(f"Rolling window must be >= 2, got {window!r}.")
    if len(y) < window:
        raise ValueError(
            f"Only {len(y)} observations for a {window}-day rolling window."
        )
    if X.shape[1] == 0:
        raise ValueError("No factor columns supplied.")
    cols = ["alpha", *X.columns]
    out = pd.DataFrame(index=y.index, columns=cols, dtype=float)
    Xv = X.values
    yv = y.values
    ones = np.ones((window, 1))
    for end in range(window, len(y) + 1):
        sl = slice(end - window, end)  # trailing only: nothing after `end`
        design = np.hstack([ones, Xv[sl]])
        coefs, *_ = np.linalg.lstsq(design, yv[sl], rcond=None)
        out.iloc[end - 1] = coefs
    return out.dropna(how="any")


# ---------------------------------------------------------------------------
# Attribution (arithmetic, model-dependent)
# ---------------------------------------------------------------------------

def factor_attribution(
    result: FactorRegressionResult,
    y_excess: pd.Series,
    X: pd.DataFrame,
) -> dict[str, float]:
    """Annualised arithmetic attribution of fitted mean excess return.

    Component for factor ``f``: ``βf × mean(f) × 252``; intercept component:
    ``αd × 252`` (deliberately linear so components sum exactly to the
    fitted annualised mean, since OLS-with-intercept residuals average to
    zero). This decomposes the *fitted mean*, not compounded wealth, and is
    model-dependent — not an exact causal decomposition.
    """
    y = pd.Series(y_excess, dtype=float).dropna()
    X = pd.DataFrame(X, dtype=float)
    common = y.index.intersection(X.index)
    y, X = y.loc[common], X.loc[common].dropna(how="any")
    y = y.loc[X.index]
    if len(y) == 0:
        raise ValueError("No overlapping data for attribution.")
    missing = [f for f in result.factor_names if f not in X.columns]
    if missing:
        raise ValueError(f"Attribution is missing factors {missing}.")
    contrib = {f: float(result.betas[f] * X[f].mean() * TRADING_DAYS)
               for f in result.factor_names}
    contrib["Alpha"] = float(result.alpha * TRADING_DAYS)
    contrib["Fitted (linear, ann.)"] = float(sum(contrib.values()))
    contrib["Realised mean (ann.)"] = float(y.mean() * TRADING_DAYS)
    return contrib


# ---------------------------------------------------------------------------
# Interpretation helpers (descriptive, never prescriptive)
# ---------------------------------------------------------------------------

def describe_market_beta(beta: float) -> str:
    """Restrained historical description of a market beta."""
    if not np.isfinite(beta):
        raise ValueError(f"Beta must be finite, got {beta!r}.")
    if beta < 0.9:
        return "Historically showed lower market sensitivity than the broad market."
    if beta <= 1.1:
        return "Historically moved roughly in step with the broad market."
    return "Historically showed greater market sensitivity than the broad market."


def describe_factor_loading(factor: str, beta: float) -> str:
    """Restrained historical description of a style-factor loading."""
    if factor not in FACTOR_MEANINGS:
        raise ValueError(f"Unknown factor {factor!r}.")
    if not np.isfinite(beta):
        raise ValueError(f"Loading must be finite, got {beta!r}.")
    magnitude = "roughly neutral on" if abs(beta) < 0.10 else \
        ("modestly tilted" if abs(beta) <= 0.30 else "pronouncedly tilted")
    direction = {
        "Mkt-RF": ("toward" if beta >= 0 else "against") + " market direction",
        "SMB": "toward smaller caps" if beta >= 0 else "toward larger caps",
        "HML": "toward value" if beta >= 0 else "toward growth",
        "RMW": "toward robust profitability" if beta >= 0 else "toward weak profitability",
        "CMA": "toward conservative investment" if beta >= 0 else "toward aggressive investment",
        "Mom": "toward past winners" if beta >= 0 else "toward past losers",
    }[factor]
    if factor == "Mkt-RF":
        return f"Historically {magnitude} {direction}."
    return (f"Historically {magnitude} {direction} "
            f"({FACTOR_MEANINGS[factor]}).")


def format_p_value(p: float) -> str:
    """Compact p-value display: ``<0.001`` for very small values."""
    if not np.isfinite(p):
        return "n/a"
    if p < 0:
        raise ValueError(f"p-value must be >= 0, got {p!r}.")
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def annualise_alpha(alpha_daily: float) -> float:
    """Exact compounded annualisation: ``(1+αd)^252 − 1`` (beta untouched)."""
    if not np.isfinite(alpha_daily):
        raise ValueError(f"Daily alpha must be finite, got {alpha_daily!r}.")
    return float((1.0 + alpha_daily) ** TRADING_DAYS - 1.0)


def is_non_us_ticker(ticker: str) -> bool:
    """Heuristic non-US check: dotted suffix other than A/B class shares."""
    sym = str(ticker).strip().upper()
    if "." not in sym:
        return False
    return sym.rsplit(".", 1)[1] not in ("A", "B")


def us_scope_note(tickers: list[str]) -> str | None:
    """US-factor scope warning, or ``None`` when nothing looks non-US.

    French factors are US equity factors; they cannot be shown to describe
    non-US holdings. (US-listed foreign firms without a suffix are
    undetectable — noted in the methodology text, not pretended away.)
    """
    foreign = sorted({t for t in tickers if is_non_us_ticker(t)})
    if not foreign:
        return None
    return (
        "This factor model uses US Fama-French factors. Results for "
        f"portfolios containing non-US assets ({', '.join(foreign)}) may be "
        "economically less meaningful."
    )


# ---------------------------------------------------------------------------
# Plots (dark terminal theme)
# ---------------------------------------------------------------------------

def plot_factor_attribution(contributions: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of annualised arithmetic attribution."""
    items = [(k, v) for k, v in contributions.items()
             if k not in ("Fitted (linear, ann.)", "Realised mean (ann.)")]
    labels = [FACTOR_LABELS.get(k, k) if k != "Alpha"
              else "Alpha Contribution (Arithmetic)"
              for k, _ in items]
    values = [v for _, v in items]
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Historical Model Attribution (annualised, arithmetic)",
                   font=dict(size=16)),
        xaxis_title="Contribution (annualised)", xaxis_tickformat=".1%",
        showlegend=False,
    )
    return fig


def plot_rolling_exposure(dates, values: pd.Series, label: str) -> go.Figure:
    """Line chart of one rolling exposure series (trailing windows)."""
    fig = go.Figure(go.Scatter(
        x=dates, y=np.asarray(values, dtype=float),
        mode="lines", name=label,
        line=dict(color="#58a6ff", width=2),
    ))
    fig.add_hline(y=0, line=dict(color="#8b949e", dash="dot", width=1))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=f"Rolling Exposure — {label} (trailing window)",
                   font=dict(size=16)),
        xaxis_title="Date", yaxis_title="Loading",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig
