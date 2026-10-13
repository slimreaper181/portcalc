"""
factors.py (data layer)
-----------------------
Kenneth R. French Data Library retrieval — network I/O only.

This module downloads the official daily US factor files and returns them
unmodified (raw **percent** units, e.g. ``0.42`` means 0.42%). Conversion
to decimal returns lives in ``analytics.factors`` where it is unit-tested.
No Streamlit, no analytics and no scraping of rendered HTML here — only
the stable downloadable datasets via ``pandas-datareader``.
"""

from __future__ import annotations

import pandas as pd

# Stable downloadable dataset names in the French library.
FRENCH_DATASETS: dict[str, str] = {
    "ff3": "F-F_Research_Data_Factors_daily",       # Mkt-RF, SMB, HML, RF
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily",  # Mkt-RF, SMB, HML, RMW, CMA, RF
    "mom": "F-F_Momentum_Factor_daily",              # Mom
}

EXPECTED_COLUMNS: dict[str, list[str]] = {
    "ff3": ["Mkt-RF", "SMB", "HML", "RF"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    "mom": ["Mom"],
}


def _parse_dataset(key: str, dataset: str, payload: dict) -> pd.DataFrame:
    """Extract and validate the return table from a reader response."""
    frame = None
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, pd.DataFrame):
                frame = value
                break
    if frame is None or frame.empty:
        raise ValueError(
            f"French dataset {dataset!r} returned no usable return table."
        )
    frame = frame.copy()
    try:
        if isinstance(frame.index, pd.PeriodIndex):
            # The 3-factor daily file arrives with a Period index.
            frame.index = frame.index.to_timestamp()
        else:
            frame.index = pd.DatetimeIndex(frame.index)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"French dataset {dataset!r} has an unparseable date index: {e}"
        )
    frame = frame.sort_index()
    missing = [c for c in EXPECTED_COLUMNS[key] if c not in frame.columns]
    if missing:
        raise ValueError(
            f"French dataset {dataset!r} is missing columns {missing}; "
            f"got {list(frame.columns)}."
        )
    frame = frame[EXPECTED_COLUMNS[key]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError(f"French dataset {dataset!r} has no valid observations.")
    return frame


def fetch_french_factors() -> dict[str, pd.DataFrame]:
    """Download daily US Fama-French factor data (raw percent units).

    Returns:
        Dict with keys ``"ff3"``, ``"ff5"``, ``"mom"`` mapping to DataFrames
        of daily factor values in **percent** (``0.42`` = 0.42%), indexed by
        date. Callers must convert via
        ``analytics.factors.to_decimal_returns`` before any regression.

    Raises:
        ValueError: on network failure or malformed/unexpected responses.
            No raw tracebacks leak — messages are UI-ready.
    """
    try:
        from pandas_datareader.famafrench import FamaFrenchReader
    except ImportError as e:
        raise ValueError(
            "Factor analysis needs the 'pandas-datareader' package, which "
            "is not installed."
        ) from e

    out: dict[str, pd.DataFrame] = {}
    for key, dataset in FRENCH_DATASETS.items():
        try:
            # Explicit full-history start: without it the reader silently
            # defaults to a ~5-year window, which would truncate long
            # backtests and rolling analyses.
            payload = FamaFrenchReader(dataset, start="1960-01-01").read()
        except Exception as e:
            raise ValueError(
                f"Could not download French dataset {dataset!r} "
                f"(network or library error: {e}). Check your connection "
                "and try again later."
            ) from e
        try:
            out[key] = _parse_dataset(key, dataset, payload)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"French dataset {dataset!r} arrived malformed: {e}"
            ) from e
    return out
