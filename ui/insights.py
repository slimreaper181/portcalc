"""insights — deterministic portfolio observations (no LLM, no advice).

Rules map a facts dict to at most MAX_INSIGHTS short observations ordered
by severity. Thresholds are centralised below. Output language is strictly
descriptive analytics — never buy/sell/hold/recommendation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INSIGHTS = 5

THRESHOLDS = {
    "single_name_warn": 0.25,      # largest position above → watch
    "top3_warn": 0.60,             # top-3 share above → watch
    "rc_concentration": 1.8,       # max RC share ÷ its weight → watch
    "corr_high": 0.85,             # pairwise correlation above → notable
    "beta_high": 1.25,             # benchmark beta above → notable
    "beta_low": 0.75,              # benchmark beta below → info
    "drawdown_watch": 0.10,        # max drawdown magnitude → watch
    "fx_concentration": 0.70,      # top quotation currency above → info
    "effective_min": 3.0,          # effective holdings below → watch
}

_ADVICE_RE = re.compile(
    r"\b(buy|buys|sell|sells|hold|holds|recommend\w*|should|dump|bet on|must own)\b")


def _assert_no_advice(text: str) -> None:
    assert not _ADVICE_RE.search(text.lower()), text


@dataclass
class Insight:
    """One observation: severity in info/watch/notable."""

    severity: str
    category: str
    text: str


def _pct(x: float) -> str:
    return f"{float(x):.1%}"


def build_insights(facts: dict) -> list:
    """Build ordered observations from a facts dict (pure)."""
    f = dict(facts or {})
    out: list = []

    def add(sev, cat, text):
        _assert_no_advice(text)
        out.append(Insight(sev, cat, text))

    n = f.get("n_assets")
    lw, lt = f.get("largest_weight"), f.get("largest_ticker")
    if lw is not None and float(lw) >= THRESHOLDS["single_name_warn"] and n and n > 1:
        add("watch", "concentration",
            f"{lt} represents {_pct(lw)} of capital — a single-name "
            f"concentration above the {_pct(THRESHOLDS['single_name_warn'])} "
            "review level.")
    t3 = f.get("top3_weight")
    if t3 is not None and float(t3) >= THRESHOLDS["top3_warn"] and n and n >= 3:
        add("watch", "concentration",
            f"Top three holdings account for {_pct(t3)} of portfolio value.")
    rc_t, rc_s, rc_w = (f.get("max_rc_ticker"), f.get("max_rc_share"),
                        f.get("max_rc_weight"))
    if rc_s is not None and rc_w not in (None, 0):
        ratio = float(rc_s) / float(rc_w)
        if ratio >= THRESHOLDS["rc_concentration"]:
            add("watch", "risk concentration",
                f"{rc_t} contributes {_pct(rc_s)} of portfolio volatility "
                f"risk from {_pct(rc_w)} of capital "
                f"(risk-to-weight {ratio:.1f}x).")
    cp, cv = f.get("max_corr_pair"), f.get("max_corr_value")
    if cv is not None and float(cv) >= THRESHOLDS["corr_high"]:
        add("notable", "correlation",
            f"Highest pairwise correlation is {float(cv):.2f} ({cp}) — "
            "these names have moved closely together.")
    beta, bench = f.get("beta"), f.get("benchmark")
    if beta is not None and bench:
        b = float(beta)
        if b >= THRESHOLDS["beta_high"]:
            add("notable", "market sensitivity",
                f"Portfolio beta is {b:.2f} versus {bench} — moves have been "
                "amplified relative to the benchmark.")
        elif b <= THRESHOLDS["beta_low"]:
            add("info", "market sensitivity",
                f"Portfolio beta is {b:.2f} versus {bench} — moves have been "
                "muted relative to the benchmark.")
    mdd = f.get("max_drawdown")
    if mdd is not None and float(mdd) <= -THRESHOLDS["drawdown_watch"]:
        add("watch", "drawdown",
            f"Maximum drawdown is {_pct(mdd)} — the historical peak-to-trough "
            "decline on the analysed window.")
    fx_c, fx_s = f.get("fx_top_ccy"), f.get("fx_top_share")
    if fx_s is not None and float(fx_s) >= THRESHOLDS["fx_concentration"]:
        add("info", "currency",
            f"{fx_c}-quoted holdings represent {_pct(fx_s)} of current "
            "market value (quotation exposure, not revenue exposure).")
    eff = f.get("effective_holdings")
    if eff is not None and n and n > 1 and float(eff) < min(
            THRESHOLDS["effective_min"], float(n)):
        add("watch", "diversification",
            f"Effective holdings: {float(eff):.1f} on {int(n)} names — "
            "capital is concentrated in a few positions.")
    sev_rank = {"notable": 0, "watch": 1, "info": 2}
    out.sort(key=lambda i: sev_rank.get(i.severity, 3))
    return out[:MAX_INSIGHTS]
