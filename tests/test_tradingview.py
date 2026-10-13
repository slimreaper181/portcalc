"""Tests for components.tradingview (resolver + safe widget HTML)."""

import json
import re

import pytest

from components.tradingview import (
    ResolvedSymbol,
    resolve_tradingview_symbol,
    tradingview_widget_html,
    validate_tradingview_symbol,
)


def test_known_us_symbols():
    assert resolve_tradingview_symbol("AAPL").symbol == "NASDAQ:AAPL"
    assert resolve_tradingview_symbol("msft").symbol == "NASDAQ:MSFT"
    assert resolve_tradingview_symbol("NVDA").symbol == "NASDAQ:NVDA"
    assert resolve_tradingview_symbol("JPM").symbol == "NYSE:JPM"
    assert resolve_tradingview_symbol("GOOGL").symbol == "NASDAQ:GOOGL"
    r = resolve_tradingview_symbol("AAPL")
    assert isinstance(r, ResolvedSymbol) and r.resolved is True


def test_lse_suffix_handling():
    assert resolve_tradingview_symbol("BARC.L").symbol == "LSE:BARC"
    assert resolve_tradingview_symbol("shel.l").symbol == "LSE:SHEL"
    assert resolve_tradingview_symbol("BARC.L").resolved is True


def test_index_symbols():
    assert resolve_tradingview_symbol("^GSPC").symbol == "SP:SPX"
    assert resolve_tradingview_symbol("^FTSE").symbol == "TVC:UKX"


def test_exchange_hint_and_fallback():
    r = resolve_tradingview_symbol("ZZZZ", exchange_hint="NMS")
    assert r.symbol == "NASDAQ:ZZZZ" and r.resolved is True
    # Unknown hint code → graceful bare fallback, never a crash.
    r2 = resolve_tradingview_symbol("ZZZZ", exchange_hint="BOGUS")
    assert r2.symbol == "ZZZZ" and r2.resolved is False and r2.note
    r3 = resolve_tradingview_symbol("ZZZZ")
    assert r3.symbol == "ZZZZ" and r3.resolved is False


def test_override_and_invalid_inputs():
    r = resolve_tradingview_symbol("^FTSE", override="TVC:UKX")
    assert r.symbol == "TVC:UKX" and r.resolved is True
    with pytest.raises(ValueError):
        resolve_tradingview_symbol("NOT A TICKER!!!")
    with pytest.raises(ValueError):
        resolve_tradingview_symbol("AAPL", override="bad symbol!!")
    with pytest.raises(ValueError):
        validate_tradingview_symbol("")


def test_widget_html_embeds_symbol_and_watchlist():
    html = tradingview_widget_html(
        "NASDAQ:AAPL", ["NASDAQ:MSFT", "NYSE:JPM", "NASDAQ:AAPL"])
    assert "NASDAQ:AAPL" in html
    assert "NASDAQ:MSFT" in html and "NYSE:JPM" in html
    assert "embed-widget-advanced-chart.js" in html
    assert '"theme":"dark"' in html and '"interval":"D"' in html
    assert '"style":"1"' in html  # candlesticks
    # Watchlist deduplicates the main symbol.
    m = re.search(r'"watchlist":(\[.*?\])', html)
    assert json.loads(m.group(1)).count("NASDAQ:AAPL") == 1
    with pytest.raises(ValueError):
        tradingview_widget_html("not a symbol!!")
    with pytest.raises(ValueError):
        tradingview_widget_html("NASDAQ:AAPL", ["evil\";x"])


def test_malicious_input_cannot_escape_config():
    evil = 'AAPL";alert(1);//'
    with pytest.raises(ValueError):
        resolve_tradingview_symbol(evil)
    with pytest.raises(ValueError):
        validate_tradingview_symbol('NASDAQ:AAPL</script><script>alert(1)</script>')
    # Even hostile-but-valid-charset strings stay inside the JSON payload:
    # exactly one script open tag and one close tag in the whole document.
    html = tradingview_widget_html("NASDAQ:AAPL", ["NYSE:JPM", "LSE:BARC"])
    assert html.count("<script") == 1 and html.count("</script>") == 1
    payload = html.split("async>", 1)[1].rsplit("</script>", 1)[0]
    assert "</script>" not in payload and "<script" not in payload
    # Payload is still valid JSON decoding to the requested config.
    assert json.loads(payload)["symbol"] == "NASDAQ:AAPL"


def test_chart_height_defaults_large():
    html = tradingview_widget_html("NASDAQ:AAPL")
    # Large default: 900px of actual chart + 32px attribution container.
    assert "height:900px" in html  # widget child = real chart height
    assert "height:932px" in html  # outer container incl. attribution
    assert tradingview_widget_html("NASDAQ:AAPL", height=650).count(
        "height:650px") == 1
    assert tradingview_widget_html("NASDAQ:AAPL", height=650).count(
        "height:682px") == 1
    with pytest.raises(ValueError):
        tradingview_widget_html("NASDAQ:AAPL", height=200)


def test_chart_height_propagates_through_embed_chain():
    # Every level of the embed must carry an explicit height so the
    # rendered candlestick area (not just the iframe) hits the target.
    html = tradingview_widget_html("NASDAQ:AAPL", height=900)
    assert "html,body" in html  # root height/margin reset present
    # Official widget target class with explicit pixel height.
    m = re.search(
        r'<div class="tradingview-widget-container__widget" style="([^"]*)"',
        html)
    assert m is not None
    assert "height:900px" in m.group(1) and "width:100%" in m.group(1)
    assert "calc(100%" not in m.group(1)  # no percentage-height reliance
    # Outer container carries chart + attribution explicitly.
    m2 = re.search(
        r'<div class="tradingview-widget-container" style="([^"]*)"', html)
    assert m2 is not None
    assert "height:932px" in m2.group(1) and "width:100%" in m2.group(1)
    # Attribution line kept (required) with its own explicit height.
    assert "tradingview-widget-copyright" in html
    # Config carries explicit dimensions alongside autosize.
    payload = html.split("async>", 1)[1].rsplit("</script>", 1)[0]
    config = json.loads(payload)
    assert config["height"] == 900 and config["width"] == "100%"


def test_chart_full_width_autosize_config():
    html = tradingview_widget_html("NASDAQ:AAPL", ["NYSE:JPM"])
    payload = html.split("async>", 1)[1].rsplit("</script>", 1)[0]
    config = json.loads(payload)
    assert config["autosize"] is True
    assert "width:100%" in html  # fluid container, no fixed pixel width
    assert "width:680" not in html and "width:900px" not in html
    # Chart controls preserved.
    assert config["theme"] == "dark" and config["style"] == "1"
    assert config["interval"] == "D" and config["timezone"] == "exchange"
    assert config["withdateranges"] is True
    assert config["hide_side_toolbar"] is False
    assert config["allow_symbol_change"] is True
    assert config["watchlist"][0] == "NASDAQ:AAPL"
    assert "support_host" in config  # TradingView attribution intact
