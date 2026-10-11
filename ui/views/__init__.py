"""views — one render module per research tab (called by the app shell).

Each module exposes ``render(ctx)`` where ``ctx`` is the shared shell
context dict (tickers, panels, weights, currency layer, ...). The Overview
renderer additionally returns an updated-state dict (history-period rebuild)
that the shell merges back into ``ctx`` for later tabs.
"""
