"""Audit regression: no function-local binding may shadow a module-level
import inside ui/views (UnboundLocalError class of bug).

Covers the release-audit P0 where ``ui/views/backtest.py::render`` held a
function-local ``from analytics.backtest import backtest_rebalanced`` while
the same name was already imported at module top level: every earlier use
in the function then raised ``UnboundLocalError`` and the whole Historical
backtest tab rendered an error card.
"""

import ast
import pathlib

VIEW = (pathlib.Path(__file__).resolve().parent.parent
        / "ui" / "views" / "backtest.py")


def _shadowed(tree: ast.Module) -> list[str]:
    top: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.update(a.asname or a.name for a in node.names)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                bound: set[str] = set()
                if isinstance(sub, ast.Import):
                    bound.update(a.asname or a.name.split(".")[0]
                                 for a in sub.names)
                elif isinstance(sub, ast.ImportFrom):
                    bound.update(a.asname or a.name for a in sub.names)
                elif isinstance(sub, (ast.Assign, ast.AnnAssign,
                                      ast.NamedExpr)):
                    targets = ([sub.target] if isinstance(
                        sub, (ast.AnnAssign, ast.NamedExpr)) else sub.targets)
                    for t in targets:
                        for n in ast.walk(t):
                            if isinstance(n, ast.Name) and isinstance(
                                    n.ctx, ast.Store):
                                bound.add(n.id)
                clash = (bound & top) - {node.name}
                if clash:
                    bad.append(f"{node.name}:{sub.lineno} shadows "
                               f"{sorted(clash)}")
    return bad


def test_backtest_view_has_no_shadowed_engine_names():
    tree = ast.parse(VIEW.read_text(encoding="utf-8"))
    assert _shadowed(tree) == []


def test_backtest_rebalanced_resolves_at_module_level():
    import ui.views.backtest as bt
    assert callable(bt.backtest_rebalanced)
