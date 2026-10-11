"""ui — presentation layer for Portfolio Calc 2.0.

Rules:
* No numerical logic here: analytics live in ``analytics/``, fetching in
  ``data/``. UI modules only format, arrange and display values computed
  elsewhere (plus deterministic display thresholds in ``insights``).
* No circular imports: ``ui`` may import ``analytics``/``data``/``components``,
  never the reverse, and ``ui`` modules must not import the app shell.
* Every helper is either pure (unit-tested) or a thin Streamlit renderer.
"""

__version__ = "2.0.0"
