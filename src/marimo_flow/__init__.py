"""marimo-flow shared utilities."""

from importlib.metadata import version

try:
    __version__ = version("marimo-flow")
except Exception:  # pragma: no cover - fallback during local dev
    __version__ = "0.0.0"


__all__ = ["__version__"]

