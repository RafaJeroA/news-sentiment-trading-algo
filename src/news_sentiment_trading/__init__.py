"""News Sentiment Trading Research package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("news-sentiment-trading-algo")
except PackageNotFoundError:  # pragma: no cover - source-tree import
    __version__ = "0+unknown"

__all__ = ["__version__"]
