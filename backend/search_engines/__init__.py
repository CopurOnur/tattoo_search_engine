"""Search engines package for tattoo image discovery."""

from .base import BaseSearchEngine, ImageResult, SearchPlatform, SearchResult
from .pinterest import PinterestSearchEngine
from .manager import SearchEngineManager

__all__ = [
    "BaseSearchEngine",
    "ImageResult",
    "SearchPlatform",
    "SearchResult",
    "PinterestSearchEngine",
    "SearchEngineManager",
]