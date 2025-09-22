"""Base classes for image search engines."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SearchPlatform(Enum):
    """Supported search platforms."""
    PINTEREST = "pinterest"
    INSTAGRAM = "instagram"
    REDDIT = "reddit"
    FLICKR = "flickr"
    DEVIANTART = "deviantart"
    GENERAL = "general"


@dataclass
class ImageResult:
    """Represents a single image search result."""
    url: str
    platform: SearchPlatform
    quality_score: float = 0.0
    width: Optional[int] = None
    height: Optional[int] = None
    title: Optional[str] = None
    source_url: Optional[str] = None

    @property
    def resolution_score(self) -> float:
        """Calculate score based on image resolution."""
        if not self.width or not self.height:
            return 0.5

        total_pixels = self.width * self.height
        if total_pixels >= 1000000:  # 1MP+
            return 1.0
        elif total_pixels >= 500000:  # 0.5MP+
            return 0.8
        elif total_pixels >= 250000:  # 0.25MP+
            return 0.6
        else:
            return 0.3


@dataclass
class SearchResult:
    """Container for all search results from multiple platforms."""
    images: List[ImageResult]
    total_found: int
    platforms_used: Set[SearchPlatform]
    search_duration: float

    def get_top_results(self, limit: int = 50) -> List[ImageResult]:
        """Get top results sorted by quality score."""
        sorted_images = sorted(self.images, key=lambda x: x.quality_score, reverse=True)
        return sorted_images[:limit]


class BaseSearchEngine(ABC):
    """Abstract base class for image search engines."""

    def __init__(self, platform: SearchPlatform):
        self.platform = platform
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> List[ImageResult]:
        """Search for images on the platform."""
        pass

    @abstractmethod
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for this platform."""
        pass

    def get_quality_score(self, url: str, **kwargs) -> float:
        """Calculate quality score for a URL (0.0 to 1.0)."""
        score = 0.5  # Base score

        # URL length penalty (very long URLs often broken)
        if len(url) > 500:
            score -= 0.2
        elif len(url) > 300:
            score -= 0.1

        # Image extension bonus
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if any(ext in url.lower() for ext in image_extensions):
            score += 0.1

        return max(0.0, min(1.0, score))