"""Pinterest-specific search engine implementation."""

import re
import time
from typing import List, Optional

from ddgs import DDGS

from .base import BaseSearchEngine, ImageResult, SearchPlatform


class PinterestSearchEngine(BaseSearchEngine):
    """Search engine for Pinterest images."""

    def __init__(self):
        super().__init__(SearchPlatform.PINTEREST)
        self.pinterest_domains = {
            "pinterest.com",
            "pinimg.com",
            "i.pinimg.com",
            "media.pinimg.com",
            "s-media-cache-ak0.pinimg.com"
        }

    def search(self, query: str, max_results: int = 20) -> List[ImageResult]:
        """Search Pinterest for tattoo images."""
        results = []

        pinterest_queries = [
            f"site:pinterest.com {query} tattoo",
            f"site:pinterest.com tattoo {query}",
        ]

        try:
            with DDGS() as ddgs:
                for i, pinterest_query in enumerate(pinterest_queries):
                    if i > 0:
                        time.sleep(2)  # Rate limiting

                    try:
                        search_results = ddgs.images(
                            pinterest_query,
                            region="wt-wt",
                            safesearch="off",
                            size="Medium",
                            max_results=max_results // 2
                        )

                        for result in search_results:
                            url = result.get("image")
                            if url and self.is_valid_url(url):
                                image_result = self._create_image_result(url, result)
                                results.append(image_result)

                        if len(results) >= max_results:
                            break

                    except Exception as e:
                        self.logger.warning(f"Pinterest query failed: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Pinterest search failed: {e}")

        return results[:max_results]

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is from Pinterest domains."""
        return any(domain in url.lower() for domain in self.pinterest_domains)

    def get_quality_score(self, url: str, **kwargs) -> float:
        """Calculate Pinterest-specific quality score."""
        score = super().get_quality_score(url)

        # Pinterest size indicators (higher resolution = higher score)
        size_patterns = {
            "/736x/": 1.0,
            "/564x/": 0.9,
            "/474x/": 0.8,
            "/236x/": 0.6
        }

        for pattern, bonus in size_patterns.items():
            if pattern in url:
                score = bonus
                break

        # Pinterest CDN reliability bonus
        if "i.pinimg.com" in url:
            score += 0.1

        return min(1.0, score)

    def _create_image_result(self, url: str, raw_result: dict) -> ImageResult:
        """Create ImageResult from raw Pinterest search result."""
        dimensions = self._extract_dimensions(url)

        return ImageResult(
            url=url,
            platform=self.platform,
            quality_score=self.get_quality_score(url),
            width=dimensions.get("width"),
            height=dimensions.get("height"),
            title=raw_result.get("title"),
            source_url=raw_result.get("source")
        )

    def _extract_dimensions(self, url: str) -> dict:
        """Extract image dimensions from Pinterest URL patterns."""
        # Pinterest URL pattern: .../236x/... or .../564x314/...
        size_match = re.search(r"/(\d+)x(\d*)/", url)
        if size_match:
            width = int(size_match.group(1))
            height = int(size_match.group(2)) if size_match.group(2) else None
            return {"width": width, "height": height}

        return {}