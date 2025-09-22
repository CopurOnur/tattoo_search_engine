"""Instagram-specific search engine implementation."""

import time
from typing import List

from ddgs import DDGS

from .base import BaseSearchEngine, ImageResult, SearchPlatform


class InstagramSearchEngine(BaseSearchEngine):
    """Search engine for Instagram images."""

    def __init__(self):
        super().__init__(SearchPlatform.INSTAGRAM)
        self.instagram_domains = {
            "instagram.com",
            "cdninstagram.com",
            "scontent.cdninstagram.com",
            "scontent-",  # Instagram CDN prefix
        }

    def search(self, query: str, max_results: int = 20) -> List[ImageResult]:
        """Search Instagram for tattoo images."""
        results = []

        # Instagram hashtag-based queries
        instagram_queries = self._build_instagram_queries(query)

        try:
            with DDGS() as ddgs:
                for i, instagram_query in enumerate(instagram_queries):
                    if i > 0:
                        time.sleep(2)  # Instagram is more sensitive to rate limiting

                    try:
                        search_results = ddgs.images(
                            instagram_query,
                            region="wt-wt",
                            safesearch="off",
                            size="Medium",
                            max_results=max_results // len(instagram_queries)
                        )

                        for result in search_results:
                            url = result.get("image")
                            if url and self.is_valid_url(url):
                                image_result = self._create_image_result(url, result)
                                results.append(image_result)

                        if len(results) >= max_results:
                            break

                    except Exception as e:
                        self.logger.warning(f"Instagram query failed: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Instagram search failed: {e}")

        return results[:max_results]

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is from Instagram domains."""
        return any(domain in url.lower() for domain in self.instagram_domains)

    def get_quality_score(self, url: str, **kwargs) -> float:
        """Calculate Instagram-specific quality score."""
        score = super().get_quality_score(url)

        # Instagram CDN URLs are generally reliable
        if "cdninstagram.com" in url or "scontent" in url:
            score += 0.15

        # Instagram posts tend to be high quality
        if "instagram.com/p/" in url:
            score += 0.1

        return min(1.0, score)

    def _build_instagram_queries(self, query: str) -> List[str]:
        """Build Instagram-specific search queries."""
        queries = []

        # General Instagram search
        queries.append(f"site:instagram.com {query} tattoo")

        # Hashtag-focused searches
        hashtag_queries = [
            f"site:instagram.com #{query.replace(' ', '')}tattoo",
            f"site:instagram.com #tattoo {query}",
        ]

        queries.extend(hashtag_queries)
        return queries

    def _create_image_result(self, url: str, raw_result: dict) -> ImageResult:
        """Create ImageResult from raw Instagram search result."""
        return ImageResult(
            url=url,
            platform=self.platform,
            quality_score=self.get_quality_score(url),
            title=raw_result.get("title"),
            source_url=raw_result.get("source")
        )