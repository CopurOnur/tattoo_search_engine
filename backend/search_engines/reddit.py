"""Reddit-specific search engine implementation."""

import time
from typing import List

from ddgs import DDGS

from .base import BaseSearchEngine, ImageResult, SearchPlatform


class RedditSearchEngine(BaseSearchEngine):
    """Search engine for Reddit images."""

    def __init__(self):
        super().__init__(SearchPlatform.REDDIT)
        self.reddit_domains = {
            "reddit.com",
            "i.redd.it",
            "i.imgur.com",
            "imgur.com"
        }
        self.tattoo_subreddits = [
            "tattoos",
            "tattoo",
            "traditionaltattoos",
            "blackwork",
            "sticknpokes",
            "tattoodesigns"
        ]

    def search(self, query: str, max_results: int = 20) -> List[ImageResult]:
        """Search Reddit for tattoo images."""
        results = []

        # Create Reddit-specific queries
        reddit_queries = self._build_reddit_queries(query)

        try:
            with DDGS() as ddgs:
                for i, reddit_query in enumerate(reddit_queries):
                    if i > 0:
                        time.sleep(1.5)  # Rate limiting

                    try:
                        search_results = ddgs.images(
                            reddit_query,
                            region="wt-wt",
                            safesearch="off",
                            size="Medium",
                            max_results=max_results // len(reddit_queries)
                        )

                        for result in search_results:
                            url = result.get("image")
                            if url and self.is_valid_url(url):
                                image_result = self._create_image_result(url, result)
                                results.append(image_result)

                        if len(results) >= max_results:
                            break

                    except Exception as e:
                        self.logger.warning(f"Reddit query failed: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Reddit search failed: {e}")

        return results[:max_results]

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is from Reddit or Reddit-linked domains."""
        return any(domain in url.lower() for domain in self.reddit_domains)

    def get_quality_score(self, url: str, **kwargs) -> float:
        """Calculate Reddit-specific quality score."""
        score = super().get_quality_score(url)

        # i.redd.it is Reddit's native image host (reliable)
        if "i.redd.it" in url:
            score += 0.2

        # Imgur is commonly used and reliable
        elif "imgur.com" in url:
            score += 0.1

        # Reddit posts tend to be higher quality
        if "reddit.com" in url:
            score += 0.1

        return min(1.0, score)

    def _build_reddit_queries(self, query: str) -> List[str]:
        """Build Reddit-specific search queries."""
        queries = []

        # General Reddit search
        queries.append(f"site:reddit.com {query} tattoo")

        # Subreddit-specific searches
        for subreddit in self.tattoo_subreddits[:3]:  # Limit to top 3 subreddits
            queries.append(f"site:reddit.com/r/{subreddit} {query}")

        return queries

    def _create_image_result(self, url: str, raw_result: dict) -> ImageResult:
        """Create ImageResult from raw Reddit search result."""
        return ImageResult(
            url=url,
            platform=self.platform,
            quality_score=self.get_quality_score(url),
            title=raw_result.get("title"),
            source_url=raw_result.get("source")
        )