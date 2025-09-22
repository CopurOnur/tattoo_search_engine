"""Search engine manager for coordinating multi-platform searches."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set

from .base import BaseSearchEngine, ImageResult, SearchPlatform, SearchResult
from .instagram import InstagramSearchEngine
from .pinterest import PinterestSearchEngine
from .reddit import RedditSearchEngine


class SearchEngineManager:
    """Manages and coordinates searches across multiple platforms."""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.engines: Dict[SearchPlatform, BaseSearchEngine] = {
            SearchPlatform.PINTEREST: PinterestSearchEngine(),
            SearchPlatform.REDDIT: RedditSearchEngine(),
            SearchPlatform.INSTAGRAM: InstagramSearchEngine(),
        }

    def search_all_platforms(
        self,
        query: str,
        max_results_per_platform: int = 20,
        platforms: Optional[Set[SearchPlatform]] = None
    ) -> SearchResult:
        """Search across multiple platforms concurrently."""
        start_time = time.time()

        if platforms is None:
            platforms = set(self.engines.keys())

        all_results = []
        platforms_used = set()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit search tasks for each platform
            future_to_platform = {
                executor.submit(
                    self._search_single_platform,
                    platform,
                    query,
                    max_results_per_platform
                ): platform
                for platform in platforms
                if platform in self.engines
            }

            # Collect results as they complete
            for future in as_completed(future_to_platform):
                platform = future_to_platform[future]
                try:
                    platform_results = future.result(timeout=30)  # 30s timeout per platform
                    if platform_results:
                        all_results.extend(platform_results)
                        platforms_used.add(platform)
                except Exception as e:
                    print(f"Platform {platform.value} search failed: {e}")

        # Remove duplicates and sort by quality
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.quality_score, reverse=True)

        search_duration = time.time() - start_time

        return SearchResult(
            images=sorted_results,
            total_found=len(sorted_results),
            platforms_used=platforms_used,
            search_duration=search_duration
        )

    def search_with_fallback(
        self,
        query: str,
        max_results: int = 50,
        min_results_threshold: int = 10
    ) -> SearchResult:
        """Search with intelligent fallback strategies."""
        # Try primary platforms first
        primary_platforms = {SearchPlatform.PINTEREST, SearchPlatform.REDDIT}
        result = self.search_all_platforms(
            query,
            max_results_per_platform=max_results // 2,
            platforms=primary_platforms
        )

        # If we don't have enough results, try additional platforms
        if len(result.images) < min_results_threshold:
            additional_platforms = {SearchPlatform.INSTAGRAM}
            additional_result = self.search_all_platforms(
                query,
                max_results_per_platform=max_results // 2,
                platforms=additional_platforms
            )

            # Merge results
            all_images = result.images + additional_result.images
            unique_images = self._deduplicate_results(all_images)
            sorted_images = sorted(unique_images, key=lambda x: x.quality_score, reverse=True)

            result = SearchResult(
                images=sorted_images,
                total_found=len(sorted_images),
                platforms_used=result.platforms_used | additional_result.platforms_used,
                search_duration=result.search_duration + additional_result.search_duration
            )

        # If still not enough, try simplified queries
        if len(result.images) < min_results_threshold:
            simplified_query = self._simplify_query(query)
            if simplified_query != query:
                fallback_result = self.search_all_platforms(
                    simplified_query,
                    max_results_per_platform=max_results // 3
                )

                # Merge with existing results
                all_images = result.images + fallback_result.images
                unique_images = self._deduplicate_results(all_images)
                sorted_images = sorted(unique_images, key=lambda x: x.quality_score, reverse=True)

                result = SearchResult(
                    images=sorted_images,
                    total_found=len(sorted_images),
                    platforms_used=result.platforms_used | fallback_result.platforms_used,
                    search_duration=result.search_duration + fallback_result.search_duration
                )

        return result

    def _search_single_platform(
        self,
        platform: SearchPlatform,
        query: str,
        max_results: int
    ) -> List[ImageResult]:
        """Search a single platform (thread-safe)."""
        engine = self.engines.get(platform)
        if not engine:
            return []

        try:
            return engine.search(query, max_results)
        except Exception as e:
            print(f"Error searching {platform.value}: {e}")
            return []

    def _deduplicate_results(self, results: List[ImageResult]) -> List[ImageResult]:
        """Remove duplicate URLs while preserving the highest quality version."""
        seen_urls = {}

        for result in results:
            if result.url in seen_urls:
                # Keep the result with higher quality score
                if result.quality_score > seen_urls[result.url].quality_score:
                    seen_urls[result.url] = result
            else:
                seen_urls[result.url] = result

        return list(seen_urls.values())

    def _simplify_query(self, query: str) -> str:
        """Simplify query by removing complex terms and keeping core concepts."""
        # Remove adjectives and keep main nouns
        words = query.split()

        # Common tattoo-related keywords to keep
        core_keywords = {
            'tattoo', 'design', 'art', 'ink', 'traditional', 'realistic', 'geometric',
            'tribal', 'watercolor', 'minimalist', 'blackwork', 'dotwork',
            'dragon', 'flower', 'skull', 'rose', 'bird', 'lion', 'butterfly'
        }

        # Keep important words and first few words
        simplified_words = []
        for i, word in enumerate(words):
            if i < 3 or word.lower() in core_keywords:
                simplified_words.append(word)

        simplified = ' '.join(simplified_words)
        return simplified if simplified else 'tattoo art'

    def get_platform_stats(self) -> Dict[str, Dict]:
        """Get statistics about available platforms."""
        stats = {}
        for platform, engine in self.engines.items():
            stats[platform.value] = {
                'available': True,
                'class': engine.__class__.__name__
            }
        return stats