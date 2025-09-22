"""Simple in-memory caching for search results."""

import hashlib
import time
from typing import Any, Dict, Optional, Tuple


class SearchCache:
    """Simple in-memory cache for search results."""

    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds (1 hour)
            max_size: Maximum number of cached items
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        value, expiry_time = self._cache[key]

        # Check if expired
        if time.time() > expiry_time:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        if ttl is None:
            ttl = self.default_ttl

        expiry_time = time.time() + ttl

        # Evict oldest entries if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_expired()

            # If still full, remove oldest entry
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

        self._cache[key] = (value, expiry_time)

    def delete(self, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()

    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry_time) in self._cache.items()
            if current_time > expiry_time
        ]

        for key in expired_keys:
            del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for _, expiry_time in self._cache.values()
            if current_time > expiry_time
        )

        return {
            'total_items': len(self._cache),
            'expired_items': expired_count,
            'active_items': len(self._cache) - expired_count,
            'max_size': self.max_size,
            'default_ttl': self.default_ttl
        }

    @staticmethod
    def create_cache_key(query: str, max_results: int, platforms: Optional[set] = None) -> str:
        """Create a cache key from search parameters."""
        # Normalize query
        normalized_query = query.lower().strip()

        # Create a string representation of platforms
        platform_str = ''
        if platforms:
            platform_str = '_'.join(sorted(p.value for p in platforms))

        # Combine all parameters
        key_string = f"{normalized_query}_{max_results}_{platform_str}"

        # Hash to create a fixed-length key
        return hashlib.md5(key_string.encode()).hexdigest()