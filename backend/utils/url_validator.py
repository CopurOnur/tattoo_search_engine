"""URL validation and health checking utilities."""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class URLValidator:
    """Validates and health-checks URLs before processing."""

    def __init__(self, max_workers: int = 10, timeout: int = 10):
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()

        # Blocked domains that consistently fail or are problematic
        self.blocked_domains = {
            'bodyartguru.com',
            'dcassetcdn.com',
            'warvox.com',
            'jenkins-tpp.blackboard.com',
            'wrdsclassroom.wharton.upenn.edu',
        }

        # User agents for health checks
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        ]

    def validate_urls(self, urls: List[str]) -> List[str]:
        """Validate multiple URLs concurrently."""
        if not urls:
            return []

        # First, filter out obviously bad URLs
        pre_filtered = self._pre_filter_urls(urls)

        if not pre_filtered:
            return []

        # Health check the remaining URLs
        valid_urls = self._health_check_urls(pre_filtered)

        logger.info(f"URL validation: {len(urls)} -> {len(pre_filtered)} -> {len(valid_urls)}")
        return valid_urls

    def _pre_filter_urls(self, urls: List[str]) -> List[str]:
        """Pre-filter URLs based on basic criteria."""
        filtered = []

        for url in urls:
            if not self._is_valid_url_format(url):
                continue

            if self._is_blocked_domain(url):
                continue

            if not self._has_image_extension(url):
                continue

            if len(url) > 500:  # Skip very long URLs
                continue

            filtered.append(url)

        return filtered

    def _health_check_urls(self, urls: List[str]) -> List[str]:
        """Perform HEAD requests to check URL accessibility."""
        valid_urls = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit health check tasks
            future_to_url = {
                executor.submit(self._check_single_url, url): url
                for url in urls
            }

            # Collect results
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    is_valid = future.result(timeout=self.timeout + 5)
                    if is_valid:
                        valid_urls.append(url)
                except Exception as e:
                    logger.debug(f"Health check failed for {url}: {e}")

                # Small delay to be respectful
                time.sleep(0.1)

        return valid_urls

    def _check_single_url(self, url: str) -> bool:
        """Check if a single URL is accessible."""
        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
                'DNT': '1',
            }

            # Add platform-specific headers
            if 'pinterest' in url.lower():
                headers.update({
                    'Referer': 'https://www.pinterest.com/',
                    'Origin': 'https://www.pinterest.com',
                })
            elif 'instagram' in url.lower():
                headers.update({
                    'Referer': 'https://www.instagram.com/',
                })
            else:
                headers['Referer'] = 'https://www.google.com/'

            response = self.session.head(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True
            )

            # Check status code
            if response.status_code not in [200, 301, 302]:
                return False

            # Check content type if available
            content_type = response.headers.get('content-type', '').lower()
            if content_type and not content_type.startswith('image/'):
                return False

            # Check content length if available
            content_length = response.headers.get('content-length')
            if content_length:
                size = int(content_length)
                if size < 1024 or size > 10 * 1024 * 1024:  # Too small or too large
                    return False

            return True

        except Exception as e:
            logger.debug(f"URL check failed for {url}: {e}")
            return False

    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL has valid format."""
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except Exception:
            return False

    def _is_blocked_domain(self, url: str) -> bool:
        """Check if URL is from a blocked domain."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return any(blocked in domain for blocked in self.blocked_domains)
        except Exception:
            return True  # Block malformed URLs

    def _has_image_extension(self, url: str) -> bool:
        """Check if URL appears to point to an image."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        url_lower = url.lower()
        return any(ext in url_lower for ext in image_extensions)

    def add_blocked_domain(self, domain: str) -> None:
        """Add a domain to the blocked list."""
        self.blocked_domains.add(domain.lower())

    def remove_blocked_domain(self, domain: str) -> None:
        """Remove a domain from the blocked list."""
        self.blocked_domains.discard(domain.lower())

    def get_blocked_domains(self) -> Set[str]:
        """Get the set of blocked domains."""
        return self.blocked_domains.copy()