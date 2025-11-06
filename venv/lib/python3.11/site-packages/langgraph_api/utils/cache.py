import asyncio
import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class LRUCache(Generic[T]):
    """LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl: float = 60):
        self._cache: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self._max_size = max_size if max_size > 0 else 1000
        self._ttl = ttl

    def _get_time(self) -> float:
        """Get current time, using loop.time() if available for better performance."""
        try:
            return asyncio.get_event_loop().time()
        except RuntimeError:
            return time.monotonic()

    def get(self, key: str) -> T | None:
        """Get item from cache, returning None if expired or not found."""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if self._get_time() - timestamp >= self._ttl:
            # Expired, remove and return None
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: T) -> None:
        """Set item in cache, evicting old entries if needed."""
        # Remove if already exists (to update timestamp)
        if key in self._cache:
            del self._cache[key]

        # Evict oldest entries if needed
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Remove oldest (FIFO)

        # Add new entry
        self._cache[key] = (value, self._get_time())

    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()
