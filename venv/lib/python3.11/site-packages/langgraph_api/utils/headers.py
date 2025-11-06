"""Shared utilities for configurable header filtering."""

import functools
import re


def translate_pattern(pat: str) -> re.Pattern[str]:
    """Translate a pattern to regex, supporting only literals and * wildcards to avoid RE DoS."""
    res = []
    i = 0
    n = len(pat)

    while i < n:
        c = pat[i]
        i += 1

        if c == "*":
            res.append(".*")
        else:
            res.append(re.escape(c))

    pattern = "".join(res)
    return re.compile(rf"(?s:{pattern})\Z")


@functools.lru_cache(maxsize=1)
def get_header_patterns(
    key: str,
) -> tuple[list[re.Pattern[str]] | None, list[re.Pattern[str]] | None]:
    """Get the configured header include/exclude patterns."""
    from langgraph_api import config

    if not config.HTTP_CONFIG:
        return None, None
    configurable = config.HTTP_CONFIG.get(key)
    if not configurable:
        return None, None
    header_includes = configurable.get("includes") or configurable.get("include") or []
    include_patterns = []
    for include in header_includes:
        include_patterns.append(translate_pattern(include))
    header_excludes = configurable.get("excludes") or configurable.get("exclude") or []
    exclude_patterns = []
    for exclude in header_excludes:
        exclude_patterns.append(translate_pattern(exclude))
    return include_patterns or None, exclude_patterns or None


@functools.lru_cache(maxsize=512)
def should_include_header(key: str) -> bool:
    """Check if a header should be included based on cached patterns.

    This function uses cached patterns from get_header_patterns() and
    provides efficient header filtering.

    Args:
        key: The header key to check

    Returns:
        True if the header should be included, False otherwise
    """
    include_patterns, exclude_patterns = get_header_patterns("configurable_headers")

    return pattern_matches(key, include_patterns, exclude_patterns)


@functools.lru_cache(maxsize=512)
def should_include_header_in_logs(key: str) -> bool:
    """Check if header should be included in logs specifically."""

    include_patterns, exclude_patterns = get_header_patterns("logging_headers")

    return pattern_matches(key, include_patterns, exclude_patterns)


def pattern_matches(
    key: str,
    include_patterns: list[re.Pattern[str]] | None,
    exclude_patterns: list[re.Pattern[str]] | None,
) -> bool:
    # Handle configurable behavior
    if exclude_patterns and any(pattern.match(key) for pattern in exclude_patterns):
        return False
    if include_patterns:
        # If include patterns are specified, only include headers matching them
        return any(pattern.match(key) for pattern in include_patterns)

    # Default behavior - include if not excluded
    return True
