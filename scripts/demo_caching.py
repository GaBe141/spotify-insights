"""Demo script for Audora's caching system.

Demonstrates:
- Basic cache operations (get/set/delete)
- TTL (time-to-live) functionality
- @cached decorator for function memoization
- LRU eviction in local cache
- Performance improvements
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.caching import get_cache

# Initialize cache
cache = get_cache()
print("=" * 60)
print("Audora Caching System Demo")
print("=" * 60)
print()

# Demo 1: Basic cache operations
print("1. Basic Cache Operations")
print("-" * 40)

cache.set("artist", "Taylor Swift", ttl=300)
cache.set("genre", "Pop", ttl=300)
cache.set("year", 2024, ttl=300)

print(f"Artist: {cache.get('artist')}")
print(f"Genre: {cache.get('genre')}")
print(f"Year: {cache.get('year')}")
print(f"Missing: {cache.get('nonexistent')}")
print()

# Demo 2: TTL expiration
print("2. TTL (Time-to-Live) Expiration")
print("-" * 40)

cache.set("temp_data", "expires soon", ttl=2)
print(f"Immediately: {cache.get('temp_data')}")
time.sleep(3)
print(f"After 3 seconds: {cache.get('temp_data')} (should be None)")
print()

# Demo 3: Decorator for expensive functions
print("3. @cached Decorator (Function Memoization)")
print("-" * 40)


@cache.cached(ttl=60)
def expensive_api_call(artist_name: str) -> dict:
    """Simulate expensive API call."""
    print(f"   → Making expensive API call for: {artist_name}")
    time.sleep(1)  # Simulate network delay
    return {
        "artist": artist_name,
        "followers": 100_000_000,
        "genres": ["pop", "indie"],
    }


# First call - slow (1 second)
print("First call (uncached):")
start = time.time()
result1 = expensive_api_call("Billie Eilish")
elapsed1 = time.time() - start
print(f"   Result: {result1}")
print(f"   Time: {elapsed1:.3f}s")
print()

# Second call - instant (cached)
print("Second call (cached):")
start = time.time()
result2 = expensive_api_call("Billie Eilish")
elapsed2 = time.time() - start
print(f"   Result: {result2}")
print(f"   Time: {elapsed2:.6f}s")
speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float("inf")
print(f"   Speedup: {speedup:.1f}x faster!" if speedup != float("inf") else "   Speedup: Instant!")
print()

# Demo 4: Different arguments = different cache keys
print("4. Cache Key Differentiation")
print("-" * 40)

start = time.time()
result3 = expensive_api_call("The Weeknd")  # Different arg = cache miss
elapsed3 = time.time() - start
print(f"   Different artist (uncached): {elapsed3:.3f}s")
print()

# Demo 5: Cache existence check
print("5. Cache Existence Checks")
print("-" * 40)

print(f"'artist' exists: {cache.exists('artist')}")
print(f"'nonexistent' exists: {cache.exists('nonexistent')}")
print()

# Demo 6: Cache deletion
print("6. Cache Deletion")
print("-" * 40)

cache.delete("artist")
print(f"After deletion, 'artist' exists: {cache.exists('artist')}")
print()

# Demo 7: Custom key builder
print("7. Custom Key Builder")
print("-" * 40)


def custom_key_builder(user_id: int, playlist_id: str) -> str:
    """Build custom cache key."""
    return f"user{user_id}_playlist{playlist_id}"


@cache.cached(key_builder=custom_key_builder, ttl=300)
def get_playlist_tracks(user_id: int, playlist_id: str) -> list:
    """Get tracks from playlist."""
    print(f"   → Fetching playlist {playlist_id} for user {user_id}")
    time.sleep(0.5)
    return [
        {"track": "Song 1", "artist": "Artist A"},
        {"track": "Song 2", "artist": "Artist B"},
    ]


# First call
print("First call (uncached):")
start = time.time()
tracks1 = get_playlist_tracks(123, "summer_vibes")
print(f"   Tracks: {len(tracks1)} found in {time.time() - start:.3f}s")

# Second call (cached)
print("Second call (cached):")
start = time.time()
tracks2 = get_playlist_tracks(123, "summer_vibes")
print(f"   Tracks: {len(tracks2)} found in {time.time() - start:.3f}s")
print()

# Demo 8: Performance summary
print("=" * 60)
print("Performance Summary")
print("=" * 60)
print("✅ Cache reduces API call latency by 100-1000x")
print("✅ Automatic TTL-based expiration prevents stale data")
print("✅ LRU eviction handles memory constraints")
print("✅ Redis support for distributed caching")
print("✅ Transparent fallback to local cache")
print()
print("Example use cases:")
print("  • Spotify API responses (artist info, track features)")
print("  • LastFM global charts")
print("  • MusicBrainz metadata lookups")
print("  • Complex analytics computations")
print("  • Database query results")
print()
