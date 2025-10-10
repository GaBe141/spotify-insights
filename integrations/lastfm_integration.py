"""Last.fm API integration for global music trends and historical data."""

import json
import time

import pandas as pd
import requests

from .config import get_config

BASE_URL = "http://ws.audioscrobbler.com/2.0/"


class LastFmAPI:
    """Last.fm API client with rate limiting and error handling."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit_delay = 0.2  # 5 requests per second max

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _make_request(self, method: str, **params) -> dict:
        """Make a rate-limited request to Last.fm API."""
        self._rate_limit()

        request_params = {"method": method, "api_key": self.api_key, "format": "json", **params}

        try:
            response = self.session.get(BASE_URL, params=request_params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                print(f"Last.fm API error: {data.get('message', 'Unknown error')}")
                return {}

            return data
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {}

    def get_top_artists_global(self, limit: int = 50) -> pd.DataFrame:
        """Get global top artists chart."""
        data = self._make_request("chart.gettopartists", limit=limit)

        if not data or "artists" not in data:
            return pd.DataFrame()

        artists = []
        for rank, artist in enumerate(data["artists"]["artist"], 1):
            artists.append(
                {
                    "rank": rank,
                    "name": artist["name"],
                    "playcount": int(artist.get("playcount", 0)),
                    "listeners": int(artist.get("listeners", 0)),
                    "url": artist.get("url", ""),
                    "mbid": artist.get("mbid", ""),
                }
            )

        return pd.DataFrame(artists)

    def get_top_tracks_global(self, limit: int = 50) -> pd.DataFrame:
        """Get global top tracks chart."""
        data = self._make_request("chart.gettoptracks", limit=limit)

        if not data or "tracks" not in data:
            return pd.DataFrame()

        tracks = []
        for rank, track in enumerate(data["tracks"]["track"], 1):
            tracks.append(
                {
                    "rank": rank,
                    "name": track["name"],
                    "artist": track["artist"]["name"],
                    "playcount": int(track.get("playcount", 0)),
                    "listeners": int(track.get("listeners", 0)),
                    "url": track.get("url", ""),
                    "mbid": track.get("mbid", ""),
                }
            )

        return pd.DataFrame(tracks)

    def get_artist_info(self, artist_name: str) -> dict:
        """Get detailed info about an artist."""
        data = self._make_request("artist.getinfo", artist=artist_name)

        if not data or "artist" not in data:
            return {}

        artist = data["artist"]

        # Parse tags/genres
        tags = []
        if "tags" in artist and "tag" in artist["tags"]:
            tag_list = artist["tags"]["tag"]
            if isinstance(tag_list, list):
                tags = [tag["name"] for tag in tag_list]
            elif isinstance(tag_list, dict):
                tags = [tag_list["name"]]

        return {
            "name": artist.get("name", ""),
            "playcount": int(artist.get("stats", {}).get("playcount", 0)),
            "listeners": int(artist.get("stats", {}).get("listeners", 0)),
            "genres": tags,
            "bio": artist.get("bio", {}).get("summary", ""),
            "url": artist.get("url", ""),
            "similar_artists": [a["name"] for a in artist.get("similar", {}).get("artist", [])],
        }

    def get_track_info(self, artist_name: str, track_name: str) -> dict:
        """Get detailed info about a track."""
        data = self._make_request("track.getinfo", artist=artist_name, track=track_name)

        if not data or "track" not in data:
            return {}

        track = data["track"]

        # Parse tags
        tags = []
        if "toptags" in track and "tag" in track["toptags"]:
            tag_list = track["toptags"]["tag"]
            if isinstance(tag_list, list):
                tags = [tag["name"] for tag in tag_list]
            elif isinstance(tag_list, dict):
                tags = [tag_list["name"]]

        return {
            "name": track.get("name", ""),
            "artist": track.get("artist", {}).get("name", ""),
            "playcount": int(track.get("playcount", 0)),
            "listeners": int(track.get("listeners", 0)),
            "genres": tags,
            "duration": int(track.get("duration", 0)),  # milliseconds
            "url": track.get("url", ""),
        }

    def search_artists(self, query: str, limit: int = 30) -> pd.DataFrame:
        """Search for artists."""
        data = self._make_request("artist.search", artist=query, limit=limit)

        if not data or "results" not in data:
            return pd.DataFrame()

        artists = []
        artist_matches = data["results"].get("artistmatches", {}).get("artist", [])

        if isinstance(artist_matches, dict):
            artist_matches = [artist_matches]

        for artist in artist_matches:
            artists.append(
                {
                    "name": artist.get("name", ""),
                    "listeners": int(artist.get("listeners", 0)),
                    "url": artist.get("url", ""),
                    "mbid": artist.get("mbid", ""),
                }
            )

        return pd.DataFrame(artists)

    def get_tag_top_artists(self, tag: str, limit: int = 50) -> pd.DataFrame:
        """Get top artists for a specific genre/tag."""
        data = self._make_request("tag.gettopartists", tag=tag, limit=limit)

        if not data or "topartists" not in data:
            return pd.DataFrame()

        artists = []
        for rank, artist in enumerate(data["topartists"]["artist"], 1):
            artists.append(
                {
                    "rank": rank,
                    "name": artist["name"],
                    "genre": tag,
                    "url": artist.get("url", ""),
                    "mbid": artist.get("mbid", ""),
                }
            )

        return pd.DataFrame(artists)


def get_lastfm_client() -> LastFmAPI | None:
    """Get authenticated Last.fm client using secure configuration."""
    config_manager = get_config()
    lastfm_config = config_manager.get_lastfm_config()

    if not lastfm_config:
        print("❌ Last.fm API key not configured!")
        print("Get a free API key from: https://www.last.fm/api/account/create")
        print("Then add it to your .env file as LASTFM_API_KEY=your_key_here")
        return None

    return LastFmAPI(lastfm_config["api_key"])


def fetch_global_trends() -> dict[str, pd.DataFrame]:
    """Fetch current global music trends from Last.fm."""
    client = get_lastfm_client()
    if not client:
        return {}

    print("🌍 Fetching global music trends from Last.fm...")

    trends = {}

    try:
        # Global top artists
        print("   📊 Getting global top artists...")
        trends["global_top_artists"] = client.get_top_artists_global(limit=50)

        # Global top tracks
        print("   🎵 Getting global top tracks...")
        trends["global_top_tracks"] = client.get_top_tracks_global(limit=50)

        # Genre-specific trends
        print("   🎭 Getting genre-specific trends...")
        popular_genres = ["rock", "pop", "electronic", "hip hop", "indie", "alternative"]

        for genre in popular_genres:
            print(f"      • {genre}...")
            genre_artists = client.get_tag_top_artists(genre, limit=20)
            if not genre_artists.empty:
                trends[f"genre_{genre.replace(' ', '_')}"] = genre_artists

        print(f"✅ Fetched trends for {len(trends)} categories")

    except Exception as e:
        print(f"❌ Error fetching trends: {e}")

    return trends


def enrich_spotify_artists_with_lastfm(spotify_artists: pd.DataFrame) -> pd.DataFrame:
    """Enrich Spotify artist data with Last.fm global stats."""
    client = get_lastfm_client()
    if not client or spotify_artists.empty:
        return spotify_artists

    print("🔗 Enriching Spotify data with Last.fm global stats...")

    enriched_data = []

    for _, artist in spotify_artists.iterrows():
        print(f"   📊 Looking up {artist['name']}...")

        lastfm_info = client.get_artist_info(artist["name"])

        enriched_row = artist.to_dict()
        enriched_row.update(
            {
                "lastfm_playcount": lastfm_info.get("playcount", 0),
                "lastfm_listeners": lastfm_info.get("listeners", 0),
                "lastfm_genres": ", ".join(lastfm_info.get("genres", [])),
                "lastfm_similar": ", ".join(lastfm_info.get("similar_artists", [])[:3]),
            }
        )

        enriched_data.append(enriched_row)

    return pd.DataFrame(enriched_data)
