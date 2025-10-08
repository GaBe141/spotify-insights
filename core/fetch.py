from typing import Literal
import pandas as pd

from .auth import get_client

TimeRange = Literal["short_term", "medium_term", "long_term"]


def fetch_top_artists(limit: int = 20, time_range: TimeRange = "short_term") -> pd.DataFrame:
    """Fetch top artists and return as a tidy DataFrame."""
    sp = get_client()
    res = sp.current_user_top_artists(limit=limit, time_range=time_range)
    items = res.get("items", [])
    rows = []
    for rank, artist in enumerate(items, start=1):
        rows.append(
            {
                "rank": rank,
                "id": artist.get("id"),
                "name": artist.get("name"),
                "genres": ", ".join(artist.get("genres", [])),
                "followers": artist.get("followers", {}).get("total"),
                "popularity": artist.get("popularity"),
            }
        )
    return pd.DataFrame(rows)


def fetch_top_tracks(limit: int = 20, time_range: TimeRange = "short_term") -> pd.DataFrame:
    """Fetch top tracks and return as a tidy DataFrame."""
    sp = get_client()
    res = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    items = res.get("items", [])
    rows = []
    for rank, track in enumerate(items, start=1):
        primary_artist = track.get("artists", [{}])[0].get("name")
        rows.append(
            {
                "rank": rank,
                "id": track.get("id"),
                "name": track.get("name"),
                "artist": primary_artist,
                "album": track.get("album", {}).get("name"),
                "popularity": track.get("popularity"),
                "duration_ms": track.get("duration_ms"),
            }
        )
    return pd.DataFrame(rows)


def fetch_recently_played(limit: int = 50) -> pd.DataFrame:
    """Fetch recently played tracks (up to 50) with timestamps."""
    sp = get_client()
    res = sp.current_user_recently_played(limit=limit)
    items = res.get("items", [])
    rows = []
    for item in items:
        track = item.get("track", {})
        primary_artist = track.get("artists", [{}])[0].get("name")
        rows.append(
            {
                "played_at": item.get("played_at"),
                "track_id": track.get("id"),
                "track_name": track.get("name"),
                "artist": primary_artist,
                "album": track.get("album", {}).get("name"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce")
    return df
