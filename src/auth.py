import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load .env once on import from the project root, overriding any stale env vars
_ENV_PATH = (Path(__file__).resolve().parent.parent / ".env").as_posix()
load_dotenv(dotenv_path=_ENV_PATH, override=True, encoding="utf-8")

DEFAULT_SCOPES = os.getenv(
    "SPOTIFY_SCOPES",
    "user-top-read user-read-recently-played playlist-read-private",
)


def get_client() -> spotipy.Spotify:
    """Create an authenticated Spotipy client using Authorization Code Flow.

    Reads credentials and config from environment variables:
    - SPOTIFY_CLIENT_ID
    - SPOTIFY_CLIENT_SECRET
    - SPOTIFY_REDIRECT_URI
    - SPOTIFY_SCOPES (optional; defaults provided)
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")

    # Fallback: parse .env directly (handles BOM/encoding edge cases)
    if not client_id or not client_secret or not redirect_uri:
        cfg = dotenv_values(_ENV_PATH, encoding="utf-8")
        # Handle UTF-8 BOM on first key
        client_id = client_id or cfg.get("SPOTIFY_CLIENT_ID") or cfg.get("\ufeffSPOTIFY_CLIENT_ID")
        client_secret = client_secret or cfg.get("SPOTIFY_CLIENT_SECRET")
        redirect_uri = redirect_uri or cfg.get("SPOTIFY_REDIRECT_URI")

    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError(
            "Missing Spotify credentials. Ensure SPOTIFY_CLIENT_ID, "
            "SPOTIFY_CLIENT_SECRET, and SPOTIFY_REDIRECT_URI are set in .env."
        )

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=DEFAULT_SCOPES,
        cache_path=".cache",  # token cache in project root
        open_browser=True,
        show_dialog=False,
    )
    return spotipy.Spotify(auth_manager=auth_manager)
