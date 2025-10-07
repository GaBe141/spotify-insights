# Spotify Insights

A small Python project to explore your Spotify listening data: top artists/tracks, recently played, genres, and audio features. Uses Spotipy for auth and data access.

## Quick start

1. Create a Spotify app at <https://developer.spotify.com/dashboard> and note your Client ID/Secret.
2. Add this Redirect URI in the dashboard (exact match):
   - `http://127.0.0.1:8888/callback`
3. Copy `.env.example` to `.env` and fill in your credentials.
4. Create/activate a virtualenv and install deps.
5. Run the sample script and log in.

## Setup (Windows PowerShell)

```powershell
# From the repo root
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt

# First run will open a browser for Spotify login
python -m src.main
```

## What's included

- `src/auth.py` – Spotify auth helper using Spotipy's OAuth
- `src/fetch.py` – Simple data fetchers (top artists/tracks, recently played)
- `src/visualize.py` – First chart: Top 10 artists bar chart
- `src/main.py` – Orchestrates auth → fetch → save CSV → plot
- `notebooks/` – Place for experiments (optional)
- `data/` – Saved CSVs and images (git-ignored)

## Environment

Create a `.env` file with:

```ini
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
SPOTIFY_SCOPES=user-top-read user-read-recently-played playlist-read-private
```

## Next ideas

- Heatmap of listening by hour/day from `recently-played`
- Audio features "mood wheel" over time
- Playlist overlap network
- Genre clusters and seasonal shifts

## Troubleshooting

- Redirect URI mismatch: ensure the URI here and in the Spotify dashboard are identical.
- TLS vs localhost: localhost/127.0.0.1 over HTTP is allowed for local dev.
- If the browser doesn't open, copy the login URL from the terminal and paste it into your browser.
