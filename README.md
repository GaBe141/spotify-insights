# Spotify Insights 🎵

A comprehensive Python project to analyze your Spotify listening data with advanced visualizations and global trend comparisons. Features secure API key management, deep musical taste analysis, and integration with Last.fm for global music trends.

## 🔐 Security First

This project implements enterprise-grade security practices:
- **Secure credential management** with comprehensive validation
- **Environment-based configuration** (never hardcode API keys)
- **Automatic security scanning** to prevent credential exposure
- **Git safety** with comprehensive .gitignore patterns

## ⚡ Quick Start

1. **Get API credentials:**
   - Spotify: [Create app](https://developer.spotify.com/dashboard)
   - Last.fm (optional): [Get API key](https://www.last.fm/api/account/create)

2. **Set up environment:**
   ```powershell
   # Clone and navigate to project
   cd spotify-insights
   
   # Create virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run security setup (creates .env template)
   python validate_security.py
   ```

3. **Configure credentials:**
   - Edit `.env` file with your API credentials
   - Set Redirect URI: `http://127.0.0.1:8888/callback`

4. **Validate and run:**
   ```powershell
   # Validate security configuration
   python validate_security.py
   
   # Run basic analysis
   python -m src.main
   
   # Run advanced analysis with global trends
   python -m src.lastfm_main
   ```

## 🎯 Features

### Core Analysis
- **Authentication**: Secure OAuth flow with fallback handling
- **Data Collection**: Top artists/tracks, recently played, listening history
- **Visualizations**: Bar charts, heatmaps, timeline analysis

### Advanced Analytics
- **Musical Maturity Score**: Quantify your taste sophistication
- **Genre Evolution**: Track how your preferences change over time
- **Seasonal Patterns**: Discover mood-based listening cycles
- **Age Analysis**: See how your musical age compares to chronological age

### Global Trends Integration
- **Last.fm Integration**: Compare your taste with global trends
- **Mainstream Analysis**: Calculate your mainstream vs. niche score
- **Discovery Insights**: Find how adventurous your listening is
- **Cultural Context**: Understand your place in the musical landscape

## What's included

- `src/auth.py` – Spotify auth helper using Spotipy's OAuth
- `src/fetch.py` – Simple data fetchers (top artists/tracks, recently played)
- `src/visualize.py` – First chart: Top 10 artists bar chart
- `src/main.py` – Orchestrates auth → fetch → save CSV → plot
- `src/config.py` – Secure configuration management for all API keys
- `src/lastfm_integration.py` – Last.fm global trends integration
- `src/musicbrainz_integration.py` – Artist metadata and relationships
- `src/audiodb_integration.py` – Rich artist profiles and biographies
- `src/spotify_charts_integration.py` – Real-time chart data scraping
- `src/multi_source_main.py` – Comprehensive multi-source analysis
- `validate_security.py` – Security validation and setup
- `simple_multi_source_demo.py` – Quick demo of multi-source capabilities
- `notebooks/` – Place for experiments (optional)
- `data/` – Saved CSVs and images (git-ignored)

## Environment

Create a `.env` file with:

```ini
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
SPOTIFY_SCOPES=user-top-read user-read-recently-played playlist-read-private

# Optional: Last.fm for global trends
LASTFM_API_KEY=your_lastfm_api_key_here

# Optional: AudioDB for rich profiles (defaults to free tier)
AUDIODB_API_KEY=123
```

## Quick Demo

Try the multi-source integration:

```powershell
# Simple working demo
python simple_multi_source_demo.py

# Test individual APIs
python src/musicbrainz_test.py
python src/audiodb_test.py

# Full security validation
python validate_security.py
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
