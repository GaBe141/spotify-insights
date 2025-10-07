"""AudioDB API integration for rich artist profiles and comprehensive music metadata."""

import requests
import pandas as pd
from typing import Dict, List, Optional
import time
try:
    from .config import get_config
except ImportError:
    # Fallback for standalone execution
    pass

BASE_URL = "https://www.theaudiodb.com/api/v1/json"


class AudioDBAPI:
    """AudioDB API client for artist biographies, discographies, and metadata."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 request per second
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to AudioDB API with rate limiting."""
        self._rate_limit()
        
        # Use API key if available, otherwise use public endpoints
        if self.api_key:
            url = f"{BASE_URL}/{self.api_key}/{endpoint}"
        else:
            url = f"{BASE_URL}/1/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"AudioDB API error: {e}")
            return None
    
    def search_artist(self, artist_name: str) -> Optional[Dict]:
        """Search for an artist by name."""
        result = self._make_request(f"search.php", {"s": artist_name})
        
        if result and result.get('artists'):
            return result['artists'][0]
        return None
    
    def get_artist_details(self, artist_name: str) -> Optional[Dict]:
        """Get detailed artist information including biography."""
        artist = self.search_artist(artist_name)
        if not artist:
            return None
        
        # AudioDB returns comprehensive data in the search result
        return {
            'name': artist.get('strArtist'),
            'biography': artist.get('strBiographyEN'),
            'genre': artist.get('strGenre'),
            'style': artist.get('strStyle'),
            'mood': artist.get('strMood'),
            'country': artist.get('strCountry'),
            'formed_year': artist.get('intFormedYear'),
            'disbanded_year': artist.get('intDiedYear'),
            'website': artist.get('strWebsite'),
            'facebook': artist.get('strFacebook'),
            'twitter': artist.get('strTwitter'),
            'thumbnail': artist.get('strArtistThumb'),
            'banner': artist.get('strArtistBanner'),
            'logo': artist.get('strArtistLogo'),
            'fanart': artist.get('strArtistFanart'),
            'members': artist.get('intMembers'),
            'audio_db_id': artist.get('idArtist')
        }
    
    def get_artist_albums(self, artist_name: str) -> List[Dict]:
        """Get all albums for an artist."""
        result = self._make_request("searchalbum.php", {"s": artist_name})
        
        if result and result.get('album'):
            albums = []
            for album in result['album']:
                albums.append({
                    'album_name': album.get('strAlbum'),
                    'artist_name': album.get('strArtist'),
                    'year': album.get('intYearReleased'),
                    'genre': album.get('strGenre'),
                    'style': album.get('strStyle'),
                    'mood': album.get('strMood'),
                    'description': album.get('strDescriptionEN'),
                    'thumbnail': album.get('strAlbumThumb'),
                    'score': album.get('intScore'),
                    'score_votes': album.get('intScoreVotes'),
                    'audio_db_album_id': album.get('idAlbum')
                })
            return albums
        return []
    
    def get_artist_tracks(self, artist_name: str) -> List[Dict]:
        """Get popular tracks for an artist."""
        result = self._make_request("track.php", {"m": artist_name})
        
        if result and result.get('track'):
            tracks = []
            for track in result['track']:
                tracks.append({
                    'track_name': track.get('strTrack'),
                    'artist_name': track.get('strArtist'),
                    'album_name': track.get('strAlbum'),
                    'genre': track.get('strGenre'),
                    'style': track.get('strStyle'),
                    'mood': track.get('strMood'),
                    'description': track.get('strDescriptionEN'),
                    'duration_ms': track.get('intDuration'),
                    'youtube_link': track.get('strMusicVid'),
                    'audio_db_track_id': track.get('idTrack')
                })
            return tracks
        return []
    
    def enrich_artist_profiles(self, artist_names: List[str]) -> pd.DataFrame:
        """Enrich multiple artists with comprehensive AudioDB data."""
        enriched_data = []
        
        print(f"🎧 Enriching {len(artist_names)} artists with AudioDB profiles...")
        
        for i, artist_name in enumerate(artist_names):
            print(f"Processing {i+1}/{len(artist_names)}: {artist_name}")
            
            artist_details = self.get_artist_details(artist_name)
            if artist_details:
                enriched_data.append(artist_details)
            else:
                # Add empty record for missing artists
                enriched_data.append({
                    'name': artist_name,
                    'biography': None,
                    'genre': None,
                    'style': None,
                    'mood': None,
                    'country': None,
                    'formed_year': None,
                    'disbanded_year': None,
                    'website': None,
                    'facebook': None,
                    'twitter': None,
                    'thumbnail': None,
                    'banner': None,
                    'logo': None,
                    'fanart': None,
                    'members': None,
                    'audio_db_id': None
                })
        
        return pd.DataFrame(enriched_data)
    
    def analyze_artist_careers(self, artist_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Analyze career patterns and discographies for multiple artists."""
        career_data = []
        album_data = []
        
        print(f"📈 Analyzing careers for {len(artist_names)} artists...")
        
        for artist_name in artist_names:
            print(f"Analyzing career: {artist_name}")
            
            # Get artist details
            details = self.get_artist_details(artist_name)
            if details:
                formed_year = details.get('formed_year')
                disbanded_year = details.get('disbanded_year')
                
                career_length = None
                if formed_year:
                    current_year = 2025
                    end_year = disbanded_year if disbanded_year else current_year
                    try:
                        career_length = int(end_year) - int(formed_year)
                    except (ValueError, TypeError):
                        pass
                
                career_data.append({
                    'artist_name': artist_name,
                    'formed_year': formed_year,
                    'disbanded_year': disbanded_year,
                    'career_length_years': career_length,
                    'country': details.get('country'),
                    'genre': details.get('genre'),
                    'members': details.get('members')
                })
            
            # Get albums for timeline analysis
            albums = self.get_artist_albums(artist_name)
            for album in albums:
                if album['year']:
                    album_data.append({
                        'artist_name': artist_name,
                        'album_name': album['album_name'],
                        'year': album['year'],
                        'genre': album['genre'],
                        'score': album['score']
                    })
        
        career_df = pd.DataFrame(career_data)
        albums_df = pd.DataFrame(album_data)
        
        # Calculate career statistics
        if not career_df.empty:
            career_stats = career_df.groupby('country').agg({
                'career_length_years': ['mean', 'median', 'count'],
                'formed_year': ['min', 'max']
            }).round(2)
        else:
            career_stats = pd.DataFrame()
        
        return {
            'careers': career_df,
            'albums': albums_df,
            'career_stats': career_stats
        }


def get_audiodb_client() -> AudioDBAPI:
    """Get AudioDB API client with optional API key."""
    # Try to get API key from environment (optional)
    try:
        import os
        api_key = os.getenv('AUDIODB_API_KEY')
        return AudioDBAPI(api_key)
    except Exception:
        return AudioDBAPI()


def enrich_spotify_artists_with_audiodb(spotify_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich Spotify artist data with AudioDB profiles."""
    client = get_audiodb_client()
    
    # Get unique artist names
    artist_names = spotify_df['artist_name'].unique().tolist()
    
    # Get enrichment data
    enrichment_df = client.enrich_artist_profiles(artist_names)
    
    # Merge with original data
    enriched_df = spotify_df.merge(
        enrichment_df,
        left_on='artist_name',
        right_on='name',
        how='left'
    )
    
    return enriched_df


def analyze_genre_evolution_with_audiodb(artist_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Analyze how artist genres and styles evolved using AudioDB data."""
    client = get_audiodb_client()
    
    print("🎭 Analyzing genre evolution with AudioDB...")
    
    # Get career analysis
    career_analysis = client.analyze_artist_careers(artist_names)
    
    careers_df = career_analysis['careers']
    albums_df = career_analysis['albums']
    
    # Analyze genre trends over time
    if not albums_df.empty and 'year' in albums_df.columns:
        # Convert year to numeric
        albums_df['year'] = pd.to_numeric(albums_df['year'], errors='coerce')
        albums_df = albums_df.dropna(subset=['year'])
        
        # Group by decade for trend analysis
        albums_df['decade'] = (albums_df['year'] // 10) * 10
        
        genre_trends = albums_df.groupby(['decade', 'genre']).size().reset_index(name='album_count')
        
        # Calculate career progression
        career_progression = albums_df.groupby('artist_name').agg({
            'year': ['min', 'max', 'count'],
            'genre': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
        }).reset_index()
        
        career_progression.columns = ['artist_name', 'first_album_year', 'latest_album_year', 'total_albums', 'dominant_genre']
    else:
        genre_trends = pd.DataFrame()
        career_progression = pd.DataFrame()
    
    return {
        'careers': careers_df,
        'albums': albums_df,
        'genre_trends': genre_trends,
        'career_progression': career_progression
    }


if __name__ == "__main__":
    # Test the AudioDB integration
    client = get_audiodb_client()
    
    test_artists = ["The Beatles", "Radiohead", "Taylor Swift"]
    
    print("Testing AudioDB integration...")
    
    # Test artist search
    for artist in test_artists:
        details = client.get_artist_details(artist)
        if details and details['name']:
            print(f"✅ Found {artist}: {details['name']}")
            if details['biography']:
                bio_preview = details['biography'][:100] + "..." if len(details['biography']) > 100 else details['biography']
                print(f"   Biography: {bio_preview}")
            print(f"   Genre: {details['genre']}, Country: {details['country']}")
        else:
            print(f"❌ Could not find {artist}")
    
    # Test enrichment
    enriched = client.enrich_artist_profiles(test_artists)
    print(f"\n✅ Enriched {len(enriched)} artists with AudioDB profiles")
    print(enriched[['name', 'country', 'genre', 'formed_year']].to_string())