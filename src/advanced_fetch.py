from typing import List
import pandas as pd
from .auth import get_client

def fetch_audio_features(track_ids: List[str]) -> pd.DataFrame:
    """Fetch audio features for a list of track IDs."""
    if not track_ids:
        return pd.DataFrame()
    
    sp = get_client()
    # Batch requests (max 100 IDs per call)
    features_list = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        features = sp.audio_features(batch)
        features_list.extend([f for f in features if f is not None])
    
    if not features_list:
        return pd.DataFrame()
    
    # Convert to DataFrame and clean up
    df = pd.DataFrame(features_list)
    
    # Select key audio features
    audio_cols = [
        'id', 'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness',
        'mode', 'key', 'time_signature', 'duration_ms'
    ]
    
    return df[audio_cols].copy()


def fetch_top_tracks_with_features(limit: int = 50, time_range: str = "medium_term") -> pd.DataFrame:
    """Fetch top tracks with their audio features."""
    sp = get_client()
    
    # Get top tracks
    top_tracks = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    tracks_data = []
    
    for rank, track in enumerate(top_tracks['items'], start=1):
        primary_artist = track.get('artists', [{}])[0]
        tracks_data.append({
            'rank': rank,
            'track_id': track.get('id'),
            'track_name': track.get('name'),
            'artist_name': primary_artist.get('name'),
            'artist_id': primary_artist.get('id'),
            'album_name': track.get('album', {}).get('name'),
            'popularity': track.get('popularity'),
            'release_date': track.get('album', {}).get('release_date'),
            'duration_ms': track.get('duration_ms')
        })
    
    tracks_df = pd.DataFrame(tracks_data)
    
    # Get audio features
    if not tracks_df.empty:
        track_ids = tracks_df['track_id'].tolist()
        features_df = fetch_audio_features(track_ids)
        
        # Merge with track info
        tracks_df = tracks_df.merge(
            features_df, 
            left_on='track_id', 
            right_on='id', 
            how='left'
        ).drop('id', axis=1)
    
    return tracks_df


def fetch_artist_genres() -> pd.DataFrame:
    """Fetch genres for top artists across all time ranges."""
    sp = get_client()
    
    all_artists = []
    time_ranges = ["short_term", "medium_term", "long_term"]
    
    for time_range in time_ranges:
        artists = sp.current_user_top_artists(limit=50, time_range=time_range)
        for rank, artist in enumerate(artists['items'], start=1):
            genres = artist.get('genres', [])
            for genre in genres:
                all_artists.append({
                    'time_range': time_range,
                    'rank': rank,
                    'artist_id': artist.get('id'),
                    'artist_name': artist.get('name'),
                    'genre': genre,
                    'popularity': artist.get('popularity'),
                    'followers': artist.get('followers', {}).get('total', 0)
                })
    
    return pd.DataFrame(all_artists)


def fetch_playlist_audio_analysis() -> pd.DataFrame:
    """Fetch playlists and analyze their audio characteristics."""
    sp = get_client()
    
    # Get user's playlists
    playlists = sp.current_user_playlists(limit=50)
    playlist_data = []
    
    for playlist in playlists['items']:
        if not playlist['owner']['id'] == sp.current_user()['id']:
            continue  # Skip playlists not owned by user
            
        playlist_id = playlist['id']
        playlist_name = playlist['name']
        
        # Get tracks from playlist
        try:
            tracks = sp.playlist_tracks(playlist_id, limit=100)
            track_ids = [
                item['track']['id'] for item in tracks['items'] 
                if item['track'] and item['track']['id']
            ]
            
            if track_ids:
                # Get audio features for playlist tracks
                features_df = fetch_audio_features(track_ids)
                
                if not features_df.empty:
                    # Calculate playlist audio characteristics
                    avg_features = features_df.select_dtypes(include=['number']).mean()
                    
                    playlist_data.append({
                        'playlist_id': playlist_id,
                        'playlist_name': playlist_name,
                        'track_count': len(track_ids),
                        **avg_features.to_dict()
                    })
        except Exception as e:
            print(f"Error processing playlist {playlist_name}: {e}")
            continue
    
    return pd.DataFrame(playlist_data)