"""Advanced visualization generator - creates multiple surprise charts from your Spotify data."""

from pathlib import Path

from .advanced_fetch import (
    fetch_artist_genres,
    fetch_playlist_audio_analysis,
    fetch_top_tracks_with_features,
)
from .advanced_viz import (
    plot_audio_dna_radar,
    plot_genre_evolution,
    plot_mood_evolution,
    plot_playlist_characteristics,
    plot_tempo_energy_dance_3d,
)
from .fetch import fetch_recently_played

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_surprise_visualizations():
    """Generate a collection of surprise visualizations."""
    print("🎵 Generating surprise visualizations...")
    generated_files = []

    try:
        print("📊 1. Analyzing your Audio DNA...")
        # Get top tracks with audio features
        top_tracks_df = fetch_top_tracks_with_features(limit=50, time_range="medium_term")
        if not top_tracks_df.empty:
            audio_features = top_tracks_df.dropna(subset=["danceability", "energy", "valence"])
            if not audio_features.empty:
                radar_path = plot_audio_dna_radar(audio_features)
                generated_files.append(radar_path)
                print(f"   ✅ Saved: {radar_path.name}")

            # 3D music space
            if not audio_features.empty and len(audio_features) > 10:
                space_3d_path = plot_tempo_energy_dance_3d(audio_features)
                generated_files.append(space_3d_path)
                print(f"   ✅ Saved: {space_3d_path.name}")

        print("🕐 2. Tracking your mood throughout the day...")
        # Get recently played with features
        recent_df = fetch_recently_played(limit=50)
        if not recent_df.empty and len(recent_df) > 5:
            # Get track IDs and fetch audio features
            from .advanced_fetch import fetch_audio_features

            track_ids = recent_df["track_id"].dropna().tolist()
            if track_ids:
                features_df = fetch_audio_features(track_ids)
                if not features_df.empty:
                    # Merge with recent plays
                    recent_with_features = recent_df.merge(
                        features_df, left_on="track_id", right_on="id", how="inner"
                    )
                    if not recent_with_features.empty:
                        mood_path = plot_mood_evolution(recent_with_features)
                        generated_files.append(mood_path)
                        print(f"   ✅ Saved: {mood_path.name}")

        print("🎭 3. Exploring your genre evolution...")
        # Genre analysis
        genres_df = fetch_artist_genres()
        if not genres_df.empty and len(genres_df) > 10:
            genre_path = plot_genre_evolution(genres_df)
            generated_files.append(genre_path)
            print(f"   ✅ Saved: {genre_path.name}")

            # Save genre data
            genre_csv = DATA_DIR / "genre_evolution.csv"
            genres_df.to_csv(genre_csv, index=False)
            print(f"   📁 Saved data: {genre_csv.name}")

        print("📱 4. Analyzing your playlist landscapes...")
        # Playlist analysis
        playlists_df = fetch_playlist_audio_analysis()
        if not playlists_df.empty and len(playlists_df) >= 3:
            playlist_path = plot_playlist_characteristics(playlists_df)
            generated_files.append(playlist_path)
            print(f"   ✅ Saved: {playlist_path.name}")

            # Save playlist data
            playlist_csv = DATA_DIR / "playlist_analysis.csv"
            playlists_df.to_csv(playlist_csv, index=False)
            print(f"   📁 Saved data: {playlist_csv.name}")

        # Save top tracks with features
        if not top_tracks_df.empty:
            tracks_csv = DATA_DIR / "top_tracks_with_features.csv"
            top_tracks_df.to_csv(tracks_csv, index=False)
            print(f"   📁 Saved data: {tracks_csv.name}")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return []

    print(f"🎉 Generated {len(generated_files)} visualizations!")
    return generated_files


if __name__ == "__main__":
    generate_surprise_visualizations()
