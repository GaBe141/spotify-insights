"""Demo script showing multi-source integration capabilities."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.auth import get_client
from src.fetch import fetch_top_artists

def demo_multi_source_capabilities():
    """Demonstrate the new multi-source integration capabilities."""
    print("🎵 Multi-Source Music Data Integration Demo")
    print("=" * 60)
    print("This demo shows what the new integration can do!")
    print("=" * 60)
    
    # Test Spotify connection
    print("\n1️⃣ Testing Spotify Connection...")
    try:
        spotify_client = get_client()
        user = spotify_client.current_user()
        print(f"✅ Connected to Spotify as: {user['display_name']}")
        
        # Get a few top artists
        top_artists = fetch_top_artists(limit=5, time_range='short_term')
        print(f"✅ Retrieved {len(top_artists)} top artists")
        print("   Sample artists:", ", ".join(top_artists['name'].head(3).tolist()))
        
    except Exception as e:
        print(f"❌ Spotify connection failed: {e}")
        return
    
    # Test MusicBrainz (no API key needed)
    print("\n2️⃣ Testing MusicBrainz Integration...")
    try:
        from musicbrainz_integration import get_musicbrainz_client
        mb_client = get_musicbrainz_client()
        
        # Test with one artist
        test_artist = top_artists['name'].iloc[0]
        print(f"   Searching for: {test_artist}")
        
        artist_info = mb_client.search_artist(test_artist)
        if artist_info:
            print(f"✅ Found in MusicBrainz: {artist_info['name']}")
            print(f"   Country: {artist_info.get('country', 'Unknown')}")
            print(f"   Type: {artist_info.get('type', 'Unknown')}")
        else:
            print(f"⚠️ {test_artist} not found in MusicBrainz")
    except Exception as e:
        print(f"❌ MusicBrainz integration error: {e}")
    
    # Test Last.fm (if configured)
    print("\n3️⃣ Testing Last.fm Integration...")
    try:
        from lastfm_integration import get_lastfm_client
        lastfm_client = get_lastfm_client()
        
        if lastfm_client:
            print("✅ Last.fm client available")
            print("   Can fetch global trends and artist popularity")
        else:
            print("⚠️ Last.fm not configured (optional)")
    except Exception as e:
        print(f"❌ Last.fm integration error: {e}")
    
    # Show what the full integration would provide
    print("\n4️⃣ Full Integration Capabilities:")
    print("=" * 40)
    
    print("📊 Data Sources Available:")
    print("  🎵 Spotify Personal Data - Your listening history, top artists/tracks")
    print("  🌍 Last.fm Global Trends - Worldwide popularity and scrobble data")
    print("  📚 MusicBrainz Metadata - Artist relationships, discography, origins")
    print("  🎧 AudioDB Profiles - Biographies, career timelines, rich metadata")
    print("  📈 Spotify Charts - Real streaming numbers by country")
    
    print("\n🎯 Analysis Capabilities:")
    print("  • Multi-platform mainstream score calculation")
    print("  • Geographic diversity analysis of your taste")
    print("  • Artist relationship network mapping")
    print("  • Cross-platform trend comparison")
    print("  • Career era and timeline analysis")
    print("  • Genre evolution tracking")
    
    print("\n📊 Visualization Outputs:")
    print("  • Mainstream comparison charts")
    print("  • Geographic diversity maps")
    print("  • Artist relationship networks")
    print("  • Era timeline visualizations")
    print("  • Cross-platform insights dashboard")
    print("  • Data coverage and completeness reports")
    
    print("\n" + "=" * 60)
    print("🚀 Ready to Run Full Analysis!")
    print("=" * 60)
    print("To run the complete multi-source analysis:")
    print("  python -m src.multi_source_main")
    print("")
    print("To test individual sources:")
    print("  python src/musicbrainz_test.py")
    print("  python src/audiodb_test.py")
    print("  python src/spotify_charts_test.py")
    print("")
    print("📝 Note: Some APIs may need additional setup:")
    print("  • AudioDB: Optional API key for premium features")
    print("  • Charts: Web scraping may need updates if site changes")
    print("  • All integrations handle failures gracefully")

if __name__ == "__main__":
    demo_multi_source_capabilities()