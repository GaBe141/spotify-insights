"""Test script for MusicBrainz integration."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.append(str(Path(__file__).parent))

    from musicbrainz_integration import get_musicbrainz_client

    def test_musicbrainz():
        """Test MusicBrainz API integration."""
        print("🎵 Testing MusicBrainz API Integration")
        print("=" * 40)

        client = get_musicbrainz_client()
        test_artists = ["The Beatles", "Radiohead", "Taylor Swift"]

        for artist_name in test_artists:
            print(f"\n🔍 Searching for: {artist_name}")

            # Test artist search
            artist = client.search_artist(artist_name)
            if artist:
                print(f"✅ Found: {artist['name']}")
                print(f"   MBID: {artist['id']}")
                print(f"   Country: {artist.get('country', 'Unknown')}")
                print(f"   Type: {artist.get('type', 'Unknown')}")

                # Test getting related artists
                print("   🔗 Finding related artists...")
                related = client.get_related_artists(artist["id"])
                if related:
                    print(f"   Found {len(related)} related artists:")
                    for rel in related[:3]:  # Show first 3
                        print(f"     - {rel['name']} ({rel['relationship_type']})")
                else:
                    print("   No related artists found")
            else:
                print(f"❌ Could not find {artist_name}")

        print("\n✅ MusicBrainz integration test complete!")

    test_musicbrainz()
