"""Test script for AudioDB integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from audiodb_integration import get_audiodb_client


def test_audiodb():
    """Test AudioDB API integration."""
    print("🎧 Testing AudioDB API Integration")
    print("=" * 40)

    client = get_audiodb_client()
    test_artists = ["The Beatles", "Radiohead", "Ed Sheeran"]

    for artist_name in test_artists:
        print(f"\n🔍 Searching for: {artist_name}")

        # Test artist details
        details = client.get_artist_details(artist_name)
        if details and details["name"]:
            print(f"✅ Found: {details['name']}")

            if details["biography"]:
                bio_preview = (
                    details["biography"][:100] + "..."
                    if len(details["biography"]) > 100
                    else details["biography"]
                )
                print(f"   Biography: {bio_preview}")

            print(f"   Genre: {details['genre'] or 'Unknown'}")
            print(f"   Country: {details['country'] or 'Unknown'}")
            print(f"   Formed: {details['formed_year'] or 'Unknown'}")

            if details["website"]:
                print(f"   Website: {details['website']}")

            # Test getting albums
            print("   📀 Getting albums...")
            albums = client.get_artist_albums(artist_name)
            if albums:
                print(f"   Found {len(albums)} albums:")
                for album in albums[:3]:  # Show first 3
                    year = album["year"] or "Unknown"
                    print(f"     - {album['album_name']} ({year})")
            else:
                print("   No albums found")

        else:
            print(f"❌ Could not find detailed info for {artist_name}")

    print("\n📈 Testing career analysis...")
    try:
        career_analysis = client.analyze_artist_careers(test_artists)
        careers_df = career_analysis["careers"]
        albums_df = career_analysis["albums"]

        print("✅ Career analysis complete:")
        print(f"   Analyzed {len(careers_df)} artist careers")
        print(f"   Found {len(albums_df)} albums total")

        if not careers_df.empty:
            print("   Career highlights:")
            for _, row in careers_df.iterrows():
                if row["formed_year"] and row["career_length_years"]:
                    print(f"     - {row['artist_name']}: {row['career_length_years']} years active")

    except Exception as e:
        print(f"❌ Error in career analysis: {e}")

    print("\n✅ AudioDB integration test complete!")


if __name__ == "__main__":
    test_audiodb()
