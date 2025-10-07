"""Test script for Spotify Charts integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from spotify_charts_integration import get_spotify_charts_client

def test_spotify_charts():
    """Test Spotify Charts scraping integration."""
    print("📈 Testing Spotify Charts Integration")
    print("=" * 40)
    
    client = get_spotify_charts_client()
    
    print("\n🌍 Testing global top 200...")
    try:
        global_charts = client.get_top_200_daily("global")
        
        if not global_charts.empty:
            print(f"✅ Successfully fetched {len(global_charts)} chart entries")
            print("\nTop 5 tracks:")
            if len(global_charts) >= 5:
                for i in range(5):
                    row = global_charts.iloc[i]
                    print(f"   {row['position']}. {row['track_name']} - {row['artist_name']}")
                    if row['streams'] > 0:
                        print(f"      Streams: {row['streams']:,}")
        else:
            print("⚠️ No chart data retrieved - this might be due to website changes")
            print("   Charts scraping requires adapting to current website structure")
    except Exception as e:
        print(f"❌ Error fetching charts: {e}")
        print("   Note: Web scraping may need updates based on site changes")
    
    print("\n🇺🇸 Testing US charts...")
    try:
        us_charts = client.get_top_200_daily("us")
        if not us_charts.empty:
            print(f"✅ Successfully fetched US charts: {len(us_charts)} entries")
        else:
            print("⚠️ No US chart data retrieved")
    except Exception as e:
        print(f"❌ Error fetching US charts: {e}")
    
    print("\n📊 Testing multi-country comparison...")
    try:
        countries = ['global', 'us']
        multi_charts = client.get_multi_country_comparison(countries)
        if not multi_charts.empty:
            print(f"✅ Successfully fetched multi-country data: {len(multi_charts)} total entries")
            countries_found = multi_charts['country'].unique()
            print(f"   Countries in dataset: {list(countries_found)}")
        else:
            print("⚠️ No multi-country data retrieved")
    except Exception as e:
        print(f"❌ Error with multi-country comparison: {e}")
    
    print("\n" + "=" * 40)
    print("📝 Note: Spotify Charts integration uses web scraping.")
    print("   If tests fail, the website structure may have changed.")
    print("   Charts data provides real streaming numbers when working.")
    print("✅ Spotify Charts integration test complete!")

if __name__ == "__main__":
    test_spotify_charts()