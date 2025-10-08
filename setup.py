#!/usr/bin/env python3
"""
Setup script for the Comprehensive Music Discovery System.
Installs dependencies, configures APIs, and runs initial tests.
"""

import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    dependencies = [
        "aiohttp",
        "aiofiles", 
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-learn",
        "statsmodels",
        "darts",
        "requests",
        "python-dotenv"
    ]
    
    for package in dependencies:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"   ⚠️ Failed to install {package}")


def create_directory_structure():
    """Create necessary directories."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "config",
        "data",
        "data/trending",
        "data/trending_visualizations", 
        "data/reports",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {directory}/")


def create_config_template():
    """Create configuration file template."""
    print("\n⚙️ Creating configuration template...")
    
    config_template = {
        "social_apis": {
            "tiktok": {
                "api_key": "",
                "secret_key": "",
                "enabled": False,
                "notes": "Get credentials at https://developers.tiktok.com/"
            },
            "youtube": {
                "api_key": "",
                "enabled": False,
                "notes": "Get credentials at https://console.developers.google.com/"
            },
            "twitter": {
                "bearer_token": "",
                "enabled": False,
                "notes": "Get credentials at https://developer.twitter.com/"
            },
            "instagram": {
                "access_token": "",
                "enabled": False,
                "notes": "Get credentials at https://developers.facebook.com/"
            },
            "reddit": {
                "client_id": "",
                "client_secret": "",
                "enabled": False,
                "notes": "Get credentials at https://www.reddit.com/prefs/apps"
            },
            "tumblr": {
                "consumer_key": "",
                "consumer_secret": "",
                "enabled": False,
                "notes": "Get credentials at https://www.tumblr.com/oauth/apps"
            }
        },
        "spotify": {
            "client_id": "",
            "client_secret": "",
            "enabled": False,
            "notes": "Get credentials at https://developer.spotify.com/"
        },
        "settings": {
            "default_region": "US",
            "discovery_interval_hours": 4,
            "max_requests_per_hour": 1000,
            "enable_continuous_monitoring": False,
            "save_raw_data": True,
            "verbose_logging": True
        }
    }
    
    config_path = Path("config/api_template.json")
    with open(config_path, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"   ✅ Template created at {config_path}")
    print("   📝 Edit this file with your API credentials")


def create_env_template():
    """Create .env template file."""
    print("\n🔐 Creating environment template...")
    
    env_template = """# Social Media API Credentials
# Copy this file to .env and fill in your credentials

# TikTok Research API
TIKTOK_API_KEY=your_tiktok_api_key_here
TIKTOK_SECRET=your_tiktok_secret_here

# YouTube Data API v3
YOUTUBE_API_KEY=your_youtube_api_key_here

# Twitter API v2
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Instagram Basic Display API
INSTAGRAM_ACCESS_TOKEN=your_instagram_access_token_here

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

# Tumblr API
TUMBLR_CONSUMER_KEY=your_tumblr_consumer_key_here
TUMBLR_CONSUMER_SECRET=your_tumblr_consumer_secret_here

# Spotify API (for existing integration)
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here

# Optional: SoundCloud, Discord, etc.
SOUNDCLOUD_CLIENT_ID=your_soundcloud_client_id_here
DISCORD_BOT_TOKEN=your_discord_bot_token_here
"""
    
    env_path = Path(".env.template")
    with open(env_path, 'w') as f:
        f.write(env_template)
    
    print(f"   ✅ Template created at {env_path}")
    print("   📝 Copy to .env and add your credentials")


def run_initial_tests():
    """Run initial system tests."""
    print("\n🧪 Running initial tests...")
    
    try:
        # Test trending schema
        print("   Testing trending schema...")
        from src.trending_schema import create_sample_trending_data
        create_sample_trending_data()
        print("   ✅ Trending schema working")
        
        # Test statistical analysis
        print("   Testing statistical analysis...")
        from src.statistical_analysis import StreamingDataQualityAnalyzer
        StreamingDataQualityAnalyzer()
        print("   ✅ Statistical analysis working")
        
        # Test API configuration
        print("   Testing API configuration...")
        from src.api_config import SocialAPIManager
        SocialAPIManager()
        print("   ✅ API configuration working")
        
        print("\n✅ All initial tests passed!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. 📝 Configure your API credentials:")
    print("   • Edit config/api_template.json with your API keys")
    print("   • Or copy .env.template to .env and fill in credentials")
    
    print("\n2. 🔑 Get API Credentials:")
    print("   • TikTok: https://developers.tiktok.com/")
    print("   • YouTube: https://console.developers.google.com/")
    print("   • Twitter: https://developer.twitter.com/")
    print("   • Instagram: https://developers.facebook.com/")
    print("   • Reddit: https://www.reddit.com/prefs/apps")
    print("   • Tumblr: https://www.tumblr.com/oauth/apps")
    
    print("\n3. 🚀 Start discovering music:")
    print("   python src/music_discovery_app.py")
    
    print("\n4. 🧪 Run tests:")
    print("   python test_statistical_analysis.py")
    print("   python src/trending_schema.py")
    
    print("\n5. 📊 Generate reports:")
    print("   python src/social_discovery_engine.py")
    
    print("\n💡 Pro Tips:")
    print("   • Start with 1-2 APIs to test the system")
    print("   • Monitor rate limits to avoid API suspensions")
    print("   • Use continuous monitoring for real-time insights")
    print("   • Check data/ folder for generated reports")
    
    print("\n🆘 Need Help?")
    print("   • Check README.md for detailed documentation")
    print("   • Run components individually to debug issues")
    print("   • Ensure all dependencies are properly installed")


def main():
    """Main setup function."""
    print("🎵 Spotify Insights - Music Discovery System Setup")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Create directory structure
    create_directory_structure()
    
    # Step 4: Create configuration templates
    create_config_template()
    create_env_template()
    
    # Step 5: Run initial tests
    tests_passed = run_initial_tests()
    
    # Step 6: Print next steps
    print_next_steps()
    
    if not tests_passed:
        print("\n⚠️ Some tests failed. The system may still work, but check for issues.")
    
    print("\n🎯 Setup complete! Ready to discover music trends! 🎵")


if __name__ == "__main__":
    main()