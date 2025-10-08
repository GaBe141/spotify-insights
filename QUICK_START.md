# 🚀 Quick Start Guide - Music Discovery System

## 5-Minute Setup

### 1. Install Dependencies
```bash
python setup.py
```

### 2. Configure APIs (Choose One Method)

#### Option A: Interactive Setup (Recommended)
```bash
python src/api_config.py
```

#### Option B: Environment File
```bash
# Copy template and edit
cp .env.template .env
# Edit .env with your API keys
```

### 3. Run the Application
```bash
python src/music_discovery_app.py
```

## Essential APIs to Get Started

### Free APIs (Start Here)
1. **YouTube Data API v3** - Free, most important
   - Go to: https://console.developers.google.com/
   - Create project → Enable YouTube Data API v3 → Get API key

2. **Reddit API** - Free, great for underground trends
   - Go to: https://www.reddit.com/prefs/apps
   - Create app → Note client_id and client_secret

3. **Twitter API v2** - Free tier available
   - Go to: https://developer.twitter.com/
   - Apply for developer account → Get Bearer Token

### Premium APIs (For Full Power)
4. **TikTok Research API** - Most valuable for Gen Z trends
   - Apply at: https://developers.tiktok.com/
   - Requires approval for research access

## First Discovery Run

```bash
# Quick test run
python -c "
import asyncio
from src.music_discovery_app import ComprehensiveMusicDiscoveryApp

async def test_discovery():
    app = ComprehensiveMusicDiscoveryApp()
    results = await app.run_full_discovery('US')
    print(f'Found {results.get(\"total_songs\", 0)} songs!')

asyncio.run(test_discovery())
"
```

## What You'll Get

- **Real-time music trends** from TikTok, YouTube, Instagram, Twitter
- **Underground discoveries** from Reddit, Tumblr, SoundCloud  
- **Viral predictions** using statistical analysis
- **Cross-platform insights** showing where songs are trending
- **Automated reports** saved to your data/ folder

## Need Help?

- Check `MUSIC_DISCOVERY_README.md` for full documentation
- Run `python src/api_config.py` to test your API setup
- Look at sample outputs in the `data/` folder

**Ready to discover the next viral hit? 🎵**