"""
API Integrations for Enhanced Music Discovery System
===================================================

This module contains integrations with various social media platforms
and music services for trend discovery and data collection.

Supported Platforms:
    - TikTok: Trending videos and audio discovery
    - YouTube: Music trending and search
    - Instagram: Story and reel music trends
    - Twitter: Music-related tweet analysis
    - Spotify: Charts and trending data
    - Last.fm: Global music statistics
    - AudioDB: Detailed music metadata
    - MusicBrainz: Comprehensive music database

Key Components:
    - api_config.py: API configuration and management
    - social_discovery_engine.py: Multi-platform discovery
    - extended_platforms.py: Extended platform support
    - trending_schema.py: Data schemas for trending content
"""

__version__ = "2.0.0"

# Import key integration classes
try:
    from .api_config import APIConfig
    from .social_discovery_engine import SocialMusicDiscoveryEngine

    __all__ = ["APIConfig", "SocialMusicDiscoveryEngine"]
except ImportError:
    __all__ = []
