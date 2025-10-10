"""
Configuration management for social media APIs.
Handles API keys, rate limiting, and platform-specific settings.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class APIConfig:
    """Configuration for individual API platforms."""

    platform: str
    api_key: str = ""
    secret_key: str = ""
    access_token: str = ""
    refresh_token: str = ""

    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000

    # Usage tracking
    last_request_time: datetime = field(default_factory=datetime.now)
    requests_today: int = 0
    requests_this_hour: int = 0
    requests_this_minute: int = 0

    # Status
    enabled: bool = True
    last_error: str = ""
    error_count: int = 0


class SocialAPIManager:
    """Manages API configurations and rate limiting for all platforms."""

    def __init__(self, config_file: str = "config/social_apis.json"):
        self.config_file = Path(config_file)
        self.configs: dict[str, APIConfig] = {}
        self.load_configs()

    def load_configs(self):
        """Load API configurations from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)

                for platform, config_data in data.items():
                    self.configs[platform] = APIConfig(platform=platform, **config_data)
            except Exception as e:
                print(f"Error loading API configs: {e}")
        else:
            self._create_default_configs()

    def _create_default_configs(self):
        """Create default configuration template."""
        default_configs = {
            "tiktok": {
                "api_key": "",
                "secret_key": "",
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "enabled": False,
            },
            "youtube": {
                "api_key": "",
                "requests_per_minute": 100,
                "requests_per_hour": 10000,
                "requests_per_day": 1000000,
                "enabled": False,
            },
            "twitter": {
                "api_key": "",
                "secret_key": "",
                "access_token": "",
                "requests_per_minute": 300,
                "requests_per_hour": 300,
                "requests_per_day": 300,
                "enabled": False,
            },
            "instagram": {
                "access_token": "",
                "requests_per_minute": 60,
                "requests_per_hour": 200,
                "requests_per_day": 200,
                "enabled": False,
            },
            "reddit": {
                "api_key": "",
                "secret_key": "",
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 1000,
                "enabled": False,
            },
            "tumblr": {
                "api_key": "",
                "secret_key": "",
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 5000,
                "enabled": False,
            },
        }

        for platform, config_data in default_configs.items():
            self.configs[platform] = APIConfig(platform=platform, **config_data)

        self.save_configs()

    def save_configs(self):
        """Save configurations to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        config_data = {}
        for platform, config in self.configs.items():
            config_data[platform] = {
                "api_key": config.api_key,
                "secret_key": config.secret_key,
                "access_token": config.access_token,
                "refresh_token": config.refresh_token,
                "requests_per_minute": config.requests_per_minute,
                "requests_per_hour": config.requests_per_hour,
                "requests_per_day": config.requests_per_day,
                "enabled": config.enabled,
                "last_error": config.last_error,
                "error_count": config.error_count,
            }

        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)

    def get_config(self, platform: str) -> APIConfig | None:
        """Get configuration for a platform."""
        return self.configs.get(platform.lower())

    def set_api_key(
        self,
        platform: str,
        api_key: str,
        secret_key: str = "",
        access_token: str = "",
        refresh_token: str = "",
    ):
        """Set API credentials for a platform."""
        platform = platform.lower()

        if platform not in self.configs:
            self.configs[platform] = APIConfig(platform=platform)

        config = self.configs[platform]
        config.api_key = api_key
        config.secret_key = secret_key
        config.access_token = access_token
        config.refresh_token = refresh_token
        config.enabled = bool(api_key or access_token)

        self.save_configs()

    def can_make_request(self, platform: str) -> bool:
        """Check if we can make a request without hitting rate limits."""
        config = self.get_config(platform)
        if not config or not config.enabled:
            return False

        now = datetime.now()

        # Reset counters if enough time has passed
        if now.date() != config.last_request_time.date():
            config.requests_today = 0

        if now.hour != config.last_request_time.hour:
            config.requests_this_hour = 0

        if now.minute != config.last_request_time.minute:
            config.requests_this_minute = 0

        # Check rate limits
        if config.requests_per_day > 0 and config.requests_today >= config.requests_per_day:
            return False

        if config.requests_per_hour > 0 and config.requests_this_hour >= config.requests_per_hour:
            return False

        if (
            config.requests_per_minute > 0
            and config.requests_this_minute >= config.requests_per_minute
        ):
            return False

        return True

    def record_request(self, platform: str, success: bool = True, error: str = ""):
        """Record a request for rate limiting tracking."""
        config = self.get_config(platform)
        if not config:
            return

        now = datetime.now()

        # Reset counters if needed
        if now.date() != config.last_request_time.date():
            config.requests_today = 0

        if now.hour != config.last_request_time.hour:
            config.requests_this_hour = 0

        if now.minute != config.last_request_time.minute:
            config.requests_this_minute = 0

        # Increment counters
        config.requests_today += 1
        config.requests_this_hour += 1
        config.requests_this_minute += 1
        config.last_request_time = now

        # Record errors
        if not success:
            config.error_count += 1
            config.last_error = error

            # Disable if too many errors
            if config.error_count >= 10:
                config.enabled = False
                print(f"⚠️ Disabled {platform} API due to repeated errors")

        self.save_configs()

    def get_status_report(self) -> dict[str, Any]:
        """Get status report for all configured APIs."""
        report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "platforms": {},
            "summary": {
                "total_platforms": len(self.configs),
                "enabled_platforms": 0,
                "disabled_platforms": 0,
                "platforms_with_errors": 0,
            },
        }

        platforms_dict: dict[str, Any] = report["platforms"]
        summary_dict: dict[str, Any] = report["summary"]

        for platform, config in self.configs.items():
            platform_status = {
                "enabled": config.enabled,
                "has_credentials": bool(config.api_key or config.access_token),
                "requests_today": config.requests_today,
                "requests_this_hour": config.requests_this_hour,
                "requests_this_minute": config.requests_this_minute,
                "rate_limits": {
                    "per_minute": config.requests_per_minute,
                    "per_hour": config.requests_per_hour,
                    "per_day": config.requests_per_day,
                },
                "error_count": config.error_count,
                "last_error": config.last_error,
                "can_make_request": self.can_make_request(platform),
            }

            platforms_dict[platform] = platform_status

            # Update summary
            if config.enabled:
                summary_dict["enabled_platforms"] += 1
            else:
                summary_dict["disabled_platforms"] += 1

            if config.error_count > 0:
                summary_dict["platforms_with_errors"] += 1

        return report


def setup_api_credentials():
    """Interactive setup for API credentials."""
    print("🔧 Social Media API Setup")
    print("=" * 50)

    manager = SocialAPIManager()

    print("\nThis will help you configure API credentials for music discovery.")
    print("You can skip any platform by pressing Enter without typing anything.\n")

    # TikTok
    print("📱 TikTok Research API:")
    print("   Get credentials at: https://developers.tiktok.com/")
    tiktok_key = input("   API Key: ").strip()
    tiktok_secret = input("   Secret Key: ").strip()

    if tiktok_key:
        manager.set_api_key("tiktok", tiktok_key, tiktok_secret)
        print("   ✅ TikTok configured")

    # YouTube
    print("\n🎥 YouTube Data API v3:")
    print("   Get credentials at: https://console.developers.google.com/")
    youtube_key = input("   API Key: ").strip()

    if youtube_key:
        manager.set_api_key("youtube", youtube_key)
        print("   ✅ YouTube configured")

    # Twitter
    print("\n🐦 Twitter API v2:")
    print("   Get credentials at: https://developer.twitter.com/")
    twitter_bearer = input("   Bearer Token: ").strip()

    if twitter_bearer:
        manager.set_api_key("twitter", access_token=twitter_bearer)
        print("   ✅ Twitter configured")

    # Instagram
    print("\n📸 Instagram Basic Display API:")
    print("   Get credentials at: https://developers.facebook.com/")
    instagram_token = input("   Access Token: ").strip()

    if instagram_token:
        manager.set_api_key("instagram", access_token=instagram_token)
        print("   ✅ Instagram configured")

    # Reddit
    print("\n🤖 Reddit API:")
    print("   Get credentials at: https://www.reddit.com/prefs/apps")
    reddit_key = input("   Client ID: ").strip()
    reddit_secret = input("   Client Secret: ").strip()

    if reddit_key:
        manager.set_api_key("reddit", reddit_key, reddit_secret)
        print("   ✅ Reddit configured")

    # Tumblr
    print("\n📝 Tumblr API:")
    print("   Get credentials at: https://www.tumblr.com/oauth/apps")
    tumblr_key = input("   Consumer Key: ").strip()
    tumblr_secret = input("   Consumer Secret: ").strip()

    if tumblr_key:
        manager.set_api_key("tumblr", tumblr_key, tumblr_secret)
        print("   ✅ Tumblr configured")

    print("\n" + "=" * 50)
    print("🎯 Configuration Complete!")

    # Show status
    status = manager.get_status_report()
    enabled_count = status["summary"]["enabled_platforms"]
    total_count = status["summary"]["total_platforms"]

    print(f"✅ {enabled_count}/{total_count} platforms configured")
    print(f"📁 Config saved to: {manager.config_file}")

    return manager


if __name__ == "__main__":
    # Run interactive setup
    manager = setup_api_credentials()

    # Show detailed status
    print("\n📊 Platform Status:")
    status = manager.get_status_report()

    for platform, details in status["platforms"].items():
        status_icon = "✅" if details["enabled"] else "❌"
        cred_icon = "🔑" if details["has_credentials"] else "🚫"

        print(
            f"   {status_icon} {platform.title()}: {cred_icon} "
            f"({details['requests_today']} requests today)"
        )

    print("\n💡 Next Steps:")
    print("   1. Run social_discovery_engine.py to start discovering music")
    print("   2. Check rate limits regularly to avoid API suspensions")
    print("   3. Monitor error counts and update credentials if needed")
