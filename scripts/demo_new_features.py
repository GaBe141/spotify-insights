#!/usr/bin/env python3
"""
🎵 AUDORA NEW FEATURES SHOWCASE
==============================

Comprehensive demonstration of all new Audora features:
1. 🎭 Mood-Based Playlist Generator
2. ⏰ Temporal Listening Analysis
3. 🚀 Enhanced Viral Predictions
4. 📊 Integrated Analytics Dashboard

This demo brings together all the latest music discovery
and analytics capabilities in one interactive experience.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.extend(
    [
        str(PROJECT_ROOT / "analytics"),
        str(PROJECT_ROOT / "core"),
    ]
)

from enhanced_viral_prediction import EnhancedViralPredictor  # noqa: E402
from mood_playlist_generator import MoodCategory, MoodPlaylistGenerator  # noqa: E402
from temporal_analysis import TemporalAnalyzer  # noqa: E402


class AudoraFeatureShowcase:
    """Comprehensive showcase of all new Audora features."""

    def __init__(self):
        """Initialize all feature modules."""
        self.data_dir = PROJECT_ROOT / "data"

        # Initialize all components
        self.mood_generator = MoodPlaylistGenerator(str(self.data_dir))
        self.temporal_analyzer = TemporalAnalyzer(str(self.data_dir))
        self.viral_predictor = EnhancedViralPredictor(str(self.data_dir))

        print("🎵 AUDORA - NEW FEATURES SHOWCASE")
        print("=" * 70)
        print("Demonstrating cutting-edge music discovery and analytics")
        print("=" * 70)

    def demo_mood_playlists(self):
        """Demonstrate mood-based playlist generation."""
        print("\n" + "🎭 " * 25)
        print("FEATURE #1: MOOD-BASED PLAYLIST GENERATOR")
        print("🎭 " * 25)

        print("\n📌 What it does:")
        print("   Analyzes audio features (valence, energy, tempo, danceability)")
        print("   to automatically categorize your music into mood-based playlists")

        # Load tracks
        self.mood_generator.load_tracks()

        # Generate playlists
        print("\n🎨 Generating mood playlists...")
        self.mood_generator.generate_mood_playlists(min_score=60.0, max_tracks_per_mood=5)

        # Show a few example playlists
        print("\n📋 SAMPLE PLAYLISTS:")

        featured_moods = [
            MoodCategory.HAPPY_UPBEAT,
            MoodCategory.CHILL_RELAXED,
            MoodCategory.FOCUS,
            MoodCategory.PARTY_DANCE,
        ]

        for mood in featured_moods:
            self.mood_generator.print_playlist(mood, max_tracks=3)

        # Statistics
        stats = self.mood_generator.get_mood_statistics()
        print("\n📊 GENERATION STATISTICS:")
        print(f"   Total playlists created: {stats['total_moods']}")
        print(f"   Total tracks analyzed: {stats['total_tracks']}")
        print(f"   Average tracks per playlist: {stats['total_tracks'] / stats['total_moods']:.1f}")

        print("\n✅ Mood playlists generated successfully!")
        print("   Export available in JSON format for integration")

    def demo_temporal_analysis(self):
        """Demonstrate temporal listening pattern analysis."""
        print("\n" + "⏰ " * 25)
        print("FEATURE #2: TEMPORAL LISTENING ANALYSIS")
        print("⏰ " * 25)

        print("\n📌 What it does:")
        print("   Analyzes when you listen to music throughout the day, week, and month")
        print("   Identifies patterns in your listening habits and preferences")

        # Load listening history
        self.temporal_analyzer.load_listening_history()

        # Generate report
        print("\n📊 Analyzing temporal patterns...")
        report = self.temporal_analyzer.generate_comprehensive_report()

        # Print insights
        self.temporal_analyzer.print_insights(report)

        # Additional highlights
        if report["hourly_patterns"]:
            hourly = report["hourly_patterns"]
            print("\n🎯 KEY INSIGHTS:")
            print(f"   • Peak listening time: {hourly['peak_listening_hour']}:00")
            print(
                f"   • Most active period: {hourly['most_active_period'].replace('_', ' ').title()}"
            )

        if report["listening_streaks"]:
            streaks = report["listening_streaks"]
            if streaks["longest_streak"] > 1:
                print(
                    f"   • Longest listening streak: {streaks['longest_streak']} consecutive days! 🔥"
                )
            print(f"   • Total active listening days: {streaks['total_listening_days']}")

        print("\n✅ Temporal analysis complete!")
        print("   Full report exported to data/reports/")

    def demo_enhanced_viral_prediction(self):
        """Demonstrate enhanced viral prediction system."""
        print("\n" + "🚀 " * 25)
        print("FEATURE #3: ENHANCED VIRAL PREDICTION")
        print("🚀 " * 25)

        print("\n📌 What it does:")
        print("   Advanced ML-powered predictions with momentum tracking, acceleration")
        print("   analysis, cross-platform velocity, and peak timing forecasts")

        # Create sample tracks with realistic data
        print("\n🎯 Analyzing viral potential...")

        sample_tracks = [
            {
                "track_name": "Vampire",
                "platform_scores": {"spotify": 92, "tiktok": 95, "youtube": 88, "instagram": 90},
                "social_signals": {"mentions": 25000, "shares": 5500, "comments": 1200},
                "audio_features": {"danceability": 0.72, "energy": 0.85, "valence": 0.45},
                "historical_data": [
                    {"timestamp": datetime.now() - timedelta(days=7), "value": 75},
                    {"timestamp": datetime.now() - timedelta(days=5), "value": 82},
                    {"timestamp": datetime.now() - timedelta(days=3), "value": 87},
                    {"timestamp": datetime.now() - timedelta(days=1), "value": 92},
                ],
            },
            {
                "track_name": "Flowers",
                "platform_scores": {"spotify": 88, "tiktok": 85, "youtube": 86},
                "social_signals": {"mentions": 18000, "shares": 3200, "comments": 850},
                "audio_features": {"danceability": 0.85, "energy": 0.78, "valence": 0.82},
            },
            {
                "track_name": "Anti-Hero",
                "platform_scores": {"spotify": 95, "tiktok": 82, "youtube": 90, "instagram": 88},
                "social_signals": {"mentions": 32000, "shares": 6800, "comments": 1500},
                "audio_features": {"danceability": 0.68, "energy": 0.65, "valence": 0.58},
            },
        ]

        # Batch prediction
        results = self.viral_predictor.batch_predict(sample_tracks)

        print("\n📊 TOP VIRAL PREDICTIONS:")
        print("=" * 70)

        for i, (track, metrics) in enumerate(results, 1):
            print(f"\n#{i}. {track['track_name']}")
            print(f"{'─' * 70}")
            print(f"   Viral Score:         {metrics.viral_score:>6.1f}/100")
            print(f"   Prediction Confidence: {metrics.confidence:>6.1%}")
            print(f"   Momentum:            {metrics.momentum:>6.1f}")
            print(f"   Acceleration:        {metrics.acceleration:>+6.1f}")
            print(f"   Cross-Platform Velocity: {metrics.cross_platform_velocity:>6.1f}")
            print(
                f"   Peak in:             {metrics.peak_eta_days:>3} days (±{(metrics.peak_confidence_interval[1] - metrics.peak_confidence_interval[0])//2} days)"
            )
            print(f"   Risk Level:          {metrics.risk_level}")
            print(f"\n   💡 {metrics.recommendation}")

        print("\n✅ Enhanced viral predictions complete!")
        print("   Investment-grade analytics for music discovery")

    def demo_integrated_insights(self):
        """Demonstrate how features work together."""
        print("\n" + "🎯 " * 25)
        print("FEATURE #4: INTEGRATED INSIGHTS")
        print("🎯 " * 25)

        print("\n📌 What it does:")
        print("   Combines all features to provide comprehensive music intelligence")

        # Get mood playlist stats
        mood_stats = self.mood_generator.get_mood_statistics()

        # Get temporal analysis
        temporal_report = self.temporal_analyzer.generate_comprehensive_report()

        print("\n🔮 COMPREHENSIVE MUSIC PROFILE:")
        print("=" * 70)

        # Music diversity
        print("\n🎨 Music Diversity:")
        print(f"   • {mood_stats['total_moods']} distinct mood categories identified")
        print(f"   • {mood_stats['total_tracks']} tracks classified")

        # Listening behavior
        if temporal_report["data_range"]["total_records"] > 0:
            print("\n⏰ Listening Behavior:")
            if temporal_report["hourly_patterns"]:
                peak_period = temporal_report["hourly_patterns"]["most_active_period"]
                print(f"   • Most active listening: {peak_period.replace('_', ' ').title()}")

            if temporal_report["weekly_patterns"]:
                weekly = temporal_report["weekly_patterns"]
                weekday_avg = weekly.get("weekday_avg", 0)
                weekend_avg = weekly.get("weekend_avg", 0)
                if weekday_avg > weekend_avg:
                    print(
                        f"   • Listening style: More active on weekdays ({weekday_avg:.1f} vs {weekend_avg:.1f} plays/day)"
                    )
                else:
                    print(
                        f"   • Listening style: Weekend music enthusiast ({weekend_avg:.1f} vs {weekday_avg:.1f} plays/day)"
                    )

        # Mood-Time correlations
        print("\n🌟 Mood-Time Insights:")
        if temporal_report.get("genre_time_patterns") and temporal_report[
            "genre_time_patterns"
        ].get("by_time_of_day"):
            # Suggest mood playlists for different times
            recommendations = {
                "Morning": MoodCategory.MORNING_ENERGY,
                "Afternoon": MoodCategory.FOCUS,
                "Evening": MoodCategory.HAPPY_UPBEAT,
                "Night": MoodCategory.LATE_NIGHT,
            }

            for time_period, mood in recommendations.items():
                playlist = self.mood_generator.get_playlist(mood)
                if playlist:
                    print(f"   • {time_period}: Perfect for {mood.value} ({len(playlist)} tracks)")

        # Overall recommendations
        print("\n💡 PERSONALIZED RECOMMENDATIONS:")
        print("   Based on your combined listening patterns and preferences:")
        print("   1. 🎧 Try Focus playlists during your peak afternoon hours")
        print("   2. 🌅 Start mornings with Morning Energy tracks")
        print("   3. 🌙 Wind down evenings with Late Night Vibes")
        print("   4. 🚀 Monitor viral predictions for early discovery of trending hits")

        print("\n✅ Integrated insights generated!")
        print("   Your complete music intelligence profile")

    def run_full_showcase(self):
        """Run the complete feature showcase."""
        try:
            # Demo 1: Mood Playlists
            self.demo_mood_playlists()

            input("\n\n⏸️  Press Enter to continue to Temporal Analysis...")

            # Demo 2: Temporal Analysis
            self.demo_temporal_analysis()

            input("\n\n⏸️  Press Enter to continue to Viral Predictions...")

            # Demo 3: Enhanced Viral Predictions
            self.demo_enhanced_viral_prediction()

            input("\n\n⏸️  Press Enter to see Integrated Insights...")

            # Demo 4: Integrated Insights
            self.demo_integrated_insights()

            # Final summary
            print("\n\n" + "=" * 70)
            print("🎉 AUDORA NEW FEATURES SHOWCASE COMPLETE!")
            print("=" * 70)
            print("\n✨ New Capabilities Added:")
            print("   ✅ Mood-Based Playlist Generation (8 mood categories)")
            print("   ✅ Temporal Listening Analysis (hour/day/week patterns)")
            print("   ✅ Enhanced Viral Predictions (ML-powered with confidence intervals)")
            print("   ✅ Integrated Music Intelligence (combining all features)")

            print("\n🚀 Next Steps:")
            print("   • Explore individual modules: python analytics/mood_playlist_generator.py")
            print("   • Run temporal analysis: python analytics/temporal_analysis.py")
            print("   • Test viral predictions: python analytics/enhanced_viral_prediction.py")
            print("   • Check exported data in: data/playlists/ and data/reports/")

            print("\n💎 Your music discovery platform is now even more powerful!")

        except KeyboardInterrupt:
            print("\n\n👋 Showcase interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during showcase: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main entry point."""
    showcase = AudoraFeatureShowcase()
    showcase.run_full_showcase()


if __name__ == "__main__":
    main()
