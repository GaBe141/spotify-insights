"""Demonstration script showing statistical analysis on real Spotify data."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
from src.statistical_analysis import StreamingDataQualityAnalyzer


def load_existing_data():
    """Load existing Spotify data files."""
    data_dir = Path("data")
    loaded_files = {}

    # List of data files to try
    data_files = [
        "simple_top_tracks.csv",
        "simple_top_artists.csv",
        "recently_played.csv",
        "lastfm_global_tracks.csv",
        "spotify_lastfm_enriched.csv",
    ]

    print("📂 Loading existing data files...")

    for filename in data_files:
        file_path = data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                loaded_files[filename] = df
                print(f"   ✅ {filename}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"   ❌ {filename}: Error - {e}")
        else:
            print(f"   ⚠️ {filename}: Not found")

    return loaded_files


def analyze_spotify_track_data(df):
    """Analyze Spotify top tracks data."""
    print(f"\n🎵 Analyzing Spotify track data ({len(df)} tracks)...")

    # Identify numeric columns for analysis
    numeric_cols = []
    potential_numeric = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    for col in potential_numeric:
        if col in df.columns and df[col].dtype in ["int64", "float64"]:
            numeric_cols.append(col)

    if not numeric_cols:
        print("   ⚠️ No numeric columns found for analysis")
        return None

    print(f"   📊 Analyzing {len(numeric_cols)} numeric features: {', '.join(numeric_cols)}")

    # Create a synthetic timestamp column for temporal analysis
    df_analysis = df.copy()
    if "date" not in df_analysis.columns:
        # Create synthetic dates (as if tracks were added daily)
        start_date = pd.to_datetime("2024-01-01")
        df_analysis["date"] = pd.date_range(start_date, periods=len(df), freq="D")

    # Run quality analysis
    analyzer = StreamingDataQualityAnalyzer()
    quality_report = analyzer.analyze_data_quality(
        data=df_analysis, timestamp_col="date", value_cols=numeric_cols
    )

    # Print insights
    print("\n   📈 Quality Analysis Results:")

    basic_stats = quality_report.get("basic_stats", {})
    print(f"      Total tracks analyzed: {basic_stats.get('total_rows', 0)}")

    missing_values = basic_stats.get("missing_values", {})
    total_missing = sum(missing_values.values())
    print(f"      Missing values: {total_missing}")

    # Feature insights
    print("\n   🎶 Music Feature Insights:")
    for col in numeric_cols[:5]:  # Show top 5 features
        if col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            print(f"      {col}: μ={col_mean:.2f}, σ={col_std:.2f}")

    # Recommendations
    recommendations = quality_report.get("recommendations", [])
    if recommendations:
        print("\n   💡 Data Quality Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. {rec}")

    return quality_report


def analyze_listening_history(df):
    """Analyze recently played listening history."""
    print(f"\n🎧 Analyzing listening history ({len(df)} plays)...")

    if "played_at" not in df.columns:
        print("   ⚠️ No 'played_at' column found for temporal analysis")
        return None

    try:
        # Convert to datetime
        df["played_at"] = pd.to_datetime(df["played_at"])
        df = df.sort_values("played_at")

        # Create daily aggregations
        df["date"] = df["played_at"].dt.date
        daily_stats = (
            df.groupby("date")
            .agg(
                {
                    "track_name": "count",  # plays per day
                    "artist_name": "nunique",  # unique artists per day
                }
            )
            .reset_index()
        )

        daily_stats.columns = ["date", "daily_plays", "unique_artists"]
        daily_stats["date"] = pd.to_datetime(daily_stats["date"])

        print(f"   📅 Aggregated to {len(daily_stats)} days of listening data")
        print(
            f"   📊 Date range: {daily_stats['date'].min().date()} to {daily_stats['date'].max().date()}"
        )

        # Add listening patterns
        daily_stats["variety_ratio"] = daily_stats["unique_artists"] / daily_stats["daily_plays"]

        # Run quality analysis
        analyzer = StreamingDataQualityAnalyzer()
        quality_report = analyzer.analyze_data_quality(
            data=daily_stats,
            timestamp_col="date",
            value_cols=["daily_plays", "unique_artists", "variety_ratio"],
        )

        print("\n   📈 Listening Pattern Analysis:")
        print(f"      Average daily plays: {daily_stats['daily_plays'].mean():.1f}")
        print(f"      Average unique artists/day: {daily_stats['unique_artists'].mean():.1f}")
        print(f"      Average variety ratio: {daily_stats['variety_ratio'].mean():.3f}")

        # Check for outlier listening days
        outlier_analysis = quality_report.get("outlier_analysis", {})
        if "daily_plays" in outlier_analysis and "iqr" in outlier_analysis["daily_plays"]:
            outlier_pct = outlier_analysis["daily_plays"]["iqr"]["percentage"]
            if outlier_pct > 10:
                print(
                    f"      🚨 UNUSUAL LISTENING DETECTED: {outlier_pct:.1f}% of days show abnormal activity"
                )
            elif outlier_pct > 5:
                print(f"      ⚠️ Some irregular listening days detected: {outlier_pct:.1f}%")

        return quality_report, daily_stats

    except Exception as e:
        print(f"   ❌ Error analyzing listening history: {e}")
        return None


def generate_streaming_insights(data_files):
    """Generate comprehensive insights from available data."""
    print("\n🧠 Generating Streaming Insights...")

    insights = {
        "data_sources_analyzed": len(data_files),
        "quality_insights": [],
        "music_insights": [],
        "listening_insights": [],
    }

    # Analyze each data source
    for filename, df in data_files.items():
        print(f"\n   Analyzing {filename}...")

        if "track" in filename.lower():
            # Track data analysis
            if "popularity" in df.columns:
                avg_popularity = df["popularity"].mean()
                insights["music_insights"].append(
                    f"Average track popularity: {avg_popularity:.1f}/100"
                )

            if "energy" in df.columns and "valence" in df.columns:
                avg_energy = df["energy"].mean()
                avg_valence = df["valence"].mean()

                if avg_energy > 0.7 and avg_valence > 0.7:
                    mood = "High-energy, positive music preference"
                elif avg_energy > 0.7 and avg_valence < 0.3:
                    mood = "High-energy, darker music preference"
                elif avg_energy < 0.3 and avg_valence > 0.7:
                    mood = "Calm, positive music preference"
                else:
                    mood = "Balanced music mood preference"

                insights["music_insights"].append(mood)

        elif "recently_played" in filename.lower():
            # Listening behavior analysis
            result = analyze_listening_history(df)
            if result and len(result) == 2:
                quality_report, daily_stats = result

                # Listening consistency
                play_std = daily_stats["daily_plays"].std()
                play_mean = daily_stats["daily_plays"].mean()
                consistency = play_std / play_mean if play_mean > 0 else 0

                if consistency < 0.3:
                    insights["listening_insights"].append("Very consistent listening habits")
                elif consistency < 0.6:
                    insights["listening_insights"].append("Moderately consistent listening habits")
                else:
                    insights["listening_insights"].append("Highly variable listening habits")

                # Music discovery
                avg_variety = daily_stats["variety_ratio"].mean()
                if avg_variety > 0.8:
                    insights["listening_insights"].append(
                        "High music discovery rate - explores many artists"
                    )
                elif avg_variety > 0.5:
                    insights["listening_insights"].append("Moderate music discovery rate")
                else:
                    insights["listening_insights"].append(
                        "Focused listening - tends to repeat favorite artists"
                    )

    return insights


def create_analysis_report(data_files, insights):
    """Create a comprehensive analysis report."""
    print("\n📊 Creating Analysis Report...")

    report = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "streaming_data_quality_analysis",
        "summary": {
            "data_sources": len(data_files),
            "total_records": sum(len(df) for df in data_files.values()),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        },
        "insights": insights,
        "data_details": {},
    }

    # Add details for each data source
    for filename, df in data_files.items():
        report["data_details"][filename] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "numeric_columns": [col for col in df.columns if df[col].dtype in ["int64", "float64"]],
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict("records") if len(df) > 0 else [],
        }

    # Save report
    report_path = Path("data") / "streaming_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"   ✅ Report saved to {report_path}")
    return report


def main():
    """Main demonstration function."""
    print("🎵 Spotify Insights - Statistical Analysis Demonstration")
    print("=" * 70)

    # Load existing data
    data_files = load_existing_data()

    if not data_files:
        print("\n❌ No data files found. Please run the basic data collection first:")
        print("   python demo_multi_source.py")
        return

    print(f"\n✅ Loaded {len(data_files)} data files")

    # Analyze each data file
    analysis_results = {}

    for filename, df in data_files.items():
        print(f"\n{'='*50}")
        print(f"ANALYZING: {filename}")
        print(f"{'='*50}")

        if "track" in filename.lower() and "popularity" in df.columns:
            result = analyze_spotify_track_data(df)
            analysis_results[filename] = result
        elif "recently_played" in filename.lower():
            result = analyze_listening_history(df)
            analysis_results[filename] = result
        else:
            # Generic analysis for other files
            print("   📊 Basic file info:")
            print(f"      Rows: {len(df)}")
            print(f"      Columns: {len(df.columns)}")
            print(
                f"      Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}"
            )

    # Generate insights
    insights = generate_streaming_insights(data_files)

    # Create comprehensive report
    create_analysis_report(data_files, insights)

    # Display summary
    print(f"\n{'='*70}")
    print("🎯 ANALYSIS SUMMARY")
    print(f"{'='*70}")

    print("\n📊 Data Overview:")
    print(f"   Sources analyzed: {insights['data_sources_analyzed']}")
    print(f"   Total records: {sum(len(df) for df in data_files.values())}")

    if insights["music_insights"]:
        print("\n🎶 Music Insights:")
        for insight in insights["music_insights"]:
            print(f"   • {insight}")

    if insights["listening_insights"]:
        print("\n🎧 Listening Insights:")
        for insight in insights["listening_insights"]:
            print(f"   • {insight}")

    if insights["quality_insights"]:
        print("\n🔍 Quality Insights:")
        for insight in insights["quality_insights"]:
            print(f"   • {insight}")

    print("\n💡 Next Steps:")
    print("   1. Review detailed report: data/streaming_analysis_report.json")
    print("   2. Install advanced libraries for forecasting: pip install statsmodels darts")
    print("   3. Run advanced forecasting analysis")
    print("   4. Use insights to optimize your music discovery")

    print("\n✅ Statistical analysis demonstration complete!")


if __name__ == "__main__":
    main()
