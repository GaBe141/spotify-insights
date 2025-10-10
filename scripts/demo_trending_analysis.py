"""
Demo script to test trending schema with real Spotify data and generate visualizations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from spotify_trending_integration import SpotifyTrendingIntegration


def main():
    print("🎵 Testing Trending Schema with Real Spotify Data")
    print("=" * 55)

    # Initialize trending integration
    integration = SpotifyTrendingIntegration()

    # Load and analyze data
    print("📊 Loading Spotify data...")
    data = integration.load_and_process_data()
    print(
        f"   📈 Processed {data.get('files_processed', 0)} files with {data.get('items_added', 0)} items"
    )

    # Perform trending analysis
    print("📈 Performing trending analysis...")
    trending_insights = integration.analyze_trending_insights()

    # Show analysis summary
    category_analysis = trending_insights.get("category_analysis", {})
    viral_content = trending_insights.get("viral_content", [])
    emerging_trends = trending_insights.get("emerging_trends", [])
    predictions = trending_insights.get("predictions", {})

    print("\n📊 Analysis Summary:")
    print(f"   🎯 Categories analyzed: {len(category_analysis)}")
    for category, data in category_analysis.items():
        print(f"      {category}: {data['total_items']} items")
        directions = data.get("directions", {})
        if directions:
            top_direction = max(directions.items(), key=lambda x: x[1])
            print(f"         Primary trend: {top_direction[0]} ({top_direction[1]} items)")

    print(f"   🔥 Viral content detected: {len(viral_content)}")
    if viral_content:
        top_viral = viral_content[0]
        print(f"      Top viral: {top_viral['name']} ({top_viral['growth_rate']:.1f}% growth)")

    print(f"   🚀 Emerging trends: {len(emerging_trends)}")
    if emerging_trends:
        top_emerging = emerging_trends[0]
        print(
            f"      Top emerging: {top_emerging['name']} (momentum: {top_emerging['momentum']:.2f})"
        )

    print(f"   🔮 Predictions generated: {len(predictions)}")

    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    report_path = integration.create_trending_report()
    print(f"   📄 Report saved to: {report_path}")

    # Try to create visualizations (if libraries available)
    print("\n🎨 Creating visualizations...")
    try:
        from src.trending_viz import TrendingVisualizationEngine

        # Create visualization directory
        viz_dir = Path("data/trending_visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualization engine
        viz_engine = TrendingVisualizationEngine()

        # Create basic visualizations
        visualizations_created = []

        try:
            # Timeline dashboard
            timeline_path = viz_dir / "trending_timeline.png"
            result = viz_engine.plot_trending_timeline(trending_insights, str(timeline_path))
            if "complete" in result:
                visualizations_created.append("Timeline dashboard")
        except Exception as e:
            print(f"   ⚠️  Timeline visualization: {e}")

        try:
            # Interactive dashboard
            interactive_path = viz_dir / "trending_dashboard.html"
            result = viz_engine.create_interactive_trending_dashboard(
                trending_insights, str(interactive_path)
            )
            if "successfully" in result:
                visualizations_created.append("Interactive dashboard")
        except Exception as e:
            print(f"   ⚠️  Interactive dashboard: {e}")

        try:
            # Prediction charts
            if predictions:
                prediction_path = viz_dir / "trend_predictions.png"
                result = viz_engine.create_trend_prediction_chart(predictions, str(prediction_path))
                if "complete" in result:
                    visualizations_created.append("Prediction charts")
        except Exception as e:
            print(f"   ⚠️  Prediction charts: {e}")

        print(f"   ✅ Created {len(visualizations_created)} visualizations:")
        for viz in visualizations_created:
            print(f"      📊 {viz}")

        if not visualizations_created:
            print("   ℹ️  Install matplotlib/plotly for visualizations:")
            print("      pip install matplotlib seaborn plotly")

    except ImportError:
        print("   ℹ️  Visualization module not available")
        print("      Install dependencies: pip install matplotlib seaborn plotly")

    # Show top findings
    print("\n🎯 Top Trending Insights:")

    if viral_content:
        print("   🔥 Most Viral Content:")
        for i, item in enumerate(viral_content[:3], 1):
            print(f"      {i}. {item['name']} ({item['growth_rate']:.1f}% growth)")

    if emerging_trends:
        print("   🚀 Top Emerging Trends:")
        for i, item in enumerate(emerging_trends[:3], 1):
            print(f"      {i}. {item['name']} (momentum: {item['momentum']:.2f})")

    # Show trending by category
    for category, data in category_analysis.items():
        top_trending = data.get("top_trending", [])
        if top_trending:
            print(f"   📈 Top {category.title()} Trends:")
            for i, item in enumerate(top_trending[:3], 1):
                direction_emoji = {
                    "rising": "📈",
                    "falling": "📉",
                    "viral": "🔥",
                    "stable": "➡️",
                }.get(item["direction"], "📊")
                print(f"      {i}. {item['name']} {direction_emoji} {item['growth_rate']:+.1f}%")

    print("\n✅ Trending analysis complete!")
    print(f"📄 Full report: {report_path}")

    return report_path


if __name__ == "__main__":
    main()
