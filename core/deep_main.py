"""Main script for deep musical taste analysis - age and seasonality patterns."""

from pathlib import Path
from .deep_analysis import create_deep_analysis_report
from .deep_viz import (
    plot_musical_maturity_radar,
    plot_generational_evolution,
    plot_seasonal_mood_matrix,
    plot_decade_preference_timeline,
    plot_energy_circadian_rhythm,
    create_musical_age_analysis_chart
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_deep_insights():
    """Generate comprehensive deep analysis of musical taste patterns."""
    print("🔬 Starting Deep Musical Analysis...")
    print("   This will analyze how your genre tastes reflect your age and seasonality")
    print()
    
    # Generate analysis report
    report = create_deep_analysis_report()
    
    if 'error' in report:
        print(f"❌ Error: {report['error']}")
        return
    
    df = report['dataset']
    generational = report['generational_patterns']
    seasonal = report['seasonal_patterns']
    maturity = report['maturity_analysis']
    
    print()
    print("🎨 Creating deep insight visualizations...")
    
    generated_files = []
    
    try:
        # 1. Musical Maturity Radar
        print("   📊 Mapping your musical sophistication...")
        maturity_path = plot_musical_maturity_radar(maturity)
        generated_files.append(maturity_path)
        print(f"      ✅ {maturity_path.name}")
        
        # 2. Era Evolution
        if 'era_preference' in generational and not generational['era_preference'].empty:
            print("   🕰️ Tracking your era evolution...")
            era_path = plot_generational_evolution(generational['era_preference'])
            generated_files.append(era_path)
            print(f"      ✅ {era_path.name}")
        
        # 3. Seasonal Mood Matrix
        if 'seasonal_preferences' in seasonal and not seasonal['seasonal_preferences'].empty:
            print("   🌅 Analyzing seasonal mood patterns...")
            seasonal_path = plot_seasonal_mood_matrix(seasonal['seasonal_preferences'])
            generated_files.append(seasonal_path)
            print(f"      ✅ {seasonal_path.name}")
        
        # 4. Decade Timeline  
        if 'decade_preference' in generational and not generational['decade_preference'].empty:
            print("   📅 Creating decade preference timeline...")
            decade_path = plot_decade_preference_timeline(generational['decade_preference'])
            generated_files.append(decade_path)
            print(f"      ✅ {decade_path.name}")
        
        # 5. Energy Circadian Rhythm
        if 'hourly_energy' in seasonal and not seasonal['hourly_energy'].empty:
            print("   🕐 Mapping your energy circadian rhythm...")
            circadian_path = plot_energy_circadian_rhythm(seasonal['hourly_energy'])
            generated_files.append(circadian_path)
            print(f"      ✅ {circadian_path.name}")
        
        # 6. Musical Age Analysis
        print("   🎂 Calculating your musical age...")
        age_path = create_musical_age_analysis_chart(df)
        generated_files.append(age_path)
        print(f"      ✅ {age_path.name}")
        
    except Exception as e:
        print(f"   ❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("🧬 DEEP ANALYSIS INSIGHTS:")
    print("=" * 50)
    
    # Musical Maturity Report
    print(f"🎓 Musical Maturity Score: {maturity['maturity_score']:.2f}/1.0")
    print(f"   • Genre Diversity: {maturity['genre_diversity']:.1f} (higher = more eclectic)")
    print(f"   • Era Diversity: {maturity['era_diversity']} different musical eras")
    print(f"   • Classic Music Preference: {maturity['classic_ratio']:.1%}")
    print(f"   • Underground Score: {maturity['underground_score']:.1%} (lower popularity)")
    print()
    
    # Genre Insights
    top_genres = df['genre'].value_counts().head(5)
    print("🎭 Top 5 Genres in Your Profile:")
    for i, (genre, count) in enumerate(top_genres.items(), 1):
        print(f"   {i}. {genre} ({count} mentions)")
    print()
    
    # Era Analysis
    era_breakdown = df.groupby('era')['weight'].sum().sort_values(ascending=False)
    print("🕰️ Musical Era Preferences:")
    for era, weight in era_breakdown.items():
        percentage = (weight / era_breakdown.sum()) * 100
        print(f"   • {era}: {percentage:.1f}%")
    print()
    
    # Temporal Patterns
    recent_data = df[df['type'] == 'recent_play']
    if not recent_data.empty:
        seasonal_breakdown = recent_data.groupby('season').size()
        if not seasonal_breakdown.empty:
            print("🌅 Recent Seasonal Listening (last 50 tracks):")
            for season, count in seasonal_breakdown.items():
                print(f"   • {season}: {count} tracks")
    
    print()
    print("📊 Analysis Summary:")
    print(f"   • Total data points analyzed: {len(df):,}")
    print(f"   • Unique genres discovered: {df['genre'].nunique()}")
    print(f"   • Musical eras represented: {df['era'].nunique()}")
    print(f"   • Visualizations created: {len(generated_files)}")
    print()
    print(f"📁 Detailed data saved to: {report['data_file'].name}")
    print("🎨 Charts saved to the 'data' folder")
    
    return generated_files


if __name__ == "__main__":
    generate_deep_insights()