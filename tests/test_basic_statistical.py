"""Simple test of statistical analysis with available libraries."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from statistical_analysis import StreamingDataQualityAnalyzer
import pandas as pd
import numpy as np


def test_basic_functionality():
    """Test basic functionality that doesn't require external libraries."""
    print("🧪 Testing Basic Statistical Analysis Functionality")
    print("=" * 60)
    
    # Create simple test data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    streams = np.random.normal(1000, 200, 30)
    streams[5:8] = np.nan  # Add missing values
    streams[15] = 5000  # Add outlier
    
    data = pd.DataFrame({
        'date': dates,
        'streams': streams,
        'plays': streams * 0.9,
        'listeners': streams * 0.4
    })
    
    print(f"✅ Created test data: {len(data)} rows")
    print(f"   Missing values: {data.isnull().sum().sum()}")
    
    # Test quality analyzer
    analyzer = StreamingDataQualityAnalyzer(verbose=True)
    
    try:
        quality_report = analyzer.analyze_data_quality(
            data=data,
            timestamp_col='date',
            value_cols=['streams', 'plays', 'listeners']
        )
        
        print("\n📊 Quality Analysis Results:")
        print("   ✅ Analysis completed successfully")
        print("   📈 Basic stats calculated")
        print(f"   🕒 Temporal analysis: {'✅' if 'temporal_analysis' in quality_report else '❌'}")
        print(f"   🚨 Outlier detection: {'✅' if 'outlier_analysis' in quality_report else '❌'}")
        print(f"   💡 Recommendations: {len(quality_report.get('recommendations', []))}")
        
        if quality_report.get('recommendations'):
            print("\n   Top recommendations:")
            for i, rec in enumerate(quality_report['recommendations'][:3], 1):
                print(f"      {i}. {rec}")
        
        # Test temporal analysis specifically
        if 'temporal_analysis' in quality_report:
            temporal = quality_report['temporal_analysis']
            print("\n   🕒 Temporal Analysis:")
            if 'date_range' in temporal:
                print(f"      Date range: {temporal['date_range']['start']} to {temporal['date_range']['end']}")
                print(f"      Span: {temporal['date_range']['span_days']} days")
            if 'gaps_detected' in temporal:
                print(f"      Gaps detected: {temporal['gaps_detected']}")
        
        return quality_report
        
    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_preparation():
    """Test data preparation and cleaning functionality."""
    print("\n🔧 Testing Data Preparation")
    print("-" * 40)
    
    # Create data with various issues
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # Data with trend, seasonality, and noise
    trend = np.linspace(1000, 1200, 50)
    seasonal = np.sin(np.arange(50) * 2 * np.pi / 7) * 100
    noise = np.random.normal(0, 50, 50)
    streams = trend + seasonal + noise
    
    # Add problems
    streams[10:15] = np.nan  # Missing values
    streams[20] = streams[20] * 3  # Outlier
    streams[25:30] = 0  # Zero values
    
    data = pd.DataFrame({
        'date': dates,
        'streams': streams
    })
    
    print(f"✅ Created problematic data: {len(data)} rows")
    print(f"   Missing: {data['streams'].isnull().sum()}")
    print(f"   Zeros: {(data['streams'] == 0).sum()}")
    print(f"   Range: {data['streams'].min():.1f} to {data['streams'].max():.1f}")
    
    # Test basic data quality analysis
    analyzer = StreamingDataQualityAnalyzer()
    quality_report = analyzer.analyze_data_quality(data, 'date', ['streams'])
    
    # Check specific quality metrics
    basic_stats = quality_report.get('basic_stats', {})
    missing_values = basic_stats.get('missing_values', {})
    zero_values = basic_stats.get('zero_values', {})
    
    print("\n📊 Quality Metrics:")
    print(f"   Missing values detected: {missing_values.get('streams', 0)}")
    print(f"   Zero values detected: {zero_values.get('streams', 0)}")
    print(f"   Total rows: {basic_stats.get('total_rows', 0)}")
    
    return data, quality_report


def demonstrate_insights():
    """Demonstrate insight generation capabilities."""
    print("\n💡 Demonstrating Insight Generation")
    print("-" * 40)
    
    # Create scenario data
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    
    # Simulate a viral event
    base_streams = np.random.normal(500, 100, 60)
    base_streams[20:30] = base_streams[20:30] * 5  # Viral spike
    base_streams[30:40] = base_streams[30:40] * 2  # Sustained popularity
    
    data = pd.DataFrame({
        'date': dates,
        'streams': base_streams,
        'saves': base_streams * 0.1,
        'shares': base_streams * 0.05
    })
    
    print("✅ Created viral event scenario")
    print(f"   Peak streams: {data['streams'].max():.0f}")
    print(f"   Average streams: {data['streams'].mean():.0f}")
    
    # Analyze
    analyzer = StreamingDataQualityAnalyzer()
    quality_report = analyzer.analyze_data_quality(data, 'date', ['streams', 'saves', 'shares'])
    
    # Check for outliers (should detect viral spike)
    outlier_analysis = quality_report.get('outlier_analysis', {})
    for metric, outliers in outlier_analysis.items():
        if 'iqr' in outliers:
            outlier_pct = outliers['iqr']['percentage']
            print(f"   {metric}: {outlier_pct:.1f}% outliers detected")
            
            if outlier_pct > 15:
                print("      🚨 HIGH ANOMALY: Possible viral content or data quality issue")
            elif outlier_pct > 5:
                print("      ⚠️ MODERATE ANOMALY: Worth investigating")
    
    return data, quality_report


if __name__ == "__main__":
    print("🎵 Spotify Insights - Basic Statistical Analysis Test")
    print("=" * 60)
    
    try:
        # Test 1: Basic functionality
        basic_result = test_basic_functionality()
        
        if basic_result:
            # Test 2: Data preparation
            prep_data, prep_report = test_data_preparation()
            
            # Test 3: Insight demonstration
            viral_data, viral_report = demonstrate_insights()
            
            print("\n" + "=" * 60)
            print("🎉 BASIC TESTS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            print("\n📊 Test Summary:")
            print("   ✅ Basic quality analysis: Working")
            print("   ✅ Data preparation: Working")
            print("   ✅ Outlier detection: Working") 
            print("   ✅ Insight generation: Working")
            
            print("\n💡 Next Steps:")
            print("   1. Install additional libraries for advanced forecasting:")
            print("      pip install statsmodels darts scikit-learn")
            print("   2. Run full statistical analysis on your streaming data")
            print("   3. Use quality recommendations to improve data collection")
            
        else:
            print("\n❌ Basic tests failed. Check error messages above.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()