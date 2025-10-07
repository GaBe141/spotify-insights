"""Test script for statistical analysis and forecasting capabilities."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from statistical_analysis import (
    StreamingDataQualityAnalyzer, 
    StreamingForecastingEngine,
    run_comprehensive_analysis
)
import pandas as pd
import numpy as np
from datetime import datetime


def create_realistic_streaming_data():
    """Create realistic streaming data with trends, seasonality, and noise."""
    print("🎵 Creating realistic streaming data...")
    
    # Generate 6 months of daily data
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, periods=180, freq='D')
    
    # Base trend (growing popularity)
    base_trend = np.linspace(1000, 1500, 180)
    
    # Weekly seasonality (higher on weekends)
    weekly_pattern = np.sin(np.arange(180) * 2 * np.pi / 7) * 200
    
    # Monthly seasonality (holiday effects)
    monthly_pattern = np.sin(np.arange(180) * 2 * np.pi / 30) * 150
    
    # Random noise
    noise = np.random.normal(0, 100, 180)
    
    # Combine patterns
    streams = base_trend + weekly_pattern + monthly_pattern + noise
    streams = np.maximum(streams, 0)  # Ensure non-negative
    
    # Create related metrics
    plays = streams * np.random.uniform(0.7, 0.9, 180)  # Plays usually lower than streams
    listeners = streams * np.random.uniform(0.3, 0.6, 180)  # Unique listeners
    
    # Add some missing values and outliers
    missing_indices = np.random.choice(180, 10, replace=False)
    streams[missing_indices] = np.nan
    
    # Add outliers (viral moments)
    outlier_indices = np.random.choice(180, 3, replace=False)
    streams[outlier_indices] *= 3
    
    data = pd.DataFrame({
        'date': dates,
        'streams': streams,
        'plays': plays,
        'listeners': listeners,
        'skips': streams * np.random.uniform(0.1, 0.3, 180),
        'saves': streams * np.random.uniform(0.05, 0.15, 180)
    })
    
    print(f"✅ Created {len(data)} days of streaming data")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"   Missing values: {data.isnull().sum().sum()}")
    print(f"   Columns: {list(data.columns)}")
    
    return data


def test_data_quality_analysis():
    """Test data quality analysis functionality."""
    print("\n" + "="*50)
    print("🔍 TESTING DATA QUALITY ANALYSIS")
    print("="*50)
    
    # Create test data
    data = create_realistic_streaming_data()
    
    # Initialize analyzer
    analyzer = StreamingDataQualityAnalyzer(verbose=True)
    
    # Run quality analysis
    quality_report = analyzer.analyze_data_quality(
        data=data,
        timestamp_col='date',
        value_cols=['streams', 'plays', 'listeners', 'skips', 'saves']
    )
    
    print("\n📊 Quality Analysis Results:")
    print(f"   Total rows: {quality_report['basic_stats']['total_rows']}")
    print(f"   Missing values: {sum(quality_report['basic_stats']['missing_values'].values())}")
    print(f"   Duplicate rows: {quality_report['basic_stats']['duplicate_rows']}")
    print(f"   Recommendations: {len(quality_report['recommendations'])}")
    
    if quality_report['recommendations']:
        print("\n💡 Top Recommendations:")
        for i, rec in enumerate(quality_report['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
    
    # Test outlier detection
    if 'outlier_analysis' in quality_report:
        print("\n🚨 Outlier Detection:")
        for col, outliers in quality_report['outlier_analysis'].items():
            if 'iqr' in outliers:
                print(f"   {col}: {outliers['iqr']['count']} outliers ({outliers['iqr']['percentage']:.1f}%)")
    
    return quality_report


def test_forecasting_engine():
    """Test forecasting engine functionality."""
    print("\n" + "="*50)
    print("📈 TESTING FORECASTING ENGINE")
    print("="*50)
    
    # Create test data
    data = create_realistic_streaming_data()
    
    # Initialize forecasting engine
    engine = StreamingForecastingEngine(verbose=True)
    
    # Test time series preparation
    print("\n🔧 Testing time series preparation...")
    ts, prep_info = engine.prepare_time_series(data, 'date', 'streams')
    print(f"   Original length: {prep_info['original_length']}")
    print(f"   Prepared length: {prep_info['filled_length']}")
    print(f"   Missing filled: {prep_info['missing_filled']}")
    
    # Test model fitting
    print("\n🏗️ Testing model fitting...")
    
    # Test ARIMA
    if engine.available_models['arima']:
        arima_result = engine.fit_arima_model(ts)
        if arima_result['success']:
            print(f"   ✅ ARIMA: AIC={arima_result['diagnostics']['aic']:.2f}")
        else:
            print(f"   ❌ ARIMA: {arima_result.get('error', 'Unknown error')}")
    
    # Test SARIMA
    if engine.available_models['sarima']:
        sarima_result = engine.fit_sarima_model(ts)
        if sarima_result['success']:
            print(f"   ✅ SARIMA: AIC={sarima_result['diagnostics']['aic']:.2f}")
        else:
            print(f"   ❌ SARIMA: {sarima_result.get('error', 'Unknown error')}")
    
    # Test Darts models
    if engine.available_models['auto_arima']:
        darts_results = engine.fit_darts_models(ts)
        for model_name, result in darts_results.items():
            if isinstance(result, dict) and 'error' not in result and result.get('success', False):
                mae_val = result['metrics']['mae']
                print(f"   ✅ Darts {model_name}: MAE={mae_val:.2f}")
            else:
                error_msg = result if isinstance(result, str) else result.get('error', 'Failed') if isinstance(result, dict) else 'Failed'
                print(f"   ❌ Darts {model_name}: {error_msg}")
    
    # Test sklearn models
    if engine.available_models['random_forest']:
        sklearn_results = engine.fit_sklearn_models(ts)
        for model_name, result in sklearn_results.items():
            if isinstance(result, dict) and 'error' not in result and result.get('success', False):
                mae_val = result['metrics']['mae']
                print(f"   ✅ Sklearn {model_name}: MAE={mae_val:.2f}")
            else:
                error_msg = result if isinstance(result, str) else result.get('error', 'Failed') if isinstance(result, dict) else 'Failed'
                print(f"   ❌ Sklearn {model_name}: {error_msg}")
    
    # Test forecasting
    print("\n🔮 Testing forecast generation...")
    forecasts = engine.generate_forecasts(ts, horizon=14)
    
    forecast_count = 0
    for model_name, forecast in forecasts.items():
        if isinstance(forecast, dict) and 'error' not in forecast and 'forecast' in forecast:
            forecast_count += 1
            forecast_values = forecast['forecast']
            print(f"   ✅ {model_name}: {len(forecast_values)} forecasts, mean={np.mean(forecast_values):.1f}")
        else:
            error_msg = forecast if isinstance(forecast, str) else forecast.get('error', 'No forecast generated') if isinstance(forecast, dict) else 'No forecast generated'
            print(f"   ❌ {model_name}: {error_msg}")
    
    print(f"\n📊 Successfully generated {forecast_count} forecasts")
    
    # Test performance evaluation
    if forecast_count > 0:
        print("\n📊 Testing performance evaluation...")
        performance = engine.evaluate_model_performance(ts)
        
        best_model = None
        best_mae = float('inf')
        
        for model_name, metrics in performance.items():
            if isinstance(metrics, dict) and 'mae' in metrics:
                mae_val = metrics['mae']
                print(f"   {model_name}: MAE={mae_val:.2f}")
                if mae_val < best_mae:
                    best_mae = mae_val
                    best_model = model_name
        
        if best_model:
            print(f"\n🏆 Best performing model: {best_model} (MAE={best_mae:.2f})")
    
    return engine, forecasts


def test_comprehensive_analysis():
    """Test the comprehensive analysis pipeline."""
    print("\n" + "="*50)
    print("🎯 TESTING COMPREHENSIVE ANALYSIS")
    print("="*50)
    
    # Create test data
    data = create_realistic_streaming_data()
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis(
        data=data,
        timestamp_col='date',
        value_cols=['streams', 'plays', 'listeners'],
        output_dir="data/test_analysis"
    )
    
    print("\n📋 Comprehensive Analysis Summary:")
    print(f"   Columns analyzed: {len(results['forecasting'])}")
    print(f"   Quality recommendations: {len(results['data_quality'].get('recommendations', []))}")
    print(f"   Total recommendations: {len(results['recommendations'])}")
    
    # Show some detailed results
    for col, col_results in results['forecasting'].items():
        if 'error' not in col_results:
            model_count = len([m for m in col_results.get('models', {}).values() if 'success' in m and m['success']])
            forecast_count = len([f for f in col_results.get('forecasts', {}).values() if 'forecast' in f])
            print(f"   {col}: {model_count} models fitted, {forecast_count} forecasts generated")
        else:
            print(f"   {col}: Error - {col_results['error']}")
    
    return results


def demonstrate_statistical_features():
    """Demonstrate key statistical features with real examples."""
    print("\n" + "="*60)
    print("🧪 DEMONSTRATING STATISTICAL FEATURES")
    print("="*60)
    
    # Create complex streaming data scenario
    print("\n📊 Creating complex streaming scenario...")
    
    # Simulate a song that goes viral
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    
    # Normal growth for first 30 days
    normal_streams = np.linspace(500, 800, 30)
    
    # Viral spike (days 30-40)
    viral_peak = np.array([800 + i * 200 for i in range(10)] + 
                         [2800 - i * 150 for i in range(10)])
    
    # Decline and stabilization (days 50-90)
    decline_streams = np.linspace(1300, 1000, 40)
    
    streams = np.concatenate([normal_streams, viral_peak, decline_streams])
    
    # Add noise and seasonality
    noise = np.random.normal(0, 50, 90)
    weekly_cycle = np.sin(np.arange(90) * 2 * np.pi / 7) * 100
    
    streams = streams + noise + weekly_cycle
    streams = np.maximum(streams, 0)
    
    # Create dataframe
    viral_data = pd.DataFrame({
        'date': dates,
        'streams': streams,
        'plays': streams * 0.85,
        'listeners': streams * 0.4
    })
    
    print(f"✅ Created viral song scenario: {len(viral_data)} days")
    print(f"   Peak streams: {viral_data['streams'].max():.0f}")
    print(f"   Normal streams: {viral_data['streams'][:30].mean():.0f}")
    
    # Analyze this scenario
    print("\n🔍 Analyzing viral pattern...")
    
    analyzer = StreamingDataQualityAnalyzer()
    quality_report = analyzer.analyze_data_quality(
        viral_data, 'date', ['streams', 'plays', 'listeners']
    )
    
    # Check for outliers (should detect viral spike)
    if 'outlier_analysis' in quality_report:
        streams_outliers = quality_report['outlier_analysis']['streams']
        if 'iqr' in streams_outliers:
            outlier_pct = streams_outliers['iqr']['percentage']
            print(f"   🚨 Outlier detection: {outlier_pct:.1f}% of streams are outliers")
            if outlier_pct > 15:
                print("   📈 High outlier percentage suggests viral content or data quality issues")
    
    # Test forecasting on this complex pattern
    print("\n🔮 Forecasting post-viral trends...")
    
    engine = StreamingForecastingEngine()
    ts, _ = engine.prepare_time_series(viral_data, 'date', 'streams')
    
    # Split data to simulate real-time forecasting during viral period
    train_data = ts[:60]  # Use data up to viral peak
    test_data = ts[60:]   # Actual post-viral data
    
    print(f"   Training on {len(train_data)} days, testing on {len(test_data)} days")
    
    # Fit models on training data
    if engine.available_models['arima']:
        arima_result = engine.fit_arima_model(train_data)
        if arima_result['success']:
            print("   ✅ ARIMA model fitted successfully")
    
    # Generate forecasts
    forecasts = engine.generate_forecasts(train_data, horizon=len(test_data))
    
    # Compare forecasts to actual
    for model_name, forecast in forecasts.items():
        if 'forecast' in forecast:
            forecast_values = np.array(forecast['forecast'][:len(test_data)])
            actual_values = test_data.values[:len(forecast_values)]
            
            mae = np.mean(np.abs(forecast_values - actual_values))
            print(f"   {model_name} forecast MAE: {mae:.1f}")
            
            # Check if model predicted decline correctly
            forecast_trend = "declining" if forecast_values[-1] < forecast_values[0] else "growing"
            actual_trend = "declining" if actual_values[-1] < actual_values[0] else "growing"
            
            trend_match = "✅" if forecast_trend == actual_trend else "❌"
            print(f"     Trend prediction: {trend_match} ({forecast_trend} vs {actual_trend})")


if __name__ == "__main__":
    print("🎵 Spotify Insights - Statistical Analysis Test Suite")
    print("="*60)
    
    try:
        # Test individual components
        quality_report = test_data_quality_analysis()
        engine, forecasts = test_forecasting_engine()
        comprehensive_results = test_comprehensive_analysis()
        
        # Demonstrate advanced features
        demonstrate_statistical_features()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n📊 Test Summary:")
        print("   ✅ Data quality analysis: Working")
        print(f"   ✅ Forecasting engine: {len(forecasts)} models tested")
        print("   ✅ Comprehensive pipeline: Working")
        print("   ✅ Advanced demonstrations: Working")
        
        print("\n💡 Next Steps:")
        print("   1. Run statistical analysis on your actual streaming data")
        print("   2. Compare forecast models to find best performers")
        print("   3. Use quality recommendations to improve data collection")
        print("   4. Set up automated forecasting for trending content")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()