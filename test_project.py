#!/usr/bin/env python3
"""
Comprehensive Audora Project Testing Suite
==========================================

Tests all major components of the Audora music discovery system.
"""

import sys
import traceback
from pathlib import Path

# Add directories to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.extend([
    str(PROJECT_ROOT / "core"),
    str(PROJECT_ROOT / "integrations"), 
    str(PROJECT_ROOT / "analytics"),
    str(PROJECT_ROOT / "visualization"),
])

def test_imports():
    """Test importing all major modules."""
    print("=" * 60)
    print("🧪 TESTING MODULE IMPORTS")
    print("=" * 60)
    
    tests = []
    
    # Test analytics modules
    try:
        from statistical_analysis import StreamingDataQualityAnalyzer
        tests.append(("StreamingDataQualityAnalyzer", "PASS", ""))
    except Exception as e:
        tests.append(("StreamingDataQualityAnalyzer", "FAIL", str(e)))
    
    try:
        from advanced_analytics import MusicTrendAnalytics
        tests.append(("MusicTrendAnalytics", "PASS", ""))
    except Exception as e:
        tests.append(("MusicTrendAnalytics", "FAIL", str(e)))
    
    # Test core modules
    try:
        from data_store import EnhancedMusicDataStore, TrendData, ViralPrediction
        tests.append(("EnhancedMusicDataStore", "PASS", ""))
    except Exception as e:
        tests.append(("EnhancedMusicDataStore", "FAIL", str(e)))
    
    try:
        from config import SecureConfig
        tests.append(("SecureConfig", "PASS", ""))
    except Exception as e:
        tests.append(("SecureConfig", "FAIL", str(e)))
    
    # Test integration modules
    try:
        from spotify_trending import SpotifyTrendingIntegration
        tests.append(("SpotifyTrendingIntegration", "PASS", ""))
    except Exception as e:
        tests.append(("SpotifyTrendingIntegration", "FAIL", str(e)))
    
    try:
        from audiodb_integration import AudioDBIntegration
        tests.append(("AudioDBIntegration", "PASS", ""))
    except Exception as e:
        tests.append(("AudioDBIntegration", "FAIL", str(e)))
    
    # Print results
    for module, status, error in tests:
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"{status_icon} {module:<30} {status}")
        if error and len(error) < 80:
            print(f"   └─ {error}")
    
    passed = sum(1 for _, status, _ in tests if status == "PASS")
    total = len(tests)
    print(f"\n📊 Import Results: {passed}/{total} modules imported successfully")
    return passed, total

def test_basic_functionality():
    """Test basic functionality of working modules."""
    print("\n" + "=" * 60)
    print("🔧 TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    tests = []
    
    # Test MusicTrendAnalytics
    try:
        from advanced_analytics import MusicTrendAnalytics
        analytics = MusicTrendAnalytics()
        
        test_track = {
            'track_name': 'Test Song',
            'artist': 'Test Artist',
            'platform_scores': {'spotify': 85, 'youtube': 92},
            'social_signals': {'mentions': 1500, 'shares': 850},
            'audio_features': {'danceability': 0.8, 'energy': 0.7}
        }
        
        result = analytics.detect_viral_patterns(test_track)
        tests.append(("MusicTrendAnalytics.detect_viral_patterns", "PASS", ""))
    except Exception as e:
        tests.append(("MusicTrendAnalytics.detect_viral_patterns", "FAIL", str(e)[:100]))
    
    # Test pandas/numpy operations
    try:
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            'track': ['A', 'B', 'C'],
            'score': [85, 92, 78]
        })
        
        mean_score = df['score'].mean()
        tests.append(("Pandas/Numpy operations", "PASS", f"Mean score: {mean_score}"))
    except Exception as e:
        tests.append(("Pandas/Numpy operations", "FAIL", str(e)[:100]))
    
    # Test scikit-learn
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        
        scaler = StandardScaler()
        cluster = DBSCAN()
        tests.append(("Scikit-learn imports", "PASS", ""))
    except Exception as e:
        tests.append(("Scikit-learn imports", "FAIL", str(e)[:100]))
    
    # Print results
    for test_name, status, info in tests:
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"{status_icon} {test_name:<40} {status}")
        if info and len(info) < 100:
            print(f"   └─ {info}")
    
    passed = sum(1 for _, status, _ in tests if status == "PASS")
    total = len(tests)
    print(f"\n📊 Functionality Results: {passed}/{total} tests passed")
    return passed, total

def test_data_files():
    """Test availability of data files."""
    print("\n" + "=" * 60)
    print("📁 TESTING DATA FILES")
    print("=" * 60)
    
    data_dir = PROJECT_ROOT / "data"
    
    expected_files = [
        "simple_top_tracks.csv",
        "simple_top_artists.csv", 
        "recently_played.csv",
        "spotify_lastfm_enriched.csv",
        "simple_insights.json"
    ]
    
    results = []
    for filename in expected_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            results.append((filename, "EXISTS", f"{size:,} bytes"))
        else:
            results.append((filename, "MISSING", ""))
    
    # Print results
    for filename, status, info in results:
        status_icon = "✅" if status == "EXISTS" else "❌"
        print(f"{status_icon} {filename:<30} {status} {info}")
    
    existing = sum(1 for _, status, _ in results if status == "EXISTS")
    total = len(results)
    print(f"\n📊 Data Files: {existing}/{total} files found")
    return existing, total

def test_configuration():
    """Test configuration files."""
    print("\n" + "=" * 60)
    print("⚙️ TESTING CONFIGURATION")
    print("=" * 60)
    
    config_dir = PROJECT_ROOT / "config"
    
    config_files = [
        "analytics_config.json",
        "database_config.json", 
        "enhanced_api_config.json",
        "notification_config.json",
        "system_config.json"
    ]
    
    results = []
    for filename in config_files:
        filepath = config_dir / filename
        if filepath.exists():
            try:
                import json
                with open(filepath, 'r') as f:
                    config = json.load(f)
                results.append((filename, "VALID", f"{len(config)} keys"))
            except Exception as e:
                results.append((filename, "INVALID", str(e)[:50]))
        else:
            results.append((filename, "MISSING", ""))
    
    # Print results
    for filename, status, info in results:
        if status == "VALID":
            status_icon = "✅"
        elif status == "INVALID":
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        print(f"{status_icon} {filename:<30} {status} {info}")
    
    valid = sum(1 for _, status, _ in results if status == "VALID")
    total = len(results)
    print(f"\n📊 Config Files: {valid}/{total} files valid")
    return valid, total

def run_full_test_suite():
    """Run the complete test suite."""
    print("🎵 AUDORA PROJECT TEST SUITE")
    print("=" * 60)
    print("Testing comprehensive music discovery system functionality...")
    print()
    
    total_passed = 0
    total_tests = 0
    
    # Run all tests
    passed, tests = test_imports()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_basic_functionality()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_data_files()
    total_passed += passed
    total_tests += tests
    
    passed, tests = test_configuration()
    total_passed += passed
    total_tests += tests
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 60)
    
    percentage = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {total_passed}")
    print(f"Tests Failed: {total_tests - total_passed}")
    print(f"Success Rate: {percentage:.1f}%")
    
    if percentage >= 80:
        print("\n🎉 EXCELLENT! Project is in great shape!")
    elif percentage >= 60:
        print("\n✅ GOOD! Most components are working.")
    elif percentage >= 40:
        print("\n⚠️ FAIR! Some components need attention.")
    else:
        print("\n❌ NEEDS WORK! Many components require fixes.")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS:")
    if total_passed < total_tests:
        print("- Fix module import issues")
        print("- Check configuration files")
        print("- Verify data file integrity")
        print("- Run: python main.py --setup")
    
    print("- Run specific demos: python main.py --demo all")
    print("- Check documentation: docs/QUICK_START.md")
    
    return total_passed, total_tests

if __name__ == "__main__":
    try:
        passed, total = run_full_test_suite()
        sys.exit(0 if passed == total else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)