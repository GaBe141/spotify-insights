# Spotify Advanced Analytics Platform

A comprehensive analytics platform for Spotify streaming data featuring statistical analysis, trending pattern detection, forecasting models, and advanced visualizations.

## 🎯 Overview

This platform provides a complete suite of tools for analyzing music streaming data with capabilities ranging from basic data quality assessment to advanced trend prediction and viral content detection.

## 🚀 Key Features

### 🔍 Statistical Analysis & Data Quality
- **Data Quality Assessment**: Missing value analysis, outlier detection, temporal consistency
- **Statistical Models**: ARIMA, SARIMA, Exponential Smoothing, Auto-ARIMA
- **Advanced Analytics**: Isolation Forest outlier detection, Z-score analysis, IQR filtering
- **Performance Metrics**: MAE, RMSE, MAPE for model evaluation

### 📈 Trending Pattern Analysis
- **Real-time Trend Detection**: Viral content identification, emerging trends analysis
- **Multi-category Support**: Artists, tracks, albums, playlists
- **Trend Classification**: Rising, falling, stable, viral, emerging, declining, volatile
- **Predictive Analytics**: Growth rate prediction, momentum analysis

### 🎨 Advanced Visualizations
- **Static Charts**: Matplotlib/Seaborn-based trending timelines, statistical reports
- **Interactive Dashboards**: Plotly-powered interactive trending analysis
- **Comprehensive Reports**: JSON exports, human-readable summaries

### 🔄 Multi-source Integration
- **Spotify Data**: Top tracks, artists, recently played, popularity metrics
- **Last.fm Integration**: Global charts, play counts, listener statistics
- **AudioDB/MusicBrainz**: Metadata enrichment (planned)
- **Extensible Architecture**: Easy integration of new data sources

## 📁 Project Structure

```
spotify-insights/
├── src/                          # Core modules
│   ├── statistical_analysis.py   # Statistical analysis engine
│   ├── statistical_viz.py        # Statistical visualizations
│   ├── trending_schema.py        # Trending pattern analysis
│   ├── trending_viz.py           # Trending visualizations
│   ├── auth.py                   # Spotify API authentication
│   ├── fetch.py                  # Data collection utilities
│   └── ...                      # Additional modules
├── data/                         # Data storage
│   ├── trending/                 # Trending analysis reports
│   ├── trending_visualizations/  # Generated charts
│   └── *.csv                    # Raw data files
├── notebooks/                    # Jupyter analysis notebooks
├── demo_*.py                     # Demonstration scripts
├── complete_platform_demo.py     # Full platform showcase
└── requirements.txt              # Python dependencies
```

## 🛠 Installation

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Basic Installation
```bash
git clone <repository-url>
cd spotify-insights
pip install -r requirements.txt
```

### Optional Advanced Dependencies
```bash
# For full statistical analysis capabilities
pip install statsmodels scikit-learn darts

# For advanced visualizations
pip install matplotlib seaborn plotly

# For extended data analysis
pip install numpy pandas jupyter
```

## 🚀 Quick Start

### 1. Basic Trending Analysis
```bash
python demo_trending_analysis.py
```

### 2. Complete Platform Demo
```bash
python complete_platform_demo.py
```

### 3. Statistical Analysis Only
```bash
python demo_statistical_analysis.py
```

### 4. Multi-source Integration
```bash
python demo_multi_source.py
```

## 📊 Usage Examples

### Statistical Analysis
```python
from src.statistical_analysis import StreamingDataQualityAnalyzer, StreamingForecastingEngine
import pandas as pd

# Load data
df = pd.read_csv('data/simple_top_tracks.csv')

# Analyze data quality
analyzer = StreamingDataQualityAnalyzer()
quality_report = analyzer.analyze_data_quality(df, 'rank', ['popularity'])

# Generate forecasts
forecaster = StreamingForecastingEngine()
forecasts = forecaster.forecast_multiple_models(df['popularity'].values, 7)
```

### Trending Analysis
```python
from spotify_trending_integration import SpotifyTrendingIntegration

# Initialize integration
integration = SpotifyTrendingIntegration()

# Load and process data
integration.load_and_process_data()

# Analyze trends
insights = integration.analyze_trending_insights()

# Generate report
report_path = integration.create_trending_report()
```

### Visualizations
```python
from src.trending_viz import TrendingVisualizationEngine

# Create visualization engine
viz_engine = TrendingVisualizationEngine()

# Generate trending timeline
viz_engine.plot_trending_timeline(trending_data, 'timeline.png')

# Create interactive dashboard
viz_engine.create_interactive_trending_dashboard(trending_data, 'dashboard.html')
```

## 📈 Trending Schema

The platform uses a comprehensive trending classification system:

### Trend Directions
- **🔥 Viral**: Explosive growth (>100% increase)
- **📈 Rising**: Significant upward trend (>20% increase)
- **📉 Falling**: Declining trend (<-20% decrease)
- **➡️ Stable**: Minimal change (-20% to +20%)
- **🚀 Emerging**: High momentum with potential
- **📉 Declining**: Consistent downward pattern
- **💫 Volatile**: High variance in values

### Metrics Calculated
- **Growth Rate**: Percentage change over time
- **Trend Strength**: Confidence in trend direction (0-1)
- **Momentum**: Rate of change acceleration
- **Viral Score**: Likelihood of viral growth

## 🔧 Configuration

### Environment Variables
```bash
# Spotify API credentials (optional for CSV-only analysis)
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Data directories
DATA_DIR=data
OUTPUT_DIR=data/trending
```

### Settings
- **Trending Thresholds**: Configurable viral/rising/falling thresholds
- **Analysis Windows**: Customizable time periods for trend analysis
- **Quality Metrics**: Adjustable data quality scoring parameters

## 📊 Output Formats

### JSON Reports
```json
{
  "report_metadata": {
    "generated_at": "2025-01-07T23:52:21",
    "data_sources": ["spotify", "lastfm"]
  },
  "trending_insights": {
    "category_analysis": {...},
    "viral_content": [...],
    "emerging_trends": [...],
    "predictions": {...}
  }
}
```

### Text Summaries
```
SPOTIFY TRENDING ANALYSIS REPORT
========================================
Generated: 2025-01-07T23:52:21

EXECUTIVE SUMMARY
Total Trending Items: 40
Viral Content: 3
Emerging Trends: 7

TOP MOVERS BY CATEGORY
ARTIST:
  🔥 Viral Artist (+150.5% growth)
  📈 Rising Star (+45.2% growth)
```

## 🧪 Testing & Validation

### Data Quality Tests
```bash
python test_statistical_analysis.py
python test_basic_statistical.py
```

### Security Validation
```bash
python validate_security.py
```

### Demo Scripts
- `demo_trending_analysis.py` - Trending pattern analysis
- `demo_statistical_analysis.py` - Statistical forecasting
- `demo_multi_source.py` - Multi-source integration
- `complete_platform_demo.py` - Full platform showcase

## 🔍 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Install statistical libraries
   pip install statsmodels scikit-learn

   # Install visualization libraries
   pip install matplotlib seaborn plotly
   ```

2. **Encoding Issues (Windows)**
   - Files are saved with UTF-8 encoding
   - Use text editors that support Unicode

3. **Memory Issues with Large Datasets**
   - Process data in chunks
   - Use sampling for initial analysis
   - Consider cloud deployment for large-scale analysis

### Performance Optimization
- Use data sampling for quick prototyping
- Enable only necessary forecasting models
- Cache intermediate results
- Parallelize analysis across multiple files

## 🚀 Advanced Features

### Extensibility
- **Custom Trend Detectors**: Implement custom trending algorithms
- **Additional Data Sources**: Integrate social media, streaming platforms
- **ML Models**: Add custom machine learning models
- **Real-time Processing**: Stream processing capabilities

### Cloud Deployment
- **Scalable Processing**: AWS/GCP/Azure deployment ready
- **API Endpoints**: RESTful API for real-time analysis
- **Scheduled Reports**: Automated report generation
- **Dashboard Hosting**: Web-based interactive dashboards

## 📚 API Reference

### Core Classes

#### `StreamingDataQualityAnalyzer`
```python
analyzer = StreamingDataQualityAnalyzer(verbose=True)
report = analyzer.analyze_data_quality(df, timestamp_col, value_cols)
```

#### `StreamingForecastingEngine`
```python
forecaster = StreamingForecastingEngine()
forecasts = forecaster.forecast_multiple_models(time_series, periods=7)
```

#### `TrendingSchema`
```python
schema = TrendingSchema()
schema.add_trending_item(item_id, name, category, current_value, historical_values)
analysis = schema.analyze_trending_patterns()
```

#### `SpotifyTrendingIntegration`
```python
integration = SpotifyTrendingIntegration(data_dir="data")
integration.load_and_process_data()
insights = integration.analyze_trending_insights()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Spotify Web API** for streaming data
- **Last.fm API** for global music statistics
- **Statsmodels** for statistical analysis
- **Darts** for time series forecasting
- **Plotly** for interactive visualizations

## 📞 Support

For questions, issues, or feature requests:
1. Check existing issues
2. Create a new issue with detailed description
3. Include sample data and error messages
4. Specify your environment (OS, Python version, dependencies)

---

**Built with ❤️ for music data enthusiasts and analytics professionals**
