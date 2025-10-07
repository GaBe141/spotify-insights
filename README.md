# 🎵 Spotify Insights - Advanced Multi-Source Music Analytics

A comprehensive music analytics platform that combines data from **5+ sources** with advanced **statistical analysis and forecasting** to provide deep insights into your music preferences and streaming patterns.

## 🌟 Features

### 📊 **Multi-Source Data Integration**
- **Spotify Web API**: Personal top tracks, artists, and recently played
- **Last.fm API**: Global charts and listening trends  
- **MusicBrainz API**: Artist metadata, relationships, and discographies
- **AudioDB API**: Rich artist profiles with biographies and multimedia
- **Spotify Charts**: Real-time global streaming data (web scraping)

### � **Advanced Statistical Analysis**
- **Data Quality Assessment**: Missing values, outliers, temporal consistency
- **Time Series Forecasting**: ARIMA, SARIMA, Exponential Smoothing, Prophet
- **Anomaly Detection**: IQR, Z-score, and Isolation Forest methods
- **Trend Analysis**: Seasonal patterns, viral content detection
- **Model Performance Evaluation**: MAE, RMSE, MAPE metrics

### 📈 **Forecasting Models**
- **ARIMA/SARIMA**: Classical time series forecasting
- **Darts Library**: Auto-ARIMA, Exponential Smoothing, Theta, Linear Regression
- **Machine Learning**: Random Forest, Neural Networks (optional)
- **Prophet**: Facebook's forecasting tool (optional)

### 🎨 **Advanced Visualizations**
- Interactive dashboards with Plotly
- Statistical quality reports
- Forecasting charts with confidence intervals
- Model performance comparisons
- Anomaly detection visualizations

### 🔒 **Enterprise Security**
- Secure credential management with validation
- Environment-based configuration
- Automatic security scanning
- Git safety with comprehensive .gitignore

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/spotify-insights.git
cd spotify-insights
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Run security setup (creates .env template)
python validate_security.py
```

Edit the generated `.env` file with your API credentials:
```env
# Spotify API (Required)
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback

# Last.fm API (Optional but recommended)
LASTFM_API_KEY=your_lastfm_api_key

# AudioDB API (Optional - use "123" for free tier)
AUDIODB_API_KEY=123
```

### 3. Basic Analysis
```bash
# Simple demonstration with working APIs
python simple_multi_source_demo.py

# Basic statistical analysis (works without advanced libraries)
python demo_statistical_analysis.py
```

### 4. Advanced Analytics
```bash
# Install advanced forecasting libraries (optional)
pip install statsmodels darts scikit-learn plotly

# Full multi-source data collection
python demo_multi_source.py

# Advanced statistical analysis with forecasting
python advanced_streaming_analytics.py
```

## 📁 Project Structure

```
spotify-insights/
├── src/                          # Core modules
│   ├── auth.py                  # Authentication handling
│   ├── fetch.py                 # Basic Spotify data fetching
│   ├── multi_source_main.py     # Multi-source data pipeline
│   ├── statistical_analysis.py  # Advanced statistical analysis
│   ├── statistical_viz.py       # Statistical visualizations
│   └── *_integration.py         # API integrations
├── data/                        # Generated data and reports
├── notebooks/                   # Jupyter notebooks (optional)
├── demo_*.py                   # Demonstration scripts
├── test_*.py                   # Test scripts
├── validate_security.py        # Security validation
└── requirements.txt            # Dependencies
```

## 🎯 Usage Examples

### Multi-Source Data Collection
```python
from src.multi_source_main import MultiSourceSpotifyAnalyzer

analyzer = MultiSourceSpotifyAnalyzer()
analyzer.collect_all_data()
insights = analyzer.analyze_cross_platform_insights()
```

### Statistical Analysis and Forecasting
```python
from src.statistical_analysis import StreamingDataQualityAnalyzer, StreamingForecastingEngine

# Data quality analysis
quality_analyzer = StreamingDataQualityAnalyzer()
quality_report = quality_analyzer.analyze_data_quality(data, 'date', ['streams', 'plays'])

# Forecasting
forecasting_engine = StreamingForecastingEngine()
ts, _ = forecasting_engine.prepare_time_series(data, 'date', 'streams')
forecasts = forecasting_engine.generate_forecasts(ts, horizon=30)
```

### Advanced Analytics Pipeline
```python
from advanced_streaming_analytics import AdvancedStreamingAnalytics

analytics = AdvancedStreamingAnalytics()
report_path = analytics.run_complete_analysis()
```

## 📊 Available Analyses

### 🔍 **Data Quality Assessment**
- **Missing Value Analysis**: Identify and quantify data gaps
- **Outlier Detection**: Multiple statistical methods for anomaly detection
- **Temporal Consistency**: Check for irregular time intervals and gaps
- **Duplicate Detection**: Identify redundant data points
- **Quality Scoring**: Overall data quality metrics (0-100)

### 📈 **Forecasting Capabilities**
- **Short-term Forecasts**: 1-30 day predictions
- **Confidence Intervals**: Statistical uncertainty quantification
- **Model Comparison**: Automatic best model selection
- **Trend Detection**: Identify growing/declining patterns
- **Seasonal Analysis**: Weekly/monthly pattern recognition

### 🎵 **Music-Specific Insights**
- **Listening Pattern Analysis**: Consistency, variety, discovery rates
- **Viral Content Detection**: Unusual streaming spikes
- **Cross-Platform Comparison**: Personal vs global trends
- **Artist Network Analysis**: Relationship mapping
- **Genre Evolution**: Preference changes over time

## 🛠️ Available Commands

| Command | Description | Requirements |
|---------|-------------|--------------|
| `python validate_security.py` | Security validation and setup | Core |
| `python simple_multi_source_demo.py` | Basic multi-source demo | Core |
| `python demo_statistical_analysis.py` | Statistical analysis with existing data | Core |
| `python test_basic_statistical.py` | Test basic statistical functions | Core |
| `python demo_multi_source.py` | Full multi-source demonstration | All APIs |
| `python advanced_streaming_analytics.py` | Full analytics pipeline | Advanced libs |
| `python src/musicbrainz_test.py` | Test MusicBrainz integration | Core |
| `python src/audiodb_test.py` | Test AudioDB integration | Core |

## 📦 Dependencies

### Core Requirements (Always Installed)
- `pandas>=2.2.2` - Data manipulation and analysis
- `requests>=2.32.0` - API communication
- `spotipy>=2.23.0` - Spotify API wrapper
- `matplotlib>=3.8.4` - Basic plotting
- `numpy>=1.24.0` - Numerical computing
- `python-dotenv>=1.0.1` - Environment management

### Statistical Analysis (Optional - Enhanced Features)
- `statsmodels>=0.14.0` - Time series analysis (ARIMA, SARIMA)
- `darts>=0.27.0` - Advanced forecasting models
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `scipy>=1.11.0` - Statistical functions
- `plotly>=5.17.0` - Interactive visualizations

### Research-Grade Features (Optional - Large Downloads)
- `prophet>=1.1.5` - Facebook's forecasting tool
- `tensorflow>=2.13.0` - Deep learning models
- `torch>=2.0.0` - PyTorch for neural networks

## 📈 Sample Outputs

### Quality Analysis Report
```json
{
  "data_quality": {
    "quality_score": 87.5,
    "missing_values": {"streams": 12, "plays": 8},
    "outlier_analysis": {
      "streams": {"percentage": 5.2, "method": "IQR"}
    },
    "recommendations": [
      "Moderate missing values in streams (5.0%). Apply interpolation.",
      "Outlier percentage acceptable for streams (5.2%)."
    ]
  }
}
```

### Forecasting Results
```json
{
  "forecasting": {
    "streams": {
      "best_model": "sarima",
      "performance": {"mae": 87.3, "rmse": 124.5},
      "forecast": [1205, 1189, 1276, 1298, ...],
      "trend": "increasing"
    }
  }
}
```

### Multi-Source Insights
```
🎵 Multi-Source Analysis Results:
✅ Analyzed 10 top artists across 3 platforms
📊 Found 3 countries represented (GB most common)
🎶 Average artist formation year: 1999
📈 3 different genres identified
```

## 🔧 Advanced Configuration

### API Setup Guide

1. **Spotify API** (Required):
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Set redirect URI to `http://127.0.0.1:8888/callback`
   - Copy Client ID and Client Secret

2. **Last.fm API** (Optional):
   - Visit [Last.fm API](https://www.last.fm/api/account/create)
   - Create account and get API key

3. **AudioDB API** (Optional):
   - Use `"123"` for free tier
   - Visit [AudioDB](https://www.theaudiodb.com/api_guide.php) for premium

### Statistical Analysis Customization
```python
# Customize forecasting models
forecasting_engine = StreamingForecastingEngine()
forecasting_engine.available_models = {
    'arima': True,
    'sarima': True,
    'prophet': False,  # Disable Prophet
    'random_forest': True
}

# Adjust quality thresholds
quality_analyzer = StreamingDataQualityAnalyzer()
# Custom outlier detection sensitivity
# Custom missing value tolerance
```

## 🔒 Security & Privacy

- **API Key Protection**: Secure credential management with `.env` files
- **Rate Limiting**: Automatic API request throttling
- **Data Privacy**: All data processed locally
- **Security Validation**: Built-in security checks (`validate_security.py`)
- **No Data Persistence**: Personal data not permanently stored
- **Git Safety**: Comprehensive .gitignore prevents credential exposure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Spotify** for the comprehensive Web API
- **Last.fm** for global music data
- **MusicBrainz** for open music metadata
- **AudioDB** for rich artist information
- **Darts** for advanced time series forecasting
- **Statsmodels** for statistical analysis tools

---

🎵 **Ready to discover insights in your music streaming data?** 

Start with basic setup: `python validate_security.py` → `python simple_multi_source_demo.py`

Then explore advanced analytics: `python demo_statistical_analysis.py` → `python advanced_streaming_analytics.py`
