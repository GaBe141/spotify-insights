"""
Visualization Components for Music Discovery System
==================================================

This module contains all visualization and charting components
for displaying music trends, analytics, and insights.

Key Features:
    - Interactive charts and graphs using Plotly
    - Statistical visualizations with Matplotlib/Seaborn
    - Real-time dashboards and monitoring displays
    - Multi-source data visualization
    - Trend timeline and growth visualizations
    - Cross-platform correlation charts

Key Components:
    - advanced_viz.py: Advanced charting and interactive visualizations
    - statistical_viz.py: Statistical charts and analysis plots
    - multi_source_viz.py: Multi-platform data visualizations
    - deep_viz.py: Deep learning model visualizations
    - trending_viz.py: Trending data specific charts
    - surprise_viz.py: Recommendation system visualizations
"""

__version__ = "2.0.0"

# Import key visualization classes
try:
    from .statistical_viz import StatisticalVisualizationEngine

    # Note: advanced_viz contains standalone functions, not classes

    __all__ = ["StatisticalVisualizationEngine"]
except ImportError:
    __all__ = []
