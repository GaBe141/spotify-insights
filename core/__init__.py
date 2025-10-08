"""
Core Components for Enhanced Music Discovery System
=================================================

This module contains the core application components including:
- Main application orchestration
- Data persistence and storage  
- Resilience and reliability systems
- Authentication and configuration
- Notification services

Key Components:
    - discovery_app.py: Main application entry point
    - data_store.py: Enterprise-grade data persistence
    - resilience.py: Circuit breakers and retry logic
    - notification_service.py: Multi-channel notifications
    - auth.py: Authentication and API management
    - config.py: Configuration management
"""

__version__ = "2.0.0"
__author__ = "Enhanced Music Discovery Team"

# Import key classes for easy access
try:
    from .data_store import EnhancedMusicDataStore
    from .resilience import EnhancedResilience  
    from .notification_service import EnhancedNotificationService
    
    __all__ = [
        "EnhancedMusicDataStore",
        "EnhancedResilience", 
        "EnhancedNotificationService"
    ]
except ImportError:
    # Allow module to be imported even if dependencies aren't installed
    __all__ = []