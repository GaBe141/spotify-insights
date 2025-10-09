"""
Enhanced Viral Prediction System
================================

Advanced ML-powered viral prediction with:
- Momentum acceleration tracking
- Cross-platform velocity analysis  
- Peak prediction timing with confidence intervals
- Multi-factor virality scoring
- Real-time trend momentum

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ViralMetrics:
    """Comprehensive viral prediction metrics."""
    viral_score: float  # 0-100 overall viral potential
    confidence: float  # 0-1 confidence in prediction
    momentum: float  # Current trend momentum
    acceleration: float  # Rate of momentum change
    cross_platform_velocity: float  # Speed of cross-platform spread
    peak_eta_days: int  # Estimated days to peak virality
    peak_confidence_interval: tuple[int, int]  # Min/max days to peak
    risk_level: str  # Low, Medium, High
    recommendation: str  # Action recommendation


class EnhancedViralPredictor:
    """Advanced viral prediction with ML features."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize enhanced viral predictor."""
        self.data_dir = Path(data_dir)
        self.historical_data: Optional[pd.DataFrame] = None
        
    def calculate_momentum(self, values: list[float], timestamps: list[datetime]) -> float:
        """
        Calculate trend momentum from time series data.
        
        Args:
            values: List of metric values over time
            timestamps: Corresponding timestamps
            
        Returns:
            Momentum score (0-100)
        """
        if len(values) < 2:
            return 0.0
        
        # Calculate rate of change
        changes = np.diff(values)
        time_diffs = np.diff([t.timestamp() for t in timestamps])
        
        # Normalize to changes per day
        rates = changes / (time_diffs / 86400)  # 86400 seconds in a day
        
        # Recent momentum matters more
        weights = np.exp(np.linspace(-1, 0, len(rates)))
        weighted_momentum = np.average(rates, weights=weights)
        
        # Normalize to 0-100 scale
        momentum = min(100, max(0, (weighted_momentum + 50)))
        
        return float(momentum)
    
    def calculate_acceleration(self, values: list[float], timestamps: list[datetime]) -> float:
        """
        Calculate acceleration (rate of change of momentum).
        
        Args:
            values: List of metric values over time
            timestamps: Corresponding timestamps
            
        Returns:
            Acceleration score (-100 to +100)
        """
        if len(values) < 3:
            return 0.0
        
        # Calculate first derivative (velocity)
        changes = np.diff(values)
        time_diffs = np.diff([t.timestamp() for t in timestamps])
        velocity = changes / (time_diffs / 86400)
        
        # Calculate second derivative (acceleration)
        accel_values = np.diff(velocity)
        accel_time_diffs = (time_diffs[:-1] + time_diffs[1:]) / 2
        acceleration = accel_values / (accel_time_diffs / 86400)
        
        # Weight recent acceleration more
        if len(acceleration) > 0:
            weights = np.exp(np.linspace(-1, 0, len(acceleration)))
            weighted_accel = np.average(acceleration, weights=weights)
        else:
            weighted_accel = 0.0
        
        # Normalize to -100 to +100
        accel_score = np.clip(weighted_accel * 10, -100, 100)
        
        return float(accel_score)
    
    def calculate_cross_platform_velocity(self, platform_scores: Dict[str, float]) -> float:
        """
        Calculate how quickly content is spreading across platforms.
        
        Args:
            platform_scores: Dict of platform: score pairs
            
        Returns:
            Cross-platform velocity (0-100)
        """
        if not platform_scores:
            return 0.0
        
        # Number of platforms with significant engagement
        active_platforms = sum(1 for score in platform_scores.values() if score > 50)
        
        # Average score across platforms
        avg_score = np.mean(list(platform_scores.values()))
        
        # Score variance (lower variance = more consistent cross-platform presence)
        score_variance = np.var(list(platform_scores.values()))
        consistency = 100 - min(100, score_variance)
        
        # Combine factors
        velocity = (active_platforms / len(platform_scores)) * 40  # 40% weight
        velocity += (avg_score / 100) * 40  # 40% weight
        velocity += (consistency / 100) * 20  # 20% weight
        
        return float(velocity)
    
    def predict_peak_timing(self, momentum: float, acceleration: float, 
                           current_score: float) -> tuple[int, tuple[int, int]]:
        """
        Predict when content will reach peak virality.
        
        Args:
            momentum: Current momentum score
            acceleration: Current acceleration score
            current_score: Current viral score
            
        Returns:
            Tuple of (estimated_days, (min_days, max_days))
        """
        # Base prediction on momentum and acceleration
        if momentum > 80 and acceleration > 50:
            # High momentum, positive acceleration: peak soon
            est_days = 3
            confidence_range = (2, 5)
        elif momentum > 60 and acceleration > 0:
            # Good momentum, positive acceleration
            est_days = 7
            confidence_range = (5, 10)
        elif momentum > 40:
            # Moderate momentum
            est_days = 14
            confidence_range = (10, 20)
        else:
            # Low momentum
            est_days = 30
            confidence_range = (20, 45)
        
        # Adjust based on current score
        if current_score > 80:
            # Already near peak
            est_days = max(1, est_days - 3)
            confidence_range = (max(1, confidence_range[0] - 2), 
                              max(est_days + 2, confidence_range[1] - 3))
        elif current_score < 40:
            # Far from peak
            est_days += 5
            confidence_range = (confidence_range[0] + 3, confidence_range[1] + 7)
        
        return est_days, confidence_range
    
    def assess_risk_level(self, viral_score: float, confidence: float, 
                         momentum: float) -> str:
        """
        Assess investment/action risk level.
        
        Args:
            viral_score: Overall viral score
            confidence: Prediction confidence
            momentum: Current momentum
            
        Returns:
            Risk level: 'Low', 'Medium', or 'High'
        """
        # High score + high confidence + high momentum = Low risk
        if viral_score > 75 and confidence > 0.8 and momentum > 70:
            return "Low"
        
        # Very low confidence or score = High risk
        if confidence < 0.5 or viral_score < 40:
            return "High"
        
        # Everything else = Medium risk
        return "Medium"
    
    def generate_recommendation(self, metrics: ViralMetrics) -> str:
        """Generate actionable recommendation based on metrics."""
        if metrics.viral_score > 85 and metrics.momentum > 70:
            return "🚀 STRONG BUY - Immediate action recommended. High viral potential with strong momentum."
        elif metrics.viral_score > 70 and metrics.acceleration > 30:
            return "📈 BUY - Good viral potential with positive acceleration. Monitor closely."
        elif metrics.viral_score > 60 and metrics.momentum > 50:
            return "👀 WATCH - Moderate potential. Track for momentum increase before action."
        elif metrics.acceleration < -30:
            return "⚠️ CAUTION - Declining momentum. May have peaked or losing traction."
        elif metrics.viral_score < 40:
            return "❌ PASS - Low viral potential. Consider alternative content."
        else:
            return "🤔 HOLD - Monitor for trend development. No immediate action needed."
    
    def predict_viral_potential(self, track_data: Dict[str, Any]) -> ViralMetrics:
        """
        Comprehensive viral prediction for a track.
        
        Args:
            track_data: Dict containing:
                - track_name: str
                - platform_scores: Dict[str, float] (spotify, tiktok, youtube, instagram)
                - social_signals: Dict[str, int] (mentions, shares, comments)
                - audio_features: Dict[str, float] (optional)
                - historical_data: List[Dict] with timestamps and values (optional)
                
        Returns:
            ViralMetrics object with comprehensive predictions
        """
        # Extract data
        platform_scores = track_data.get('platform_scores', {})
        social_signals = track_data.get('social_signals', {})
        audio_features = track_data.get('audio_features', {})
        historical_data = track_data.get('historical_data', [])
        
        # Base viral score calculation
        platform_avg = np.mean(list(platform_scores.values())) if platform_scores else 50
        
        # Social momentum score
        total_mentions = social_signals.get('mentions', 0)
        total_shares = social_signals.get('shares', 0)
        total_comments = social_signals.get('comments', 0)
        social_score = min(100, (total_mentions / 100 + total_shares / 50 + total_comments / 25))
        
        # Audio appeal (if available)
        if audio_features:
            danceability = audio_features.get('danceability', 0.5)
            energy = audio_features.get('energy', 0.5)
            valence = audio_features.get('valence', 0.5)
            audio_score = (danceability + energy + valence) / 3 * 100
        else:
            audio_score = 50
        
        # Weighted viral score
        viral_score = (
            platform_avg * 0.40 +
            social_score * 0.35 +
            audio_score * 0.25
        )
        
        # Calculate momentum and acceleration if historical data available
        if historical_data and len(historical_data) > 1:
            values = [d['value'] for d in historical_data]
            timestamps = [datetime.fromisoformat(d['timestamp']) 
                         if isinstance(d['timestamp'], str)
                         else d['timestamp'] for d in historical_data]
            
            momentum = self.calculate_momentum(values, timestamps)
            acceleration = self.calculate_acceleration(values, timestamps)
        else:
            # Estimate based on current scores
            momentum = viral_score * 0.8  # Assume some momentum if score is high
            acceleration = (viral_score - 50) * 0.5  # Positive if above average
        
        # Cross-platform velocity
        cross_platform_velocity = self.calculate_cross_platform_velocity(platform_scores)
        
        # Prediction confidence
        # Higher confidence with more platforms, more data, higher consistency
        num_platforms = len(platform_scores)
        platform_confidence = min(1.0, num_platforms / 4)  # Up to 4 platforms
        
        data_confidence = min(1.0, len(historical_data) / 10)  # More historical data = higher confidence
        
        score_consistency = 1.0 - (np.std(list(platform_scores.values())) / 100 if platform_scores else 0.5)
        
        confidence = (platform_confidence * 0.4 + 
                     data_confidence * 0.3 + 
                     score_consistency * 0.3)
        
        # Peak timing prediction
        peak_days, confidence_interval = self.predict_peak_timing(
            momentum, acceleration, viral_score
        )
        
        # Risk assessment
        risk_level = self.assess_risk_level(viral_score, confidence, momentum)
        
        # Create metrics object
        metrics = ViralMetrics(
            viral_score=viral_score,
            confidence=confidence,
            momentum=momentum,
            acceleration=acceleration,
            cross_platform_velocity=cross_platform_velocity,
            peak_eta_days=peak_days,
            peak_confidence_interval=confidence_interval,
            risk_level=risk_level,
            recommendation=""  # Will be set next
        )
        
        # Generate recommendation
        metrics.recommendation = self.generate_recommendation(metrics)
        
        return metrics
    
    def batch_predict(self, tracks: list[Dict[str, Any]]) -> list[tuple[Dict[str, Any], ViralMetrics]]:
        """
        Batch prediction for multiple tracks.
        
        Args:
            tracks: List of track data dictionaries
            
        Returns:
            List of (track_data, ViralMetrics) tuples, sorted by viral score
        """
        results = []
        
        for track in tracks:
            metrics = self.predict_viral_potential(track)
            results.append((track, metrics))
        
        # Sort by viral score descending
        results.sort(key=lambda x: x[1].viral_score, reverse=True)
        
        return results
    
    def print_prediction(self, track_name: str, metrics: ViralMetrics):
        """Print formatted prediction results."""
        print(f"\n{'=' * 60}")
        print(f"🎵 {track_name}")
        print(f"{'=' * 60}")
        print(f"Viral Score:        {metrics.viral_score:>6.1f}/100")
        print(f"Confidence:         {metrics.confidence:>6.1%}")
        print(f"Momentum:           {metrics.momentum:>6.1f}")
        print(f"Acceleration:       {metrics.acceleration:>+6.1f}")
        print(f"Platform Velocity:  {metrics.cross_platform_velocity:>6.1f}")
        print(f"Peak ETA:           {metrics.peak_eta_days:>3} days (range: {metrics.peak_confidence_interval[0]}-{metrics.peak_confidence_interval[1]} days)")
        print(f"Risk Level:         {metrics.risk_level}")
        print(f"\n💡 {metrics.recommendation}")


def demo():
    """Demonstration of enhanced viral prediction."""
    print("🚀 AUDORA ENHANCED VIRAL PREDICTION SYSTEM")
    print("=" * 60)
    
    predictor = EnhancedViralPredictor()
    
    # Sample track data with historical trends
    sample_track = {
        'track_name': 'Viral Hit Example',
        'platform_scores': {
            'spotify': 85,
            'tiktok': 92,
            'youtube': 78,
            'instagram': 88
        },
        'social_signals': {
            'mentions': 15000,
            'shares': 3500,
            'comments': 850
        },
        'audio_features': {
            'danceability': 0.85,
            'energy': 0.78,
            'valence': 0.72
        },
        'historical_data': [
            {'timestamp': datetime.now() - timedelta(days=7), 'value': 45},
            {'timestamp': datetime.now() - timedelta(days=6), 'value': 52},
            {'timestamp': datetime.now() - timedelta(days=5), 'value': 61},
            {'timestamp': datetime.now() - timedelta(days=4), 'value': 68},
            {'timestamp': datetime.now() - timedelta(days=3), 'value': 75},
            {'timestamp': datetime.now() - timedelta(days=2), 'value': 82},
            {'timestamp': datetime.now() - timedelta(days=1), 'value': 88},
        ]
    }
    
    # Predict
    metrics = predictor.predict_viral_potential(sample_track)
    predictor.print_prediction(sample_track['track_name'], metrics)
    
    # Batch prediction example
    print("\n\n" + "=" * 60)
    print("📊 BATCH PREDICTION - TOP 3 VIRAL CANDIDATES")
    print("=" * 60)
    
    batch_tracks = [
        {
            'track_name': 'Rising Star Track',
            'platform_scores': {'spotify': 72, 'tiktok': 85, 'youtube': 68},
            'social_signals': {'mentions': 8000, 'shares': 1200, 'comments': 450},
            'audio_features': {'danceability': 0.78, 'energy': 0.82, 'valence': 0.65}
        },
        {
            'track_name': 'Established Hit',
            'platform_scores': {'spotify': 95, 'tiktok': 88, 'youtube': 92, 'instagram': 90},
            'social_signals': {'mentions': 25000, 'shares': 5000, 'comments': 1200},
            'audio_features': {'danceability': 0.92, 'energy': 0.85, 'valence': 0.88}
        },
        {
            'track_name': 'Slow Burner',
            'platform_scores': {'spotify': 55, 'tiktok': 48},
            'social_signals': {'mentions': 2000, 'shares': 350, 'comments': 120},
            'audio_features': {'danceability': 0.45, 'energy': 0.38, 'valence': 0.42}
        }
    ]
    
    results = predictor.batch_predict(batch_tracks)
    
    for i, (track, metrics) in enumerate(results, 1):
        print(f"\n#{i}. {track['track_name']}")
        print(f"    Viral Score: {metrics.viral_score:.1f} | "
              f"Confidence: {metrics.confidence:.1%} | "
              f"Risk: {metrics.risk_level}")
        print(f"    {metrics.recommendation}")
    
    print("\n✨ Enhanced viral prediction complete!")


if __name__ == "__main__":
    demo()
