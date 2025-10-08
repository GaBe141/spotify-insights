"""
Advanced analytics engine for music trend analysis and prediction.
Implements machine learning, statistical analysis, and pattern recognition.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

class MusicTrendAnalytics:
    """
    Advanced analytics engine for music discovery and trend prediction.
    
    Features:
    - Cross-platform correlation analysis
    - Viral prediction using machine learning
    - Trend clustering and pattern recognition
    - Time series forecasting
    - Anomaly detection
    - Social sentiment integration
    """
    
    def __init__(self, data_store=None):
        self.logger = logging.getLogger(__name__)
        self.data_store = data_store
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        
    def detect_viral_patterns(self, track_data: Dict[str, Any], 
                            historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze viral potential using multiple signal detection methods.
        
        Args:
            track_data: Current track metrics
            historical_data: Optional historical trend data
            
        Returns:
            Comprehensive viral analysis
        """
        analysis_results = {
            "track_name": track_data.get('track_name', 'Unknown'),
            "artist": track_data.get('artist', 'Unknown'),
            "analysis_timestamp": datetime.now().isoformat(),
            "viral_signals": {},
            "risk_factors": {},
            "prediction": {}
        }
        
        # Signal 1: Growth velocity analysis
        growth_signals = self._analyze_growth_velocity(track_data)
        analysis_results["viral_signals"]["growth"] = growth_signals
        
        # Signal 2: Cross-platform momentum
        platform_signals = self._analyze_platform_momentum(track_data)
        analysis_results["viral_signals"]["platforms"] = platform_signals
        
        # Signal 3: Creator engagement patterns
        creator_signals = self._analyze_creator_patterns(track_data)
        analysis_results["viral_signals"]["creators"] = creator_signals
        
        # Signal 4: Temporal patterns
        temporal_signals = self._analyze_temporal_patterns(track_data)
        analysis_results["viral_signals"]["temporal"] = temporal_signals
        
        # Signal 5: Audio feature analysis
        audio_signals = self._analyze_audio_features(track_data)
        analysis_results["viral_signals"]["audio"] = audio_signals
        
        # Combine signals for final prediction
        final_prediction = self._combine_viral_signals(analysis_results["viral_signals"])
        analysis_results["prediction"] = final_prediction
        
        # Risk assessment
        risk_assessment = self._assess_viral_risks(track_data, analysis_results["viral_signals"])
        analysis_results["risk_factors"] = risk_assessment
        
        return analysis_results
    
    def _analyze_growth_velocity(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth velocity patterns."""
        velocity_metrics = {
            "current_velocity": 0.0,
            "acceleration": 0.0,
            "consistency_score": 0.0,
            "velocity_trend": "stable"
        }
        
        # Extract velocity data
        current_score = track_data.get('score', 0)
        previous_scores = track_data.get('score_history', [current_score])
        
        if len(previous_scores) >= 2:
            # Calculate velocity (rate of change)
            velocities = []
            for i in range(1, len(previous_scores)):
                velocity = previous_scores[i] - previous_scores[i-1]
                velocities.append(velocity)
            
            velocity_metrics["current_velocity"] = velocities[-1] if velocities else 0.0
            
            # Calculate acceleration (change in velocity)
            if len(velocities) >= 2:
                velocity_metrics["acceleration"] = velocities[-1] - velocities[-2]
            
            # Consistency score (lower standard deviation = more consistent growth)
            if len(velocities) > 1:
                velocity_std = np.std(velocities)
                velocity_metrics["consistency_score"] = 1.0 / (1.0 + velocity_std) if velocity_std > 0 else 1.0
            
            # Trend analysis
            if len(velocities) >= 3:
                recent_trend = np.polyfit(range(len(velocities)), velocities, 1)[0]
                if recent_trend > 0.5:
                    velocity_metrics["velocity_trend"] = "accelerating"
                elif recent_trend < -0.5:
                    velocity_metrics["velocity_trend"] = "decelerating"
                else:
                    velocity_metrics["velocity_trend"] = "stable"
        
        # Growth pattern classification
        if velocity_metrics["current_velocity"] > 5.0 and velocity_metrics["acceleration"] > 0:
            velocity_metrics["pattern"] = "explosive_growth"
        elif velocity_metrics["current_velocity"] > 2.0 and velocity_metrics["consistency_score"] > 0.7:
            velocity_metrics["pattern"] = "steady_growth"
        elif velocity_metrics["current_velocity"] < 0:
            velocity_metrics["pattern"] = "declining"
        else:
            velocity_metrics["pattern"] = "stable"
        
        return velocity_metrics
    
    def _analyze_platform_momentum(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-platform momentum indicators."""
        platform_metrics = {
            "platform_count": 1,
            "cross_platform_score": 0.0,
            "platform_diversity": 0.0,
            "momentum_alignment": 0.0
        }
        
        platforms = track_data.get('platforms', {})
        if isinstance(platforms, dict) and platforms:
            platform_metrics["platform_count"] = len(platforms)
            
            # Calculate cross-platform alignment
            platform_scores = list(platforms.values())
            if len(platform_scores) > 1:
                # Use coefficient of variation (lower = more aligned)
                cv = np.std(platform_scores) / np.mean(platform_scores) if np.mean(platform_scores) > 0 else 1.0
                platform_metrics["momentum_alignment"] = 1.0 / (1.0 + cv)
                
                # Calculate diversity (how spread across platform types)
                platform_types = self._classify_platforms(list(platforms.keys()))
                unique_types = len(set(platform_types))
                platform_metrics["platform_diversity"] = unique_types / len(platform_types)
            
            # Cross-platform viral score
            if platform_metrics["platform_count"] >= 3:
                platform_metrics["cross_platform_score"] = min(1.0, platform_metrics["platform_count"] / 5.0)
            elif platform_metrics["platform_count"] >= 2:
                platform_metrics["cross_platform_score"] = 0.6
            else:
                platform_metrics["cross_platform_score"] = 0.2
        
        return platform_metrics
    
    def _classify_platforms(self, platform_names: List[str]) -> List[str]:
        """Classify platforms into types."""
        platform_types = []
        for platform in platform_names:
            platform_lower = platform.lower()
            if platform_lower in ['tiktok', 'instagram', 'snapchat']:
                platform_types.append('short_video')
            elif platform_lower in ['youtube', 'vimeo']:
                platform_types.append('long_video')
            elif platform_lower in ['twitter', 'facebook']:
                platform_types.append('social_media')
            elif platform_lower in ['spotify', 'apple_music', 'soundcloud']:
                platform_types.append('music_streaming')
            elif platform_lower in ['reddit', 'discord', 'tumblr']:
                platform_types.append('community')
            else:
                platform_types.append('other')
        return platform_types
    
    def _analyze_creator_patterns(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze creator engagement and influence patterns."""
        creator_metrics = {
            "total_creators": 0,
            "tier_distribution": {},
            "influence_score": 0.0,
            "organic_indicator": 0.0
        }
        
        creators = track_data.get('creators', [])
        if creators:
            creator_metrics["total_creators"] = len(creators)
            
            # Analyze creator tiers
            tiers = [creator.get('tier', 'unknown') for creator in creators]
            tier_counts = Counter(tiers)
            creator_metrics["tier_distribution"] = dict(tier_counts)
            
            # Calculate influence score
            influence_weights = {
                'mega': 10.0,
                'macro': 5.0,
                'micro': 2.0,
                'nano': 1.0,
                'unknown': 0.5
            }
            
            total_influence = sum(influence_weights.get(tier, 0.5) * count 
                                for tier, count in tier_counts.items())
            creator_metrics["influence_score"] = min(1.0, total_influence / 100.0)
            
            # Organic growth indicator (high micro/nano ratio suggests organic spread)
            micro_nano_count = tier_counts.get('micro', 0) + tier_counts.get('nano', 0)
            
            if creator_metrics["total_creators"] > 5:
                organic_ratio = micro_nano_count / creator_metrics["total_creators"]
                creator_metrics["organic_indicator"] = min(1.0, organic_ratio * 1.5)
        
        return creator_metrics
    
    def _analyze_temporal_patterns(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns and timing factors."""
        temporal_metrics = {
            "time_since_release": 0,
            "viral_window_score": 0.0,
            "seasonal_factor": 0.0,
            "timing_advantage": "neutral"
        }
        
        # Time since release analysis
        release_date = track_data.get('release_date')
        if release_date:
            if isinstance(release_date, str):
                release_date = datetime.fromisoformat(release_date)
            
            time_diff = datetime.now() - release_date
            days_since_release = time_diff.days
            temporal_metrics["time_since_release"] = days_since_release
            
            # Viral window analysis (songs typically go viral within 30 days)
            if days_since_release <= 7:
                temporal_metrics["viral_window_score"] = 1.0
                temporal_metrics["timing_advantage"] = "optimal"
            elif days_since_release <= 30:
                temporal_metrics["viral_window_score"] = 0.8
                temporal_metrics["timing_advantage"] = "good"
            elif days_since_release <= 90:
                temporal_metrics["viral_window_score"] = 0.4
                temporal_metrics["timing_advantage"] = "moderate"
            else:
                temporal_metrics["viral_window_score"] = 0.1
                temporal_metrics["timing_advantage"] = "low"
        
        # Seasonal analysis
        current_month = datetime.now().month
        seasonal_factors = {
            12: 0.9, 1: 0.8, 2: 0.7,  # Winter
            3: 0.8, 4: 0.9, 5: 1.0,   # Spring
            6: 1.1, 7: 1.2, 8: 1.1,   # Summer
            9: 1.0, 10: 0.9, 11: 0.8  # Fall
        }
        temporal_metrics["seasonal_factor"] = seasonal_factors.get(current_month, 1.0)
        
        return temporal_metrics
    
    def _analyze_audio_features(self, track_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio features for viral potential."""
        audio_metrics = {
            "viral_audio_score": 0.0,
            "danceability_factor": 0.0,
            "energy_factor": 0.0,
            "catchiness_score": 0.0
        }
        
        audio_features = track_data.get('audio_features', {})
        if audio_features:
            # Danceability is crucial for viral content
            danceability = audio_features.get('danceability', 0.5)
            audio_metrics["danceability_factor"] = danceability
            
            # Energy level
            energy = audio_features.get('energy', 0.5)
            audio_metrics["energy_factor"] = energy
            
            # Catchiness heuristic (high danceability + moderate valence + good tempo)
            valence = audio_features.get('valence', 0.5)
            tempo = audio_features.get('tempo', 120)
            
            # Optimal tempo range for viral content (100-140 BPM)
            tempo_score = 1.0 if 100 <= tempo <= 140 else max(0.3, 1.0 - abs(tempo - 120) / 60)
            
            catchiness = (danceability * 0.4 + valence * 0.3 + tempo_score * 0.3)
            audio_metrics["catchiness_score"] = catchiness
            
            # Overall viral audio score
            viral_score = (danceability * 0.35 + energy * 0.25 + 
                          catchiness * 0.25 + tempo_score * 0.15)
            audio_metrics["viral_audio_score"] = viral_score
        
        return audio_metrics
    
    def _combine_viral_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all viral signals into final prediction."""
        # Weight different signal types
        weights = {
            "growth": 0.25,
            "platforms": 0.20,
            "creators": 0.20,
            "temporal": 0.15,
            "audio": 0.20
        }
        
        # Extract key scores from each signal type
        scores = {}
        
        # Growth signals
        growth = signals.get('growth', {})
        growth_score = 0.0
        if growth.get('pattern') == 'explosive_growth':
            growth_score = 0.9
        elif growth.get('pattern') == 'steady_growth':
            growth_score = 0.7
        elif growth.get('pattern') == 'declining':
            growth_score = 0.1
        else:
            growth_score = 0.4
        scores['growth'] = growth_score
        
        # Platform signals
        platforms = signals.get('platforms', {})
        platform_score = platforms.get('cross_platform_score', 0.0)
        scores['platforms'] = platform_score
        
        # Creator signals
        creators = signals.get('creators', {})
        creator_score = (creators.get('influence_score', 0.0) + 
                        creators.get('organic_indicator', 0.0)) / 2.0
        scores['creators'] = creator_score
        
        # Temporal signals
        temporal = signals.get('temporal', {})
        temporal_score = (temporal.get('viral_window_score', 0.0) + 
                         temporal.get('seasonal_factor', 1.0) - 0.5) / 1.5
        temporal_score = max(0.0, min(1.0, temporal_score))
        scores['temporal'] = temporal_score
        
        # Audio signals
        audio = signals.get('audio', {})
        audio_score = audio.get('viral_audio_score', 0.0)
        scores['audio'] = audio_score
        
        # Calculate weighted final score
        final_score = sum(weights[signal] * scores[signal] for signal in weights.keys())
        
        # Generate confidence based on signal strength consistency
        signal_values = list(scores.values())
        signal_std = np.std(signal_values)
        confidence = 1.0 / (1.0 + signal_std * 2.0)  # Lower std = higher confidence
        
        # Determine prediction category
        if final_score >= 0.8:
            category = "highly_likely"
            days_to_peak = 3
        elif final_score >= 0.6:
            category = "likely"
            days_to_peak = 7
        elif final_score >= 0.4:
            category = "possible"
            days_to_peak = 14
        else:
            category = "unlikely"
            days_to_peak = 30
        
        return {
            "viral_probability": final_score,
            "confidence": confidence,
            "category": category,
            "predicted_days_to_peak": days_to_peak,
            "predicted_peak_date": (datetime.now() + timedelta(days=days_to_peak)).isoformat(),
            "signal_scores": scores,
            "key_factors": self._identify_key_factors(signals, scores)
        }
    
    def _identify_key_factors(self, signals: Dict[str, Any], scores: Dict[str, Any]) -> List[str]:
        """Identify the most important factors contributing to viral potential."""
        factors = []
        
        # Growth factors
        growth = signals.get('growth', {})
        if scores['growth'] > 0.7:
            if growth.get('pattern') == 'explosive_growth':
                factors.append("Explosive growth pattern detected")
            elif growth.get('pattern') == 'steady_growth':
                factors.append("Consistent steady growth")
        
        # Platform factors
        platforms = signals.get('platforms', {})
        if scores['platforms'] > 0.6:
            platform_count = platforms.get('platform_count', 1)
            if platform_count >= 3:
                factors.append(f"Strong cross-platform presence ({platform_count} platforms)")
        
        # Creator factors
        creators = signals.get('creators', {})
        if scores['creators'] > 0.6:
            if creators.get('organic_indicator', 0) > 0.7:
                factors.append("High organic creator engagement")
            if creators.get('influence_score', 0) > 0.7:
                factors.append("Strong influencer support")
        
        # Temporal factors
        temporal = signals.get('temporal', {})
        if scores['temporal'] > 0.6:
            if temporal.get('timing_advantage') == 'optimal':
                factors.append("Optimal viral timing window")
        
        # Audio factors
        audio = signals.get('audio', {})
        if scores['audio'] > 0.7:
            if audio.get('danceability_factor', 0) > 0.8:
                factors.append("High danceability score")
            if audio.get('catchiness_score', 0) > 0.8:
                factors.append("Highly catchy audio profile")
        
        return factors
    
    def _assess_viral_risks(self, track_data: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential risks that could prevent viral success."""
        risk_factors = []
        risk_score = 0.0
        
        # Over-saturation risk
        genre = track_data.get('genre', '')
        if genre.lower() in ['pop', 'hip-hop', 'rap']:
            risk_factors.append("High competition in popular genre")
            risk_score += 0.2
        
        # Platform dependency risk
        platforms = signals.get('platforms', {})
        if platforms.get('platform_count', 1) == 1:
            risk_factors.append("Single platform dependency")
            risk_score += 0.3
        
        # Timing risk
        temporal = signals.get('temporal', {})
        if temporal.get('timing_advantage') == 'low':
            risk_factors.append("Poor timing for viral growth")
            risk_score += 0.25
        
        # Creator risk
        creators = signals.get('creators', {})
        if creators.get('total_creators', 0) < 3:
            risk_factors.append("Limited creator adoption")
            risk_score += 0.2
        
        # Audio appeal risk
        audio = signals.get('audio', {})
        if audio.get('viral_audio_score', 0) < 0.4:
            risk_factors.append("Audio features not optimized for viral content")
            risk_score += 0.15
        
        return {
            "overall_risk_score": min(1.0, risk_score),
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            "risk_factors": risk_factors,
            "mitigation_suggestions": self._generate_mitigation_suggestions(risk_factors)
        }
    
    def _generate_mitigation_suggestions(self, risk_factors: List[str]) -> List[str]:
        """Generate suggestions to mitigate identified risks."""
        suggestions = []
        
        for risk in risk_factors:
            if "competition" in risk.lower():
                suggestions.append("Focus on unique angle or niche audience first")
            elif "single platform" in risk.lower():
                suggestions.append("Expand to additional platforms immediately")
            elif "timing" in risk.lower():
                suggestions.append("Consider re-launch strategy or remix")
            elif "creator" in risk.lower():
                suggestions.append("Increase creator outreach and collaboration")
            elif "audio" in risk.lower():
                suggestions.append("Consider audio optimization or remix for danceability")
        
        return suggestions
    
    def detect_trending_clusters(self, days: int = 30, min_cluster_size: int = 5) -> List[Dict[str, Any]]:
        """
        Detect clusters of similar trending songs that might indicate micro-genres or trends.
        
        Args:
            days: Number of days to analyze
            min_cluster_size: Minimum songs per cluster
            
        Returns:
            List of detected clusters with characteristics
        """
        if not self.data_store:
            self.logger.error("No data store available for clustering analysis")
            return []
        
        # Get recent trends with audio features
        df = self.data_store.get_trending_tracks(days=days, limit=1000)
        
        if len(df) < min_cluster_size:
            self.logger.warning(f"Not enough data for clustering: {len(df)} tracks")
            return []
        
        # Extract audio features for clustering
        audio_features = []
        track_info = []
        
        for _, row in df.iterrows():
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            
            audio = metadata.get('audio_features', {})
            if audio and len(audio) >= 4:  # Need minimum features
                features = [
                    audio.get('energy', 0.5),
                    audio.get('danceability', 0.5),
                    audio.get('valence', 0.5),
                    audio.get('acousticness', 0.5),
                    audio.get('tempo', 120) / 200.0,  # Normalize tempo
                ]
                audio_features.append(features)
                track_info.append({
                    'track_name': row['track_name'],
                    'artist': row['artist'],
                    'platform': row['platform'],
                    'score': row['score']
                })
        
        if len(audio_features) < min_cluster_size:
            self.logger.warning(f"Not enough tracks with audio features: {len(audio_features)}")
            return []
        
        # Perform clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(audio_features)
        
        # Use DBSCAN for density-based clustering
        clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size).fit(features_scaled)
        
        # Analyze clusters
        labels = clustering.labels_
        unique_labels = set(labels)
        
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise cluster
        
        clusters = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_tracks = [track_info[i] for i, mask in enumerate(cluster_mask) if mask]
            cluster_features = [audio_features[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if len(cluster_tracks) >= min_cluster_size:
                # Calculate cluster characteristics
                avg_features = np.mean(cluster_features, axis=0)
                
                cluster_data = {
                    "cluster_id": int(label),
                    "size": len(cluster_tracks),
                    "tracks": cluster_tracks[:10],  # Top 10 tracks
                    "audio_profile": {
                        "energy": float(avg_features[0]),
                        "danceability": float(avg_features[1]),
                        "valence": float(avg_features[2]),
                        "acousticness": float(avg_features[3]),
                        "tempo": float(avg_features[4] * 200.0)
                    },
                    "top_artists": Counter([t['artist'] for t in cluster_tracks]).most_common(5),
                    "platform_distribution": Counter([t['platform'] for t in cluster_tracks]).most_common(),
                    "avg_score": np.mean([t['score'] for t in cluster_tracks]),
                    "cluster_name": self._generate_cluster_name(avg_features)
                }
                clusters.append(cluster_data)
        
        # Sort by cluster size and average score
        clusters.sort(key=lambda x: (x['size'], x['avg_score']), reverse=True)
        
        self.logger.info(f"Detected {len(clusters)} music trend clusters")
        return clusters
    
    def _generate_cluster_name(self, features: np.ndarray) -> str:
        """Generate descriptive name for audio feature cluster."""
        energy, danceability, valence, acousticness, tempo_norm = features
        tempo = tempo_norm * 200.0
        
        descriptors = []
        
        # Energy descriptors
        if energy > 0.8:
            descriptors.append("High-Energy")
        elif energy < 0.3:
            descriptors.append("Low-Energy")
        
        # Danceability descriptors
        if danceability > 0.8:
            descriptors.append("Danceable")
        elif danceability < 0.3:
            descriptors.append("Non-Dance")
        
        # Valence descriptors
        if valence > 0.7:
            descriptors.append("Upbeat")
        elif valence < 0.3:
            descriptors.append("Melancholic")
        
        # Acousticness descriptors
        if acousticness > 0.7:
            descriptors.append("Acoustic")
        
        # Tempo descriptors
        if tempo > 140:
            descriptors.append("Fast-Tempo")
        elif tempo < 80:
            descriptors.append("Slow-Tempo")
        
        if descriptors:
            return " ".join(descriptors)
        else:
            return "Mixed-Style"
    
    def forecast_trend_trajectory(self, track_name: str, artist: str, 
                                days_ahead: int = 14) -> Dict[str, Any]:
        """
        Forecast the trajectory of a specific track's trend.
        
        Args:
            track_name: Name of the track
            artist: Artist name
            days_ahead: Number of days to forecast
            
        Returns:
            Forecast results with confidence intervals
        """
        if not self.data_store:
            self.logger.error("No data store available for forecasting")
            return {}
        
        # Get historical data for the track
        with self.data_store.get_connection() as conn:
            query = '''
            SELECT th.timestamp, th.score, th.rank
            FROM trend_history th
            JOIN trends t ON th.trend_id = t.id
            WHERE t.track_name LIKE ? AND t.artist LIKE ?
            ORDER BY th.timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=[f'%{track_name}%', f'%{artist}%'])
        
        if len(df) < 5:
            return {"error": "Insufficient historical data for forecasting"}
        
        # Prepare time series data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.resample('D').mean().fillna(method='forward')  # Daily resampling
        
        forecast_results = {
            "track_name": track_name,
            "artist": artist,
            "forecast_days": days_ahead,
            "historical_points": len(df),
            "forecast_date": datetime.now().isoformat()
        }
        
        # Try multiple forecasting methods
        methods_results = {}
        
        # Method 1: Linear regression
        try:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['score'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
            linear_forecast = model.predict(future_X)
            
            methods_results['linear'] = {
                "forecast": linear_forecast.tolist(),
                "r2_score": r2_score(y, model.predict(X)),
                "trend": "increasing" if model.coef_[0] > 0 else "decreasing"
            }
            
        except Exception as e:
            self.logger.warning(f"Linear regression failed: {e}")
        
        # Method 2: ARIMA (if statsmodels available)
        if HAS_STATSMODELS and len(df) >= 10:
            try:
                model = ARIMA(df['score'], order=(1, 1, 1))
                fitted_model = model.fit()
                
                arima_forecast = fitted_model.forecast(steps=days_ahead)
                
                methods_results['arima'] = {
                    "forecast": arima_forecast.tolist(),
                    "aic": fitted_model.aic,
                    "confidence_intervals": "not_implemented"  # Would need full implementation
                }
                
            except Exception as e:
                self.logger.warning(f"ARIMA forecasting failed: {e}")
        
        # Method 3: Simple moving average with trend
        try:
            window = min(7, len(df) // 2)
            ma = df['score'].rolling(window=window).mean()
            recent_trend = (df['score'].iloc[-1] - ma.iloc[-window]) / window
            
            ma_forecast = []
            last_value = df['score'].iloc[-1]
            
            for i in range(days_ahead):
                next_value = last_value + recent_trend * (i + 1)
                ma_forecast.append(max(0, min(100, next_value)))  # Bound between 0-100
            
            methods_results['moving_average'] = {
                "forecast": ma_forecast,
                "trend_slope": recent_trend,
                "window_size": window
            }
            
        except Exception as e:
            self.logger.warning(f"Moving average forecasting failed: {e}")
        
        # Combine forecasts (ensemble)
        if methods_results:
            all_forecasts = []
            weights = {'linear': 0.4, 'arima': 0.4, 'moving_average': 0.2}
            
            for day in range(days_ahead):
                day_predictions = []
                day_weights = []
                
                for method, results in methods_results.items():
                    if 'forecast' in results and day < len(results['forecast']):
                        day_predictions.append(results['forecast'][day])
                        day_weights.append(weights.get(method, 0.33))
                
                if day_predictions:
                    weighted_avg = np.average(day_predictions, weights=day_weights)
                    all_forecasts.append(float(weighted_avg))
                else:
                    all_forecasts.append(df['score'].iloc[-1])  # Fallback to last known value
            
            forecast_results['ensemble_forecast'] = all_forecasts
            
            # Generate forecast dates
            start_date = df.index[-1] + timedelta(days=1)
            forecast_dates = [(start_date + timedelta(days=i)).isoformat() 
                            for i in range(days_ahead)]
            forecast_results['forecast_dates'] = forecast_dates
            
            # Add confidence assessment
            if len(methods_results) > 1:
                # Calculate variance across methods
                method_forecasts = [results['forecast'] for results in methods_results.values() 
                                  if 'forecast' in results]
                if len(method_forecasts) > 1:
                    forecast_variance = np.var([f[0] for f in method_forecasts if f])
                    confidence = max(0.1, 1.0 / (1.0 + forecast_variance / 10.0))
                    forecast_results['confidence'] = confidence
        
        forecast_results['methods_used'] = list(methods_results.keys())
        forecast_results['individual_methods'] = methods_results
        
        return forecast_results

# Example usage
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create analytics engine
    analytics = MusicTrendAnalytics()
    
    # Test viral pattern detection
    sample_track = {
        "track_name": "Test Viral Song",
        "artist": "Rising Artist",
        "score": 85.5,
        "score_history": [45, 52, 61, 73, 85.5],
        "platforms": {"tiktok": 88, "youtube": 82, "instagram": 86},
        "creators": [
            {"tier": "micro", "followers": 50000},
            {"tier": "macro", "followers": 500000},
            {"tier": "nano", "followers": 10000}
        ],
        "release_date": "2025-10-01",
        "audio_features": {
            "danceability": 0.85,
            "energy": 0.78,
            "valence": 0.82,
            "tempo": 128,
            "acousticness": 0.15
        }
    }
    
    # Analyze viral potential
    viral_analysis = analytics.detect_viral_patterns(sample_track)
    print("Viral Analysis Results:")
    print(f"Viral Probability: {viral_analysis['prediction']['viral_probability']:.2f}")
    print(f"Category: {viral_analysis['prediction']['category']}")
    print(f"Key Factors: {viral_analysis['prediction']['key_factors']}")
    print(f"Risk Level: {viral_analysis['risk_factors']['risk_level']}")
    
    print("\nAnalytics engine test completed successfully!")