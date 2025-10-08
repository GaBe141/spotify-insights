"""
Enhanced data persistence layer for music discovery data.
Handles trending data, viral predictions, and cross-platform analysis.
"""

import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import json
from contextlib import contextmanager

@dataclass
class TrendData:
    """Data class for trend information."""
    platform: str
    track_id: str
    track_name: str
    artist: str
    score: float
    rank: int
    region: str
    trend_date: datetime
    metadata: Dict[str, Any]
    first_detected: datetime

@dataclass
class ViralPrediction:
    """Data class for viral predictions."""
    track_id: str
    track_name: str
    artist: str
    confidence: float
    predicted_peak_date: datetime
    predicted_peak_score: float
    prediction_features: Dict[str, Any]
    prediction_date: datetime
    actual_peak_score: Optional[float] = None
    accuracy_score: Optional[float] = None

class EnhancedMusicDataStore:
    """
    Enhanced data persistence system for music discovery.
    
    Features:
    - ACID transactions
    - Data validation
    - Efficient querying with indexes
    - Backup and restore capabilities
    - Data export in multiple formats
    - Analytics-ready data structures
    """
    
    def __init__(self, db_path: str = "enhanced_music_trends.db", backup_dir: str = "backups"):
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._initialize_database()
        self._create_indexes()
        
        # Enable WAL mode for better concurrency
        with self.get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def _initialize_database(self) -> None:
        """Create all database tables with enhanced schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enhanced trends table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                track_id TEXT,
                track_name TEXT NOT NULL,
                artist TEXT NOT NULL,
                score REAL NOT NULL,
                rank INTEGER,
                region TEXT NOT NULL DEFAULT 'global',
                trend_date TEXT NOT NULL,
                first_detected TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                metadata TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(platform, track_id, region, trend_date) ON CONFLICT REPLACE
            )
            ''')
            
            # Trend history with better granularity
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trend_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trend_id INTEGER,
                timestamp TEXT NOT NULL,
                score REAL NOT NULL,
                rank INTEGER,
                velocity REAL,  -- Rate of change
                momentum REAL,  -- Sustained growth indicator
                cross_platform_count INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trend_id) REFERENCES trends (id) ON DELETE CASCADE
            )
            ''')
            
            # Enhanced viral predictions
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS viral_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT,
                track_name TEXT NOT NULL,
                artist TEXT NOT NULL,
                confidence REAL NOT NULL,
                prediction_date TEXT NOT NULL,
                predicted_peak_date TEXT,
                predicted_peak_score REAL,
                prediction_features TEXT,  -- JSON of features used
                actual_peak_date TEXT,
                actual_peak_score REAL,
                accuracy_score REAL,
                status TEXT DEFAULT 'pending',  -- pending, confirmed, failed
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Cross-platform correlation tracking
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_platform_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_name TEXT NOT NULL,
                artist TEXT NOT NULL,
                source_platform TEXT NOT NULL,
                target_platform TEXT NOT NULL,
                correlation_coefficient REAL,
                lag_hours REAL,  -- Time difference between platform appearances
                propagation_strength REAL,
                analysis_date TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(track_name, artist, source_platform, target_platform, analysis_date) ON CONFLICT REPLACE
            )
            ''')
            
            # Platform performance metrics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS platform_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                date TEXT NOT NULL,
                total_tracks INTEGER DEFAULT 0,
                new_discoveries INTEGER DEFAULT 0,
                viral_hits INTEGER DEFAULT 0,
                average_prediction_accuracy REAL,
                api_success_rate REAL,
                avg_response_time REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(platform, date) ON CONFLICT REPLACE
            )
            ''')
            
            # User alerts and notifications
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                conditions TEXT NOT NULL,  -- JSON query conditions
                notification_channels TEXT,  -- JSON list of channels
                is_active BOOLEAN DEFAULT 1,
                last_triggered TEXT,
                trigger_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Data quality and validation logs
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                validation_type TEXT NOT NULL,
                validation_result TEXT NOT NULL,  -- pass, fail, warning
                details TEXT,
                row_count INTEGER,
                issue_count INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            self.logger.info("Database tables initialized successfully")
    
    def _create_indexes(self) -> None:
        """Create database indexes for better query performance."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Indexes for trends table
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_trends_platform ON trends(platform)",
                "CREATE INDEX IF NOT EXISTS idx_trends_artist ON trends(artist)",
                "CREATE INDEX IF NOT EXISTS idx_trends_score ON trends(score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_trends_date ON trends(trend_date)",
                "CREATE INDEX IF NOT EXISTS idx_trends_region ON trends(region)",
                "CREATE INDEX IF NOT EXISTS idx_trends_active ON trends(is_active)",
                "CREATE INDEX IF NOT EXISTS idx_trends_composite ON trends(platform, region, trend_date)",
                
                # Indexes for trend_history
                "CREATE INDEX IF NOT EXISTS idx_history_trend_id ON trend_history(trend_id)",
                "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON trend_history(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_history_score ON trend_history(score DESC)",
                
                # Indexes for viral_predictions
                "CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON viral_predictions(confidence DESC)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_date ON viral_predictions(prediction_date)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_status ON viral_predictions(status)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_artist ON viral_predictions(artist)",
                
                # Indexes for cross_platform_correlations
                "CREATE INDEX IF NOT EXISTS idx_correlations_track ON cross_platform_correlations(track_name, artist)",
                "CREATE INDEX IF NOT EXISTS idx_correlations_platforms ON cross_platform_correlations(source_platform, target_platform)",
                "CREATE INDEX IF NOT EXISTS idx_correlations_date ON cross_platform_correlations(analysis_date)",
                
                # Indexes for platform_metrics
                "CREATE INDEX IF NOT EXISTS idx_metrics_platform_date ON platform_metrics(platform, date)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_accuracy ON platform_metrics(average_prediction_accuracy DESC)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            self.logger.info("Database indexes created successfully")
    
    def save_trend(self, trend_data: TrendData) -> int:
        """
        Save a trend with full validation and error handling.
        
        Args:
            trend_data: TrendData object
            
        Returns:
            ID of saved trend
        """
        # Validate data
        self._validate_trend_data(trend_data)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                INSERT INTO trends 
                (platform, track_id, track_name, artist, score, rank, region, 
                 trend_date, first_detected, last_updated, metadata, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trend_data.platform,
                    trend_data.track_id,
                    trend_data.track_name,
                    trend_data.artist,
                    trend_data.score,
                    trend_data.rank,
                    trend_data.region,
                    trend_data.trend_date.isoformat(),
                    trend_data.first_detected.isoformat(),
                    datetime.now().isoformat(),
                    json.dumps(trend_data.metadata),
                    True
                ))
                
                trend_id = cursor.lastrowid
                
                # Add initial history entry
                cursor.execute('''
                INSERT INTO trend_history
                (trend_id, timestamp, score, rank, velocity, momentum, cross_platform_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trend_id,
                    datetime.now().isoformat(),
                    trend_data.score,
                    trend_data.rank,
                    0.0,  # Initial velocity
                    1.0,  # Initial momentum
                    1     # Initially on one platform
                ))
                
                conn.commit()
                self.logger.info(f"Saved trend: {trend_data.track_name} by {trend_data.artist}")
                return trend_id
                
            except sqlite3.IntegrityError as e:
                self.logger.warning(f"Trend already exists, updating: {e}")
                # Update existing trend
                return self._update_existing_trend(trend_data, conn)
    
    def _validate_trend_data(self, trend_data: TrendData) -> None:
        """Validate trend data before saving."""
        if not trend_data.track_name or not trend_data.artist:
            raise ValueError("Track name and artist are required")
        
        if not 0 <= trend_data.score <= 100:
            raise ValueError("Score must be between 0 and 100")
        
        if trend_data.rank < 0:
            raise ValueError("Rank cannot be negative")
        
        if not trend_data.platform:
            raise ValueError("Platform is required")
    
    def _update_existing_trend(self, trend_data: TrendData, conn: sqlite3.Connection) -> int:
        """Update an existing trend and add history entry."""
        cursor = conn.cursor()
        
        # Get existing trend
        cursor.execute('''
        SELECT id, score FROM trends 
        WHERE platform = ? AND track_name = ? AND artist = ? AND region = ?
        ORDER BY last_updated DESC LIMIT 1
        ''', (trend_data.platform, trend_data.track_name, trend_data.artist, trend_data.region))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError("Trend not found for update")
        
        trend_id, old_score = row
        
        # Calculate velocity (rate of change)
        velocity = trend_data.score - old_score
        
        # Update trend
        cursor.execute('''
        UPDATE trends SET 
        score = ?, rank = ?, last_updated = ?, metadata = ?
        WHERE id = ?
        ''', (
            trend_data.score,
            trend_data.rank,
            datetime.now().isoformat(),
            json.dumps(trend_data.metadata),
            trend_id
        ))
        
        # Add history entry
        cursor.execute('''
        INSERT INTO trend_history
        (trend_id, timestamp, score, rank, velocity, momentum)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            trend_id,
            datetime.now().isoformat(),
            trend_data.score,
            trend_data.rank,
            velocity,
            max(0.1, 1.0 + velocity / 10.0)  # Simple momentum calculation
        ))
        
        conn.commit()
        return trend_id
    
    def save_viral_prediction(self, prediction: ViralPrediction) -> int:
        """Save a viral prediction with validation."""
        if not 0 <= prediction.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO viral_predictions
            (track_id, track_name, artist, confidence, prediction_date,
             predicted_peak_date, predicted_peak_score, prediction_features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.track_id,
                prediction.track_name,
                prediction.artist,
                prediction.confidence,
                prediction.prediction_date.isoformat(),
                prediction.predicted_peak_date.isoformat(),
                prediction.predicted_peak_score,
                json.dumps(prediction.prediction_features)
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            
            self.logger.info(f"Saved viral prediction: {prediction.track_name} (confidence: {prediction.confidence:.2f})")
            return prediction_id
    
    def get_trending_tracks(self, 
                           platform: Optional[str] = None,
                           region: Optional[str] = None,
                           days: int = 7,
                           min_score: float = 0.0,
                           limit: int = 50) -> pd.DataFrame:
        """
        Get trending tracks with advanced filtering.
        
        Args:
            platform: Filter by platform
            region: Filter by region
            days: Number of days to look back
            min_score: Minimum score threshold
            limit: Maximum number of results
            
        Returns:
            DataFrame with trending tracks
        """
        with self.get_connection() as conn:
            # Build dynamic query
            conditions = ["datetime(trend_date) >= datetime('now', ?)", "score >= ?", "is_active = 1"]
            params = [f'-{days} days', min_score]
            
            if platform:
                conditions.append("platform = ?")
                params.append(platform)
            
            if region:
                conditions.append("region = ?")
                params.append(region)
            
            query = f'''
            SELECT 
                platform, track_name, artist, score, rank, region, trend_date,
                metadata, first_detected,
                (SELECT COUNT(*) FROM trend_history th WHERE th.trend_id = t.id) as data_points,
                (SELECT AVG(velocity) FROM trend_history th WHERE th.trend_id = t.id) as avg_velocity
            FROM trends t
            WHERE {' AND '.join(conditions)}
            ORDER BY score DESC, trend_date DESC
            LIMIT ?
            '''
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Parse metadata if it exists
            if not df.empty and 'metadata' in df.columns:
                df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
            
            return df
    
    def get_viral_predictions(self, 
                             confidence_threshold: float = 0.7,
                             status: Optional[str] = None,
                             days: int = 30,
                             limit: int = 20) -> pd.DataFrame:
        """Get viral predictions with filtering."""
        with self.get_connection() as conn:
            conditions = ["confidence >= ?", "datetime(prediction_date) >= datetime('now', ?)"]
            params = [confidence_threshold, f'-{days} days']
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            query = f'''
            SELECT 
                track_name, artist, confidence, prediction_date,
                predicted_peak_date, predicted_peak_score,
                actual_peak_date, actual_peak_score, accuracy_score, status,
                prediction_features
            FROM viral_predictions
            WHERE {' AND '.join(conditions)}
            ORDER BY prediction_date DESC, confidence DESC
            LIMIT ?
            '''
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # Parse prediction features
            if not df.empty and 'prediction_features' in df.columns:
                df['prediction_features'] = df['prediction_features'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
    
    def analyze_cross_platform_spread(self, 
                                    track_name: str, 
                                    artist: str,
                                    days: int = 30) -> Dict[str, Any]:
        """
        Analyze how a track spreads across platforms.
        
        Args:
            track_name: Name of the track
            artist: Artist name
            days: Number of days to analyze
            
        Returns:
            Cross-platform analysis results
        """
        with self.get_connection() as conn:
            # Get all platform appearances for this track
            query = '''
            SELECT 
                platform, 
                MIN(datetime(first_detected)) as first_appearance,
                MAX(score) as peak_score,
                COUNT(*) as data_points
            FROM trends
            WHERE track_name LIKE ? AND artist LIKE ?
            AND datetime(trend_date) >= datetime('now', ?)
            GROUP BY platform
            ORDER BY first_appearance
            '''
            
            df = pd.read_sql_query(query, conn, params=[f'%{track_name}%', f'%{artist}%', f'-{days} days'])
            
            if df.empty:
                return {"message": "No cross-platform data found"}
            
            platforms = df['platform'].tolist()
            
            # Calculate propagation times
            propagation_analysis = []
            if len(platforms) > 1:
                df['first_appearance'] = pd.to_datetime(df['first_appearance'])
                df = df.sort_values('first_appearance')
                
                for i in range(1, len(df)):
                    current = df.iloc[i]
                    previous = df.iloc[i-1]
                    
                    time_diff = (current['first_appearance'] - previous['first_appearance']).total_seconds() / 3600
                    
                    propagation_analysis.append({
                        "from_platform": previous['platform'],
                        "to_platform": current['platform'],
                        "hours_difference": round(time_diff, 1),
                        "score_change": current['peak_score'] - previous['peak_score']
                    })
            
            # Store correlation data
            for prop in propagation_analysis:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO cross_platform_correlations
                (track_name, artist, source_platform, target_platform, 
                 lag_hours, propagation_strength, analysis_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    track_name, artist,
                    prop['from_platform'], prop['to_platform'],
                    prop['hours_difference'],
                    abs(prop['score_change']) / 10.0,  # Normalize strength
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            
            return {
                "track_name": track_name,
                "artist": artist,
                "platforms": platforms,
                "platform_count": len(platforms),
                "first_platform": platforms[0] if platforms else None,
                "propagation_pattern": propagation_analysis,
                "total_propagation_time": sum(p['hours_difference'] for p in propagation_analysis),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def create_backup(self) -> str:
        """Create a backup of the database."""
        backup_filename = f"music_trends_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = self.backup_dir / backup_filename
        
        with self.get_connection() as source_conn:
            with sqlite3.connect(backup_path) as backup_conn:
                source_conn.backup(backup_conn)
        
        self.logger.info(f"Database backup created: {backup_path}")
        return str(backup_path)
    
    def export_to_csv(self, 
                     table: str, 
                     filepath: str,
                     days: Optional[int] = None) -> str:
        """Export table data to CSV."""
        with self.get_connection() as conn:
            if days:
                query = f'''
                SELECT * FROM {table}
                WHERE datetime(created_at) >= datetime('now', '-{days} days')
                ORDER BY created_at DESC
                '''
            else:
                query = f"SELECT * FROM {table} ORDER BY created_at DESC"
            
            df = pd.read_sql_query(query, conn)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            df.to_csv(filepath, index=False)
            self.logger.info(f"Exported {len(df)} rows from {table} to {filepath}")
            
        return filepath
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table row counts
            table_stats = {}
            tables = ['trends', 'trend_history', 'viral_predictions', 'cross_platform_correlations']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_stats[table] = cursor.fetchone()[0]
            
            # Data quality checks
            quality_issues = []
            
            # Check for missing track names or artists
            cursor.execute("SELECT COUNT(*) FROM trends WHERE track_name = '' OR artist = ''")
            missing_data = cursor.fetchone()[0]
            if missing_data > 0:
                quality_issues.append(f"{missing_data} trends with missing track name or artist")
            
            # Check for score outliers
            cursor.execute("SELECT COUNT(*) FROM trends WHERE score < 0 OR score > 100")
            invalid_scores = cursor.fetchone()[0]
            if invalid_scores > 0:
                quality_issues.append(f"{invalid_scores} trends with invalid scores")
            
            # Check for orphaned history records
            cursor.execute('''
            SELECT COUNT(*) FROM trend_history th
            LEFT JOIN trends t ON th.trend_id = t.id
            WHERE t.id IS NULL
            ''')
            orphaned_history = cursor.fetchone()[0]
            if orphaned_history > 0:
                quality_issues.append(f"{orphaned_history} orphaned history records")
            
            # Platform distribution
            cursor.execute('''
            SELECT platform, COUNT(*) as count
            FROM trends
            WHERE datetime(trend_date) >= datetime('now', '-7 days')
            GROUP BY platform
            ORDER BY count DESC
            ''')
            platform_distribution = dict(cursor.fetchall())
            
            # Log quality report
            cursor.execute('''
            INSERT INTO data_quality_logs
            (table_name, validation_type, validation_result, details, row_count, issue_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                'all_tables',
                'comprehensive_check',
                'pass' if not quality_issues else 'warning',
                json.dumps(quality_issues),
                sum(table_stats.values()),
                len(quality_issues)
            ))
            
            conn.commit()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "table_statistics": table_stats,
                "quality_issues": quality_issues,
                "platform_distribution": platform_distribution,
                "total_records": sum(table_stats.values()),
                "issue_count": len(quality_issues)
            }

# Example usage
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data store
    store = EnhancedMusicDataStore("test_enhanced.db")
    
    # Test saving trend data
    trend = TrendData(
        platform="tiktok",
        track_id="test123",
        track_name="Test Song",
        artist="Test Artist",
        score=85.5,
        rank=3,
        region="US",
        trend_date=datetime.now(),
        metadata={"views": 100000, "likes": 5000},
        first_detected=datetime.now()
    )
    
    trend_id = store.save_trend(trend)
    print(f"Saved trend with ID: {trend_id}")
    
    # Test viral prediction
    prediction = ViralPrediction(
        track_id="test123",
        track_name="Test Song",
        artist="Test Artist",
        confidence=0.85,
        predicted_peak_date=datetime.now() + timedelta(days=7),
        predicted_peak_score=95.0,
        prediction_features={"growth_rate": 1.5, "platform_count": 2},
        prediction_date=datetime.now()
    )
    
    pred_id = store.save_viral_prediction(prediction)
    print(f"Saved prediction with ID: {pred_id}")
    
    # Test querying
    df = store.get_trending_tracks(limit=10)
    print(f"Found {len(df)} trending tracks")
    
    # Test cross-platform analysis
    analysis = store.analyze_cross_platform_spread("Test Song", "Test Artist")
    print(f"Cross-platform analysis: {analysis}")
    
    # Generate quality report
    quality_report = store.get_data_quality_report()
    print(f"Data quality report: {quality_report}")
    
    # Create backup
    backup_path = store.create_backup()
    print(f"Backup created: {backup_path}")