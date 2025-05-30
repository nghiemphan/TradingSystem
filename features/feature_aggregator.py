"""
SMC Feature Aggregator
Combines all SMC features into unified feature vectors for AI models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from features.smc_calculator import MarketStructureAnalyzer, TrendDirection
from features.order_blocks import OrderBlockAnalyzer, OrderBlockType, OrderBlockStatus
from data.storage.cache_manager import get_cache_manager
from data.storage.file_storage import get_file_storage_manager

logger = logging.getLogger(__name__)

@dataclass
class SMCFeatureSet:
    """Complete SMC feature set for a single timeframe"""
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Market Structure Features
    trend: int  # -1: bearish, 0: sideways, 1: bullish
    trend_strength: float
    structure_quality: float
    recent_bos_count: int
    recent_choch_count: int
    recent_msb_count: int
    
    # Order Block Features
    total_obs: int
    fresh_obs_count: int
    tested_obs_count: int
    respected_obs_count: int
    bullish_obs_count: int
    bearish_obs_count: int
    ob_respect_rate: float
    ob_avg_strength: float
    
    # Confluence Features
    current_price_confluence: float
    nearest_ob_distance: float
    nearest_ob_type: int  # -1: bearish, 0: none, 1: bullish
    nearest_ob_strength: float
    
    # Bias Features
    structural_bias: int  # -1: bearish, 0: neutral, 1: bullish
    bias_strength: float
    bias_consistency: float
    
    # Recent Activity Features (last 10 bars)
    recent_hh_count: int
    recent_hl_count: int
    recent_lh_count: int
    recent_ll_count: int
    
    # Time-based Features
    hour_of_day: int
    day_of_week: int
    is_session_open: int  # 1 if major session active

class SMCFeatureAggregator:
    """
    Aggregates all SMC features into unified feature vectors
    """
    
    def __init__(self, cache_features: bool = True, save_features: bool = True):
        self.cache_features = cache_features
        self.save_features = save_features
        
        # Initialize analyzers
        self.ms_analyzer = MarketStructureAnalyzer()
        self.ob_analyzer = OrderBlockAnalyzer()
        
        # Initialize storage
        self.cache_manager = get_cache_manager() if cache_features else None
        self.file_manager = get_file_storage_manager() if save_features else None
        
        # Feature cache
        self._feature_cache = {}
        
    def extract_features(self, symbol: str, timeframe: str, data: pd.DataFrame,
                        use_cache: bool = True) -> Optional[SMCFeatureSet]:
        """
        Extract complete SMC feature set from market data
        
        Args:
            symbol: Trading symbol
            timeframe: Time frame
            data: OHLCV DataFrame
            use_cache: Whether to use cached features
            
        Returns:
            SMCFeatureSet or None if extraction fails
        """
        if data.empty or len(data) < 50:
            logger.warning(f"Insufficient data for feature extraction: {len(data)} bars")
            return None
        
        try:
            # Check cache first
            if use_cache and self.cache_manager:
                cached_features = self._get_cached_features(symbol, timeframe, data.index[-1])
                if cached_features:
                    return cached_features
            
            # Extract fresh features
            feature_set = self._extract_fresh_features(symbol, timeframe, data)
            
            if feature_set:
                # Cache features
                if self.cache_manager:
                    self._cache_features(feature_set)
                
                # Save features to file
                if self.file_manager:
                    self._save_features(feature_set)
                
                logger.debug(f"Extracted features for {symbol} {timeframe}")
                
            return feature_set
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol} {timeframe}: {e}")
            return None
    
    def _extract_fresh_features(self, symbol: str, timeframe: str, 
                               data: pd.DataFrame) -> Optional[SMCFeatureSet]:
        """Extract fresh features from market data"""
        
        # Run SMC analyses
        ms_analysis = self.ms_analyzer.analyze_market_structure(data)
        ob_analysis = self.ob_analyzer.analyze_order_blocks(data)
        
        if not ms_analysis or not ob_analysis:
            logger.error("SMC analysis failed")
            return None
        
        # Extract timestamp and current price
        timestamp = data.index[-1]
        current_price = data['Close'].iloc[-1]
        
        # Market Structure Features
        trend_value = self._encode_trend(ms_analysis.get('trend', TrendDirection.SIDEWAYS))
        trend_strength = ms_analysis.get('trend_strength', 0.0)
        structure_quality = ms_analysis.get('structure_quality', 0.0)
        
        # Count recent events (last 20 bars)
        recent_cutoff = timestamp - timedelta(hours=20)  # Adjust based on timeframe
        
        recent_bos = [e for e in ms_analysis.get('bos_events', []) 
                     if e['timestamp'] > recent_cutoff]
        recent_choch = [e for e in ms_analysis.get('choch_events', [])
                       if e['timestamp'] > recent_cutoff]
        recent_msb = [e for e in ms_analysis.get('msb_events', [])
                     if e['timestamp'] > recent_cutoff]
        
        # Order Block Features
        ob_metrics = ob_analysis.get('metrics', {})
        
        # Confluence Features
        confluence_score = self.ob_analyzer.get_confluence_score(current_price)
        nearby_obs = self.ob_analyzer.get_order_blocks_near_price(current_price, 0.001)
        
        nearest_ob_distance = 0.0
        nearest_ob_type = 0
        nearest_ob_strength = 0.0
        
        if nearby_obs:
            nearest_ob = nearby_obs[0]
            nearest_ob_distance = min(
                abs(current_price - nearest_ob.top),
                abs(current_price - nearest_ob.bottom)
            )
            nearest_ob_type = 1 if nearest_ob.ob_type == OrderBlockType.BULLISH else -1
            nearest_ob_strength = nearest_ob.strength
        
        # Bias Features
        bias_info = self.ms_analyzer.get_current_bias()
        structural_bias = self._encode_trend(bias_info.get('trend', TrendDirection.SIDEWAYS))
        bias_strength = bias_info.get('bias_strength', 0.0)
        
        # Calculate bias consistency
        bias_consistency = self._calculate_bias_consistency(ms_analysis)
        
        # Structure Point Counts
        structure_points = ms_analysis.get('structure_points', [])
        recent_points = [p for p in structure_points if p.timestamp > recent_cutoff]
        
        recent_hh = len([p for p in recent_points if 'HH' in str(p.structure_type)])
        recent_hl = len([p for p in recent_points if 'HL' in str(p.structure_type)])
        recent_lh = len([p for p in recent_points if 'LH' in str(p.structure_type)])
        recent_ll = len([p for p in recent_points if 'LL' in str(p.structure_type)])
        
        # Time-based Features
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        is_session_open = self._is_major_session_open(timestamp)
        
        # Create feature set
        feature_set = SMCFeatureSet(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            
            # Market Structure
            trend=trend_value,
            trend_strength=trend_strength,
            structure_quality=structure_quality,
            recent_bos_count=len(recent_bos),
            recent_choch_count=len(recent_choch),
            recent_msb_count=len(recent_msb),
            
            # Order Blocks
            total_obs=ob_metrics.get('total_obs', 0),
            fresh_obs_count=ob_metrics.get('fresh_count', 0),
            tested_obs_count=ob_metrics.get('tested_count', 0),
            respected_obs_count=ob_metrics.get('respected_count', 0),
            bullish_obs_count=ob_metrics.get('bullish_count', 0),
            bearish_obs_count=ob_metrics.get('bearish_count', 0),
            ob_respect_rate=ob_metrics.get('respect_rate', 0.0),
            ob_avg_strength=ob_metrics.get('avg_strength', 0.0),
            
            # Confluence
            current_price_confluence=confluence_score,
            nearest_ob_distance=nearest_ob_distance,
            nearest_ob_type=nearest_ob_type,
            nearest_ob_strength=nearest_ob_strength,
            
            # Bias
            structural_bias=structural_bias,
            bias_strength=bias_strength,
            bias_consistency=bias_consistency,
            
            # Recent Activity
            recent_hh_count=recent_hh,
            recent_hl_count=recent_hl,
            recent_lh_count=recent_lh,
            recent_ll_count=recent_ll,
            
            # Time
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_session_open=is_session_open
        )
        
        return feature_set
    
    def extract_multi_timeframe_features(self, symbol: str, 
                                       timeframes: List[str],
                                       data_dict: Dict[str, pd.DataFrame]) -> Dict[str, SMCFeatureSet]:
        """
        Extract features for multiple timeframes
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze
            data_dict: Dictionary mapping timeframes to data
            
        Returns:
            Dictionary mapping timeframes to feature sets
        """
        features = {}
        
        for tf in timeframes:
            if tf in data_dict and not data_dict[tf].empty:
                feature_set = self.extract_features(symbol, tf, data_dict[tf])
                if feature_set:
                    features[tf] = feature_set
                else:
                    logger.warning(f"Failed to extract features for {symbol} {tf}")
            else:
                logger.warning(f"No data available for {symbol} {tf}")
        
        return features
    
    def features_to_vector(self, feature_set: SMCFeatureSet) -> np.ndarray:
        """
        Convert feature set to numerical vector for ML models
        
        Args:
            feature_set: SMC feature set
            
        Returns:
            Numerical feature vector
        """
        vector = np.array([
            # Market Structure (6 features)
            feature_set.trend,
            feature_set.trend_strength,
            feature_set.structure_quality,
            feature_set.recent_bos_count,
            feature_set.recent_choch_count,
            feature_set.recent_msb_count,
            
            # Order Blocks (8 features)
            feature_set.total_obs,
            feature_set.fresh_obs_count,
            feature_set.tested_obs_count,
            feature_set.respected_obs_count,
            feature_set.bullish_obs_count,
            feature_set.bearish_obs_count,
            feature_set.ob_respect_rate,
            feature_set.ob_avg_strength,
            
            # Confluence (4 features)
            feature_set.current_price_confluence,
            feature_set.nearest_ob_distance,
            feature_set.nearest_ob_type,
            feature_set.nearest_ob_strength,
            
            # Bias (3 features)
            feature_set.structural_bias,
            feature_set.bias_strength,
            feature_set.bias_consistency,
            
            # Recent Activity (4 features)
            feature_set.recent_hh_count,
            feature_set.recent_hl_count,
            feature_set.recent_lh_count,
            feature_set.recent_ll_count,
            
            # Time (3 features)
            feature_set.hour_of_day / 24.0,  # Normalize
            feature_set.day_of_week / 7.0,   # Normalize
            feature_set.is_session_open
        ])
        
        return vector
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for ML models"""
        return [
            # Market Structure
            'trend', 'trend_strength', 'structure_quality',
            'recent_bos_count', 'recent_choch_count', 'recent_msb_count',
            
            # Order Blocks
            'total_obs', 'fresh_obs_count', 'tested_obs_count', 'respected_obs_count',
            'bullish_obs_count', 'bearish_obs_count', 'ob_respect_rate', 'ob_avg_strength',
            
            # Confluence
            'current_price_confluence', 'nearest_ob_distance', 'nearest_ob_type', 'nearest_ob_strength',
            
            # Bias
            'structural_bias', 'bias_strength', 'bias_consistency',
            
            # Recent Activity
            'recent_hh_count', 'recent_hl_count', 'recent_lh_count', 'recent_ll_count',
            
            # Time
            'hour_of_day_norm', 'day_of_week_norm', 'is_session_open'
        ]
    
    def create_feature_dataframe(self, feature_sets: List[SMCFeatureSet]) -> pd.DataFrame:
        """
        Create pandas DataFrame from list of feature sets
        
        Args:
            feature_sets: List of SMC feature sets
            
        Returns:
            DataFrame with features
        """
        if not feature_sets:
            return pd.DataFrame()
        
        data = []
        for fs in feature_sets:
            vector = self.features_to_vector(fs)
            row = {
                'timestamp': fs.timestamp,
                'symbol': fs.symbol,
                'timeframe': fs.timeframe,
                **dict(zip(self.get_feature_names(), vector))
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _encode_trend(self, trend: TrendDirection) -> int:
        """Encode trend direction to integer"""
        if trend == TrendDirection.BULLISH:
            return 1
        elif trend == TrendDirection.BEARISH:
            return -1
        else:
            return 0
    
    def _calculate_bias_consistency(self, ms_analysis: Dict) -> float:
        """Calculate bias consistency score"""
        structure_points = ms_analysis.get('structure_points', [])
        
        if len(structure_points) < 3:
            return 0.0
        
        # Get recent points
        recent_points = structure_points[-5:]
        
        # Count bullish vs bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        for point in recent_points:
            if 'HH' in str(point.structure_type) or 'HL' in str(point.structure_type):
                bullish_signals += 1
            elif 'LH' in str(point.structure_type) or 'LL' in str(point.structure_type):
                bearish_signals += 1
        
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return 0.0
        
        # Calculate consistency as dominance of one direction
        max_direction = max(bullish_signals, bearish_signals)
        consistency = max_direction / total_signals
        
        return consistency
    
    def _is_major_session_open(self, timestamp: datetime) -> int:
        """Check if major trading session is open"""
        hour_utc = timestamp.hour
        
        # Major sessions (UTC):
        # London: 8:00-17:00
        # New York: 13:00-22:00
        # Overlap: 13:00-17:00
        
        london_open = 8 <= hour_utc < 17
        ny_open = 13 <= hour_utc < 22
        
        return 1 if (london_open or ny_open) else 0
    
    def _get_cached_features(self, symbol: str, timeframe: str, 
                           timestamp: datetime) -> Optional[SMCFeatureSet]:
        """Get cached features if available and fresh"""
        if not self.cache_manager:
            return None
        
        try:
            cached_data = self.cache_manager.get_cached_features(symbol, timeframe)
            if cached_data and '_cached_at' in cached_data:
                cached_time = datetime.fromisoformat(cached_data['_cached_at'])
                age_minutes = (datetime.now() - cached_time).total_seconds() / 60
                
                # Use cache if less than 5 minutes old
                if age_minutes < 5:
                    # Convert cached data back to SMCFeatureSet
                    # This would need custom serialization/deserialization
                    pass
        except Exception as e:
            logger.error(f"Error retrieving cached features: {e}")
        
        return None
    
    def _cache_features(self, feature_set: SMCFeatureSet):
        """Cache feature set"""
        if not self.cache_manager:
            return
        
        try:
            # Convert feature set to dictionary for caching
            feature_dict = {
                'timestamp': feature_set.timestamp.isoformat(),
                'symbol': feature_set.symbol,
                'timeframe': feature_set.timeframe,
                'features': self.features_to_vector(feature_set).tolist(),
                'feature_names': self.get_feature_names()
            }
            
            self.cache_manager.cache_features(
                feature_set.symbol,
                feature_set.timeframe,
                feature_dict,
                ttl=300  # 5 minutes
            )
        except Exception as e:
            logger.error(f"Error caching features: {e}")
    
    def _save_features(self, feature_set: SMCFeatureSet):
        """Save features to file storage"""
        if not self.file_manager:
            return
        
        try:
            # Convert to dictionary for file storage
            feature_dict = {
                name: value for name, value in zip(
                    self.get_feature_names(),
                    self.features_to_vector(feature_set)
                )
            }
            
            self.file_manager.save_features(
                feature_set.symbol,
                feature_set.timeframe,
                feature_dict,
                feature_set.timestamp
            )
        except Exception as e:
            logger.error(f"Error saving features: {e}")

# Export main classes
__all__ = [
    'SMCFeatureAggregator',
    'SMCFeatureSet'
]