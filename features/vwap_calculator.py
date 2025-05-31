"""
VWAP Calculator - Smart Money Concepts Component
Calculates Volume Weighted Average Price with bands and SMC integration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VWAPType(Enum):
    """VWAP calculation types"""
    STANDARD = "standard"           # Standard VWAP from session start
    ANCHORED = "anchored"          # VWAP from specific anchor point
    ROLLING = "rolling"            # Rolling VWAP over N periods
    DAILY = "daily"               # Daily VWAP reset
    WEEKLY = "weekly"             # Weekly VWAP reset
    MONTHLY = "monthly"           # Monthly VWAP reset

class VWAPBandType(Enum):
    """VWAP band types"""
    STANDARD_DEVIATION = "std_dev"     # Standard deviation bands
    PERCENTAGE = "percentage"          # Percentage-based bands
    ATR = "atr"                       # ATR-based bands
    DYNAMIC = "dynamic"               # Dynamic bands based on volatility

class VWAPZone(Enum):
    """Price zones relative to VWAP"""
    ABOVE_UPPER_BAND = "above_upper_band"
    BETWEEN_VWAP_UPPER = "between_vwap_upper"
    AT_VWAP = "at_vwap"
    BETWEEN_LOWER_VWAP = "between_lower_vwap"
    BELOW_LOWER_BAND = "below_lower_band"

@dataclass
class VWAPBands:
    """VWAP bands configuration and values"""
    upper_band_1: float
    upper_band_2: float
    vwap_line: float
    lower_band_1: float
    lower_band_2: float
    
    # Band properties
    band_type: VWAPBandType
    band_width: float
    band_multiplier: float
    volatility_adjustment: float = 0.0

@dataclass
class VWAPAnalysis:
    """Complete VWAP analysis result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    vwap_type: VWAPType
    
    # Core VWAP values
    vwap_value: float
    vwap_bands: VWAPBands
    
    # Current market context
    current_price: float
    current_zone: VWAPZone
    distance_from_vwap: float  # Percentage
    distance_from_bands: Dict[str, float]
    
    # VWAP trend analysis
    vwap_slope: float  # Positive = rising, Negative = falling
    vwap_trend: str   # RISING, FALLING, SIDEWAYS
    trend_strength: float  # 0.0 to 1.0
    
    # Volume analysis
    total_volume: int
    volume_profile: Dict[str, float]  # Above/below VWAP volume distribution
    institutional_volume: float  # Large volume transactions
    
    # SMC integration
    support_resistance_levels: List[float]
    confluence_zones: List[Tuple[float, float]]  # (price_start, price_end)
    institutional_interest: float  # 0.0 to 1.0
    
    # Trading signals
    vwap_bias: str  # BULLISH, BEARISH, NEUTRAL
    signal_strength: float  # 0.0 to 1.0
    trading_recommendation: str
    entry_zones: List[Tuple[float, float]]
    target_zones: List[float]
    
    # Analysis quality
    confidence: float  # 0.0 to 1.0
    data_quality: float  # 0.0 to 1.0
    invalidation_level: Optional[float] = None

class VWAPCalculator:
    """
    VWAP Calculator with SMC integration
    Provides comprehensive VWAP analysis for institutional trading levels
    """
    
    def __init__(self,
                 # VWAP calculation parameters
                 vwap_source: str = "hlc3",  # hlc3, ohlc4, close, typical
                 anchor_session: bool = True,  # Anchor to session start
                 
                 # Band parameters
                 default_band_type: VWAPBandType = VWAPBandType.STANDARD_DEVIATION,
                 std_dev_multipliers: Tuple[float, float] = (1.0, 2.0),
                 percentage_bands: Tuple[float, float] = (0.5, 1.0),
                 atr_multipliers: Tuple[float, float] = (1.5, 2.5),
                 
                 # Volume analysis
                 institutional_volume_threshold: float = 2.0,  # 2x average volume
                 volume_confirmation: bool = True,
                 
                 # SMC integration
                 confluence_distance: float = 0.002,  # 0.2% for confluence
                 support_resistance_strength: float = 0.6,
                 
                 # Analysis parameters
                 min_bars_required: int = 20,
                 trend_lookback: int = 10,
                 volatility_adjustment: bool = True):
        
        self.vwap_source = vwap_source
        self.anchor_session = anchor_session
        
        self.default_band_type = default_band_type
        self.std_dev_multipliers = std_dev_multipliers
        self.percentage_bands = percentage_bands
        self.atr_multipliers = atr_multipliers
        
        self.institutional_volume_threshold = institutional_volume_threshold
        self.volume_confirmation = volume_confirmation
        
        self.confluence_distance = confluence_distance
        self.support_resistance_strength = support_resistance_strength
        
        self.min_bars_required = min_bars_required
        self.trend_lookback = trend_lookback
        self.volatility_adjustment = volatility_adjustment
        
        # Calculation cache
        self._vwap_cache = {}
        
    def calculate_vwap(self, data: pd.DataFrame, 
                      vwap_type: VWAPType = VWAPType.STANDARD,
                      anchor_point: Optional[datetime] = None) -> Optional[VWAPAnalysis]:
        """
        Main VWAP calculation and analysis function
        
        Args:
            data: OHLCV DataFrame
            vwap_type: Type of VWAP calculation
            anchor_point: Specific anchor point for anchored VWAP
            
        Returns:
            VWAPAnalysis or None if calculation fails
        """
        if data.empty or len(data) < self.min_bars_required:
            logger.warning(f"Insufficient data for VWAP calculation: {len(data)} bars")
            return None
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            logger.error("Missing required OHLCV columns for VWAP")
            return None
        
        try:
            # Step 1: Calculate price source
            price_series = self._calculate_price_source(data)
            
            # Step 2: Calculate VWAP based on type
            vwap_series = self._calculate_vwap_series(data, price_series, vwap_type, anchor_point)
            
            if vwap_series is None or vwap_series.empty:
                logger.error("Failed to calculate VWAP series")
                return None
            
            # Step 3: Calculate VWAP bands
            vwap_bands = self._calculate_vwap_bands(data, vwap_series, price_series)
            
            # Step 4: Analyze current market context
            current_price = data['Close'].iloc[-1]
            current_vwap = vwap_series.iloc[-1]
            current_zone = self._determine_vwap_zone(current_price, vwap_bands)
            
            # Step 5: Calculate distances and positions
            distance_from_vwap = ((current_price - current_vwap) / current_vwap) * 100
            distance_from_bands = self._calculate_band_distances(current_price, vwap_bands)
            
            # Step 6: VWAP trend analysis
            vwap_slope, vwap_trend, trend_strength = self._analyze_vwap_trend(vwap_series)
            
            # Step 7: Volume analysis
            volume_analysis = self._analyze_volume_profile(data, vwap_series)
            
            # Step 8: SMC integration
            smc_analysis = self._analyze_smc_confluence(data, vwap_series, vwap_bands)
            
            # Step 9: Trading signal generation
            trading_signals = self._generate_trading_signals(
                current_price, vwap_bands, vwap_trend, volume_analysis, smc_analysis
            )
            
            # Step 10: Quality assessment
            confidence, data_quality = self._assess_analysis_quality(data, vwap_series, volume_analysis)
            
            # Create comprehensive analysis
            analysis = VWAPAnalysis(
                timestamp=data.index[-1],
                symbol=getattr(data, 'symbol', 'UNKNOWN'),
                timeframe=getattr(data, 'timeframe', 'UNKNOWN'),
                vwap_type=vwap_type,
                
                # Core VWAP values
                vwap_value=current_vwap,
                vwap_bands=vwap_bands,
                
                # Current context
                current_price=current_price,
                current_zone=current_zone,
                distance_from_vwap=distance_from_vwap,
                distance_from_bands=distance_from_bands,
                
                # Trend analysis
                vwap_slope=vwap_slope,
                vwap_trend=vwap_trend,
                trend_strength=trend_strength,
                
                # Volume analysis
                total_volume=volume_analysis['total_volume'],
                volume_profile=volume_analysis['profile'],
                institutional_volume=volume_analysis['institutional_volume'],
                
                # SMC integration
                support_resistance_levels=smc_analysis['support_resistance'],
                confluence_zones=smc_analysis['confluence_zones'],
                institutional_interest=smc_analysis['institutional_interest'],
                
                # Trading signals
                vwap_bias=trading_signals['bias'],
                signal_strength=trading_signals['strength'],
                trading_recommendation=trading_signals['recommendation'],
                entry_zones=trading_signals['entry_zones'],
                target_zones=trading_signals['target_zones'],
                invalidation_level=trading_signals['invalidation_level'],
                
                # Quality metrics
                confidence=confidence,
                data_quality=data_quality
            )
            
            logger.debug(f"VWAP analysis completed - VWAP: {current_vwap:.5f}, Zone: {current_zone.value}, Trend: {vwap_trend}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"VWAP calculation failed: {e}")
            return None
    
    def _calculate_price_source(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price source for VWAP calculation"""
        try:
            if self.vwap_source == "hlc3":
                return (data['High'] + data['Low'] + data['Close']) / 3
            elif self.vwap_source == "ohlc4":
                return (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
            elif self.vwap_source == "close":
                return data['Close']
            elif self.vwap_source == "typical":
                return (data['High'] + data['Low'] + data['Close']) / 3
            else:
                logger.warning(f"Unknown VWAP source: {self.vwap_source}, using hlc3")
                return (data['High'] + data['Low'] + data['Close']) / 3
                
        except Exception as e:
            logger.error(f"Error calculating price source: {e}")
            return data['Close']
    
    def _calculate_vwap_series(self, data: pd.DataFrame, price_series: pd.Series, 
                              vwap_type: VWAPType, anchor_point: Optional[datetime] = None) -> Optional[pd.Series]:
        """Calculate VWAP series based on type"""
        try:
            if vwap_type == VWAPType.STANDARD:
                return self._calculate_standard_vwap(data, price_series)
            elif vwap_type == VWAPType.ANCHORED:
                return self._calculate_anchored_vwap(data, price_series, anchor_point)
            elif vwap_type == VWAPType.ROLLING:
                return self._calculate_rolling_vwap(data, price_series)
            elif vwap_type == VWAPType.DAILY:
                return self._calculate_session_vwap(data, price_series, 'D')
            elif vwap_type == VWAPType.WEEKLY:
                return self._calculate_session_vwap(data, price_series, 'W')
            elif vwap_type == VWAPType.MONTHLY:
                return self._calculate_session_vwap(data, price_series, 'M')
            else:
                logger.warning(f"Unknown VWAP type: {vwap_type}, using standard")
                return self._calculate_standard_vwap(data, price_series)
                
        except Exception as e:
            logger.error(f"Error calculating VWAP series: {e}")
            return None
    
    def _calculate_standard_vwap(self, data: pd.DataFrame, price_series: pd.Series) -> pd.Series:
        """Calculate standard VWAP from start of data"""
        volume_price = price_series * data['Volume']
        cumulative_volume_price = volume_price.cumsum()
        cumulative_volume = data['Volume'].cumsum()
        
        # Avoid division by zero
        vwap = cumulative_volume_price / cumulative_volume.replace(0, np.nan)
        
        # FIXED: Replace deprecated fillna method
        return vwap.ffill()  # Changed from fillna(method='ffill')
    
    def _calculate_anchored_vwap(self, data: pd.DataFrame, price_series: pd.Series, 
                        anchor_point: Optional[datetime] = None) -> pd.Series:
        """Calculate anchored VWAP from specific point"""
        if anchor_point is None:
            anchor_point = data.index[0]
        
        # Find anchor index
        try:
            anchor_idx = data.index.get_loc(anchor_point)
        except KeyError:
            anchor_idx = data.index.get_indexer([anchor_point], method='nearest')[0]
        
        # Calculate VWAP from anchor point
        vwap_series = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            if i < anchor_idx:
                vwap_series.iloc[i] = np.nan
            else:
                subset_data = data.iloc[anchor_idx:i+1]
                subset_price = price_series.iloc[anchor_idx:i+1]
                
                volume_price = subset_price * subset_data['Volume']
                total_volume_price = volume_price.sum()
                total_volume = subset_data['Volume'].sum()
                
                if total_volume > 0:
                    vwap_series.iloc[i] = total_volume_price / total_volume
                else:
                    vwap_series.iloc[i] = subset_price.iloc[-1]
        
        # FIXED: Replace deprecated fillna method
        return vwap_series.ffill()  # Changed from fillna(method='ffill')
    
    def _calculate_rolling_vwap(self, data: pd.DataFrame, price_series: pd.Series, 
                           window: int = 20) -> pd.Series:
        """Calculate rolling VWAP over specified window"""
        volume_price = price_series * data['Volume']
        
        rolling_volume_price = volume_price.rolling(window=window).sum()
        rolling_volume = data['Volume'].rolling(window=window).sum()
        
        vwap = rolling_volume_price / rolling_volume.replace(0, np.nan)
        
        # FIXED: Replace deprecated fillna method
        return vwap.ffill()  # Changed from fillna(method='ffill')
    
    def _calculate_session_vwap(self, data: pd.DataFrame, price_series: pd.Series, 
                           frequency: str) -> pd.Series:
        """Calculate VWAP reset by session (daily, weekly, monthly)"""
        groups = data.groupby(pd.Grouper(freq=frequency))
        
        vwap_series = pd.Series(index=data.index, dtype=float)
        
        for name, group in groups:
            if len(group) == 0:
                continue
            
            group_price = price_series.loc[group.index]
            volume_price = group_price * group['Volume']
            
            cumulative_volume_price = volume_price.cumsum()
            cumulative_volume = group['Volume'].cumsum()
            
            group_vwap = cumulative_volume_price / cumulative_volume.replace(0, np.nan)
            vwap_series.loc[group.index] = group_vwap
        
        # FIXED: Replace deprecated fillna method
        return vwap_series.ffill()  # Changed from fillna(method='ffill')
    
    def _calculate_vwap_bands(self, data: pd.DataFrame, vwap_series: pd.Series, 
                             price_series: pd.Series) -> VWAPBands:
        """Calculate VWAP bands based on specified method"""
        try:
            current_vwap = vwap_series.iloc[-1]
            
            if self.default_band_type == VWAPBandType.STANDARD_DEVIATION:
                bands = self._calculate_std_dev_bands(vwap_series, price_series)
            elif self.default_band_type == VWAPBandType.PERCENTAGE:
                bands = self._calculate_percentage_bands(current_vwap)
            elif self.default_band_type == VWAPBandType.ATR:
                bands = self._calculate_atr_bands(data, current_vwap)
            elif self.default_band_type == VWAPBandType.DYNAMIC:
                bands = self._calculate_dynamic_bands(data, vwap_series, price_series)
            else:
                bands = self._calculate_std_dev_bands(vwap_series, price_series)
            
            return bands
            
        except Exception as e:
            logger.error(f"Error calculating VWAP bands: {e}")
            return self._create_default_bands(vwap_series.iloc[-1])
    
    def _calculate_std_dev_bands(self, vwap_series: pd.Series, price_series: pd.Series) -> VWAPBands:
        """Calculate standard deviation based VWAP bands"""
        current_vwap = vwap_series.iloc[-1]
        
        # Calculate standard deviation of price from VWAP
        price_deviation = price_series - vwap_series
        std_dev = price_deviation.std()
        
        if np.isnan(std_dev) or std_dev == 0:
            std_dev = current_vwap * 0.01  # Default to 1% of VWAP
        
        # Apply volatility adjustment if enabled
        if self.volatility_adjustment:
            recent_volatility = price_series.tail(20).std()
            avg_volatility = price_series.std()
            vol_adjustment = recent_volatility / avg_volatility if avg_volatility > 0 else 1.0
            std_dev *= vol_adjustment
        
        upper_1 = current_vwap + (std_dev * self.std_dev_multipliers[0])
        upper_2 = current_vwap + (std_dev * self.std_dev_multipliers[1])
        lower_1 = current_vwap - (std_dev * self.std_dev_multipliers[0])
        lower_2 = current_vwap - (std_dev * self.std_dev_multipliers[1])
        
        return VWAPBands(
            upper_band_1=upper_1,
            upper_band_2=upper_2,
            vwap_line=current_vwap,
            lower_band_1=lower_1,
            lower_band_2=lower_2,
            band_type=VWAPBandType.STANDARD_DEVIATION,
            band_width=upper_2 - lower_2,
            band_multiplier=self.std_dev_multipliers[1],
            volatility_adjustment=std_dev
        )
    
    def _calculate_percentage_bands(self, current_vwap: float) -> VWAPBands:
        """Calculate percentage-based VWAP bands"""
        upper_1 = current_vwap * (1 + self.percentage_bands[0] / 100)
        upper_2 = current_vwap * (1 + self.percentage_bands[1] / 100)
        lower_1 = current_vwap * (1 - self.percentage_bands[0] / 100)
        lower_2 = current_vwap * (1 - self.percentage_bands[1] / 100)
        
        return VWAPBands(
            upper_band_1=upper_1,
            upper_band_2=upper_2,
            vwap_line=current_vwap,
            lower_band_1=lower_1,
            lower_band_2=lower_2,
            band_type=VWAPBandType.PERCENTAGE,
            band_width=upper_2 - lower_2,
            band_multiplier=self.percentage_bands[1]
        )
    
    def _calculate_atr_bands(self, data: pd.DataFrame, current_vwap: float) -> VWAPBands:
        """Calculate ATR-based VWAP bands"""
        # Calculate ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1]
        
        if np.isnan(atr) or atr == 0:
            atr = current_vwap * 0.02  # Default to 2% of VWAP
        
        upper_1 = current_vwap + (atr * self.atr_multipliers[0])
        upper_2 = current_vwap + (atr * self.atr_multipliers[1])
        lower_1 = current_vwap - (atr * self.atr_multipliers[0])
        lower_2 = current_vwap - (atr * self.atr_multipliers[1])
        
        return VWAPBands(
            upper_band_1=upper_1,
            upper_band_2=upper_2,
            vwap_line=current_vwap,
            lower_band_1=lower_1,
            lower_band_2=lower_2,
            band_type=VWAPBandType.ATR,
            band_width=upper_2 - lower_2,
            band_multiplier=self.atr_multipliers[1],
            volatility_adjustment=atr
        )
    
    def _calculate_dynamic_bands(self, data: pd.DataFrame, vwap_series: pd.Series, 
                                price_series: pd.Series) -> VWAPBands:
        """Calculate dynamic bands combining multiple methods"""
        current_vwap = vwap_series.iloc[-1]
        
        # Combine standard deviation and ATR
        std_bands = self._calculate_std_dev_bands(vwap_series, price_series)
        atr_bands = self._calculate_atr_bands(data, current_vwap)
        
        # Average the band distances
        std_distance = (std_bands.upper_band_1 - current_vwap)
        atr_distance = (atr_bands.upper_band_1 - current_vwap)
        avg_distance_1 = (std_distance + atr_distance) / 2
        
        std_distance_2 = (std_bands.upper_band_2 - current_vwap)
        atr_distance_2 = (atr_bands.upper_band_2 - current_vwap)
        avg_distance_2 = (std_distance_2 + atr_distance_2) / 2
        
        return VWAPBands(
            upper_band_1=current_vwap + avg_distance_1,
            upper_band_2=current_vwap + avg_distance_2,
            vwap_line=current_vwap,
            lower_band_1=current_vwap - avg_distance_1,
            lower_band_2=current_vwap - avg_distance_2,
            band_type=VWAPBandType.DYNAMIC,
            band_width=avg_distance_2 * 2,
            band_multiplier=2.0,
            volatility_adjustment=(std_bands.volatility_adjustment + atr_bands.volatility_adjustment) / 2
        )
    
    def _create_default_bands(self, current_vwap: float) -> VWAPBands:
        """Create default bands when calculation fails"""
        default_distance = current_vwap * 0.01  # 1% of VWAP
        
        return VWAPBands(
            upper_band_1=current_vwap + default_distance,
            upper_band_2=current_vwap + (default_distance * 2),
            vwap_line=current_vwap,
            lower_band_1=current_vwap - default_distance,
            lower_band_2=current_vwap - (default_distance * 2),
            band_type=VWAPBandType.PERCENTAGE,
            band_width=default_distance * 4,
            band_multiplier=2.0
        )
    
    def _determine_vwap_zone(self, current_price: float, vwap_bands: VWAPBands) -> VWAPZone:
        """Determine which VWAP zone current price is in"""
        if current_price >= vwap_bands.upper_band_2:
            return VWAPZone.ABOVE_UPPER_BAND
        elif current_price >= vwap_bands.vwap_line:
            return VWAPZone.BETWEEN_VWAP_UPPER
        elif abs(current_price - vwap_bands.vwap_line) / vwap_bands.vwap_line < 0.001:  # Within 0.1%
            return VWAPZone.AT_VWAP
        elif current_price >= vwap_bands.lower_band_2:
            return VWAPZone.BETWEEN_LOWER_VWAP
        else:
            return VWAPZone.BELOW_LOWER_BAND
    
    def _calculate_band_distances(self, current_price: float, vwap_bands: VWAPBands) -> Dict[str, float]:
        """Calculate distances from current price to all bands"""
        distances = {}
        
        # Distance to VWAP
        distances['vwap'] = ((current_price - vwap_bands.vwap_line) / vwap_bands.vwap_line) * 100
        
        # Distance to bands
        distances['upper_band_1'] = ((vwap_bands.upper_band_1 - current_price) / current_price) * 100
        distances['upper_band_2'] = ((vwap_bands.upper_band_2 - current_price) / current_price) * 100
        distances['lower_band_1'] = ((current_price - vwap_bands.lower_band_1) / current_price) * 100
        distances['lower_band_2'] = ((current_price - vwap_bands.lower_band_2) / current_price) * 100
        
        return distances
    
    def _analyze_vwap_trend(self, vwap_series: pd.Series) -> Tuple[float, str, float]:
        """Analyze VWAP trend direction and strength"""
        try:
            if len(vwap_series) < self.trend_lookback:
                return 0.0, "SIDEWAYS", 0.0
            
            recent_vwap = vwap_series.tail(self.trend_lookback)
            
            # Calculate slope using linear regression
            x = np.arange(len(recent_vwap))
            y = recent_vwap.values
            
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                # Normalize slope relative to VWAP value
                normalized_slope = slope / recent_vwap.iloc[-1] * 100
                
                # Determine trend direction
                if normalized_slope > 0.01:  # More than 0.01% per bar
                    trend = "RISING"
                elif normalized_slope < -0.01:
                    trend = "FALLING"
                else:
                    trend = "SIDEWAYS"
                
                # Calculate trend strength
                trend_strength = min(abs(normalized_slope) / 0.1, 1.0)  # Max at 0.1% per bar
                
                return normalized_slope, trend, trend_strength
            else:
                return 0.0, "SIDEWAYS", 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing VWAP trend: {e}")
            return 0.0, "SIDEWAYS", 0.0
    
    def _analyze_volume_profile(self, data: pd.DataFrame, vwap_series: pd.Series) -> Dict:
        """Analyze volume distribution relative to VWAP"""
        try:
            total_volume = data['Volume'].sum()
            
            # Volume above and below VWAP
            above_vwap_mask = data['Close'] > vwap_series
            below_vwap_mask = data['Close'] <= vwap_series
            
            volume_above = data.loc[above_vwap_mask, 'Volume'].sum()
            volume_below = data.loc[below_vwap_mask, 'Volume'].sum()
            
            # Calculate percentages
            if total_volume > 0:
                above_percentage = (volume_above / total_volume) * 100
                below_percentage = (volume_below / total_volume) * 100
            else:
                above_percentage = below_percentage = 50.0
            
            # Identify institutional volume (large transactions)
            average_volume = data['Volume'].mean()
            institutional_threshold = average_volume * self.institutional_volume_threshold
            institutional_volume_total = data[data['Volume'] >= institutional_threshold]['Volume'].sum()
            institutional_percentage = (institutional_volume_total / total_volume * 100) if total_volume > 0 else 0.0
            
            return {
                'total_volume': int(total_volume),
                'profile': {
                    'above_vwap': above_percentage,
                    'below_vwap': below_percentage,
                    'volume_above': int(volume_above),
                    'volume_below': int(volume_below)
                },
                'institutional_volume': institutional_percentage,
                'average_volume': average_volume,
                'institutional_threshold': institutional_threshold
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return {
                'total_volume': 0,
                'profile': {'above_vwap': 50.0, 'below_vwap': 50.0, 'volume_above': 0, 'volume_below': 0},
                'institutional_volume': 0.0,
                'average_volume': 0.0,
                'institutional_threshold': 0.0
            }
    
    def _analyze_smc_confluence(self, data: pd.DataFrame, vwap_series: pd.Series, 
                               vwap_bands: VWAPBands) -> Dict:
        """Analyze SMC confluence with VWAP levels"""
        try:
            # Key VWAP levels for confluence analysis
            key_levels = [
                vwap_bands.upper_band_2,
                vwap_bands.upper_band_1,
                vwap_bands.vwap_line,
                vwap_bands.lower_band_1,
                vwap_bands.lower_band_2
            ]
            
            # Find support and resistance levels
            support_resistance_levels = []
            confluence_zones = []
            
            for level in key_levels:
                # Count touches and bounces at this level
                touches = self._count_level_touches(data, level, self.confluence_distance)
                
                if touches['total_touches'] >= 2:
                    bounce_rate = touches['bounces'] / touches['total_touches'] if touches['total_touches'] > 0 else 0
                    
                    if bounce_rate >= self.support_resistance_strength:
                        support_resistance_levels.append(level)
                        
                        # Create confluence zone around the level
                        zone_width = level * self.confluence_distance
                        confluence_zones.append((level - zone_width, level + zone_width))
            
            # Calculate institutional interest
            institutional_interest = self._calculate_institutional_interest(data, vwap_series, key_levels)
            
            return {
                'support_resistance': support_resistance_levels,
                'confluence_zones': confluence_zones,
                'institutional_interest': institutional_interest,
                'key_levels': key_levels
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SMC confluence: {e}")
            return {
                'support_resistance': [],
                'confluence_zones': [],
                'institutional_interest': 0.0,
                'key_levels': []
            }
    
    def _count_level_touches(self, data: pd.DataFrame, level: float, tolerance: float) -> Dict[str, int]:
        """Count touches and bounces at a specific price level"""
        try:
            level_tolerance = level * tolerance
            lower_bound = level - level_tolerance
            upper_bound = level + level_tolerance
            
            # Find bars that touched the level
            touches = data[(data['Low'] <= upper_bound) & (data['High'] >= lower_bound)]
            
            total_touches = len(touches)
            bounces = 0
            
            for idx, touch in touches.iterrows():
                # Look at next bar to see if it bounced
                next_idx = data.index.get_loc(idx) + 1
                
                if next_idx < len(data):
                    next_bar = data.iloc[next_idx]
                    
                    # Determine if it was a bounce based on close relative to level
                    if touch['Low'] <= level <= touch['High']:
                        if (level < touch['Close'] and next_bar['Close'] > level) or \
                           (level > touch['Close'] and next_bar['Close'] < level):
                            bounces += 1
            
            return {
                'total_touches': total_touches,
                'bounces': bounces,
                'breaks': total_touches - bounces
            }
            
        except Exception as e:
            logger.error(f"Error counting level touches: {e}")
            return {'total_touches': 0, 'bounces': 0, 'breaks': 0}
    
    def _calculate_institutional_interest(self, data: pd.DataFrame, vwap_series: pd.Series, 
                                        key_levels: List[float]) -> float:
        """Calculate institutional interest score based on volume and VWAP interaction"""
        try:
            institutional_score = 0.0
            
            # Volume concentration near VWAP
            vwap_current = vwap_series.iloc[-1]
            vwap_tolerance = vwap_current * 0.002  # 0.2% tolerance
            
            near_vwap_data = data[
                (data['Low'] <= vwap_current + vwap_tolerance) & 
                (data['High'] >= vwap_current - vwap_tolerance)
            ]
            
            if len(near_vwap_data) > 0:
                near_vwap_volume = near_vwap_data['Volume'].sum()
                total_volume = data['Volume'].sum()
                
                if total_volume > 0:
                    vwap_volume_concentration = near_vwap_volume / total_volume
                    institutional_score += min(vwap_volume_concentration * 2, 0.4)  # Max 0.4
            
            # Large volume transactions
            average_volume = data['Volume'].mean()
            large_volume_threshold = average_volume * self.institutional_volume_threshold
            large_volume_bars = len(data[data['Volume'] >= large_volume_threshold])
            
            if len(data) > 0:
                large_volume_percentage = large_volume_bars / len(data)
                institutional_score += min(large_volume_percentage * 2, 0.3)  # Max 0.3
            
            # VWAP respect (price tends to return to VWAP)
            distances_from_vwap = np.abs(data['Close'] - vwap_series) / vwap_series
            average_distance = distances_from_vwap.mean()
            
            # Lower average distance indicates higher VWAP respect
            vwap_respect = max(0, 1 - (average_distance * 100))  # Convert to 0-1 scale
            institutional_score += vwap_respect * 0.3  # Max 0.3
            
            return min(institutional_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating institutional interest: {e}")
            return 0.0
    
    def _generate_trading_signals(self, current_price: float, vwap_bands: VWAPBands, 
                                 vwap_trend: str, volume_analysis: Dict, smc_analysis: Dict) -> Dict:
        """Generate trading signals based on VWAP analysis"""
        try:
            # Determine bias
            bias = "NEUTRAL"
            strength = 0.0
            
            # Price position bias
            if current_price > vwap_bands.vwap_line:
                position_bias = "BULLISH"
                position_strength = min((current_price - vwap_bands.vwap_line) / vwap_bands.vwap_line * 10, 1.0)
            elif current_price < vwap_bands.vwap_line:
                position_bias = "BEARISH"
                position_strength = min((vwap_bands.vwap_line - current_price) / vwap_bands.vwap_line * 10, 1.0)
            else:
                position_bias = "NEUTRAL"
                position_strength = 0.0
            
            # Trend bias
            trend_bias = vwap_trend
            trend_strength_factor = 0.5 if trend_bias != "SIDEWAYS" else 0.0
            
            # Volume bias
            volume_profile = volume_analysis['profile']
            if volume_profile['above_vwap'] > 60:
                volume_bias = "BULLISH"
                volume_strength = (volume_profile['above_vwap'] - 50) / 50
            elif volume_profile['below_vwap'] > 60:
                volume_bias = "BEARISH"
                volume_strength = (volume_profile['below_vwap'] - 50) / 50
            else:
                volume_bias = "NEUTRAL"
                volume_strength = 0.0
            
            # Combine biases
            bias_scores = {
                "BULLISH": 0,
                "BEARISH": 0,
                "NEUTRAL": 0
            }
            
            # Weight the biases
            if position_bias != "NEUTRAL":
                bias_scores[position_bias] += position_strength * 0.4
            
            if trend_bias != "SIDEWAYS":
                trend_bias_direction = "BULLISH" if trend_bias == "RISING" else "BEARISH"
                bias_scores[trend_bias_direction] += trend_strength_factor * 0.3
            
            if volume_bias != "NEUTRAL":
                bias_scores[volume_bias] += volume_strength * 0.3
            
            # Determine final bias
            max_bias = max(bias_scores, key=bias_scores.get)
            max_score = bias_scores[max_bias]
            
            if max_score > 0.3:
                bias = max_bias
                strength = min(max_score, 1.0)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(bias, strength, current_price, vwap_bands)
            
            # Generate entry zones
            entry_zones = self._generate_entry_zones(bias, vwap_bands, smc_analysis)
            
            # Generate target zones
            target_zones = self._generate_target_zones(bias, vwap_bands, current_price)
            
            # Generate invalidation level
            invalidation_level = self._generate_invalidation_level(bias, vwap_bands, current_price)
            
            return {
                'bias': bias,
                'strength': strength,
                'recommendation': recommendation,
                'entry_zones': entry_zones,
                'target_zones': target_zones,
                'invalidation_level': invalidation_level
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {
                'bias': "NEUTRAL",
                'strength': 0.0,
                'recommendation': "NO_CLEAR_SIGNAL",
                'entry_zones': [],
                'target_zones': [],
                'invalidation_level': None
            }
    
    def _generate_recommendation(self, bias: str, strength: float, current_price: float, 
                               vwap_bands: VWAPBands) -> str:
        """Generate trading recommendation"""
        if bias == "NEUTRAL" or strength < 0.3:
            return "WAIT_FOR_SETUP"
        
        # Check if price is at extreme levels
        distance_from_vwap = abs(current_price - vwap_bands.vwap_line) / vwap_bands.vwap_line
        
        if distance_from_vwap > 0.02:  # More than 2% from VWAP
            if bias == "BULLISH" and current_price < vwap_bands.vwap_line:
                return "BUY_PULLBACK_TO_VWAP"
            elif bias == "BEARISH" and current_price > vwap_bands.vwap_line:
                return "SELL_PULLBACK_TO_VWAP"
            elif bias == "BULLISH" and current_price > vwap_bands.upper_band_1:
                return "WAIT_FOR_PULLBACK"
            elif bias == "BEARISH" and current_price < vwap_bands.lower_band_1:
                return "WAIT_FOR_PULLBACK"
        
        # Standard recommendations
        if strength >= 0.7:
            return f"STRONG_{bias}_SIGNAL"
        elif strength >= 0.5:
            return f"MODERATE_{bias}_SIGNAL"
        else:
            return f"WEAK_{bias}_SIGNAL"
    
    def _generate_entry_zones(self, bias: str, vwap_bands: VWAPBands, smc_analysis: Dict) -> List[Tuple[float, float]]:
        """Generate entry zones based on bias and VWAP levels"""
        entry_zones = []
        
        if bias == "BULLISH":
            # Entry zones for bullish bias
            entry_zones.append((vwap_bands.lower_band_1, vwap_bands.vwap_line))
            
            # Add confluence zones if available
            for zone_start, zone_end in smc_analysis.get('confluence_zones', []):
                if zone_start < vwap_bands.vwap_line:
                    entry_zones.append((zone_start, zone_end))
                    
        elif bias == "BEARISH":
            # Entry zones for bearish bias
            entry_zones.append((vwap_bands.vwap_line, vwap_bands.upper_band_1))
            
            # Add confluence zones if available
            for zone_start, zone_end in smc_analysis.get('confluence_zones', []):
                if zone_end > vwap_bands.vwap_line:
                    entry_zones.append((zone_start, zone_end))
        
        return entry_zones[:3]  # Limit to 3 entry zones
    
    def _generate_target_zones(self, bias: str, vwap_bands: VWAPBands, current_price: float) -> List[float]:
        """Generate target zones based on bias"""
        targets = []
        
        if bias == "BULLISH":
            if current_price < vwap_bands.vwap_line:
                targets.append(vwap_bands.vwap_line)
            targets.extend([vwap_bands.upper_band_1, vwap_bands.upper_band_2])
            
        elif bias == "BEARISH":
            if current_price > vwap_bands.vwap_line:
                targets.append(vwap_bands.vwap_line)
            targets.extend([vwap_bands.lower_band_1, vwap_bands.lower_band_2])
        
        return targets
    
    def _generate_invalidation_level(self, bias: str, vwap_bands: VWAPBands, current_price: float) -> Optional[float]:
        """Generate invalidation level for the bias"""
        if bias == "BULLISH":
            return vwap_bands.lower_band_2
        elif bias == "BEARISH":
            return vwap_bands.upper_band_2
        else:
            return None
    
    def _assess_analysis_quality(self, data: pd.DataFrame, vwap_series: pd.Series, 
                                volume_analysis: Dict) -> Tuple[float, float]:
        """Assess quality and confidence of VWAP analysis"""
        try:
            confidence = 0.0
            data_quality = 0.0
            
            # Data quality assessment
            if len(data) >= 50:
                data_quality += 0.4
            elif len(data) >= 20:
                data_quality += 0.2
            
            # Volume quality
            total_volume = volume_analysis['total_volume']
            if total_volume > 10000:
                data_quality += 0.3
            elif total_volume > 1000:
                data_quality += 0.2
            
            # Data completeness
            if not vwap_series.isna().any():
                data_quality += 0.3
            
            # Confidence assessment
            # Volume distribution clarity
            volume_profile = volume_analysis['profile']
            volume_imbalance = abs(volume_profile['above_vwap'] - 50)
            if volume_imbalance > 20:
                confidence += 0.3
            elif volume_imbalance > 10:
                confidence += 0.2
            
            # VWAP stability (less volatile VWAP = higher confidence)
            vwap_volatility = vwap_series.pct_change().std()
            if vwap_volatility < 0.01:
                confidence += 0.3
            elif vwap_volatility < 0.02:
                confidence += 0.2
            
            # Institutional volume presence
            institutional_volume = volume_analysis['institutional_volume']
            if institutional_volume > 20:
                confidence += 0.2
            elif institutional_volume > 10:
                confidence += 0.1
            
            # Time factor (more recent data = higher confidence)
            if len(data) > 0:
                latest_time = data.index[-1]
                time_factor = min(1.0, 24 / max(1, (datetime.now() - latest_time).total_seconds() / 3600))
                confidence += time_factor * 0.2
            
            return min(confidence, 1.0), min(data_quality, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing analysis quality: {e}")
            return 0.5, 0.5

# Utility functions for integration with other SMC components
def get_vwap_levels(analysis: VWAPAnalysis) -> Dict[str, float]:
    """Get key VWAP levels for integration with other components"""
    if not analysis:
        return {}
    
    return {
        'vwap': analysis.vwap_value,
        'upper_band_1': analysis.vwap_bands.upper_band_1,
        'upper_band_2': analysis.vwap_bands.upper_band_2,
        'lower_band_1': analysis.vwap_bands.lower_band_1,
        'lower_band_2': analysis.vwap_bands.lower_band_2
    }

def calculate_vwap_bias_integration(analysis: VWAPAnalysis) -> Dict[str, Union[str, float]]:
    """Calculate VWAP bias for integration with BIAS analyzer"""
    if not analysis:
        return {'direction': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
    
    # Convert VWAP bias to standardized format
    direction_mapping = {
        'BULLISH': 'BULLISH',
        'BEARISH': 'BEARISH',
        'NEUTRAL': 'NEUTRAL'
    }
    
    return {
        'direction': direction_mapping.get(analysis.vwap_bias, 'NEUTRAL'),
        'strength': analysis.signal_strength,
        'confidence': analysis.confidence,
        'vwap_level': analysis.vwap_value,
        'current_zone': analysis.current_zone.value,
        'trend': analysis.vwap_trend,
        'institutional_interest': analysis.institutional_interest
    }

def get_vwap_confluence_score(analysis: VWAPAnalysis, price_level: float, tolerance: float = 0.001) -> float:
    """Calculate confluence score for a price level with VWAP elements"""
    if not analysis:
        return 0.0
    
    confluence_score = 0.0
    vwap_levels = get_vwap_levels(analysis)
    
    for level_name, level_value in vwap_levels.items():
        distance = abs(price_level - level_value) / level_value
        
        if distance <= tolerance:
            # Weight different levels
            if level_name == 'vwap':
                confluence_score += 0.4
            elif 'band_1' in level_name:
                confluence_score += 0.3
            elif 'band_2' in level_name:
                confluence_score += 0.2
    
    # Add institutional interest bonus
    confluence_score += analysis.institutional_interest * 0.1
    
    return min(confluence_score, 1.0)

# Export main classes and functions
__all__ = [
    'VWAPCalculator',
    'VWAPAnalysis',
    'VWAPBands',
    'VWAPType',
    'VWAPBandType',
    'VWAPZone',
    'get_vwap_levels',
    'calculate_vwap_bias_integration',
    'get_vwap_confluence_score'
]