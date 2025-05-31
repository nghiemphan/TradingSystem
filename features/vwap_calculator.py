"""
VWAP (Volume Weighted Average Price) Calculator
Calculates various VWAP types and bands for institutional trading analysis
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
    STANDARD = "standard"        # Standard VWAP from session start
    ANCHORED = "anchored"        # VWAP from specific anchor point
    ROLLING = "rolling"          # Rolling VWAP over fixed period
    WEEKLY = "weekly"           # Weekly VWAP reset
    MONTHLY = "monthly"         # Monthly VWAP reset

class VWAPBandType(Enum):
    """VWAP band types"""
    STANDARD_DEVIATION = "std_dev"
    PERCENTAGE = "percentage"
    ATR = "atr"
    FIXED_POINTS = "fixed_points"

@dataclass
class VWAPBands:
    """VWAP bands data structure"""
    upper_band_1: float
    upper_band_2: float
    upper_band_3: float
    vwap: float
    lower_band_1: float
    lower_band_2: float
    lower_band_3: float
    band_width: float
    
    @property
    def upper_bands(self) -> List[float]:
        """Get all upper bands"""
        return [self.upper_band_1, self.upper_band_2, self.upper_band_3]
    
    @property
    def lower_bands(self) -> List[float]:
        """Get all lower bands"""
        return [self.lower_band_1, self.lower_band_2, self.lower_band_3]

@dataclass
class VWAPAnalysis:
    """Complete VWAP analysis result"""
    timestamp: datetime
    vwap_type: VWAPType
    anchor_time: Optional[datetime]
    
    # Core VWAP data
    vwap: float
    volume_sum: int
    pv_sum: float  # Price * Volume sum
    
    # VWAP bands
    bands: VWAPBands
    
    # Price relationship
    current_price: float
    price_vs_vwap: str  # "above", "below", "at"
    distance_from_vwap: float
    distance_percentage: float
    
    # Volume analysis
    volume_weighted_trend: str  # "bullish", "bearish", "neutral"
    volume_momentum: float
    
    # Trading signals
    vwap_slope: float
    trend_strength: float
    reversal_signal: bool
    continuation_signal: bool

class VWAPCalculator:
    """
    Volume Weighted Average Price calculator with multiple VWAP types and bands
    """
    
    def __init__(self,
                 # Band configuration
                 std_dev_multipliers: List[float] = [1.0, 2.0, 2.5],
                 percentage_bands: List[float] = [0.1, 0.2, 0.3],  # 0.1%, 0.2%, 0.3%
                 
                 # Rolling VWAP periods
                 rolling_periods: List[int] = [20, 50, 200],
                 
                 # Session times (UTC)
                 session_start_hour: int = 0,
                 
                 # Trend analysis
                 slope_periods: int = 20,
                 momentum_periods: int = 10):
        
        self.std_dev_multipliers = std_dev_multipliers
        self.percentage_bands = percentage_bands
        self.rolling_periods = rolling_periods
        self.session_start_hour = session_start_hour
        self.slope_periods = slope_periods
        self.momentum_periods = momentum_periods
        
        # Storage for anchored VWAPs
        self.anchored_vwaps = {}
        
    def calculate_vwap(self, data: pd.DataFrame, 
                      vwap_type: VWAPType = VWAPType.STANDARD,
                      anchor_time: Optional[datetime] = None,
                      rolling_period: int = 20,
                      band_type: VWAPBandType = VWAPBandType.STANDARD_DEVIATION) -> Optional[VWAPAnalysis]:
        """
        Calculate VWAP with specified parameters
        
        Args:
            data: OHLCV DataFrame
            vwap_type: Type of VWAP calculation
            anchor_time: Anchor point for anchored VWAP
            rolling_period: Period for rolling VWAP
            band_type: Type of bands to calculate
            
        Returns:
            VWAPAnalysis result
        """
        if data.empty or len(data) < 2:
            logger.warning("Insufficient data for VWAP calculation")
            return None
        
        try:
            # Determine calculation period
            calc_data = self._get_calculation_data(data, vwap_type, anchor_time, rolling_period)
            
            if calc_data.empty:
                return None
            
            # Calculate VWAP
            vwap_value, volume_sum, pv_sum = self._calculate_core_vwap(calc_data)
            
            if vwap_value is None:
                return None
            
            # Calculate VWAP bands
            bands = self._calculate_vwap_bands(calc_data, vwap_value, band_type)
            
            # Current price analysis
            current_price = data['Close'].iloc[-1]
            price_analysis = self._analyze_price_vs_vwap(current_price, vwap_value)
            
            # Volume analysis
            volume_analysis = self._analyze_volume_trend(calc_data)
            
            # Trading signals
            signals = self._generate_trading_signals(data, vwap_value, bands, vwap_type)
            
            # Create analysis result
            analysis = VWAPAnalysis(
                timestamp=data.index[-1],
                vwap_type=vwap_type,
                anchor_time=anchor_time,
                vwap=vwap_value,
                volume_sum=volume_sum,
                pv_sum=pv_sum,
                bands=bands,
                current_price=current_price,
                price_vs_vwap=price_analysis['position'],
                distance_from_vwap=price_analysis['distance'],
                distance_percentage=price_analysis['distance_pct'],
                volume_weighted_trend=volume_analysis['trend'],
                volume_momentum=volume_analysis['momentum'],
                vwap_slope=signals['slope'],
                trend_strength=signals['trend_strength'],
                reversal_signal=signals['reversal_signal'],
                continuation_signal=signals['continuation_signal']
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return None
    
    def _get_calculation_data(self, data: pd.DataFrame, vwap_type: VWAPType,
                            anchor_time: Optional[datetime], rolling_period: int) -> pd.DataFrame:
        """Get data for VWAP calculation based on type"""
        
        if vwap_type == VWAPType.STANDARD:
            # Standard VWAP from session start
            return self._get_session_data(data)
        
        elif vwap_type == VWAPType.ANCHORED:
            # Anchored VWAP from specific time
            if anchor_time is None:
                return data
            return data[data.index >= anchor_time]
        
        elif vwap_type == VWAPType.ROLLING:
            # Rolling VWAP over fixed period
            return data.tail(rolling_period)
        
        elif vwap_type == VWAPType.WEEKLY:
            # Weekly VWAP (last 7 days)
            week_ago = data.index[-1] - timedelta(days=7)
            return data[data.index >= week_ago]
        
        elif vwap_type == VWAPType.MONTHLY:
            # Monthly VWAP (last 30 days)
            month_ago = data.index[-1] - timedelta(days=30)
            return data[data.index >= month_ago]
        
        else:
            return data
    
    def _get_session_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get data from current session start"""
        # Find most recent session start
        current_time = data.index[-1]
        current_date = current_time.date()
        
        # Session starts at specified hour UTC
        session_start = datetime.combine(current_date, datetime.min.time().replace(hour=self.session_start_hour))
        session_start = session_start.replace(tzinfo=current_time.tzinfo)
        
        # If current time is before session start, use previous day
        if current_time < session_start:
            session_start -= timedelta(days=1)
        
        return data[data.index >= session_start]
    
    def _calculate_core_vwap(self, data: pd.DataFrame) -> Tuple[Optional[float], int, float]:
        """Calculate core VWAP value"""
        if data.empty:
            return None, 0, 0.0
        
        # Calculate typical price (HLC/3)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Get volume (default to 1 if not available)
        volume = data.get('Volume', pd.Series(1, index=data.index))
        volume = volume.fillna(1)  # Replace NaN with 1
        
        # Calculate price * volume
        pv = typical_price * volume
        
        # Calculate VWAP
        volume_sum = volume.sum()
        pv_sum = pv.sum()
        
        if volume_sum == 0:
            return None, 0, 0.0
        
        vwap = pv_sum / volume_sum
        
        return float(vwap), int(volume_sum), float(pv_sum)
    
    def _calculate_vwap_bands(self, data: pd.DataFrame, vwap: float,
                            band_type: VWAPBandType) -> VWAPBands:
        """Calculate VWAP bands"""
        
        if band_type == VWAPBandType.STANDARD_DEVIATION:
            return self._calculate_std_dev_bands(data, vwap)
        elif band_type == VWAPBandType.PERCENTAGE:
            return self._calculate_percentage_bands(vwap)
        elif band_type == VWAPBandType.ATR:
            return self._calculate_atr_bands(data, vwap)
        else:
            # Default to standard deviation
            return self._calculate_std_dev_bands(data, vwap)
    
    def _calculate_std_dev_bands(self, data: pd.DataFrame, vwap: float) -> VWAPBands:
        """Calculate standard deviation based VWAP bands"""
        # Calculate typical price
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        volume = data.get('Volume', pd.Series(1, index=data.index))
        
        # Calculate volume-weighted variance
        price_diff_sq = (typical_price - vwap) ** 2
        volume_weighted_variance = (price_diff_sq * volume).sum() / volume.sum()
        std_dev = np.sqrt(volume_weighted_variance)
        
        # Calculate bands
        upper_1 = vwap + (std_dev * self.std_dev_multipliers[0])
        upper_2 = vwap + (std_dev * self.std_dev_multipliers[1])
        upper_3 = vwap + (std_dev * self.std_dev_multipliers[2])
        
        lower_1 = vwap - (std_dev * self.std_dev_multipliers[0])
        lower_2 = vwap - (std_dev * self.std_dev_multipliers[1])
        lower_3 = vwap - (std_dev * self.std_dev_multipliers[2])
        
        return VWAPBands(
            upper_band_1=upper_1,
            upper_band_2=upper_2,
            upper_band_3=upper_3,
            vwap=vwap,
            lower_band_1=lower_1,
            lower_band_2=lower_2,
            lower_band_3=lower_3,
            band_width=upper_1 - lower_1
        )
    
    def _calculate_percentage_bands(self, vwap: float) -> VWAPBands:
        """Calculate percentage-based VWAP bands"""
        upper_1 = vwap * (1 + self.percentage_bands[0] / 100)
        upper_2 = vwap * (1 + self.percentage_bands[1] / 100)
        upper_3 = vwap * (1 + self.percentage_bands[2] / 100)
        
        lower_1 = vwap * (1 - self.percentage_bands[0] / 100)
        lower_2 = vwap * (1 - self.percentage_bands[1] / 100)
        lower_3 = vwap * (1 - self.percentage_bands[2] / 100)
        
        return VWAPBands(
            upper_band_1=upper_1,
            upper_band_2=upper_2,
            upper_band_3=upper_3,
            vwap=vwap,
            lower_band_1=lower_1,
            lower_band_2=lower_2,
            lower_band_3=lower_3,
            band_width=upper_1 - lower_1
        )
    
    def _calculate_atr_bands(self, data: pd.DataFrame, vwap: float) -> VWAPBands:
        """Calculate ATR-based VWAP bands"""
        # Calculate ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=14).mean().iloc[-1]
        
        if pd.isna(atr) or atr == 0:
            atr = (data['High'] - data['Low']).mean()
        
        # Calculate bands using ATR multiples
        upper_1 = vwap + (atr * 1.0)
        upper_2 = vwap + (atr * 2.0)
        upper_3 = vwap + (atr * 2.5)
        
        lower_1 = vwap - (atr * 1.0)
        lower_2 = vwap - (atr * 2.0)
        lower_3 = vwap - (atr * 2.5)
        
        return VWAPBands(
            upper_band_1=upper_1,
            upper_band_2=upper_2,
            upper_band_3=upper_3,
            vwap=vwap,
            lower_band_1=lower_1,
            lower_band_2=lower_2,
            lower_band_3=lower_3,
            band_width=upper_1 - lower_1
        )
    
    def _analyze_price_vs_vwap(self, current_price: float, vwap: float) -> Dict:
        """Analyze current price relative to VWAP"""
        distance = current_price - vwap
        distance_pct = (distance / vwap) * 100 if vwap != 0 else 0
        
        if abs(distance_pct) < 0.01:  # Within 0.01%
            position = "at"
        elif distance > 0:
            position = "above"
        else:
            position = "below"
        
        return {
            'position': position,
            'distance': distance,
            'distance_pct': distance_pct
        }
    
    def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze volume-weighted trend"""
        if len(data) < self.momentum_periods:
            return {'trend': 'neutral', 'momentum': 0.0}
        
        # Calculate volume-weighted price change
        recent_data = data.tail(self.momentum_periods)
        typical_price = (recent_data['High'] + recent_data['Low'] + recent_data['Close']) / 3
        volume = recent_data.get('Volume', pd.Series(1, index=recent_data.index))
        
        # Volume-weighted price momentum
        price_changes = typical_price.diff()
        volume_weighted_changes = (price_changes * volume).sum()
        total_volume = volume.sum()
        
        momentum = volume_weighted_changes / total_volume if total_volume > 0 else 0
        
        # Determine trend
        if momentum > 0.0001:  # 0.01% threshold
            trend = "bullish"
        elif momentum < -0.0001:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            'trend': trend,
            'momentum': float(momentum)
        }
    
    def _generate_trading_signals(self, data: pd.DataFrame, vwap: float,
                                bands: VWAPBands, vwap_type: VWAPType) -> Dict:
        """Generate trading signals from VWAP analysis"""
        signals = {
            'slope': 0.0,
            'trend_strength': 0.0,
            'reversal_signal': False,
            'continuation_signal': False
        }
        
        if len(data) < self.slope_periods:
            return signals
        
        # Calculate VWAP slope
        recent_data = data.tail(self.slope_periods)
        vwap_values = []
        
        for i in range(len(recent_data)):
            subset = recent_data.iloc[:i+1]
            subset_vwap, _, _ = self._calculate_core_vwap(subset)
            if subset_vwap is not None:
                vwap_values.append(subset_vwap)
        
        if len(vwap_values) >= 3:
            # Calculate slope using linear regression
            x = np.arange(len(vwap_values))
            slope = np.polyfit(x, vwap_values, 1)[0]
            signals['slope'] = float(slope)
            
            # Trend strength based on R-squared
            y_pred = np.poly1d(np.polyfit(x, vwap_values, 1))(x)
            ss_res = np.sum((vwap_values - y_pred) ** 2)
            ss_tot = np.sum((vwap_values - np.mean(vwap_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            signals['trend_strength'] = max(0, min(1, r_squared))
        
        # Signal generation
        current_price = data['Close'].iloc[-1]
        
        # Reversal signals
        if current_price > bands.upper_band_2:
            # Price far above VWAP - potential reversal down
            signals['reversal_signal'] = True
        elif current_price < bands.lower_band_2:
            # Price far below VWAP - potential reversal up
            signals['reversal_signal'] = True
        
        # Continuation signals
        if (current_price > vwap and signals['slope'] > 0 and 
            bands.lower_band_1 < current_price < bands.upper_band_1):
            # Bullish continuation
            signals['continuation_signal'] = True
        elif (current_price < vwap and signals['slope'] < 0 and
              bands.lower_band_1 < current_price < bands.upper_band_1):
            # Bearish continuation
            signals['continuation_signal'] = True
        
        return signals
    
    def calculate_multiple_vwaps(self, data: pd.DataFrame) -> Dict[str, VWAPAnalysis]:
        """Calculate multiple VWAP types for comprehensive analysis"""
        results = {}
        
        # Standard session VWAP
        standard_vwap = self.calculate_vwap(data, VWAPType.STANDARD)
        if standard_vwap:
            results['standard'] = standard_vwap
        
        # Rolling VWAPs
        for period in self.rolling_periods:
            rolling_vwap = self.calculate_vwap(data, VWAPType.ROLLING, rolling_period=period)
            if rolling_vwap:
                results[f'rolling_{period}'] = rolling_vwap
        
        # Weekly VWAP
        weekly_vwap = self.calculate_vwap(data, VWAPType.WEEKLY)
        if weekly_vwap:
            results['weekly'] = weekly_vwap
        
        # Monthly VWAP
        monthly_vwap = self.calculate_vwap(data, VWAPType.MONTHLY)
        if monthly_vwap:
            results['monthly'] = monthly_vwap
        
        return results
    
    def create_anchored_vwap(self, data: pd.DataFrame, anchor_time: datetime,
                           name: str) -> Optional[VWAPAnalysis]:
        """Create and store anchored VWAP"""
        anchored_vwap = self.calculate_vwap(data, VWAPType.ANCHORED, anchor_time)
        
        if anchored_vwap:
            self.anchored_vwaps[name] = {
                'anchor_time': anchor_time,
                'analysis': anchored_vwap
            }
        
        return anchored_vwap
    
    def get_vwap_confluence(self, data: pd.DataFrame, current_price: float,
                          tolerance: float = 0.0001) -> Dict:
        """Find VWAP confluence zones"""
        vwaps = self.calculate_multiple_vwaps(data)
        
        confluence_zones = []
        vwap_levels = []
        
        # Collect all VWAP levels
        for name, analysis in vwaps.items():
            vwap_levels.append({
                'name': name,
                'vwap': analysis.vwap,
                'bands': analysis.bands
            })
        
        # Find confluence zones
        for i, level1 in enumerate(vwap_levels):
            confluence_count = 1
            confluent_levels = [level1['name']]
            
            for j, level2 in enumerate(vwap_levels[i+1:], i+1):
                if abs(level1['vwap'] - level2['vwap']) <= tolerance:
                    confluence_count += 1
                    confluent_levels.append(level2['name'])
            
            if confluence_count >= 2:  # At least 2 VWAPs confluent
                confluence_zones.append({
                    'price': level1['vwap'],
                    'confluence_count': confluence_count,
                    'confluent_levels': confluent_levels,
                    'distance_from_current': abs(current_price - level1['vwap'])
                })
        
        # Sort by confluence count and proximity
        confluence_zones.sort(key=lambda x: (x['confluence_count'], -x['distance_from_current']), reverse=True)
        
        return {
            'confluence_zones': confluence_zones,
            'total_zones': len(confluence_zones),
            'strongest_confluence': confluence_zones[0] if confluence_zones else None
        }
    
    def get_vwap_summary(self, analysis: VWAPAnalysis) -> Dict:
        """Get summary of VWAP analysis"""
        if not analysis:
            return {'error': 'No analysis data'}
        
        summary = {
            'vwap_type': analysis.vwap_type.value,
            'vwap': analysis.vwap,
            'current_price': analysis.current_price,
            'price_position': analysis.price_vs_vwap,
            'distance_pips': abs(analysis.distance_from_vwap * 10000),  # Convert to pips
            'distance_percentage': analysis.distance_percentage,
            'volume_trend': analysis.volume_weighted_trend,
            'vwap_slope': 'rising' if analysis.vwap_slope > 0 else 'falling' if analysis.vwap_slope < 0 else 'flat',
            'trend_strength': analysis.trend_strength,
            'reversal_signal': analysis.reversal_signal,
            'continuation_signal': analysis.continuation_signal,
            'upper_bands': analysis.bands.upper_bands,
            'lower_bands': analysis.bands.lower_bands,
            'band_width_pips': analysis.bands.band_width * 10000
        }
        
        return summary

# Export main classes
__all__ = [
    'VWAPCalculator',
    'VWAPAnalysis',
    'VWAPBands',
    'VWAPType',
    'VWAPBandType'
]