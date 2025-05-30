"""
Premium/Discount Zone Analysis
Identifies market ranges and premium/discount zones for SMC trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ZoneType(Enum):
    """Zone classification types"""
    PREMIUM = "premium"
    DISCOUNT = "discount" 
    EQUILIBRIUM = "equilibrium"
    EXTREME_PREMIUM = "extreme_premium"
    EXTREME_DISCOUNT = "extreme_discount"

class RangeQuality(Enum):
    """Range quality assessment"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class PremiumDiscountRange:
    """Premium/Discount range data structure"""
    timestamp: datetime
    range_high: float
    range_low: float
    fifty_percent_level: float
    current_zone: ZoneType
    range_size: float
    range_age_hours: float
    touch_count: int
    quality: RangeQuality
    strength: float  # 0.0 to 1.0
    volume_profile: Dict = None
    
    @property
    def range_mid(self) -> float:
        """Get range midpoint (50% level)"""
        return self.fifty_percent_level
    
    @property
    def premium_zone_start(self) -> float:
        """Premium zone starts at 50% level"""
        return self.fifty_percent_level
    
    @property
    def discount_zone_end(self) -> float:
        """Discount zone ends at 50% level"""
        return self.fifty_percent_level
    
    @property
    def premium_threshold(self) -> float:
        """75% level for premium classification"""
        return self.range_low + (self.range_size * 0.75)
    
    @property
    def discount_threshold(self) -> float:
        """25% level for discount classification"""
        return self.range_low + (self.range_size * 0.25)
    
    @property
    def extreme_premium_threshold(self) -> float:
        """95% level for extreme premium"""
        return self.range_low + (self.range_size * 0.95)
    
    @property
    def extreme_discount_threshold(self) -> float:
        """5% level for extreme discount"""
        return self.range_low + (self.range_size * 0.05)

class PremiumDiscountAnalyzer:
    """
    Analyzes Premium and Discount zones in market ranges
    """
    
    def __init__(self,
                min_range_size: float = 0.002,  # 200 pips for majors
                min_range_duration: int = 24,   # 24 hours minimum
                swing_lookback: int = 20,
                equilibrium_threshold: float = 0.25,  # UPDATED: 25% around 50% level
                premium_threshold: float = 0.75,      # UPDATED: 75% for premium
                discount_threshold: float = 0.25,     # UPDATED: 25% for discount
                extreme_premium_threshold: float = 0.95,  # NEW: 95% for extreme premium
                extreme_discount_threshold: float = 0.05): # NEW: 5% for extreme discount
        
        self.min_range_size = min_range_size
        self.min_range_duration = min_range_duration
        self.swing_lookback = swing_lookback
        self.equilibrium_threshold = equilibrium_threshold
        self.premium_threshold = premium_threshold
        self.discount_threshold = discount_threshold
        self.extreme_premium_threshold = extreme_premium_threshold
        self.extreme_discount_threshold = extreme_discount_threshold
        
        # Storage for identified ranges
        self.active_ranges = []
        
    def analyze_premium_discount(self, data: pd.DataFrame) -> Dict:
        """
        Main function to analyze premium/discount zones
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            Dictionary with premium/discount analysis
        """
        if len(data) < self.swing_lookback * 2:
            logger.warning("Insufficient data for premium/discount analysis")
            return self._empty_analysis()
        
        try:
            # Step 1: Identify significant ranges
            ranges = self._identify_ranges(data)
            
            # Step 2: Calculate 50% levels and zone classifications
            analyzed_ranges = self._analyze_ranges(ranges, data)
            
            # Step 3: Assess current market position
            current_assessment = self._assess_current_position(analyzed_ranges, data)
            
            # Step 4: Update active ranges
            self._update_active_ranges(analyzed_ranges, data)
            
            # Step 5: Calculate zone metrics
            zone_metrics = self._calculate_zone_metrics(data)
            
            return {
                'timestamp': data.index[-1],
                'current_price': data['Close'].iloc[-1],
                'ranges': analyzed_ranges,
                'active_ranges': self.active_ranges,
                'current_assessment': current_assessment,
                'zone_metrics': zone_metrics,
                'summary': self._generate_summary(current_assessment)
            }
            
        except Exception as e:
            logger.error(f"Error in premium/discount analysis: {e}")
            return self._empty_analysis()
    
    def _identify_ranges(self, data: pd.DataFrame) -> List[Dict]:
        """Identify significant price ranges"""
        ranges = []
        
        # Find swing highs and lows
        swing_highs = self._find_swing_highs(data)
        swing_lows = self._find_swing_lows(data)
        
        # Combine and analyze potential ranges
        for i, high in enumerate(swing_highs):
            for j, low in enumerate(swing_lows):
                # Only consider ranges where low comes before high or vice versa
                time_diff = abs((high['timestamp'] - low['timestamp']).total_seconds() / 3600)
                
                if time_diff >= self.min_range_duration:
                    range_size = high['price'] - low['price']
                    
                    if range_size >= self.min_range_size:
                        # Determine range start and end times
                        start_time = min(high['timestamp'], low['timestamp'])
                        end_time = max(high['timestamp'], low['timestamp'])
                        
                        ranges.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'high': high['price'],
                            'low': low['price'],
                            'size': range_size,
                            'duration_hours': time_diff,
                            'high_timestamp': high['timestamp'],
                            'low_timestamp': low['timestamp']
                        })
        
        # Sort ranges by recency and size
        ranges.sort(key=lambda x: (x['end_time'], x['size']), reverse=True)
        
        # Keep only the most significant ranges (top 10)
        return ranges[:10]
    
    def _find_swing_highs(self, data: pd.DataFrame) -> List[Dict]:
        """Find swing high points"""
        swing_highs = []
        highs = data['High'].values
        times = data.index
        
        for i in range(self.swing_lookback, len(data) - self.swing_lookback):
            current_high = highs[i]
            
            # Check if current bar is higher than surrounding bars
            is_swing_high = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and highs[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Calculate significance
                left_max = max(highs[i - self.swing_lookback:i])
                right_max = max(highs[i + 1:i + self.swing_lookback + 1])
                surrounding_max = max(left_max, right_max)
                
                if surrounding_max > 0:
                    significance = (current_high - surrounding_max) / surrounding_max
                else:
                    significance = 0.5
                
                if significance >= 0.001:  # Minimum 0.1% significance
                    swing_highs.append({
                        'timestamp': times[i],
                        'price': current_high,
                        'significance': significance
                    })
        
        return swing_highs
    
    def _find_swing_lows(self, data: pd.DataFrame) -> List[Dict]:
        """Find swing low points"""
        swing_lows = []
        lows = data['Low'].values
        times = data.index
        
        for i in range(self.swing_lookback, len(data) - self.swing_lookback):
            current_low = lows[i]
            
            # Check if current bar is lower than surrounding bars
            is_swing_low = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and lows[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Calculate significance
                left_min = min(lows[i - self.swing_lookback:i])
                right_min = min(lows[i + 1:i + self.swing_lookback + 1])
                surrounding_min = min(left_min, right_min)
                
                if current_low > 0:
                    significance = (surrounding_min - current_low) / current_low
                else:
                    significance = 0.5
                
                if significance >= 0.001:
                    swing_lows.append({
                        'timestamp': times[i],
                        'price': current_low,
                        'significance': significance
                    })
        
        return swing_lows
    
    def _analyze_ranges(self, ranges: List[Dict], data: pd.DataFrame) -> List[PremiumDiscountRange]:
        """Analyze ranges and create PremiumDiscountRange objects"""
        analyzed_ranges = []
        
        for range_data in ranges:
            # Calculate 50% level
            fifty_percent = range_data['low'] + (range_data['size'] / 2)
            
            # Assess range quality
            quality = self._assess_range_quality(range_data, data)
            
            # Calculate touch count
            touch_count = self._calculate_touch_count(range_data, data)
            
            # Calculate strength based on multiple factors
            strength = self._calculate_range_strength(range_data, quality, touch_count)
            
            # Determine current zone for latest price
            current_price = data['Close'].iloc[-1]
            current_zone = self._classify_zone_position(current_price, range_data)
            
            # Create PremiumDiscountRange object
            pd_range = PremiumDiscountRange(
                timestamp=range_data['end_time'],
                range_high=range_data['high'],
                range_low=range_data['low'],
                fifty_percent_level=fifty_percent,
                current_zone=current_zone,
                range_size=range_data['size'],
                range_age_hours=range_data['duration_hours'],
                touch_count=touch_count,
                quality=quality,
                strength=strength
            )
            
            analyzed_ranges.append(pd_range)
        
        return analyzed_ranges
    
    def _classify_zone_position(self, price: float, range_data: Dict) -> ZoneType:
        """
        Classify current price position within range
        Uses configurable thresholds for consistent classification
        """
        range_size = range_data['size']
        relative_position = (price - range_data['low']) / range_size
        
        # Use instance thresholds for consistency
        if relative_position >= self.extreme_premium_threshold:
            return ZoneType.EXTREME_PREMIUM
        elif relative_position >= self.premium_threshold:
            return ZoneType.PREMIUM
        elif relative_position <= self.extreme_discount_threshold:
            return ZoneType.EXTREME_DISCOUNT
        elif relative_position <= self.discount_threshold:
            return ZoneType.DISCOUNT
        else:
            return ZoneType.EQUILIBRIUM
    
    def _assess_range_quality(self, range_data: Dict, data: pd.DataFrame) -> RangeQuality:
        """Assess the quality of a range"""
        # Factors: size, duration, touch count, age
        
        size_score = min(range_data['size'] / self.min_range_size, 3.0) / 3.0
        duration_score = min(range_data['duration_hours'] / (self.min_range_duration * 3), 1.0)
        
        # Check how recent the range is
        latest_time = data.index[-1]
        age_hours = (latest_time - range_data['end_time']).total_seconds() / 3600
        age_score = max(0, 1 - (age_hours / (24 * 7)))  # Decay over a week
        
        # Combined quality score
        quality_score = (size_score + duration_score + age_score) / 3
        
        if quality_score >= 0.7:
            return RangeQuality.HIGH
        elif quality_score >= 0.4:
            return RangeQuality.MEDIUM
        else:
            return RangeQuality.LOW
    
    def _calculate_touch_count(self, range_data: Dict, data: pd.DataFrame) -> int:
        """Calculate how many times price touched range boundaries"""
        touch_count = 0
        tolerance = range_data['size'] * 0.01  # 1% tolerance
        
        # Get data within the range timeframe
        range_start = range_data['start_time']
        range_end = range_data['end_time']
        range_data_subset = data[(data.index >= range_start) & (data.index <= range_end)]
        
        for _, candle in range_data_subset.iterrows():
            # Check touches of high
            if abs(candle['High'] - range_data['high']) <= tolerance:
                touch_count += 1
            # Check touches of low
            elif abs(candle['Low'] - range_data['low']) <= tolerance:
                touch_count += 1
        
        return touch_count
    
    def _calculate_range_strength(self, range_data: Dict, quality: RangeQuality, touch_count: int) -> float:
        """Calculate overall range strength"""
        # Base strength from size
        size_strength = min(range_data['size'] / (self.min_range_size * 2), 1.0)
        
        # Quality multiplier
        quality_multiplier = {
            RangeQuality.HIGH: 1.0,
            RangeQuality.MEDIUM: 0.8,
            RangeQuality.LOW: 0.6
        }[quality]
        
        # Touch count factor (more touches = stronger range)
        touch_factor = min(touch_count / 10, 1.0)
        
        # Duration factor
        duration_factor = min(range_data['duration_hours'] / (24 * 3), 1.0)  # 3 days max
        
        # Combined strength
        strength = (size_strength * 0.4 + 
                   touch_factor * 0.3 + 
                   duration_factor * 0.3) * quality_multiplier
        
        return min(strength, 1.0)
    
    def _assess_current_position(self, ranges: List[PremiumDiscountRange], data: pd.DataFrame) -> Dict:
        """Assess current market position relative to premium/discount zones"""
        if not ranges:
            return {
                'zone_type': ZoneType.EQUILIBRIUM,
                'zone_strength': 0.0,
                'distance_to_equilibrium': 0.0,
                'nearest_range': None,
                'trading_bias': 'neutral'
            }
        
        current_price = data['Close'].iloc[-1]
        
        # Find the most relevant range (highest strength, most recent)
        relevant_range = max(ranges, key=lambda x: (x.strength, x.timestamp))
        
        # Calculate relative position within this range
        relative_position = (current_price - relevant_range.range_low) / relevant_range.range_size
        distance_to_equilibrium = abs(relative_position - 0.5)
        
        # Determine zone type and strength
        zone_type = self._classify_zone_position(current_price, {
            'low': relevant_range.range_low,
            'high': relevant_range.range_high,
            'size': relevant_range.range_size
        })
        
        # Calculate zone strength based on how deep in the zone price is
        if zone_type == ZoneType.EXTREME_PREMIUM:
            zone_strength = (relative_position - 0.9) / 0.1
        elif zone_type == ZoneType.PREMIUM:
            zone_strength = (relative_position - 0.7) / 0.2
        elif zone_type == ZoneType.EXTREME_DISCOUNT:
            zone_strength = (0.1 - relative_position) / 0.1
        elif zone_type == ZoneType.DISCOUNT:
            zone_strength = (0.3 - relative_position) / 0.2
        else:  # EQUILIBRIUM
            zone_strength = 1 - (distance_to_equilibrium / 0.2)  # Max strength at 50%
        
        zone_strength = max(0, min(zone_strength, 1.0))
        
        # Determine trading bias
        if zone_type in [ZoneType.EXTREME_PREMIUM, ZoneType.PREMIUM]:
            trading_bias = 'bearish'  # Look for shorts in premium
        elif zone_type in [ZoneType.EXTREME_DISCOUNT, ZoneType.DISCOUNT]:
            trading_bias = 'bullish'  # Look for longs in discount
        else:
            trading_bias = 'neutral'
        
        return {
            'zone_type': zone_type,
            'zone_strength': zone_strength,
            'distance_to_equilibrium': distance_to_equilibrium,
            'nearest_range': relevant_range,
            'trading_bias': trading_bias,
            'relative_position': relative_position,
            'price_levels': {
                'current_price': current_price,
                'fifty_percent': relevant_range.fifty_percent_level,
                'premium_threshold': relevant_range.premium_threshold,
                'discount_threshold': relevant_range.discount_threshold,
                'extreme_premium': relevant_range.extreme_premium_threshold,
                'extreme_discount': relevant_range.extreme_discount_threshold
            }
        }
    
    def _update_active_ranges(self, new_ranges: List[PremiumDiscountRange], data: pd.DataFrame):
        """Update list of active ranges"""
        current_time = data.index[-1]
        
        # Remove old ranges (older than 1 week)
        week_ago = current_time - timedelta(days=7)
        self.active_ranges = [r for r in self.active_ranges if r.timestamp > week_ago]
        
        # Add new ranges that meet quality criteria
        for new_range in new_ranges:
            if new_range.quality in [RangeQuality.HIGH, RangeQuality.MEDIUM]:
                # Check if this range is a duplicate
                is_duplicate = False
                for existing_range in self.active_ranges:
                    if (abs(existing_range.range_high - new_range.range_high) < new_range.range_size * 0.05 and
                        abs(existing_range.range_low - new_range.range_low) < new_range.range_size * 0.05):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    self.active_ranges.append(new_range)
        
        # Keep only top 10 ranges by strength
        self.active_ranges.sort(key=lambda x: x.strength, reverse=True)
        self.active_ranges = self.active_ranges[:10]
    
    def _calculate_zone_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate premium/discount zone metrics"""
        if not self.active_ranges:
            return {
                'total_ranges': 0,
                'avg_range_size': 0.0,
                'avg_range_strength': 0.0,
                'premium_zones': 0,
                'discount_zones': 0,
                'equilibrium_zones': 0,
                'zone_distribution': {}
            }
        
        current_price = data['Close'].iloc[-1]
        
        # Count zones by type
        zone_counts = {
            'premium': 0,
            'discount': 0,
            'equilibrium': 0,
            'extreme_premium': 0,
            'extreme_discount': 0
        }
        
        for range_obj in self.active_ranges:
            zone_type = self._classify_zone_position(current_price, {
                'low': range_obj.range_low,
                'high': range_obj.range_high,
                'size': range_obj.range_size
            })
            
            zone_counts[zone_type.value] += 1
        
        total_ranges = len(self.active_ranges)
        avg_size = sum(r.range_size for r in self.active_ranges) / total_ranges
        avg_strength = sum(r.strength for r in self.active_ranges) / total_ranges
        
        return {
            'total_ranges': total_ranges,
            'avg_range_size': avg_size,
            'avg_range_strength': avg_strength,
            'premium_zones': zone_counts['premium'] + zone_counts['extreme_premium'],
            'discount_zones': zone_counts['discount'] + zone_counts['extreme_discount'],
            'equilibrium_zones': zone_counts['equilibrium'],
            'zone_distribution': zone_counts
        }
    
    def _generate_summary(self, current_assessment: Dict) -> Dict:
        """Generate summary of premium/discount analysis"""
        if not current_assessment.get('nearest_range'):
            return {
                'market_position': 'no_range_identified',
                'trading_recommendation': 'wait_for_range',
                'confidence': 0.0
            }
        
        zone_type = current_assessment['zone_type']
        zone_strength = current_assessment['zone_strength']
        trading_bias = current_assessment['trading_bias']
        
        # Generate confidence based on zone strength and range quality
        nearest_range = current_assessment['nearest_range']
        confidence = (zone_strength + nearest_range.strength) / 2
        
        # Generate trading recommendation
        if zone_type in [ZoneType.EXTREME_PREMIUM, ZoneType.EXTREME_DISCOUNT]:
            recommendation = 'high_probability_reversal'
        elif zone_type in [ZoneType.PREMIUM, ZoneType.DISCOUNT]:
            recommendation = 'moderate_reversal_opportunity'
        else:
            recommendation = 'wait_for_clear_zone'
        
        return {
            'market_position': zone_type.value,
            'trading_bias': trading_bias,
            'trading_recommendation': recommendation,
            'confidence': confidence,
            'zone_strength': zone_strength,
            'range_quality': nearest_range.quality.value
        }
    
    def get_zone_at_price(self, price: float, tolerance: float = 0.0001) -> Optional[PremiumDiscountRange]:
        """Get the range/zone that contains the specified price"""
        for range_obj in self.active_ranges:
            if (range_obj.range_low - tolerance <= price <= range_obj.range_high + tolerance):
                return range_obj
        return None
    
    def get_nearest_fifty_percent_level(self, price: float) -> Optional[Tuple[float, float]]:
        """Get nearest 50% level and its distance"""
        if not self.active_ranges:
            return None
        
        nearest_level = None
        min_distance = float('inf')
        
        for range_obj in self.active_ranges:
            distance = abs(price - range_obj.fifty_percent_level)
            if distance < min_distance:
                min_distance = distance
                nearest_level = range_obj.fifty_percent_level
        
        return (nearest_level, min_distance) if nearest_level else None
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'timestamp': datetime.now(),
            'current_price': 0.0,
            'ranges': [],
            'active_ranges': [],
            'current_assessment': {
                'zone_type': ZoneType.EQUILIBRIUM,
                'zone_strength': 0.0,
                'distance_to_equilibrium': 0.0,
                'nearest_range': None,
                'trading_bias': 'neutral'
            },
            'zone_metrics': {
                'total_ranges': 0,
                'avg_range_size': 0.0,
                'avg_range_strength': 0.0,
                'premium_zones': 0,
                'discount_zones': 0,
                'equilibrium_zones': 0
            },
            'summary': {
                'market_position': 'no_data',
                'trading_recommendation': 'insufficient_data',
                'confidence': 0.0
            }
        }

# Export main classes
__all__ = [
    'PremiumDiscountAnalyzer',
    'PremiumDiscountRange',
    'ZoneType',
    'RangeQuality'
]