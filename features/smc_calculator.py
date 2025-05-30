"""
Smart Money Concepts (SMC) Market Structure Detection
Implements BOS, MSB, CHoCH, and trend analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction enumeration"""
    BULLISH = 1
    BEARISH = -1
    SIDEWAYS = 0

class StructureType(Enum):
    """Market structure types"""
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL" 
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EH"
    EQUAL_LOW = "EL"

@dataclass
class StructurePoint:
    """Market structure point data"""
    timestamp: datetime
    price: float
    structure_type: StructureType
    significance: float  # 0.0 to 1.0
    volume: int = 0
    confirmed: bool = False

@dataclass
class MarketStructureState:
    """Current market structure state"""
    trend: TrendDirection
    last_hh: Optional[StructurePoint] = None
    last_hl: Optional[StructurePoint] = None
    last_lh: Optional[StructurePoint] = None
    last_ll: Optional[StructurePoint] = None
    structure_breaks: List[Dict] = None
    choch_events: List[Dict] = None
    
    def __post_init__(self):
        if self.structure_breaks is None:
            self.structure_breaks = []
        if self.choch_events is None:
            self.choch_events = []

class MarketStructureAnalyzer:
    """
    Analyzes market structure for SMC concepts
    """
    
    def __init__(self, 
                 swing_lookback: int = 20,
                 min_swing_size: float = 0.0005,
                 structure_confirmation_bars: int = 3):
        
        self.swing_lookback = swing_lookback
        self.min_swing_size = min_swing_size
        self.structure_confirmation_bars = structure_confirmation_bars
        
        # Initialize state
        self.current_state = MarketStructureState(trend=TrendDirection.SIDEWAYS)
        
    def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """
        Main function to analyze market structure
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            Dictionary with market structure analysis
        """
        if len(data) < self.swing_lookback * 2:
            logger.warning("Insufficient data for market structure analysis")
            return self._empty_analysis()
        
        try:
            # Step 1: Identify swing points
            swing_highs, swing_lows = self._identify_swing_points(data)
            
            # Step 2: Classify structure points
            structure_points = self._classify_structure_points(swing_highs, swing_lows, data)
            
            # Step 3: Determine trend
            current_trend = self._determine_trend(structure_points)
            
            # Step 4: Detect breaks of structure (BOS)
            bos_events = self._detect_bos(structure_points, data)
            
            # Step 5: Detect change of character (CHoCH)
            choch_events = self._detect_choch(structure_points, data)
            
            # Step 6: Calculate market structure break (MSB)
            msb_events = self._detect_msb(structure_points, data)
            
            # Update internal state
            self._update_state(structure_points, current_trend, bos_events, choch_events)
            
            return {
                'timestamp': data.index[-1],
                'trend': current_trend,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'structure_points': structure_points,
                'bos_events': bos_events,
                'choch_events': choch_events,
                'msb_events': msb_events,
                'trend_strength': self._calculate_trend_strength(structure_points),
                'structure_quality': self._assess_structure_quality(structure_points),
                'current_state': self.current_state
            }
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return self._empty_analysis()
    
    def _identify_swing_points(self, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Identify swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        highs = data['High'].values
        lows = data['Low'].values
        volumes = data.get('Volume', pd.Series(0, index=data.index)).values
        times = data.index
        
        # Find swing highs
        for i in range(self.swing_lookback, len(data) - self.swing_lookback):
            current_high = highs[i]
            
            # Check if current bar is higher than surrounding bars
            is_swing_high = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and highs[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Calculate significance based on how much higher it is
                left_max = max(highs[i - self.swing_lookback:i])
                right_max = max(highs[i + 1:i + self.swing_lookback + 1])
                surrounding_max = max(left_max, right_max)
                
                if surrounding_max > 0:
                    significance = min((current_high - surrounding_max) / surrounding_max, 1.0)
                else:
                    significance = 0.5
                
                # Only include significant swings
                if significance >= 0.001:  # Minimum 0.1% significance
                    swing_highs.append({
                        'timestamp': times[i],
                        'price': current_high,
                        'significance': significance,
                        'volume': volumes[i],
                        'index': i
                    })
        
        # Find swing lows
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
                    significance = min((surrounding_min - current_low) / current_low, 1.0)
                else:
                    significance = 0.5
                
                if significance >= 0.001:
                    swing_lows.append({
                        'timestamp': times[i],
                        'price': current_low,
                        'significance': significance,
                        'volume': volumes[i],
                        'index': i
                    })
        
        # Sort by timestamp
        swing_highs.sort(key=lambda x: x['timestamp'])
        swing_lows.sort(key=lambda x: x['timestamp'])
        
        logger.debug(f"Identified {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
        
        return swing_highs, swing_lows
    
    def _classify_structure_points(self, swing_highs: List[Dict], 
                                 swing_lows: List[Dict], 
                                 data: pd.DataFrame) -> List[StructurePoint]:
        """Classify swing points into structure types"""
        structure_points = []
        
        # Combine and sort all swing points
        all_swings = []
        
        for sh in swing_highs:
            all_swings.append({
                **sh,
                'type': 'high'
            })
        
        for sl in swing_lows:
            all_swings.append({
                **sl,
                'type': 'low'
            })
        
        all_swings.sort(key=lambda x: x['timestamp'])
        
        # Classify structure points
        prev_high = None
        prev_low = None
        
        for swing in all_swings:
            structure_type = None
            
            if swing['type'] == 'high':
                if prev_high is not None:
                    price_diff = abs(swing['price'] - prev_high['price'])
                    relative_diff = price_diff / prev_high['price']
                    
                    if swing['price'] > prev_high['price']:
                        if relative_diff > 0.0001:  # Minimum 0.01% difference
                            structure_type = StructureType.HIGHER_HIGH
                        else:
                            structure_type = StructureType.EQUAL_HIGH
                    else:
                        structure_type = StructureType.LOWER_HIGH
                
                prev_high = swing
                
            else:  # swing['type'] == 'low'
                if prev_low is not None:
                    price_diff = abs(swing['price'] - prev_low['price'])
                    relative_diff = price_diff / prev_low['price']
                    
                    if swing['price'] < prev_low['price']:
                        if relative_diff > 0.0001:
                            structure_type = StructureType.LOWER_LOW
                        else:
                            structure_type = StructureType.EQUAL_LOW
                    else:
                        structure_type = StructureType.HIGHER_LOW
                
                prev_low = swing
            
            if structure_type is not None:
                structure_points.append(StructurePoint(
                    timestamp=swing['timestamp'],
                    price=swing['price'],
                    structure_type=structure_type,
                    significance=swing['significance'],
                    volume=swing['volume'],
                    confirmed=True  # Will be updated based on confirmation logic
                ))
        
        return structure_points
    
    def _determine_trend(self, structure_points: List[StructurePoint]) -> TrendDirection:
        """Determine overall trend based on structure points"""
        if len(structure_points) < 4:
            return TrendDirection.SIDEWAYS
        
        # Get recent structure points (last 10 or all if less)
        recent_points = structure_points[-10:]
        
        bullish_signals = 0
        bearish_signals = 0
        
        for point in recent_points:
            if point.structure_type in [StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW]:
                bullish_signals += point.significance
            elif point.structure_type in [StructureType.LOWER_HIGH, StructureType.LOWER_LOW]:
                bearish_signals += point.significance
        
        # Determine trend based on weighted signals
        signal_difference = bullish_signals - bearish_signals
        
        if signal_difference > 0.5:
            return TrendDirection.BULLISH
        elif signal_difference < -0.5:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def _detect_bos(self, structure_points: List[StructurePoint], data: pd.DataFrame) -> List[Dict]:
        """Detect Break of Structure (BOS) events"""
        bos_events = []
        
        if len(structure_points) < 2:
            return bos_events
        
        # Look for breaks of previous structure levels
        for i, point in enumerate(structure_points[1:], 1):
            prev_point = structure_points[i-1]
            
            # Bullish BOS: Current HL > Previous LH
            if (point.structure_type == StructureType.HIGHER_LOW and 
                prev_point.structure_type == StructureType.LOWER_HIGH and
                point.price > prev_point.price):
                
                bos_events.append({
                    'timestamp': point.timestamp,
                    'direction': 'bullish',
                    'break_price': prev_point.price,
                    'current_price': point.price,
                    'strength': point.significance + prev_point.significance,
                    'type': 'BOS'
                })
            
            # Bearish BOS: Current LH < Previous HL  
            elif (point.structure_type == StructureType.LOWER_HIGH and
                  prev_point.structure_type == StructureType.HIGHER_LOW and
                  point.price < prev_point.price):
                
                bos_events.append({
                    'timestamp': point.timestamp,
                    'direction': 'bearish',
                    'break_price': prev_point.price,
                    'current_price': point.price,
                    'strength': point.significance + prev_point.significance,
                    'type': 'BOS'
                })
        
        return bos_events
    
    def _detect_choch(self, structure_points: List[StructurePoint], data: pd.DataFrame) -> List[Dict]:
        """Detect Change of Character (CHoCH) events"""
        choch_events = []
        
        if len(structure_points) < 4:
            return choch_events
        
        # Look for trend changes based on structure sequence
        for i in range(3, len(structure_points)):
            # Get last 4 points to analyze pattern
            recent_points = structure_points[i-3:i+1]
            
            # Check for bullish CHoCH pattern: LL -> HL -> LH -> HH
            if (len(recent_points) == 4 and
                recent_points[0].structure_type == StructureType.LOWER_LOW and
                recent_points[1].structure_type == StructureType.HIGHER_LOW and
                recent_points[2].structure_type == StructureType.LOWER_HIGH and
                recent_points[3].structure_type == StructureType.HIGHER_HIGH):
                
                choch_events.append({
                    'timestamp': recent_points[3].timestamp,
                    'direction': 'bullish',
                    'from_trend': 'bearish',
                    'to_trend': 'bullish',
                    'confirmation_price': recent_points[3].price,
                    'strength': sum(p.significance for p in recent_points) / 4,
                    'type': 'CHoCH'
                })
            
            # Check for bearish CHoCH pattern: HH -> LH -> HL -> LL
            elif (len(recent_points) == 4 and
                  recent_points[0].structure_type == StructureType.HIGHER_HIGH and
                  recent_points[1].structure_type == StructureType.LOWER_HIGH and
                  recent_points[2].structure_type == StructureType.HIGHER_LOW and
                  recent_points[3].structure_type == StructureType.LOWER_LOW):
                
                choch_events.append({
                    'timestamp': recent_points[3].timestamp,
                    'direction': 'bearish',
                    'from_trend': 'bullish',
                    'to_trend': 'bearish',
                    'confirmation_price': recent_points[3].price,
                    'strength': sum(p.significance for p in recent_points) / 4,
                    'type': 'CHoCH'
                })
        
        return choch_events
    
    def _detect_msb(self, structure_points: List[StructurePoint], data: pd.DataFrame) -> List[Dict]:
        """Detect Market Structure Break (MSB) events"""
        msb_events = []
        
        # MSB is similar to BOS but with stronger confirmation
        bos_events = self._detect_bos(structure_points, data)
        
        for bos in bos_events:
            # Upgrade BOS to MSB if strength is high enough
            if bos['strength'] > 0.7:  # High significance threshold
                msb_event = bos.copy()
                msb_event['type'] = 'MSB'
                msb_events.append(msb_event)
        
        return msb_events
    
    def _calculate_trend_strength(self, structure_points: List[StructurePoint]) -> float:
        """Calculate trend strength based on structure consistency"""
        if len(structure_points) < 3:
            return 0.0
        
        recent_points = structure_points[-6:]  # Last 6 points
        
        # Count consistent structure types
        bullish_count = sum(1 for p in recent_points 
                           if p.structure_type in [StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW])
        bearish_count = sum(1 for p in recent_points
                           if p.structure_type in [StructureType.LOWER_HIGH, StructureType.LOWER_LOW])
        
        total_count = len(recent_points)
        
        if total_count == 0:
            return 0.0
        
        # Calculate strength as consistency ratio
        max_consistency = max(bullish_count, bearish_count)
        strength = max_consistency / total_count
        
        return min(strength, 1.0)
    
    def _assess_structure_quality(self, structure_points: List[StructurePoint]) -> float:
        """Assess overall quality of market structure"""
        if len(structure_points) < 2:
            return 0.0
        
        # Calculate average significance
        avg_significance = sum(p.significance for p in structure_points) / len(structure_points)
        
        # Check for structure consistency
        recent_points = structure_points[-5:]
        if len(recent_points) < 2:
            return avg_significance
        
        # Bonus for consistent structure
        consistency_bonus = 0.0
        for i in range(1, len(recent_points)):
            current = recent_points[i]
            previous = recent_points[i-1]
            
            # Reward consistent trend direction
            if ((current.structure_type in [StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW] and
                 previous.structure_type in [StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW]) or
                (current.structure_type in [StructureType.LOWER_HIGH, StructureType.LOWER_LOW] and
                 previous.structure_type in [StructureType.LOWER_HIGH, StructureType.LOWER_LOW])):
                consistency_bonus += 0.1
        
        quality = min(avg_significance + consistency_bonus, 1.0)
        return quality
    
    def _update_state(self, structure_points: List[StructurePoint], 
                     current_trend: TrendDirection,
                     bos_events: List[Dict], 
                     choch_events: List[Dict]):
        """Update internal market structure state"""
        self.current_state.trend = current_trend
        self.current_state.structure_breaks.extend(bos_events)
        self.current_state.choch_events.extend(choch_events)
        
        # Update latest structure points
        for point in structure_points:
            if point.structure_type == StructureType.HIGHER_HIGH:
                self.current_state.last_hh = point
            elif point.structure_type == StructureType.HIGHER_LOW:
                self.current_state.last_hl = point
            elif point.structure_type == StructureType.LOWER_HIGH:
                self.current_state.last_lh = point
            elif point.structure_type == StructureType.LOWER_LOW:
                self.current_state.last_ll = point
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'timestamp': datetime.now(),
            'trend': TrendDirection.SIDEWAYS,
            'swing_highs': [],
            'swing_lows': [],
            'structure_points': [],
            'bos_events': [],
            'choch_events': [],
            'msb_events': [],
            'trend_strength': 0.0,
            'structure_quality': 0.0,
            'current_state': self.current_state
        }
    
    def get_current_bias(self) -> Dict:
        """Get current market bias based on structure"""
        return {
            'trend': self.current_state.trend,
            'bias_strength': self._calculate_trend_strength([
                p for p in [
                    self.current_state.last_hh,
                    self.current_state.last_hl, 
                    self.current_state.last_lh,
                    self.current_state.last_ll
                ] if p is not None
            ]),
            'last_structure_break': (
                self.current_state.structure_breaks[-1] 
                if self.current_state.structure_breaks 
                else None
            ),
            'last_choch': (
                self.current_state.choch_events[-1]
                if self.current_state.choch_events
                else None
            )
        }

# Export main class
__all__ = [
    'MarketStructureAnalyzer',
    'TrendDirection', 
    'StructureType',
    'StructurePoint',
    'MarketStructureState'
]