# features/fibonacci_levels.py
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FibonacciType(Enum):
    """Fibonacci analysis types"""
    RETRACEMENT = "retracement"
    EXTENSION = "extension"
    PROJECTION = "projection"

class FibonacciDirection(Enum):
    """Fibonacci direction based on swing"""
    BULLISH = 1  # Low to High swing
    BEARISH = -1  # High to Low swing

@dataclass
class FibonacciLevel:
    """Individual Fibonacci level"""
    level: float  # Fibonacci ratio (0.236, 0.382, etc.)
    price: float  # Actual price level
    level_type: FibonacciType
    direction: FibonacciDirection
    distance_from_current: float  # Distance from current price
    confluence_score: float = 0.0  # Confluence with other levels/zones
    
@dataclass
class SwingPoint:
    """Swing high/low point for Fibonacci calculation"""
    price: float
    timestamp: pd.Timestamp
    index: int
    swing_type: str  # 'high' or 'low'
    strength: float = 1.0  # Swing strength based on bars around it

@dataclass
class FibonacciZone:
    """Complete Fibonacci analysis result"""
    swing_high: SwingPoint
    swing_low: SwingPoint
    direction: FibonacciDirection
    fib_type: FibonacciType
    levels: List[FibonacciLevel]
    quality_score: float
    timestamp: pd.Timestamp

class FibonacciAnalyzer:
    """
    Advanced Fibonacci analysis for SMC trading
    Detects swing points and calculates key Fibonacci levels
    """
    
    def __init__(self, 
                 swing_lookback: int = 10,
                 min_swing_size: float = 0.002,  # 20 pips for forex
                 confluence_threshold: float = 0.0005):  # 5 pips confluence zone
        self.swing_lookback = swing_lookback
        self.min_swing_size = min_swing_size
        self.confluence_threshold = confluence_threshold
        
        # Standard Fibonacci levels
        self.retracement_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.extension_levels = [1.0, 1.236, 1.382, 1.5, 1.618, 2.0, 2.618]
        self.projection_levels = [0.618, 1.0, 1.382, 1.618, 2.0, 2.618]
        
        logger.debug("FibonacciAnalyzer initialized")

    def analyze_fibonacci(self, data: pd.DataFrame, 
                         max_swings: int = 3) -> Optional[List[FibonacciZone]]:
        """
        Main Fibonacci analysis method
        
        Args:
            data: OHLCV DataFrame
            max_swings: Maximum number of swing patterns to analyze
            
        Returns:
            List of FibonacciZone objects or None if analysis fails
        """
        try:
            if data is None or len(data) < 20:
                logger.warning("Insufficient data for Fibonacci analysis")
                return None
                
            # Detect swing points
            swing_highs = self._detect_swing_highs(data)
            swing_lows = self._detect_swing_lows(data)
            
            if not swing_highs or not swing_lows:
                logger.warning("No significant swing points found")
                return None
                
            # Find valid swing combinations
            fibonacci_zones = self._create_fibonacci_zones(
                data, swing_highs, swing_lows, max_swings
            )
            
            if fibonacci_zones:
                # Calculate confluence scores
                self._calculate_confluence_scores(fibonacci_zones)
                
                # Sort by quality
                fibonacci_zones.sort(key=lambda x: x.quality_score, reverse=True)
                
                logger.debug(f"Generated {len(fibonacci_zones)} Fibonacci zones")
                return fibonacci_zones
            else:
                logger.warning("No valid Fibonacci zones created")
                return None
                
        except Exception as e:
            logger.error(f"Fibonacci analysis failed: {e}")
            return None

    def _detect_swing_highs(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Detect swing high points"""
        swing_highs = []
        
        try:
            for i in range(self.swing_lookback, len(data) - self.swing_lookback):
                current_high = data.iloc[i]['High']
                
                # Check if current high is higher than surrounding bars
                is_swing_high = True
                for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                    if j != i and data.iloc[j]['High'] >= current_high:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    # Calculate swing strength
                    strength = self._calculate_swing_strength(data, i, 'high')
                    
                    # Validate swing size
                    if self._validate_swing_size(data, i, 'high'):
                        swing_point = SwingPoint(
                            price=current_high,
                            timestamp=data.index[i],
                            index=i,
                            swing_type='high',
                            strength=strength
                        )
                        swing_highs.append(swing_point)
            
            # Sort by strength and keep most significant
            swing_highs.sort(key=lambda x: x.strength, reverse=True)
            return swing_highs[:10]  # Keep top 10 swing highs
            
        except Exception as e:
            logger.error(f"Swing high detection failed: {e}")
            return []

    def _detect_swing_lows(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Detect swing low points"""
        swing_lows = []
        
        try:
            for i in range(self.swing_lookback, len(data) - self.swing_lookback):
                current_low = data.iloc[i]['Low']
                
                # Check if current low is lower than surrounding bars
                is_swing_low = True
                for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                    if j != i and data.iloc[j]['Low'] <= current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    # Calculate swing strength
                    strength = self._calculate_swing_strength(data, i, 'low')
                    
                    # Validate swing size
                    if self._validate_swing_size(data, i, 'low'):
                        swing_point = SwingPoint(
                            price=current_low,
                            timestamp=data.index[i],
                            index=i,
                            swing_type='low',
                            strength=strength
                        )
                        swing_lows.append(swing_point)
            
            # Sort by strength and keep most significant
            swing_lows.sort(key=lambda x: x.strength, reverse=True)
            return swing_lows[:10]  # Keep top 10 swing lows
            
        except Exception as e:
            logger.error(f"Swing low detection failed: {e}")
            return []

    def _calculate_swing_strength(self, data: pd.DataFrame, index: int, swing_type: str) -> float:
        """Calculate swing point strength based on surrounding price action"""
        try:
            strength = 1.0
            lookback_range = min(self.swing_lookback * 2, len(data) - index - 1, index)
            
            if swing_type == 'high':
                current_price = data.iloc[index]['High']
                
                # Count how many bars this high is higher than
                higher_count = 0
                for i in range(max(0, index - lookback_range), 
                              min(len(data), index + lookback_range + 1)):
                    if i != index and data.iloc[i]['High'] < current_price:
                        higher_count += 1
                
                # Calculate volume factor if available
                volume_factor = 1.0
                if 'Volume' in data.columns:
                    avg_volume = data['Volume'].rolling(20).mean().iloc[index]
                    current_volume = data.iloc[index]['Volume']
                    if avg_volume > 0:
                        volume_factor = min(2.0, current_volume / avg_volume)
                
                strength = (higher_count / (lookback_range * 2)) * volume_factor
                
            else:  # swing_type == 'low'
                current_price = data.iloc[index]['Low']
                
                # Count how many bars this low is lower than
                lower_count = 0
                for i in range(max(0, index - lookback_range), 
                              min(len(data), index + lookback_range + 1)):
                    if i != index and data.iloc[i]['Low'] > current_price:
                        lower_count += 1
                
                # Calculate volume factor if available
                volume_factor = 1.0
                if 'Volume' in data.columns:
                    avg_volume = data['Volume'].rolling(20).mean().iloc[index]
                    current_volume = data.iloc[index]['Volume']
                    if avg_volume > 0:
                        volume_factor = min(2.0, current_volume / avg_volume)
                
                strength = (lower_count / (lookback_range * 2)) * volume_factor
            
            return min(10.0, max(0.1, strength))  # Clamp between 0.1 and 10.0
            
        except Exception as e:
            logger.error(f"Swing strength calculation failed: {e}")
            return 1.0

    def _validate_swing_size(self, data: pd.DataFrame, index: int, swing_type: str) -> bool:
        """Validate if swing move is significant enough"""
        try:
            lookback = min(self.swing_lookback, index)
            
            if swing_type == 'high':
                current_high = data.iloc[index]['High']
                recent_low = data.iloc[max(0, index - lookback):index + 1]['Low'].min()
                swing_size = abs(current_high - recent_low) / recent_low
            else:
                current_low = data.iloc[index]['Low']
                recent_high = data.iloc[max(0, index - lookback):index + 1]['High'].max()
                swing_size = abs(recent_high - current_low) / recent_high
            
            return swing_size >= self.min_swing_size
            
        except Exception as e:
            logger.error(f"Swing size validation failed: {e}")
            return False

    def _create_fibonacci_zones(self, data: pd.DataFrame, 
                               swing_highs: List[SwingPoint], 
                               swing_lows: List[SwingPoint],
                               max_swings: int) -> List[FibonacciZone]:
        """Create Fibonacci zones from swing combinations"""
        fibonacci_zones = []
        
        try:
            current_price = data.iloc[-1]['Close']
            current_time = data.index[-1]
            
            # Try different swing combinations
            combinations_tried = 0
            
            for swing_high in swing_highs:
                for swing_low in swing_lows:
                    if combinations_tried >= max_swings * 5:  # Limit combinations
                        break
                        
                    # Ensure proper chronological order
                    if abs(swing_high.index - swing_low.index) < 5:  # Too close
                        continue
                        
                    # Calculate swing size
                    swing_size = abs(swing_high.price - swing_low.price) / min(swing_high.price, swing_low.price)
                    if swing_size < self.min_swing_size:
                        continue
                    
                    combinations_tried += 1
                    
                    # Determine direction and create zones
                    if swing_low.index < swing_high.index:  # Bullish swing
                        direction = FibonacciDirection.BULLISH
                        
                        # Retracement levels (from high back to low)
                        retracement_zone = self._create_fibonacci_zone(
                            swing_high, swing_low, direction, 
                            FibonacciType.RETRACEMENT, current_price, current_time
                        )
                        if retracement_zone:
                            fibonacci_zones.append(retracement_zone)
                        
                        # Extension levels (beyond high)
                        extension_zone = self._create_fibonacci_zone(
                            swing_high, swing_low, direction,
                            FibonacciType.EXTENSION, current_price, current_time
                        )
                        if extension_zone:
                            fibonacci_zones.append(extension_zone)
                            
                    else:  # Bearish swing
                        direction = FibonacciDirection.BEARISH
                        
                        # Retracement levels (from low back to high)
                        retracement_zone = self._create_fibonacci_zone(
                            swing_high, swing_low, direction,
                            FibonacciType.RETRACEMENT, current_price, current_time
                        )
                        if retracement_zone:
                            fibonacci_zones.append(retracement_zone)
                        
                        # Extension levels (beyond low)
                        extension_zone = self._create_fibonacci_zone(
                            swing_high, swing_low, direction,
                            FibonacciType.EXTENSION, current_price, current_time
                        )
                        if extension_zone:
                            fibonacci_zones.append(extension_zone)
            
            return fibonacci_zones
            
        except Exception as e:
            logger.error(f"Fibonacci zone creation failed: {e}")
            return []

    def _create_fibonacci_zone(self, swing_high: SwingPoint, swing_low: SwingPoint,
                              direction: FibonacciDirection, fib_type: FibonacciType,
                              current_price: float, current_time: pd.Timestamp) -> Optional[FibonacciZone]:
        """Create individual Fibonacci zone"""
        try:
            levels = []
            swing_range = abs(swing_high.price - swing_low.price)
            
            # Select appropriate Fibonacci ratios
            if fib_type == FibonacciType.RETRACEMENT:
                fib_ratios = self.retracement_levels
            elif fib_type == FibonacciType.EXTENSION:
                fib_ratios = self.extension_levels
            else:
                fib_ratios = self.projection_levels
            
            # Calculate Fibonacci levels
            for ratio in fib_ratios:
                if direction == FibonacciDirection.BULLISH:
                    if fib_type == FibonacciType.RETRACEMENT:
                        # Retracement from high back toward low
                        price_level = swing_high.price - (swing_range * ratio)
                    else:  # Extension
                        # Extension beyond high
                        price_level = swing_high.price + (swing_range * (ratio - 1.0))
                else:  # BEARISH
                    if fib_type == FibonacciType.RETRACEMENT:
                        # Retracement from low back toward high
                        price_level = swing_low.price + (swing_range * ratio)
                    else:  # Extension
                        # Extension beyond low
                        price_level = swing_low.price - (swing_range * (ratio - 1.0))
                
                # Calculate distance from current price
                distance_from_current = abs(price_level - current_price) / current_price
                
                level = FibonacciLevel(
                    level=ratio,
                    price=price_level,
                    level_type=fib_type,
                    direction=direction,
                    distance_from_current=distance_from_current
                )
                levels.append(level)
            
            # Calculate quality score based on swing strength and recency
            time_factor = max(0.1, 1.0 - (current_time - max(swing_high.timestamp, swing_low.timestamp)).days / 30)
            swing_strength_factor = (swing_high.strength + swing_low.strength) / 2
            size_factor = min(2.0, swing_range / (current_price * 0.01))  # Prefer bigger swings
            
            quality_score = time_factor * swing_strength_factor * size_factor
            
            fibonacci_zone = FibonacciZone(
                swing_high=swing_high,
                swing_low=swing_low,
                direction=direction,
                fib_type=fib_type,
                levels=levels,
                quality_score=quality_score,
                timestamp=current_time
            )
            
            return fibonacci_zone
            
        except Exception as e:
            logger.error(f"Individual Fibonacci zone creation failed: {e}")
            return None

    def _calculate_confluence_scores(self, fibonacci_zones: List[FibonacciZone]) -> None:
        """Calculate confluence scores between Fibonacci levels"""
        try:
            all_levels = []
            
            # Collect all levels from all zones
            for zone in fibonacci_zones:
                for level in zone.levels:
                    all_levels.append((level, zone))
            
            # Calculate confluence for each level
            for level, parent_zone in all_levels:
                confluence_count = 0
                
                for other_level, other_zone in all_levels:
                    if level != other_level and parent_zone != other_zone:
                        price_difference = abs(level.price - other_level.price)
                        relative_difference = price_difference / level.price
                        
                        if relative_difference <= self.confluence_threshold:
                            confluence_count += 1
                
                # Confluence score based on number of nearby levels
                level.confluence_score = min(10.0, confluence_count * 2.0)
                
                # Boost quality of parent zone based on confluence
                if level.confluence_score > 0:
                    parent_zone.quality_score += level.confluence_score * 0.1
                    
        except Exception as e:
            logger.error(f"Confluence score calculation failed: {e}")

    def get_key_levels_near_price(self, fibonacci_zones: List[FibonacciZone], 
                                 current_price: float, 
                                 distance_threshold: float = 0.02) -> List[FibonacciLevel]:
        """Get key Fibonacci levels near current price"""
        try:
            near_levels = []
            
            for zone in fibonacci_zones:
                for level in zone.levels:
                    if level.distance_from_current <= distance_threshold:
                        near_levels.append(level)
            
            # Sort by distance from current price
            near_levels.sort(key=lambda x: x.distance_from_current)
            
            return near_levels
            
        except Exception as e:
            logger.error(f"Getting key levels near price failed: {e}")
            return []

    def get_fibonacci_summary(self, fibonacci_zones: List[FibonacciZone]) -> Dict:
        """Generate summary of Fibonacci analysis"""
        try:
            if not fibonacci_zones:
                return {
                    'total_zones': 0,
                    'key_levels_count': 0,
                    'high_quality_zones': 0,
                    'confluence_levels': 0
                }
            
            high_quality_zones = len([z for z in fibonacci_zones if z.quality_score > 2.0])
            confluence_levels = sum(len([l for l in zone.levels if l.confluence_score > 0]) 
                                  for zone in fibonacci_zones)
            
            total_levels = sum(len(zone.levels) for zone in fibonacci_zones)
            
            summary = {
                'total_zones': len(fibonacci_zones),
                'total_levels': total_levels,
                'high_quality_zones': high_quality_zones,
                'confluence_levels': confluence_levels,
                'avg_quality_score': np.mean([z.quality_score for z in fibonacci_zones]),
                'best_zone_quality': max(z.quality_score for z in fibonacci_zones)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Fibonacci summary generation failed: {e}")
            return {}


# Testing function for development
def test_fibonacci_analyzer():
    """Test function for Fibonacci analyzer"""
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from data.connectors.mt5_connector import get_mt5_connector
        from config.mt5_config import MT5_CONNECTION, get_trading_symbols
        
        # Initialize analyzer
        fib_analyzer = FibonacciAnalyzer()
        
        # Get real MT5 data
        connector = get_mt5_connector(MT5_CONNECTION)
        symbols = get_trading_symbols("major")[:2]  # Test with 2 major pairs
        
        for symbol in symbols:
            print(f"\n=== Testing Fibonacci Analysis for {symbol} ===")
            
            # Get data for different timeframes
            timeframes = ["H4", "H1"]
            
            for tf in timeframes:
                print(f"\n--- {symbol} {tf} ---")
                
                data = connector.get_rates(symbol, tf, 200)
                if data is not None and len(data) > 50:
                    
                    # Perform Fibonacci analysis
                    fib_zones = fib_analyzer.analyze_fibonacci(data, max_swings=3)
                    
                    if fib_zones:
                        summary = fib_analyzer.get_fibonacci_summary(fib_zones)
                        print(f"Fibonacci Analysis Results:")
                        print(f"  Total zones: {summary['total_zones']}")
                        print(f"  Total levels: {summary['total_levels']}")
                        print(f"  High quality zones: {summary['high_quality_zones']}")
                        print(f"  Confluence levels: {summary['confluence_levels']}")
                        print(f"  Best zone quality: {summary.get('best_zone_quality', 0):.2f}")
                        
                        # Show top 3 zones
                        print("\n  Top Fibonacci Zones:")
                        for i, zone in enumerate(fib_zones[:3]):
                            print(f"    Zone {i+1}: {zone.fib_type.value} {zone.direction.name}")
                            print(f"      Quality: {zone.quality_score:.2f}")
                            print(f"      Swing: {zone.swing_low.price:.5f} -> {zone.swing_high.price:.5f}")
                            
                            # Show key levels
                            current_price = data.iloc[-1]['Close']
                            key_levels = fib_analyzer.get_key_levels_near_price(
                                [zone], current_price, distance_threshold=0.03
                            )
                            
                            if key_levels:
                                print(f"      Key levels near price ({current_price:.5f}):")
                                for level in key_levels[:3]:
                                    print(f"        {level.level:.3f}: {level.price:.5f} "
                                          f"(distance: {level.distance_from_current:.1%})")
                    else:
                        print("  No Fibonacci zones detected")
                else:
                    print(f"  Failed to get data for {symbol} {tf}")
        
        print("\n=== Fibonacci Analysis Test Completed ===")
        return True
        
    except Exception as e:
        print(f"Fibonacci test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fibonacci_analyzer()