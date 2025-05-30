"""
Liquidity Detection and Analysis for SMC Trading
Identifies liquidity pools, sweeps, and grabs
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LiquidityType(Enum):
    """Types of liquidity"""
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"

class LiquidityEvent(Enum):
    """Liquidity events"""
    SWEEP = "sweep"
    GRAB = "grab"
    HUNT = "hunt"
    PURGE = "purge"

@dataclass
class LiquidityPool:
    """Liquidity pool data structure"""
    timestamp: datetime
    price: float
    liquidity_type: LiquidityType
    strength: float  # 0.0 to 1.0
    volume: int
    touches: int = 0
    last_test: Optional[datetime] = None
    swept: bool = False
    sweep_timestamp: Optional[datetime] = None
    expected_stops: int = 0  # Estimated stop orders
    
    @property
    def age_hours(self) -> float:
        """Get age of liquidity pool in hours"""
        return (datetime.now() - self.timestamp).total_seconds() / 3600
    
    @property
    def is_fresh(self) -> bool:
        """Check if liquidity is still fresh (untested)"""
        return self.touches == 0 and not self.swept

@dataclass
class LiquiditySweep:
    """Liquidity sweep event"""
    timestamp: datetime
    sweep_type: LiquidityEvent
    price: float
    liquidity_pool: LiquidityPool
    penetration_distance: float
    reversal_strength: float
    volume_spike: bool = False

class LiquidityAnalyzer:
    """
    Analyzes liquidity patterns and sweep events
    """
    
    def __init__(self,
                 equal_level_threshold: float = 0.0002,  # 2 pips tolerance
                 min_liquidity_gap: int = 5,  # Minimum bars between levels
                 sweep_confirmation_bars: int = 3,
                 min_sweep_distance: float = 0.0001):  # 1 pip minimum
        
        self.equal_level_threshold = equal_level_threshold
        self.min_liquidity_gap = min_liquidity_gap
        self.sweep_confirmation_bars = sweep_confirmation_bars
        self.min_sweep_distance = min_sweep_distance
        
        # Storage for identified liquidity
        self.liquidity_pools = []
        self.sweep_events = []
        
    def analyze_liquidity(self, data: pd.DataFrame) -> Dict:
        """
        Main function to analyze liquidity patterns
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            Dictionary with liquidity analysis
        """
        if len(data) < self.min_liquidity_gap * 4:
            logger.warning("Insufficient data for liquidity analysis")
            return self._empty_analysis()
        
        try:
            # Step 1: Identify equal highs and lows
            equal_highs = self._identify_equal_highs(data)
            equal_lows = self._identify_equal_lows(data)
            
            # Step 2: Find double/triple tops and bottoms
            multiple_tops = self._identify_multiple_tops(data)
            multiple_bottoms = self._identify_multiple_bottoms(data)
            
            # Step 3: Combine all liquidity pools
            all_liquidity = equal_highs + equal_lows + multiple_tops + multiple_bottoms
            
            # Step 4: Update existing pools and add new ones
            self._update_liquidity_pools(all_liquidity, data)
            
            # Step 5: Detect liquidity sweeps
            new_sweeps = self._detect_liquidity_sweeps(data)
            self.sweep_events.extend(new_sweeps)
            
            # Step 6: Calculate liquidity metrics
            liquidity_metrics = self._calculate_liquidity_metrics(data)
            
            # Step 7: Assess current liquidity zones
            current_zones = self._assess_current_zones(data)
            
            return {
                'timestamp': data.index[-1],
                'liquidity_pools': self.liquidity_pools,
                'equal_highs': equal_highs,
                'equal_lows': equal_lows,
                'multiple_tops': multiple_tops,
                'multiple_bottoms': multiple_bottoms,
                'sweep_events': new_sweeps,
                'all_sweeps': self.sweep_events,
                'metrics': liquidity_metrics,
                'current_zones': current_zones,
                'summary': self._generate_liquidity_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return self._empty_analysis()
    
    def _identify_equal_highs(self, data: pd.DataFrame) -> List[LiquidityPool]:
        """Identify equal highs patterns"""
        equal_highs = []
        highs = data['High'].values
        volumes = data.get('Volume', pd.Series(0, index=data.index)).values
        times = data.index
        
        # Find potential equal highs
        for i in range(self.min_liquidity_gap, len(data) - self.min_liquidity_gap):
            current_high = highs[i]
            
            # Look for similar highs within reasonable distance
            for j in range(i + self.min_liquidity_gap, min(i + 50, len(data))):
                compare_high = highs[j]
                
                # Check if highs are approximately equal
                price_diff = abs(current_high - compare_high)
                if price_diff <= self.equal_level_threshold:
                    
                    # Calculate strength based on volume and price action
                    avg_volume = (volumes[i] + volumes[j]) / 2
                    max_volume = max(volumes[i], volumes[j])
                    volume_strength = min(max_volume / (avg_volume + 1), 2.0) / 2.0
                    
                    # Distance strength (closer levels = stronger)
                    distance_bars = j - i
                    distance_strength = max(0.1, 1.0 - (distance_bars / 50.0))
                    
                    # Combined strength
                    strength = (volume_strength + distance_strength) / 2
                    
                    # Estimate stops above equal highs
                    price_level = max(current_high, compare_high)
                    estimated_stops = self._estimate_stops_above(price_level, data, i, j)
                    
                    liquidity_pool = LiquidityPool(
                        timestamp=times[i],
                        price=price_level,
                        liquidity_type=LiquidityType.EQUAL_HIGHS,
                        strength=strength,
                        volume=int(avg_volume),
                        expected_stops=estimated_stops
                    )
                    
                    equal_highs.append(liquidity_pool)
                    break  # Found match, move to next potential level
        
        return equal_highs
    
    def _identify_equal_lows(self, data: pd.DataFrame) -> List[LiquidityPool]:
        """Identify equal lows patterns"""
        equal_lows = []
        lows = data['Low'].values
        volumes = data.get('Volume', pd.Series(0, index=data.index)).values
        times = data.index
        
        # Find potential equal lows
        for i in range(self.min_liquidity_gap, len(data) - self.min_liquidity_gap):
            current_low = lows[i]
            
            # Look for similar lows within reasonable distance
            for j in range(i + self.min_liquidity_gap, min(i + 50, len(data))):
                compare_low = lows[j]
                
                # Check if lows are approximately equal
                price_diff = abs(current_low - compare_low)
                if price_diff <= self.equal_level_threshold:
                    
                    # Calculate strength
                    avg_volume = (volumes[i] + volumes[j]) / 2
                    max_volume = max(volumes[i], volumes[j])
                    volume_strength = min(max_volume / (avg_volume + 1), 2.0) / 2.0
                    
                    distance_bars = j - i
                    distance_strength = max(0.1, 1.0 - (distance_bars / 50.0))
                    
                    strength = (volume_strength + distance_strength) / 2
                    
                    # Estimate stops below equal lows
                    price_level = min(current_low, compare_low)
                    estimated_stops = self._estimate_stops_below(price_level, data, i, j)
                    
                    liquidity_pool = LiquidityPool(
                        timestamp=times[i],
                        price=price_level,
                        liquidity_type=LiquidityType.EQUAL_LOWS,
                        strength=strength,
                        volume=int(avg_volume),
                        expected_stops=estimated_stops
                    )
                    
                    equal_lows.append(liquidity_pool)
                    break
        
        return equal_lows
    
    def _identify_multiple_tops(self, data: pd.DataFrame) -> List[LiquidityPool]:
        """Identify double/triple tops"""
        multiple_tops = []
        highs = data['High'].values
        times = data.index
        
        # Look for multiple top patterns
        for i in range(20, len(data) - 20):
            current_high = highs[i]
            
            # Check if this is a significant high
            if not self._is_significant_high(i, highs, 10):
                continue
            
            # Look for other highs at similar level
            similar_highs = []
            for j in range(max(0, i - 40), min(len(data), i + 40)):
                if abs(j - i) < 10:  # Skip too close bars
                    continue
                
                if (self._is_significant_high(j, highs, 5) and 
                    abs(highs[j] - current_high) <= self.equal_level_threshold):
                    similar_highs.append(j)
            
            # Classify based on number of similar highs
            if len(similar_highs) >= 1:  # Double top
                liquidity_type = LiquidityType.DOUBLE_TOP
                if len(similar_highs) >= 2:  # Triple top
                    liquidity_type = LiquidityType.TRIPLE_TOP
                
                # Calculate strength based on number of touches
                strength = min(0.3 + (len(similar_highs) * 0.3), 1.0)
                
                estimated_stops = len(similar_highs) * 50  # Estimate based on pattern
                
                liquidity_pool = LiquidityPool(
                    timestamp=times[i],
                    price=current_high,
                    liquidity_type=liquidity_type,
                    strength=strength,
                    volume=0,  # Will be updated with volume data
                    touches=len(similar_highs),
                    expected_stops=estimated_stops
                )
                
                multiple_tops.append(liquidity_pool)
        
        return multiple_tops
    
    def _identify_multiple_bottoms(self, data: pd.DataFrame) -> List[LiquidityPool]:
        """Identify double/triple bottoms"""
        multiple_bottoms = []
        lows = data['Low'].values
        times = data.index
        
        # Look for multiple bottom patterns
        for i in range(20, len(data) - 20):
            current_low = lows[i]
            
            # Check if this is a significant low
            if not self._is_significant_low(i, lows, 10):
                continue
            
            # Look for other lows at similar level
            similar_lows = []
            for j in range(max(0, i - 40), min(len(data), i + 40)):
                if abs(j - i) < 10:
                    continue
                
                if (self._is_significant_low(j, lows, 5) and 
                    abs(lows[j] - current_low) <= self.equal_level_threshold):
                    similar_lows.append(j)
            
            # Classify based on number of similar lows
            if len(similar_lows) >= 1:  # Double bottom
                liquidity_type = LiquidityType.DOUBLE_BOTTOM
                if len(similar_lows) >= 2:  # Triple bottom
                    liquidity_type = LiquidityType.TRIPLE_BOTTOM
                
                strength = min(0.3 + (len(similar_lows) * 0.3), 1.0)
                estimated_stops = len(similar_lows) * 50
                
                liquidity_pool = LiquidityPool(
                    timestamp=times[i],
                    price=current_low,
                    liquidity_type=liquidity_type,
                    strength=strength,
                    volume=0,
                    touches=len(similar_lows),
                    expected_stops=estimated_stops
                )
                
                multiple_bottoms.append(liquidity_pool)
        
        return multiple_bottoms
    
    def _is_significant_high(self, index: int, highs: np.ndarray, lookback: int) -> bool:
        """Check if index represents a significant high"""
        if index < lookback or index >= len(highs) - lookback:
            return False
        
        current_high = highs[index]
        
        # Check if it's higher than surrounding bars
        for i in range(index - lookback, index + lookback + 1):
            if i != index and highs[i] >= current_high:
                return False
        
        return True
    
    def _is_significant_low(self, index: int, lows: np.ndarray, lookback: int) -> bool:
        """Check if index represents a significant low"""
        if index < lookback or index >= len(lows) - lookback:
            return False
        
        current_low = lows[index]
        
        # Check if it's lower than surrounding bars
        for i in range(index - lookback, index + lookback + 1):
            if i != index and lows[i] <= current_low:
                return False
        
        return True
    
    def _estimate_stops_above(self, price_level: float, data: pd.DataFrame, 
                            start_idx: int, end_idx: int) -> int:
        """Estimate number of stop orders above price level"""
        # Simple estimation based on price action and volume
        volume_data = data['Volume'].iloc[start_idx:end_idx+1]
        avg_volume = volume_data.mean()
        
        # Higher volume = more participants = more stops
        volume_factor = min(avg_volume / 100, 10) if avg_volume > 0 else 1
        
        # Distance between levels factor
        bars_distance = end_idx - start_idx
        distance_factor = max(1, bars_distance / 10)
        
        estimated_stops = int(volume_factor * distance_factor * 20)
        return min(estimated_stops, 1000)  # Cap at reasonable number
    
    def _estimate_stops_below(self, price_level: float, data: pd.DataFrame,
                            start_idx: int, end_idx: int) -> int:
        """Estimate number of stop orders below price level"""
        return self._estimate_stops_above(price_level, data, start_idx, end_idx)
    
    def _update_liquidity_pools(self, new_liquidity: List[LiquidityPool], data: pd.DataFrame):
        """Update existing liquidity pools and add new ones"""
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1]
        
        # Update existing pools
        for pool in self.liquidity_pools:
            # Check if pool has been tested
            distance_to_pool = abs(current_price - pool.price)
            if distance_to_pool <= self.equal_level_threshold * 2:
                pool.touches += 1
                pool.last_test = current_time
        
        # Add new pools that don't duplicate existing ones
        for new_pool in new_liquidity:
            if not self._is_duplicate_pool(new_pool):
                self.liquidity_pools.append(new_pool)
        
        # Clean up old pools
        self._cleanup_old_pools()
    
    def _is_duplicate_pool(self, new_pool: LiquidityPool) -> bool:
        """Check if liquidity pool already exists"""
        for existing_pool in self.liquidity_pools:
            if (abs(existing_pool.price - new_pool.price) <= self.equal_level_threshold and
                existing_pool.liquidity_type == new_pool.liquidity_type):
                return True
        return False
    
    def _cleanup_old_pools(self):
        """Remove old or swept liquidity pools"""
        current_time = datetime.now()
        
        # Remove pools older than 7 days or already swept
        self.liquidity_pools = [
            pool for pool in self.liquidity_pools
            if ((current_time - pool.timestamp).total_seconds() < 604800 and  # 7 days
                not pool.swept)
        ]
        
        # Keep only top 50 pools by strength
        if len(self.liquidity_pools) > 50:
            self.liquidity_pools.sort(key=lambda x: x.strength, reverse=True)
            self.liquidity_pools = self.liquidity_pools[:50]
    
    def _detect_liquidity_sweeps(self, data: pd.DataFrame) -> List[LiquiditySweep]:
        """Detect liquidity sweep events"""
        sweeps = []
        
        if len(data) < self.sweep_confirmation_bars:
            return sweeps
        
        current_time = data.index[-1]
        recent_data = data.tail(self.sweep_confirmation_bars + 5)
        
        for pool in self.liquidity_pools:
            if pool.swept:
                continue
            
            # Check for sweep above (for highs) or below (for lows)
            if pool.liquidity_type in [LiquidityType.EQUAL_HIGHS, LiquidityType.DOUBLE_TOP, LiquidityType.TRIPLE_TOP]:
                sweep = self._check_upward_sweep(pool, recent_data)
            else:  # EQUAL_LOWS, DOUBLE_BOTTOM, TRIPLE_BOTTOM
                sweep = self._check_downward_sweep(pool, recent_data)
            
            if sweep:
                pool.swept = True
                pool.sweep_timestamp = current_time
                sweeps.append(sweep)
        
        return sweeps
    
    def _check_upward_sweep(self, pool: LiquidityPool, data: pd.DataFrame) -> Optional[LiquiditySweep]:
        """Check for upward liquidity sweep"""
        highs = data['High'].values
        closes = data['Close'].values
        times = data.index
        
        # Look for penetration above pool level
        for i, high in enumerate(highs):
            if high > pool.price + self.min_sweep_distance:
                # Check for reversal after sweep
                penetration = high - pool.price
                
                # Look for reversal in subsequent bars
                reversal_strength = 0.0
                for j in range(i + 1, min(i + self.sweep_confirmation_bars + 1, len(closes))):
                    if closes[j] < pool.price:  # Closed back below level
                        reversal_strength = (high - closes[j]) / high
                        break
                
                if reversal_strength > 0.001:  # Minimum reversal required
                    return LiquiditySweep(
                        timestamp=times[i],
                        sweep_type=LiquidityEvent.SWEEP,
                        price=high,
                        liquidity_pool=pool,
                        penetration_distance=penetration,
                        reversal_strength=reversal_strength
                    )
        
        return None
    
    def _check_downward_sweep(self, pool: LiquidityPool, data: pd.DataFrame) -> Optional[LiquiditySweep]:
        """Check for downward liquidity sweep"""
        lows = data['Low'].values
        closes = data['Close'].values
        times = data.index
        
        # Look for penetration below pool level
        for i, low in enumerate(lows):
            if low < pool.price - self.min_sweep_distance:
                # Check for reversal after sweep
                penetration = pool.price - low
                
                # Look for reversal in subsequent bars
                reversal_strength = 0.0
                for j in range(i + 1, min(i + self.sweep_confirmation_bars + 1, len(closes))):
                    if closes[j] > pool.price:  # Closed back above level
                        reversal_strength = (closes[j] - low) / closes[j]
                        break
                
                if reversal_strength > 0.001:
                    return LiquiditySweep(
                        timestamp=times[i],
                        sweep_type=LiquidityEvent.SWEEP,
                        price=low,
                        liquidity_pool=pool,
                        penetration_distance=penetration,
                        reversal_strength=reversal_strength
                    )
        
        return None
    
    def _calculate_liquidity_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate liquidity-related metrics"""
        current_price = data['Close'].iloc[-1]
        
        # Count different types of liquidity
        total_pools = len(self.liquidity_pools)
        fresh_pools = len([p for p in self.liquidity_pools if p.is_fresh])
        tested_pools = len([p for p in self.liquidity_pools if p.touches > 0])
        swept_pools = len([p for p in self.liquidity_pools if p.swept])
        
        # Calculate average strength
        avg_strength = (sum(p.strength for p in self.liquidity_pools) / total_pools 
                       if total_pools > 0 else 0.0)
        
        # Count by type
        equal_highs_count = len([p for p in self.liquidity_pools 
                               if p.liquidity_type == LiquidityType.EQUAL_HIGHS])
        equal_lows_count = len([p for p in self.liquidity_pools 
                              if p.liquidity_type == LiquidityType.EQUAL_LOWS])
        
        # Recent sweep activity
        recent_sweeps = len([s for s in self.sweep_events 
                           if (data.index[-1] - s.timestamp).total_seconds() < 86400])  # Last 24h
        
        return {
            'total_pools': total_pools,
            'fresh_pools': fresh_pools,
            'tested_pools': tested_pools,
            'swept_pools': swept_pools,
            'avg_strength': avg_strength,
            'equal_highs_count': equal_highs_count,
            'equal_lows_count': equal_lows_count,
            'recent_sweeps': recent_sweeps,
            'total_sweeps': len(self.sweep_events)
        }
    
    def _assess_current_zones(self, data: pd.DataFrame) -> Dict:
        """Assess current liquidity zones relative to price"""
        current_price = data['Close'].iloc[-1]
        
        # Find nearby liquidity pools
        nearby_pools = []
        for pool in self.liquidity_pools:
            distance = abs(current_price - pool.price)
            if distance <= 0.005:  # Within 50 pips
                nearby_pools.append({
                    'pool': pool,
                    'distance': distance,
                    'direction': 'above' if pool.price > current_price else 'below'
                })
        
        # Sort by distance
        nearby_pools.sort(key=lambda x: x['distance'])
        
        # Identify immediate zones
        resistance_zones = [p for p in nearby_pools if p['direction'] == 'above'][:3]
        support_zones = [p for p in nearby_pools if p['direction'] == 'below'][:3]
        
        return {
            'current_price': current_price,
            'nearby_pools': nearby_pools[:10],  # Top 10 closest
            'resistance_zones': resistance_zones,
            'support_zones': support_zones,
            'in_liquidity_zone': len(nearby_pools) > 0
        }
    
    def _generate_liquidity_summary(self) -> Dict:
        """Generate summary of liquidity state"""
        if not self.liquidity_pools:
            return {
                'dominant_liquidity': None,
                'sweep_tendency': None,
                'liquidity_quality': 0.0
            }
        
        # Determine dominant liquidity type
        high_liquidity = len([p for p in self.liquidity_pools 
                            if p.liquidity_type in [LiquidityType.EQUAL_HIGHS, 
                                                  LiquidityType.DOUBLE_TOP, 
                                                  LiquidityType.TRIPLE_TOP]])
        low_liquidity = len([p for p in self.liquidity_pools 
                           if p.liquidity_type in [LiquidityType.EQUAL_LOWS,
                                                 LiquidityType.DOUBLE_BOTTOM,
                                                 LiquidityType.TRIPLE_BOTTOM]])
        
        if high_liquidity > low_liquidity:
            dominant_liquidity = "resistance_heavy"
        elif low_liquidity > high_liquidity:
            dominant_liquidity = "support_heavy"
        else:
            dominant_liquidity = "balanced"
        
        # Assess sweep tendency
        recent_sweeps = [s for s in self.sweep_events 
                        if (datetime.now() - s.timestamp).total_seconds() < 86400]
        
        if len(recent_sweeps) > 3:
            sweep_tendency = "high_activity"
        elif len(recent_sweeps) > 1:
            sweep_tendency = "moderate_activity"
        else:
            sweep_tendency = "low_activity"
        
        # Calculate overall liquidity quality
        avg_strength = sum(p.strength for p in self.liquidity_pools) / len(self.liquidity_pools)
        fresh_ratio = len([p for p in self.liquidity_pools if p.is_fresh]) / len(self.liquidity_pools)
        liquidity_quality = (avg_strength + fresh_ratio) / 2
        
        return {
            'dominant_liquidity': dominant_liquidity,
            'sweep_tendency': sweep_tendency,
            'liquidity_quality': liquidity_quality
        }
    
    def get_liquidity_near_price(self, price: float, distance: float = 0.001) -> List[LiquidityPool]:
        """Get liquidity pools near specified price"""
        nearby_pools = []
        
        for pool in self.liquidity_pools:
            if abs(pool.price - price) <= distance and not pool.swept:
                nearby_pools.append(pool)
        
        # Sort by distance to price
        nearby_pools.sort(key=lambda x: abs(x.price - price))
        return nearby_pools
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'timestamp': datetime.now(),
            'liquidity_pools': [],
            'equal_highs': [],
            'equal_lows': [],
            'multiple_tops': [],
            'multiple_bottoms': [],
            'sweep_events': [],
            'all_sweeps': [],
            'metrics': {
                'total_pools': 0,
                'fresh_pools': 0,
                'tested_pools': 0,
                'swept_pools': 0,
                'avg_strength': 0.0,
                'equal_highs_count': 0,
                'equal_lows_count': 0,
                'recent_sweeps': 0,
                'total_sweeps': 0
            },
            'current_zones': {
                'current_price': 0.0,
                'nearby_pools': [],
                'resistance_zones': [],
                'support_zones': [],
                'in_liquidity_zone': False
            },
            'summary': {
                'dominant_liquidity': None,
                'sweep_tendency': None,
                'liquidity_quality': 0.0
            }
        }

# Export main classes
__all__ = [
    'LiquidityAnalyzer',
    'LiquidityPool',
    'LiquiditySweep',
    'LiquidityType',
    'LiquidityEvent'
]