"""
Order Block Detection and Analysis
Identifies institutional order blocks for SMC trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderBlockType(Enum):
    """Order block types"""
    BULLISH = "bullish"
    BEARISH = "bearish"

class OrderBlockStatus(Enum):
    """Order block status"""
    FRESH = "fresh"
    TESTED = "tested"  
    BROKEN = "broken"
    RESPECTED = "respected"

@dataclass
class OrderBlock:
    """Order block data structure"""
    timestamp: datetime
    ob_type: OrderBlockType
    top: float
    bottom: float
    body_top: float
    body_bottom: float
    volume: int
    strength: float  # 0.0 to 1.0
    status: OrderBlockStatus = OrderBlockStatus.FRESH
    test_count: int = 0
    creation_candle_index: int = 0
    reaction_strength: float = 0.0
    mitigation_level: float = 0.0
    
    @property
    def mid_point(self) -> float:
        """Get order block midpoint"""
        return (self.top + self.bottom) / 2
    
    @property
    def size(self) -> float:
        """Get order block size"""
        return self.top - self.bottom
    
    @property
    def body_size(self) -> float:
        """Get order block body size"""
        return abs(self.body_top - self.body_bottom)

class OrderBlockAnalyzer:
    """
    Analyzes and identifies order blocks in price action
    """
    
    def __init__(self,
                 min_reaction_pips: float = 10.0,
                 min_ob_body_size: float = 0.0005,
                 lookback_period: int = 50,
                 confirmation_bars: int = 3,
                 max_test_distance: float = 0.0002):
        
        self.min_reaction_pips = min_reaction_pips
        self.min_ob_body_size = min_ob_body_size
        self.lookback_period = lookback_period
        self.confirmation_bars = confirmation_bars
        self.max_test_distance = max_test_distance
        
        # Storage for identified order blocks
        self.order_blocks = []
        
    def analyze_order_blocks(self, data: pd.DataFrame) -> Dict:
        """
        Main function to analyze order blocks
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            Dictionary with order block analysis
        """
        if len(data) < self.lookback_period:
            logger.warning("Insufficient data for order block analysis")
            return self._empty_analysis()
        
        try:
            # Step 1: Identify potential order blocks
            potential_obs = self._identify_potential_order_blocks(data)
            
            # Step 2: Validate order blocks with reaction analysis
            valid_obs = self._validate_order_blocks(potential_obs, data)
            
            # Step 3: Update existing order blocks status
            self._update_order_block_status(data)
            
            # Step 4: Add new valid order blocks
            for ob in valid_obs:
                if not self._is_duplicate_order_block(ob):
                    self.order_blocks.append(ob)
            
            # Step 5: Clean up old/broken order blocks
            self._cleanup_order_blocks()
            
            # Step 6: Calculate order block metrics
            ob_metrics = self._calculate_order_block_metrics(data)
            
            return {
                'timestamp': data.index[-1],
                'order_blocks': self.order_blocks,
                'fresh_obs': [ob for ob in self.order_blocks if ob.status == OrderBlockStatus.FRESH],
                'tested_obs': [ob for ob in self.order_blocks if ob.status == OrderBlockStatus.TESTED],
                'respected_obs': [ob for ob in self.order_blocks if ob.status == OrderBlockStatus.RESPECTED],
                'new_obs': valid_obs,
                'metrics': ob_metrics,
                'summary': self._generate_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in order block analysis: {e}")
            return self._empty_analysis()
    
    def _identify_potential_order_blocks(self, data: pd.DataFrame) -> List[OrderBlock]:
        """Identify potential order blocks based on price action"""
        potential_obs = []
        
        opens = data['Open'].values
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        volumes = data.get('Volume', pd.Series(0, index=data.index)).values
        times = data.index
        
        # Look for order blocks in recent price action
        for i in range(self.confirmation_bars, len(data) - self.confirmation_bars):
            
            # Bullish Order Block: Last bearish candle before bullish move
            if self._is_potential_bullish_ob(i, opens, highs, lows, closes):
                ob = self._create_bullish_order_block(i, data, volumes[i])
                if ob:
                    potential_obs.append(ob)
            
            # Bearish Order Block: Last bullish candle before bearish move  
            if self._is_potential_bearish_ob(i, opens, highs, lows, closes):
                ob = self._create_bearish_order_block(i, data, volumes[i])
                if ob:
                    potential_obs.append(ob)
        
        logger.debug(f"Identified {len(potential_obs)} potential order blocks")
        return potential_obs
    
    def _is_potential_bullish_ob(self, index: int, opens: np.ndarray, 
                                highs: np.ndarray, lows: np.ndarray, 
                                closes: np.ndarray) -> bool:
        """Check if candle could be a bullish order block"""
        
        # Current candle should be bearish (close < open)
        if closes[index] >= opens[index]:
            return False
        
        # Check for bullish reaction in following candles
        bullish_reaction = False
        for j in range(index + 1, min(index + self.confirmation_bars + 1, len(closes))):
            if closes[j] > opens[index]:  # Price moves above OB candle open
                bullish_reaction = True
                break
        
        if not bullish_reaction:
            return False
        
        # Check minimum body size
        body_size = abs(opens[index] - closes[index])
        if body_size < self.min_ob_body_size:
            return False
        
        # Additional validation: should be a significant move
        total_range = highs[index] - lows[index]
        if body_size / total_range < 0.6:  # Body should be at least 60% of total range
            return False
        
        return True
    
    def _is_potential_bearish_ob(self, index: int, opens: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray) -> bool:
        """Check if candle could be a bearish order block"""
        
        # Current candle should be bullish (close > open)
        if closes[index] <= opens[index]:
            return False
        
        # Check for bearish reaction in following candles
        bearish_reaction = False
        for j in range(index + 1, min(index + self.confirmation_bars + 1, len(closes))):
            if closes[j] < opens[index]:  # Price moves below OB candle open
                bearish_reaction = True
                break
        
        if not bearish_reaction:
            return False
        
        # Check minimum body size
        body_size = abs(opens[index] - closes[index])
        if body_size < self.min_ob_body_size:
            return False
        
        # Additional validation
        total_range = highs[index] - lows[index]
        if body_size / total_range < 0.6:
            return False
        
        return True
    
    def _create_bullish_order_block(self, index: int, data: pd.DataFrame, volume: int) -> Optional[OrderBlock]:
        """Create bullish order block from candle data"""
        try:
            candle = data.iloc[index]
            
            # Calculate reaction strength
            reaction_strength = self._calculate_reaction_strength(index, data, OrderBlockType.BULLISH)
            
            ob = OrderBlock(
                timestamp=candle.name,
                ob_type=OrderBlockType.BULLISH,
                top=candle['High'],
                bottom=candle['Low'],
                body_top=candle['Open'],  # For bearish candle, open is body top
                body_bottom=candle['Close'],  # Close is body bottom
                volume=volume,
                strength=reaction_strength,
                creation_candle_index=index,
                reaction_strength=reaction_strength,
                mitigation_level=candle['Close']  # 50% level for bullish OB
            )
            
            return ob
            
        except Exception as e:
            logger.error(f"Error creating bullish order block: {e}")
            return None
    
    def _create_bearish_order_block(self, index: int, data: pd.DataFrame, volume: int) -> Optional[OrderBlock]:
        """Create bearish order block from candle data"""
        try:
            candle = data.iloc[index]
            
            # Calculate reaction strength
            reaction_strength = self._calculate_reaction_strength(index, data, OrderBlockType.BEARISH)
            
            ob = OrderBlock(
                timestamp=candle.name,
                ob_type=OrderBlockType.BEARISH,
                top=candle['High'],
                bottom=candle['Low'], 
                body_top=candle['Close'],  # For bullish candle, close is body top
                body_bottom=candle['Open'],  # Open is body bottom
                volume=volume,
                strength=reaction_strength,
                creation_candle_index=index,
                reaction_strength=reaction_strength,
                mitigation_level=candle['Open']  # 50% level for bearish OB
            )
            
            return ob
            
        except Exception as e:
            logger.error(f"Error creating bearish order block: {e}")
            return None
    
    def _calculate_reaction_strength(self, index: int, data: pd.DataFrame, ob_type: OrderBlockType) -> float:
        """Calculate strength of reaction from order block"""
        try:
            if index + self.confirmation_bars >= len(data):
                return 0.0
            
            ob_candle = data.iloc[index]
            
            # Calculate move from order block
            max_reaction = 0.0
            
            for i in range(index + 1, min(index + self.confirmation_bars + 5, len(data))):
                current_candle = data.iloc[i]
                
                if ob_type == OrderBlockType.BULLISH:
                    # Measure upward reaction from OB low
                    reaction = (current_candle['High'] - ob_candle['Low']) / ob_candle['Low']
                else:
                    # Measure downward reaction from OB high
                    reaction = (ob_candle['High'] - current_candle['Low']) / ob_candle['High']
                
                max_reaction = max(max_reaction, reaction)
            
            # Normalize reaction strength (0.0 to 1.0)
            # Reactions above 1% get maximum strength
            normalized_strength = min(max_reaction / 0.01, 1.0)
            
            return normalized_strength
            
        except Exception as e:
            logger.error(f"Error calculating reaction strength: {e}")
            return 0.0
    
    def _validate_order_blocks(self, potential_obs: List[OrderBlock], data: pd.DataFrame) -> List[OrderBlock]:
        """Validate potential order blocks with additional criteria"""
        valid_obs = []
        
        for ob in potential_obs:
            # Minimum strength requirement
            if ob.strength < 0.3:  # At least 30% strength
                continue
            
            # Check if order block has sufficient reaction
            if ob.reaction_strength < 0.005:  # At least 0.5% reaction
                continue
            
            # Volume validation (if available)
            if ob.volume > 0:
                # Compare with average volume
                recent_data = data.tail(20)
                avg_volume = recent_data['Volume'].mean()
                if avg_volume > 0 and ob.volume < avg_volume * 0.5:  # Below 50% of average
                    continue
            
            # Time-based validation - not too old
            latest_time = data.index[-1]
            time_diff = (latest_time - ob.timestamp).total_seconds() / 3600  # Hours
            if time_diff > 168:  # Older than 1 week
                continue
            
            valid_obs.append(ob)
        
        logger.debug(f"Validated {len(valid_obs)} order blocks from {len(potential_obs)} potential")
        return valid_obs
    
    def _update_order_block_status(self, data: pd.DataFrame):
        """Update status of existing order blocks based on current price action"""
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1]
        
        for ob in self.order_blocks:
            if ob.status == OrderBlockStatus.BROKEN:
                continue  # Skip already broken OBs
            
            # Check if order block has been tested
            if self._is_order_block_tested(ob, data):
                if ob.status == OrderBlockStatus.FRESH:
                    ob.status = OrderBlockStatus.TESTED
                    ob.test_count += 1
                
                # Check if respected after test
                if self._is_order_block_respected(ob, data):
                    ob.status = OrderBlockStatus.RESPECTED
            
            # Check if order block is broken
            if self._is_order_block_broken(ob, current_price):
                ob.status = OrderBlockStatus.BROKEN
    
    def _is_order_block_tested(self, ob: OrderBlock, data: pd.DataFrame) -> bool:
        """Check if order block has been tested by price"""
        # Get recent price data after OB creation
        ob_time = ob.timestamp
        recent_data = data[data.index > ob_time].tail(20)  # Last 20 bars after OB
        
        if recent_data.empty:
            return False
        
        for _, candle in recent_data.iterrows():
            if ob.ob_type == OrderBlockType.BULLISH:
                # Price touched or entered the OB zone
                if candle['Low'] <= ob.top and candle['High'] >= ob.bottom:
                    return True
            else:  # Bearish OB
                # Price touched or entered the OB zone  
                if candle['High'] >= ob.bottom and candle['Low'] <= ob.top:
                    return True
        
        return False
    
    def _is_order_block_respected(self, ob: OrderBlock, data: pd.DataFrame) -> bool:
        """Check if order block was respected (price reacted from it)"""
        if not self._is_order_block_tested(ob, data):
            return False
        
        # Get data after OB was tested
        ob_time = ob.timestamp
        recent_data = data[data.index > ob_time].tail(10)
        
        if recent_data.empty:
            return False
        
        # Look for reaction after touching OB
        for i, (_, candle) in enumerate(recent_data.iterrows()):
            if ob.ob_type == OrderBlockType.BULLISH:
                # Check if price bounced up from OB
                if (candle['Low'] <= ob.top and 
                    i < len(recent_data) - 1):
                    next_candle = recent_data.iloc[i + 1]
                    if next_candle['Close'] > candle['Close']:  # Bullish reaction
                        return True
            else:  # Bearish OB
                # Check if price dropped from OB
                if (candle['High'] >= ob.bottom and 
                    i < len(recent_data) - 1):
                    next_candle = recent_data.iloc[i + 1]
                    if next_candle['Close'] < candle['Close']:  # Bearish reaction
                        return True
        
        return False
    
    def _is_order_block_broken(self, ob: OrderBlock, current_price: float) -> bool:
        """Check if order block is broken"""
        if ob.ob_type == OrderBlockType.BULLISH:
            # Bullish OB broken if price closes below the low
            return current_price < ob.bottom
        else:  # Bearish OB
            # Bearish OB broken if price closes above the high
            return current_price > ob.top
    
    def _is_duplicate_order_block(self, new_ob: OrderBlock) -> bool:
        """Check if order block already exists"""
        for existing_ob in self.order_blocks:
            # Check if OBs are at similar levels and same type
            if (existing_ob.ob_type == new_ob.ob_type and
                abs(existing_ob.top - new_ob.top) < self.max_test_distance and
                abs(existing_ob.bottom - new_ob.bottom) < self.max_test_distance):
                return True
        return False
    
    def _cleanup_order_blocks(self):
        """Remove old or broken order blocks"""
        current_time = datetime.now()
        
        # Remove broken OBs older than 24 hours
        self.order_blocks = [
            ob for ob in self.order_blocks
            if not (ob.status == OrderBlockStatus.BROKEN and 
                   (current_time - ob.timestamp).total_seconds() > 86400)
        ]
        
        # Keep only recent OBs (last 7 days)
        self.order_blocks = [
            ob for ob in self.order_blocks
            if (current_time - ob.timestamp).total_seconds() < 604800
        ]
        
        # Keep only top 20 OBs by strength
        if len(self.order_blocks) > 20:
            self.order_blocks.sort(key=lambda x: x.strength, reverse=True)
            self.order_blocks = self.order_blocks[:20]
    
    def _calculate_order_block_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate order block performance metrics"""
        if not self.order_blocks:
            return {
                'total_obs': 0,
                'fresh_count': 0,
                'tested_count': 0,
                'respected_count': 0,
                'broken_count': 0,
                'respect_rate': 0.0,
                'avg_strength': 0.0,
                'bullish_count': 0,
                'bearish_count': 0
            }
        
        total_obs = len(self.order_blocks)
        fresh_count = sum(1 for ob in self.order_blocks if ob.status == OrderBlockStatus.FRESH)
        tested_count = sum(1 for ob in self.order_blocks if ob.status == OrderBlockStatus.TESTED)
        respected_count = sum(1 for ob in self.order_blocks if ob.status == OrderBlockStatus.RESPECTED)
        broken_count = sum(1 for ob in self.order_blocks if ob.status == OrderBlockStatus.BROKEN)
        
        # Calculate respect rate
        tested_or_respected = tested_count + respected_count
        respect_rate = respected_count / tested_or_respected if tested_or_respected > 0 else 0.0
        
        # Average strength
        avg_strength = sum(ob.strength for ob in self.order_blocks) / total_obs
        
        # Type counts
        bullish_count = sum(1 for ob in self.order_blocks if ob.ob_type == OrderBlockType.BULLISH)
        bearish_count = sum(1 for ob in self.order_blocks if ob.ob_type == OrderBlockType.BEARISH)
        
        return {
            'total_obs': total_obs,
            'fresh_count': fresh_count,
            'tested_count': tested_count,
            'respected_count': respected_count,
            'broken_count': broken_count,
            'respect_rate': respect_rate,
            'avg_strength': avg_strength,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }
    
    def _generate_summary(self) -> Dict:
        """Generate summary of current order block state"""
        current_obs = [ob for ob in self.order_blocks if ob.status != OrderBlockStatus.BROKEN]
        
        if not current_obs:
            return {
                'active_obs_count': 0,
                'dominant_type': None,
                'strongest_ob': None,
                'nearest_ob': None
            }
        
        # Find dominant type
        bullish_active = [ob for ob in current_obs if ob.ob_type == OrderBlockType.BULLISH]
        bearish_active = [ob for ob in current_obs if ob.ob_type == OrderBlockType.BEARISH]
        
        if len(bullish_active) > len(bearish_active):
            dominant_type = OrderBlockType.BULLISH
        elif len(bearish_active) > len(bullish_active):
            dominant_type = OrderBlockType.BEARISH
        else:
            dominant_type = None
        
        # Find strongest OB
        strongest_ob = max(current_obs, key=lambda x: x.strength) if current_obs else None
        
        # Find nearest OB (this would need current price)
        # For now, just get most recent
        nearest_ob = max(current_obs, key=lambda x: x.timestamp) if current_obs else None
        
        return {
            'active_obs_count': len(current_obs),
            'dominant_type': dominant_type,
            'strongest_ob': strongest_ob,
            'nearest_ob': nearest_ob
        }
    
    def get_order_blocks_near_price(self, price: float, distance: float = 0.001) -> List[OrderBlock]:
        """Get order blocks near specified price"""
        nearby_obs = []
        
        for ob in self.order_blocks:
            if ob.status == OrderBlockStatus.BROKEN:
                continue
            
            # Check if price is within distance of OB
            if (price >= ob.bottom - distance and 
                price <= ob.top + distance):
                nearby_obs.append(ob)
        
        # Sort by distance to price
        nearby_obs.sort(key=lambda x: min(abs(price - x.top), abs(price - x.bottom)))
        
        return nearby_obs
    
    def get_confluence_score(self, price: float) -> float:
        """Calculate confluence score at price level based on nearby OBs"""
        nearby_obs = self.get_order_blocks_near_price(price, distance=0.0005)
        
        if not nearby_obs:
            return 0.0
        
        # Calculate weighted confluence based on OB strength and proximity
        confluence_score = 0.0
        
        for ob in nearby_obs:
            # Distance factor (closer = higher score)
            distance = min(abs(price - ob.top), abs(price - ob.bottom))
            distance_factor = max(0, 1 - (distance / 0.0005))
            
            # Status factor
            status_multiplier = {
                OrderBlockStatus.FRESH: 1.0,
                OrderBlockStatus.TESTED: 0.8,
                OrderBlockStatus.RESPECTED: 1.2,
                OrderBlockStatus.BROKEN: 0.0
            }
            
            ob_contribution = (ob.strength * distance_factor * 
                             status_multiplier.get(ob.status, 0.5))
            confluence_score += ob_contribution
        
        return min(confluence_score, 1.0)
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'timestamp': datetime.now(),
            'order_blocks': [],
            'fresh_obs': [],
            'tested_obs': [],
            'respected_obs': [],
            'new_obs': [],
            'metrics': self._calculate_order_block_metrics(pd.DataFrame()),
            'summary': {
                'active_obs_count': 0,
                'dominant_type': None,
                'strongest_ob': None,
                'nearest_ob': None
            }
        }

# Export main classes
__all__ = [
    'OrderBlockAnalyzer',
    'OrderBlock',
    'OrderBlockType',
    'OrderBlockStatus'
]