"""
Volume Profile Analyzer - Smart Money Concepts Component
Analyzes volume distribution across price levels to identify key support/resistance zones
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VolumeNodeType(Enum):
    """Volume node classification"""
    HIGH_VOLUME_NODE = "high_volume_node"    # Above threshold concentration
    LOW_VOLUME_NODE = "low_volume_node"      # Below threshold concentration
    POINT_OF_CONTROL = "point_of_control"    # Highest volume price level
    VALUE_AREA_HIGH = "value_area_high"      # Upper value area boundary
    VALUE_AREA_LOW = "value_area_low"        # Lower value area boundary

class VolumeProfileType(Enum):
    """Volume profile analysis types"""
    SESSION_PROFILE = "session_profile"      # Single session analysis
    COMPOSITE_PROFILE = "composite_profile"  # Multi-session analysis
    DEVELOPING_PROFILE = "developing_profile" # Real-time developing profile

@dataclass
class VolumeNode:
    """Individual volume node at specific price level"""
    price_level: float
    volume: int
    percentage: float  # Percentage of total volume
    node_type: VolumeNodeType
    bar_count: int  # Number of bars at this price level
    timestamp_first: datetime
    timestamp_last: datetime
    
    # SMC context
    is_institutional_level: bool = False
    confluence_score: float = 0.0
    touched_count: int = 0  # How many times price revisited this level

@dataclass
class ValueArea:
    """Value Area (70% of volume concentration)"""
    value_area_high: float  # VAH - Upper boundary
    value_area_low: float   # VAL - Lower boundary
    value_area_volume: int  # Total volume in value area
    value_area_percentage: float  # Should be ~70%
    
    # Value area characteristics
    width: float  # VAH - VAL
    midpoint: float  # (VAH + VAL) / 2
    skew: float  # Positive = upper heavy, Negative = lower heavy

@dataclass
class VolumeProfileAnalysis:
    """Complete volume profile analysis result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    profile_type: VolumeProfileType
    
    # Core components
    point_of_control: float  # POC - Highest volume price level
    poc_volume: int
    poc_percentage: float
    
    # Value area analysis
    value_area: ValueArea
    
    # Volume nodes
    volume_nodes: List[VolumeNode]
    high_volume_nodes: List[VolumeNode]
    low_volume_nodes: List[VolumeNode]
    
    # Profile characteristics
    total_volume: int
    price_range: Tuple[float, float]  # (lowest, highest)
    volume_distribution: Dict[str, float]  # Distribution statistics
    
    # SMC integration
    institutional_levels: List[float]
    support_levels: List[float]
    resistance_levels: List[float]
    volume_imbalances: List[Tuple[float, float]]  # (price_start, price_end)
    
    # Trading context
    current_price_context: str  # Above/Below/At POC/Value Area
    trading_bias: str  # Based on volume profile
    confidence: float  # Analysis confidence (0.0-1.0)

class VolumeProfileAnalyzer:
    """
    Volume Profile analyzer for SMC integration
    Analyzes volume distribution to identify institutional activity zones
    """
    
    def __init__(self,
                 # Volume profile parameters
                 price_levels: int = 50,  # Number of price levels for analysis
                 value_area_percentage: float = 0.70,  # 70% value area standard
                 
                 # Node classification thresholds
                 high_volume_threshold: float = 2.0,  # 2x average volume
                 low_volume_threshold: float = 0.5,   # 0.5x average volume
                 
                 # SMC integration parameters
                 institutional_threshold: float = 3.0,  # 3x average for institutional
                 confluence_enabled: bool = True,
                 
                 # Analysis parameters
                 min_bars_required: int = 20,
                 volume_smoothing: bool = True,
                 normalize_volume: bool = True):
        
        self.price_levels = max(price_levels, 10)  # Minimum 10 levels
        self.value_area_percentage = np.clip(value_area_percentage, 0.5, 0.9)
        
        self.high_volume_threshold = high_volume_threshold
        self.low_volume_threshold = low_volume_threshold
        self.institutional_threshold = institutional_threshold
        
        self.confluence_enabled = confluence_enabled
        self.min_bars_required = min_bars_required
        self.volume_smoothing = volume_smoothing
        self.normalize_volume = normalize_volume
        
        # Analysis cache
        self._profile_cache = {}
        
    def analyze_volume_profile(self, data: pd.DataFrame, 
                             profile_type: VolumeProfileType = VolumeProfileType.SESSION_PROFILE) -> Optional[VolumeProfileAnalysis]:
        """
        Main volume profile analysis function
        
        Args:
            data: OHLCV DataFrame
            profile_type: Type of volume profile analysis
            
        Returns:
            VolumeProfileAnalysis or None if analysis fails
        """
        if data.empty or len(data) < self.min_bars_required:
            logger.warning(f"Insufficient data for volume profile: {len(data)} bars")
            return None
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            logger.error("Missing required OHLCV columns for volume profile")
            return None
        
        try:
            # Step 1: Create price-volume distribution
            price_volume_dist = self._create_price_volume_distribution(data)
            
            if not price_volume_dist:
                logger.error("Failed to create price-volume distribution")
                return None
            
            # Step 2: Identify Point of Control (POC)
            poc_price, poc_volume = self._find_point_of_control(price_volume_dist)
            
            # Step 3: Calculate Value Area
            value_area = self._calculate_value_area(price_volume_dist, poc_price)
            
            # Step 4: Classify volume nodes
            volume_nodes = self._classify_volume_nodes(price_volume_dist, data)
            
            # Step 5: SMC integration analysis
            institutional_levels = self._identify_institutional_levels(volume_nodes)
            support_resistance = self._identify_support_resistance_levels(volume_nodes, data)
            volume_imbalances = self._identify_volume_imbalances(price_volume_dist)
            
            # Step 6: Trading context analysis
            current_price = data['Close'].iloc[-1]
            price_context = self._analyze_price_context(current_price, poc_price, value_area)
            trading_bias = self._determine_trading_bias(current_price, value_area, volume_nodes)
            confidence = self._calculate_analysis_confidence(price_volume_dist, volume_nodes)
            
            # Step 7: Create comprehensive result
            total_volume = sum(vol for vol in price_volume_dist.values())
            price_range = (min(price_volume_dist.keys()), max(price_volume_dist.keys()))
            
            # Volume distribution statistics
            volume_distribution = self._calculate_volume_statistics(price_volume_dist, value_area)
            
            # Filter nodes by type
            high_volume_nodes = [node for node in volume_nodes if node.node_type == VolumeNodeType.HIGH_VOLUME_NODE]
            low_volume_nodes = [node for node in volume_nodes if node.node_type == VolumeNodeType.LOW_VOLUME_NODE]
            
            analysis = VolumeProfileAnalysis(
                timestamp=data.index[-1],
                symbol=getattr(data, 'symbol', 'UNKNOWN'),
                timeframe=getattr(data, 'timeframe', 'UNKNOWN'),
                profile_type=profile_type,
                
                # Core components
                point_of_control=poc_price,
                poc_volume=poc_volume,
                poc_percentage=(poc_volume / total_volume * 100) if total_volume > 0 else 0.0,
                
                # Value area
                value_area=value_area,
                
                # Volume nodes
                volume_nodes=volume_nodes,
                high_volume_nodes=high_volume_nodes,
                low_volume_nodes=low_volume_nodes,
                
                # Profile characteristics
                total_volume=total_volume,
                price_range=price_range,
                volume_distribution=volume_distribution,
                
                # SMC integration
                institutional_levels=institutional_levels,
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance'],
                volume_imbalances=volume_imbalances,
                
                # Trading context
                current_price_context=price_context,
                trading_bias=trading_bias,
                confidence=confidence
            )
            
            logger.debug(f"Volume profile analysis completed - POC: {poc_price:.5f}, VAH: {value_area.value_area_high:.5f}, VAL: {value_area.value_area_low:.5f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Volume profile analysis failed: {e}")
            return None
    
    def _create_price_volume_distribution(self, data: pd.DataFrame) -> Dict[float, int]:
        """Create price-volume distribution across specified price levels"""
        try:
            # Get price range
            high_price = data['High'].max()
            low_price = data['Low'].min()
            price_range = high_price - low_price
            
            if price_range <= 0:
                logger.error("Invalid price range for volume profile")
                return {}
            
            # Create price levels
            price_step = price_range / self.price_levels
            price_levels = [low_price + i * price_step for i in range(self.price_levels + 1)]
            
            # Initialize volume distribution
            volume_distribution = {level: 0 for level in price_levels}
            
            # Distribute volume across price levels
            for idx, row in data.iterrows():
                bar_volume = row['Volume']
                bar_high = row['High']
                bar_low = row['Low']
                
                # For each price level, check if it's within this bar's range
                for price_level in price_levels:
                    if bar_low <= price_level <= bar_high:
                        # Distribute volume proportionally
                        if bar_high > bar_low:
                            # Weight volume based on proximity to close price
                            close_distance = abs(price_level - row['Close'])
                            range_distance = bar_high - bar_low
                            weight = max(0.1, 1.0 - (close_distance / range_distance))
                            volume_distribution[price_level] += int(bar_volume * weight / self.price_levels)
                        else:
                            # If high == low, assign all volume to that level
                            volume_distribution[price_level] += bar_volume
            
            # Remove zero volume levels and apply smoothing if enabled
            volume_distribution = {k: v for k, v in volume_distribution.items() if v > 0}
            
            if self.volume_smoothing and len(volume_distribution) > 3:
                volume_distribution = self._smooth_volume_distribution(volume_distribution)
            
            return volume_distribution
            
        except Exception as e:
            logger.error(f"Error creating price-volume distribution: {e}")
            return {}
    
    def _smooth_volume_distribution(self, volume_dist: Dict[float, int]) -> Dict[float, int]:
        """Apply smoothing to volume distribution to reduce noise"""
        try:
            prices = sorted(volume_dist.keys())
            volumes = [volume_dist[price] for price in prices]
            
            # Simple moving average smoothing
            window = min(3, len(volumes) // 5)  # Adaptive window size
            
            if window >= 2:
                smoothed_volumes = []
                for i in range(len(volumes)):
                    start_idx = max(0, i - window // 2)
                    end_idx = min(len(volumes), i + window // 2 + 1)
                    avg_volume = int(np.mean(volumes[start_idx:end_idx]))
                    smoothed_volumes.append(avg_volume)
                
                return dict(zip(prices, smoothed_volumes))
            
            return volume_dist
            
        except Exception as e:
            logger.error(f"Error smoothing volume distribution: {e}")
            return volume_dist
    
    def _find_point_of_control(self, volume_dist: Dict[float, int]) -> Tuple[float, int]:
        """Find Point of Control (highest volume price level)"""
        if not volume_dist:
            return 0.0, 0
        
        poc_price = max(volume_dist.keys(), key=lambda k: volume_dist[k])
        poc_volume = volume_dist[poc_price]
        
        return poc_price, poc_volume
    
    def _calculate_value_area(self, volume_dist: Dict[float, int], poc_price: float) -> ValueArea:
        """Calculate Value Area (70% volume concentration around POC)"""
        try:
            total_volume = sum(volume_dist.values())
            target_volume = total_volume * self.value_area_percentage
            
            # Sort prices by volume (descending)
            sorted_by_volume = sorted(volume_dist.items(), key=lambda x: x[1], reverse=True)
            
            # Start with POC and expand to include neighboring high-volume areas
            value_area_volume = 0
            included_prices = []
            
            for price, volume in sorted_by_volume:
                value_area_volume += volume
                included_prices.append(price)
                
                if value_area_volume >= target_volume:
                    break
            
            # Determine value area boundaries
            if included_prices:
                vah = max(included_prices)  # Value Area High
                val = min(included_prices)  # Value Area Low
            else:
                vah = val = poc_price
            
            # Calculate value area statistics
            width = vah - val
            midpoint = (vah + val) / 2
            
            # Calculate skew (distribution within value area)
            poc_position = (poc_price - val) / width if width > 0 else 0.5
            skew = (poc_position - 0.5) * 2  # -1 to 1 scale
            
            return ValueArea(
                value_area_high=vah,
                value_area_low=val,
                value_area_volume=value_area_volume,
                value_area_percentage=(value_area_volume / total_volume * 100) if total_volume > 0 else 0.0,
                width=width,
                midpoint=midpoint,
                skew=skew
            )
            
        except Exception as e:
            logger.error(f"Error calculating value area: {e}")
            return ValueArea(
                value_area_high=poc_price,
                value_area_low=poc_price,
                value_area_volume=0,
                value_area_percentage=0.0,
                width=0.0,
                midpoint=poc_price,
                skew=0.0
            )
    
    def _classify_volume_nodes(self, volume_dist: Dict[float, int], data: pd.DataFrame) -> List[VolumeNode]:
        """Classify volume nodes based on volume concentration"""
        try:
            if not volume_dist:
                return []
            
            total_volume = sum(volume_dist.values())
            average_volume = total_volume / len(volume_dist)
            
            volume_nodes = []
            
            for price_level, volume in volume_dist.items():
                percentage = (volume / total_volume * 100) if total_volume > 0 else 0.0
                
                # Classify node type
                if volume >= average_volume * self.high_volume_threshold:
                    node_type = VolumeNodeType.HIGH_VOLUME_NODE
                elif volume <= average_volume * self.low_volume_threshold:
                    node_type = VolumeNodeType.LOW_VOLUME_NODE
                else:
                    continue  # Skip medium volume nodes
                
                # Find first and last occurrence of this price level
                price_touches = data[(data['Low'] <= price_level) & (data['High'] >= price_level)]
                
                if len(price_touches) > 0:
                    timestamp_first = price_touches.index[0]
                    timestamp_last = price_touches.index[-1]
                    bar_count = len(price_touches)
                    touched_count = len(price_touches)
                else:
                    timestamp_first = data.index[0]
                    timestamp_last = data.index[-1]
                    bar_count = 1
                    touched_count = 1
                
                # Determine if institutional level
                is_institutional = volume >= average_volume * self.institutional_threshold
                
                # Calculate confluence score if enabled
                confluence_score = 0.0
                if self.confluence_enabled:
                    confluence_score = self._calculate_node_confluence(price_level, data)
                
                node = VolumeNode(
                    price_level=price_level,
                    volume=volume,
                    percentage=percentage,
                    node_type=node_type,
                    bar_count=bar_count,
                    timestamp_first=timestamp_first,
                    timestamp_last=timestamp_last,
                    is_institutional_level=is_institutional,
                    confluence_score=confluence_score,
                    touched_count=touched_count
                )
                
                volume_nodes.append(node)
            
            # Sort nodes by volume (descending)
            volume_nodes.sort(key=lambda x: x.volume, reverse=True)
            
            return volume_nodes
            
        except Exception as e:
            logger.error(f"Error classifying volume nodes: {e}")
            return []
    
    def _calculate_node_confluence(self, price_level: float, data: pd.DataFrame) -> float:
        """Calculate confluence score for a volume node"""
        try:
            confluence_score = 0.0
            current_price = data['Close'].iloc[-1]
            
            # Distance factor (closer to current price = higher score)
            price_distance = abs(price_level - current_price) / current_price
            distance_score = max(0, 1.0 - price_distance * 100)  # Reduce score for distant levels
            confluence_score += distance_score * 0.3
            
            # Recency factor (more recent touches = higher score)
            recent_data = data.tail(50)  # Last 50 bars
            recent_touches = recent_data[(recent_data['Low'] <= price_level) & (recent_data['High'] >= price_level)]
            recency_score = min(len(recent_touches) / 10, 1.0)  # Max score at 10+ touches
            confluence_score += recency_score * 0.4
            
            # Support/Resistance factor
            support_resistance_score = self._calculate_support_resistance_strength(price_level, data)
            confluence_score += support_resistance_score * 0.3
            
            return min(confluence_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confluence score: {e}")
            return 0.0
    
    def _calculate_support_resistance_strength(self, price_level: float, data: pd.DataFrame) -> float:
        """Calculate support/resistance strength for a price level"""
        try:
            touches = data[(data['Low'] <= price_level * 1.001) & (data['High'] >= price_level * 0.999)]
            
            if len(touches) < 2:
                return 0.0
            
            # Count bounces vs breaks
            bounces = 0
            breaks = 0
            
            for idx, touch in touches.iterrows():
                next_idx = data.index.get_loc(idx) + 1
                if next_idx < len(data):
                    next_bar = data.iloc[next_idx]
                    
                    if touch['Low'] <= price_level <= touch['High']:
                        if next_bar['Close'] > price_level:
                            bounces += 1
                        else:
                            breaks += 1
            
            total_interactions = bounces + breaks
            if total_interactions == 0:
                return 0.0
            
            bounce_rate = bounces / total_interactions
            return bounce_rate
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance strength: {e}")
            return 0.0
    
    def _identify_institutional_levels(self, volume_nodes: List[VolumeNode]) -> List[float]:
        """Identify institutional activity levels from volume nodes"""
        institutional_levels = []
        
        for node in volume_nodes:
            if node.is_institutional_level and node.confluence_score > 0.5:
                institutional_levels.append(node.price_level)
        
        # Sort by volume (most institutional first)
        institutional_levels.sort(key=lambda price: next(
            node.volume for node in volume_nodes if node.price_level == price
        ), reverse=True)
        
        return institutional_levels[:10]  # Top 10 institutional levels
    
    def _identify_support_resistance_levels(self, volume_nodes: List[VolumeNode], data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify support and resistance levels from volume profile"""
        current_price = data['Close'].iloc[-1]
        
        support_levels = []
        resistance_levels = []
        
        for node in volume_nodes:
            if node.node_type == VolumeNodeType.HIGH_VOLUME_NODE:
                if node.price_level < current_price:
                    support_levels.append(node.price_level)
                else:
                    resistance_levels.append(node.price_level)
        
        # Sort support levels (highest first), resistance levels (lowest first)
        support_levels.sort(reverse=True)
        resistance_levels.sort()
        
        return {
            'support': support_levels[:5],  # Top 5 support levels
            'resistance': resistance_levels[:5]  # Top 5 resistance levels
        }
    
    def _identify_volume_imbalances(self, volume_dist: Dict[float, int]) -> List[Tuple[float, float]]:
        """Identify volume imbalance areas (gaps in volume profile)"""
        try:
            if len(volume_dist) < 3:
                return []
            
            sorted_prices = sorted(volume_dist.keys())
            average_volume = sum(volume_dist.values()) / len(volume_dist)
            min_volume_threshold = average_volume * 0.2  # 20% of average
            
            imbalances = []
            imbalance_start = None
            
            for i, price in enumerate(sorted_prices):
                volume = volume_dist[price]
                
                if volume < min_volume_threshold:
                    if imbalance_start is None:
                        imbalance_start = price
                else:
                    if imbalance_start is not None:
                        imbalances.append((imbalance_start, price))
                        imbalance_start = None
            
            # Close any open imbalance at the end
            if imbalance_start is not None:
                imbalances.append((imbalance_start, sorted_prices[-1]))
            
            # Filter out small imbalances
            significant_imbalances = []
            for start, end in imbalances:
                gap_size = abs(end - start)
                avg_price = (start + end) / 2
                gap_percentage = (gap_size / avg_price) * 100
                
                if gap_percentage > 0.1:  # At least 0.1% price gap
                    significant_imbalances.append((start, end))
            
            return significant_imbalances
            
        except Exception as e:
            logger.error(f"Error identifying volume imbalances: {e}")
            return []
    
    def _analyze_price_context(self, current_price: float, poc_price: float, value_area: ValueArea) -> str:
        """Analyze current price context relative to volume profile"""
        try:
            # Price relative to POC
            if abs(current_price - poc_price) / poc_price < 0.001:  # Within 0.1%
                poc_context = "AT_POC"
            elif current_price > poc_price:
                poc_context = "ABOVE_POC"
            else:
                poc_context = "BELOW_POC"
            
            # Price relative to Value Area
            if value_area.value_area_low <= current_price <= value_area.value_area_high:
                va_context = "INSIDE_VALUE_AREA"
            elif current_price > value_area.value_area_high:
                va_context = "ABOVE_VALUE_AREA"
            else:
                va_context = "BELOW_VALUE_AREA"
            
            return f"{va_context}_{poc_context}"
            
        except Exception as e:
            logger.error(f"Error analyzing price context: {e}")
            return "UNKNOWN_CONTEXT"
    
    def _determine_trading_bias(self, current_price: float, value_area: ValueArea, volume_nodes: List[VolumeNode]) -> str:
        """Determine trading bias based on volume profile analysis"""
        try:
            # Price position bias
            if current_price > value_area.value_area_high:
                position_bias = "BULLISH"
            elif current_price < value_area.value_area_low:
                position_bias = "BEARISH"
            else:
                position_bias = "NEUTRAL"
            
            # Volume distribution bias
            upper_half_volume = sum(node.volume for node in volume_nodes 
                                  if node.price_level > value_area.midpoint)
            lower_half_volume = sum(node.volume for node in volume_nodes 
                                  if node.price_level < value_area.midpoint)
            
            total_volume = upper_half_volume + lower_half_volume
            
            if total_volume > 0:
                upper_percentage = upper_half_volume / total_volume
                
                if upper_percentage > 0.6:
                    volume_bias = "BULLISH"
                elif upper_percentage < 0.4:
                    volume_bias = "BEARISH"
                else:
                    volume_bias = "NEUTRAL"
            else:
                volume_bias = "NEUTRAL"
            
            # Combine biases
            if position_bias == volume_bias and position_bias != "NEUTRAL":
                return f"STRONG_{position_bias}"
            elif position_bias != "NEUTRAL":
                return f"MODERATE_{position_bias}"
            elif volume_bias != "NEUTRAL":
                return f"WEAK_{volume_bias}"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Error determining trading bias: {e}")
            return "NEUTRAL"
    
    def _calculate_analysis_confidence(self, volume_dist: Dict[float, int], volume_nodes: List[VolumeNode]) -> float:
        """Calculate confidence in volume profile analysis"""
        try:
            confidence = 0.0
            
            # Data quality factor
            total_volume = sum(volume_dist.values())
            price_levels_count = len(volume_dist)
            
            if total_volume > 10000 and price_levels_count >= 10:
                data_quality = 0.9
            elif total_volume > 1000 and price_levels_count >= 5:
                data_quality = 0.7
            else:
                data_quality = 0.5
            
            confidence += data_quality * 0.4
            
            # Node quality factor
            high_volume_nodes = [node for node in volume_nodes if node.node_type == VolumeNodeType.HIGH_VOLUME_NODE]
            institutional_nodes = [node for node in volume_nodes if node.is_institutional_level]
            
            if len(institutional_nodes) >= 3:
                node_quality = 0.9
            elif len(high_volume_nodes) >= 5:
                node_quality = 0.7
            elif len(high_volume_nodes) >= 2:
                node_quality = 0.5
            else:
                node_quality = 0.3
            
            confidence += node_quality * 0.4
            
            # Distribution quality factor
            volume_values = list(volume_dist.values())
            if len(volume_values) > 1:
                volume_std = np.std(volume_values)
                volume_mean = np.mean(volume_values)
                coefficient_of_variation = volume_std / volume_mean if volume_mean > 0 else 0
                
                # Good distribution should have some variation but not too much
                if 0.3 <= coefficient_of_variation <= 1.5:
                    distribution_quality = 0.8
                elif 0.1 <= coefficient_of_variation <= 2.0:
                    distribution_quality = 0.6
                else:
                    distribution_quality = 0.4
            else:
                distribution_quality = 0.2
            
            confidence += distribution_quality * 0.2
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {e}")
            return 0.5
    
    def _calculate_volume_statistics(self, volume_dist: Dict[float, int], value_area: ValueArea) -> Dict[str, float]:
        """Calculate volume distribution statistics"""
        try:
            volumes = list(volume_dist.values())
            total_volume = sum(volumes)
            
            if not volumes:
                return {}
            
            stats = {
                'total_volume': total_volume,
                'mean_volume': np.mean(volumes),
                'median_volume': np.median(volumes),
                'std_volume': np.std(volumes),
                'max_volume': max(volumes),
                'min_volume': min(volumes),
                'volume_range': max(volumes) - min(volumes),
                'coefficient_of_variation': np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0,
                'value_area_percentage': value_area.value_area_percentage,
                'value_area_width_percentage': (value_area.width / ((max(volume_dist.keys()) - min(volume_dist.keys())) or 1)) * 100,
                'skewness': value_area.skew
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating volume statistics: {e}")
            return {}
    
    def get_volume_profile_summary(self, analysis: VolumeProfileAnalysis) -> Dict:
        """Get summary of volume profile analysis for quick reference"""
        if not analysis:
            return {}
        
        return {
            'poc_price': analysis.point_of_control,
            'poc_volume_percentage': analysis.poc_percentage,
            'value_area_range': (analysis.value_area.value_area_low, analysis.value_area.value_area_high),
            'value_area_width': analysis.value_area.width,
            'institutional_levels_count': len(analysis.institutional_levels),
            'high_volume_nodes_count': len(analysis.high_volume_nodes),
            'support_levels_count': len(analysis.support_levels),
            'resistance_levels_count': len(analysis.resistance_levels),
            'volume_imbalances_count': len(analysis.volume_imbalances),
            'trading_bias': analysis.trading_bias,
            'price_context': analysis.current_price_context,
            'confidence': analysis.confidence,
            'total_volume': analysis.total_volume
        }
    
    def get_trading_levels(self, analysis: VolumeProfileAnalysis, current_price: float) -> Dict[str, List[float]]:
        """Get key trading levels from volume profile analysis"""
        if not analysis:
            return {'support': [], 'resistance': [], 'targets': []}
        
        # Get nearest levels to current price
        all_levels = []
        
        # Add POC
        all_levels.append(analysis.point_of_control)
        
        # Add value area boundaries
        all_levels.extend([analysis.value_area.value_area_high, analysis.value_area.value_area_low])
        
        # Add institutional levels
        all_levels.extend(analysis.institutional_levels)
        
        # Add high volume nodes
        all_levels.extend([node.price_level for node in analysis.high_volume_nodes])
        
        # Remove duplicates and sort
        unique_levels = sorted(set(all_levels))
        
        # Categorize relative to current price
        support_levels = [level for level in unique_levels if level < current_price]
        resistance_levels = [level for level in unique_levels if level > current_price]
        
        # Get nearest levels
        nearest_support = sorted(support_levels, reverse=True)[:5]  # 5 nearest support
        nearest_resistance = sorted(resistance_levels)[:5]  # 5 nearest resistance
        
        # Target levels (further resistance/support)
        target_levels = []
        if len(resistance_levels) > 5:
            target_levels.extend(resistance_levels[5:8])
        if len(support_levels) > 5:
            target_levels.extend(support_levels[-8:-5])
        
        return {
            'support': nearest_support,
            'resistance': nearest_resistance,
            'targets': sorted(target_levels)
        }
    
    def analyze_volume_confluence(self, analysis: VolumeProfileAnalysis, price_level: float, tolerance: float = 0.001) -> Dict:
        """Analyze volume confluence at a specific price level"""
        if not analysis:
            return {'confluence_score': 0.0, 'factors': []}
        
        confluence_factors = []
        confluence_score = 0.0
        
        # Check proximity to POC
        poc_distance = abs(price_level - analysis.point_of_control) / analysis.point_of_control
        if poc_distance <= tolerance:
            confluence_factors.append('POC')
            confluence_score += 0.3
        
        # Check proximity to Value Area boundaries
        vah_distance = abs(price_level - analysis.value_area.value_area_high) / analysis.value_area.value_area_high
        val_distance = abs(price_level - analysis.value_area.value_area_low) / analysis.value_area.value_area_low
        
        if vah_distance <= tolerance:
            confluence_factors.append('VAH')
            confluence_score += 0.25
        
        if val_distance <= tolerance:
            confluence_factors.append('VAL')
            confluence_score += 0.25
        
        # Check proximity to high volume nodes
        for node in analysis.high_volume_nodes:
            node_distance = abs(price_level - node.price_level) / node.price_level
            if node_distance <= tolerance:
                confluence_factors.append(f'HIGH_VOLUME_NODE')
                confluence_score += 0.1
        
        # Check proximity to institutional levels
        for inst_level in analysis.institutional_levels:
            inst_distance = abs(price_level - inst_level) / inst_level
            if inst_distance <= tolerance:
                confluence_factors.append('INSTITUTIONAL_LEVEL')
                confluence_score += 0.15
        
        return {
            'confluence_score': min(confluence_score, 1.0),
            'factors': confluence_factors,
            'factor_count': len(confluence_factors)
        }

# Utility functions for integration with other SMC components
def get_volume_support_resistance(analysis: VolumeProfileAnalysis, current_price: float) -> Dict[str, float]:
    """Get nearest volume-based support and resistance levels"""
    if not analysis:
        return {'nearest_support': 0.0, 'nearest_resistance': 0.0}
    
    # Find nearest support (highest price below current)
    support_candidates = [
        analysis.point_of_control,
        analysis.value_area.value_area_low,
        *[node.price_level for node in analysis.high_volume_nodes if node.price_level < current_price]
    ]
    
    valid_support = [level for level in support_candidates if level < current_price]
    nearest_support = max(valid_support) if valid_support else 0.0
    
    # Find nearest resistance (lowest price above current)
    resistance_candidates = [
        analysis.point_of_control,
        analysis.value_area.value_area_high,
        *[node.price_level for node in analysis.high_volume_nodes if node.price_level > current_price]
    ]
    
    valid_resistance = [level for level in resistance_candidates if level > current_price]
    nearest_resistance = min(valid_resistance) if valid_resistance else 0.0
    
    return {
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance
    }

def calculate_volume_profile_bias(analysis: VolumeProfileAnalysis) -> Dict[str, Union[str, float]]:
    """Calculate volume profile bias for integration with BIAS analyzer"""
    if not analysis:
        return {'direction': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
    
    # Convert trading bias to standardized format
    bias_mapping = {
        'STRONG_BULLISH': {'direction': 'BULLISH', 'strength': 0.8},
        'MODERATE_BULLISH': {'direction': 'BULLISH', 'strength': 0.6},
        'WEAK_BULLISH': {'direction': 'BULLISH', 'strength': 0.4},
        'STRONG_BEARISH': {'direction': 'BEARISH', 'strength': 0.8},
        'MODERATE_BEARISH': {'direction': 'BEARISH', 'strength': 0.6},
        'WEAK_BEARISH': {'direction': 'BEARISH', 'strength': 0.4},
        'NEUTRAL': {'direction': 'NEUTRAL', 'strength': 0.0}
    }
    
    bias_info = bias_mapping.get(analysis.trading_bias, {'direction': 'NEUTRAL', 'strength': 0.0})
    
    return {
        'direction': bias_info['direction'],
        'strength': bias_info['strength'],
        'confidence': analysis.confidence,
        'poc_price': analysis.point_of_control,
        'value_area_range': (analysis.value_area.value_area_low, analysis.value_area.value_area_high),
        'institutional_levels': analysis.institutional_levels[:3]  # Top 3 institutional levels
    }

# Export main classes and functions
__all__ = [
    'VolumeProfileAnalyzer',
    'VolumeProfileAnalysis',
    'VolumeNode',
    'ValueArea',
    'VolumeNodeType',
    'VolumeProfileType',
    'get_volume_support_resistance',
    'calculate_volume_profile_bias'
]