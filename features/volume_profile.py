"""
Volume Profile Analysis for SMC Trading
Analyzes volume distribution at different price levels to identify key zones
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VolumeProfileType(Enum):
    """Volume profile analysis types"""
    SESSION = "session"
    FIXED_RANGE = "fixed_range"
    VISIBLE_RANGE = "visible_range"
    ANCHORED = "anchored"

class POCType(Enum):
    """Point of Control types"""
    SINGLE = "single"
    MULTIPLE = "multiple"
    DEVELOPING = "developing"

@dataclass
class VolumeNode:
    """Individual volume node at a price level"""
    price: float
    volume: int
    percentage: float  # Percentage of total volume
    bar_count: int     # Number of bars that touched this price
    first_time: datetime
    last_time: datetime
    
    @property
    def volume_per_bar(self) -> float:
        """Average volume per bar at this price level"""
        return self.volume / self.bar_count if self.bar_count > 0 else 0

@dataclass
class ValueArea:
    """Value Area analysis (70% of volume)"""
    value_area_high: float
    value_area_low: float
    value_area_volume: int
    value_area_percentage: float
    poc_price: float
    poc_volume: int
    
    @property
    def value_area_range(self) -> float:
        """Range of value area"""
        return self.value_area_high - self.value_area_low
    
    @property
    def poc_position(self) -> str:
        """Position of POC within value area"""
        va_range = self.value_area_range
        if va_range == 0:
            return "center"
        
        poc_position = (self.poc_price - self.value_area_low) / va_range
        
        if poc_position < 0.3:
            return "lower"
        elif poc_position > 0.7:
            return "upper"
        else:
            return "center"

@dataclass
class VolumeProfile:
    """Complete volume profile analysis"""
    timestamp: datetime
    profile_type: VolumeProfileType
    start_time: datetime
    end_time: datetime
    
    # Core data
    volume_nodes: List[VolumeNode]
    total_volume: int
    total_bars: int
    
    # Key levels
    poc: VolumeNode  # Point of Control
    value_area: ValueArea
    
    # Price levels
    high_volume_nodes: List[VolumeNode]  # Top 20% by volume
    low_volume_nodes: List[VolumeNode]   # Bottom 20% by volume
    
    # Market structure
    volume_imbalances: List[Dict]  # Areas of low volume
    acceptance_levels: List[float]  # Levels with sustained volume
    rejection_levels: List[float]   # Levels with quick rejection
    
    @property
    def price_range(self) -> float:
        """Total price range of the profile"""
        if not self.volume_nodes:
            return 0.0
        prices = [node.price for node in self.volume_nodes]
        return max(prices) - min(prices)
    
    @property
    def volume_distribution_balance(self) -> str:
        """Balance of volume distribution"""
        poc_position = self.value_area.poc_position
        
        if poc_position == "upper":
            return "bearish_distribution"  # Volume concentrated at top
        elif poc_position == "lower":
            return "bullish_distribution"  # Volume concentrated at bottom
        else:
            return "balanced_distribution"

class VolumeProfileAnalyzer:
    """
    Analyzes volume profiles for institutional trading insights
    """
    
    def __init__(self,
                 price_resolution: int = 50,        # Number of price levels
                 min_volume_threshold: float = 0.01, # Minimum 1% volume for node
                 value_area_percentage: float = 0.70, # 70% for value area
                 high_volume_threshold: float = 0.80,  # Top 20% volume nodes
                 low_volume_threshold: float = 0.20,   # Bottom 20% volume nodes
                 imbalance_threshold: float = 0.30):   # 30% below average for imbalance
        
        self.price_resolution = price_resolution
        self.min_volume_threshold = min_volume_threshold
        self.value_area_percentage = value_area_percentage
        self.high_volume_threshold = high_volume_threshold
        self.low_volume_threshold = low_volume_threshold
        self.imbalance_threshold = imbalance_threshold
        
        # Storage for profiles
        self.session_profiles = []
        self.anchored_profiles = []
    
    def analyze_volume_profile(self, data: pd.DataFrame, 
                             profile_type: VolumeProfileType = VolumeProfileType.SESSION,
                             anchor_time: Optional[datetime] = None) -> Optional[VolumeProfile]:
        """
        Main function to analyze volume profile
        
        Args:
            data: OHLCV DataFrame with datetime index
            profile_type: Type of volume profile analysis
            anchor_time: Anchor point for anchored profiles
            
        Returns:
            VolumeProfile analysis or None if failed
        """
        if data.empty or len(data) < 10:
            logger.warning("Insufficient data for volume profile analysis")
            return None
        
        try:
            # Determine analysis period
            if profile_type == VolumeProfileType.ANCHORED and anchor_time:
                analysis_data = data[data.index >= anchor_time]
                start_time = anchor_time
            else:
                analysis_data = data
                start_time = data.index[0]
            
            if analysis_data.empty:
                return None
            
            end_time = analysis_data.index[-1]
            
            # Step 1: Create price bins
            price_bins = self._create_price_bins(analysis_data)
            
            # Step 2: Distribute volume to price levels
            volume_nodes = self._calculate_volume_distribution(analysis_data, price_bins)
            
            if not volume_nodes:
                return None
            
            # Step 3: Find Point of Control (POC)
            poc = max(volume_nodes, key=lambda x: x.volume)
            
            # Step 4: Calculate Value Area
            value_area = self._calculate_value_area(volume_nodes)
            
            # Step 5: Identify key volume levels
            high_volume_nodes = self._identify_high_volume_nodes(volume_nodes)
            low_volume_nodes = self._identify_low_volume_nodes(volume_nodes)
            
            # Step 6: Find volume imbalances
            volume_imbalances = self._find_volume_imbalances(volume_nodes)
            
            # Step 7: Identify acceptance/rejection levels
            acceptance_levels = self._identify_acceptance_levels(volume_nodes, analysis_data)
            rejection_levels = self._identify_rejection_levels(volume_nodes, analysis_data)
            
            # Create volume profile
            profile = VolumeProfile(
                timestamp=end_time,
                profile_type=profile_type,
                start_time=start_time,
                end_time=end_time,
                volume_nodes=volume_nodes,
                total_volume=sum(node.volume for node in volume_nodes),
                total_bars=len(analysis_data),
                poc=poc,
                value_area=value_area,
                high_volume_nodes=high_volume_nodes,
                low_volume_nodes=low_volume_nodes,
                volume_imbalances=volume_imbalances,
                acceptance_levels=acceptance_levels,
                rejection_levels=rejection_levels
            )
            
            # Store profile based on type
            if profile_type == VolumeProfileType.SESSION:
                self.session_profiles.append(profile)
                self._cleanup_old_profiles()
            elif profile_type == VolumeProfileType.ANCHORED:
                self.anchored_profiles.append(profile)
            
            logger.info(f"Volume profile analysis completed: {len(volume_nodes)} nodes, POC at {poc.price:.5f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {e}")
            return None
    
    def _create_price_bins(self, data: pd.DataFrame) -> np.ndarray:
        """Create price bins for volume distribution"""
        price_low = data['Low'].min()
        price_high = data['High'].max()
        
        # Create evenly spaced price levels
        price_bins = np.linspace(price_low, price_high, self.price_resolution + 1)
        
        return price_bins
    
    def _calculate_volume_distribution(self, data: pd.DataFrame, 
                                     price_bins: np.ndarray) -> List[VolumeNode]:
        """Calculate volume distribution across price levels"""
        volume_nodes = []
        
        # Initialize volume for each price level
        for i in range(len(price_bins) - 1):
            price_level = (price_bins[i] + price_bins[i + 1]) / 2
            volume_nodes.append({
                'price': price_level,
                'volume': 0,
                'bar_count': 0,
                'first_time': None,
                'last_time': None,
                'price_low': price_bins[i],
                'price_high': price_bins[i + 1]
            })
        
        # Distribute volume to price levels
        for timestamp, bar in data.iterrows():
            bar_volume = bar.get('Volume', 1)  # Default to 1 if no volume
            if bar_volume <= 0:
                bar_volume = 1
            
            # Distribute volume across the bar's price range
            bar_low = bar['Low']
            bar_high = bar['High']
            bar_range = bar_high - bar_low
            
            if bar_range <= 0:
                # If no range, assign all volume to close price
                close_price = bar['Close']
                for node in volume_nodes:
                    if node['price_low'] <= close_price <= node['price_high']:
                        node['volume'] += bar_volume
                        node['bar_count'] += 1
                        if node['first_time'] is None:
                            node['first_time'] = timestamp
                        node['last_time'] = timestamp
                        break
            else:
                # Distribute volume proportionally across overlapping price levels
                for node in volume_nodes:
                    # Calculate overlap between bar range and price level
                    overlap_low = max(bar_low, node['price_low'])
                    overlap_high = min(bar_high, node['price_high'])
                    
                    if overlap_high > overlap_low:  # There is overlap
                        overlap_range = overlap_high - overlap_low
                        volume_proportion = overlap_range / bar_range
                        allocated_volume = int(bar_volume * volume_proportion)
                        
                        if allocated_volume > 0:
                            node['volume'] += allocated_volume
                            node['bar_count'] += 1
                            if node['first_time'] is None:
                                node['first_time'] = timestamp
                            node['last_time'] = timestamp
        
        # Convert to VolumeNode objects and filter by minimum threshold
        total_volume = sum(node['volume'] for node in volume_nodes)
        min_volume = total_volume * self.min_volume_threshold
        
        filtered_nodes = []
        for node in volume_nodes:
            if node['volume'] >= min_volume and node['bar_count'] > 0:
                percentage = (node['volume'] / total_volume) * 100 if total_volume > 0 else 0
                
                volume_node = VolumeNode(
                    price=node['price'],
                    volume=node['volume'],
                    percentage=percentage,
                    bar_count=node['bar_count'],
                    first_time=node['first_time'],
                    last_time=node['last_time']
                )
                filtered_nodes.append(volume_node)
        
        return filtered_nodes
    
    def _calculate_value_area(self, volume_nodes: List[VolumeNode]) -> ValueArea:
        """Calculate Value Area (70% of volume around POC)"""
        if not volume_nodes:
            return ValueArea(0, 0, 0, 0, 0, 0)
        
        # Find POC (highest volume node)
        poc_node = max(volume_nodes, key=lambda x: x.volume)
        
        # Sort nodes by price
        sorted_nodes = sorted(volume_nodes, key=lambda x: x.price)
        poc_index = next(i for i, node in enumerate(sorted_nodes) if node.price == poc_node.price)
        
        # Calculate total volume
        total_volume = sum(node.volume for node in volume_nodes)
        target_volume = total_volume * self.value_area_percentage
        
        # Expand around POC until we reach target volume
        value_area_volume = poc_node.volume
        lower_index = poc_index
        upper_index = poc_index
        
        while value_area_volume < target_volume:
            # Determine which direction to expand
            lower_volume = 0
            upper_volume = 0
            
            if lower_index > 0:
                lower_volume = sorted_nodes[lower_index - 1].volume
            
            if upper_index < len(sorted_nodes) - 1:
                upper_volume = sorted_nodes[upper_index + 1].volume
            
            # Expand in direction with higher volume
            if lower_volume > upper_volume and lower_index > 0:
                lower_index -= 1
                value_area_volume += lower_volume
            elif upper_volume > 0 and upper_index < len(sorted_nodes) - 1:
                upper_index += 1
                value_area_volume += upper_volume
            elif lower_volume > 0 and lower_index > 0:
                lower_index -= 1
                value_area_volume += lower_volume
            else:
                break  # Can't expand further
        
        value_area_low = sorted_nodes[lower_index].price
        value_area_high = sorted_nodes[upper_index].price
        value_area_percentage = (value_area_volume / total_volume) * 100 if total_volume > 0 else 0
        
        return ValueArea(
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            value_area_volume=value_area_volume,
            value_area_percentage=value_area_percentage,
            poc_price=poc_node.price,
            poc_volume=poc_node.volume
        )
    
    def _identify_high_volume_nodes(self, volume_nodes: List[VolumeNode]) -> List[VolumeNode]:
        """Identify high volume nodes (top 20% by volume)"""
        if not volume_nodes:
            return []
        
        # Sort by volume descending
        sorted_by_volume = sorted(volume_nodes, key=lambda x: x.volume, reverse=True)
        
        # Take top 20%
        high_volume_count = max(1, int(len(sorted_by_volume) * (1 - self.high_volume_threshold)))
        
        return sorted_by_volume[:high_volume_count]
    
    def _identify_low_volume_nodes(self, volume_nodes: List[VolumeNode]) -> List[VolumeNode]:
        """Identify low volume nodes (bottom 20% by volume)"""
        if not volume_nodes:
            return []
        
        # Sort by volume ascending
        sorted_by_volume = sorted(volume_nodes, key=lambda x: x.volume)
        
        # Take bottom 20%
        low_volume_count = max(1, int(len(sorted_by_volume) * self.low_volume_threshold))
        
        return sorted_by_volume[:low_volume_count]
    
    def _find_volume_imbalances(self, volume_nodes: List[VolumeNode]) -> List[Dict]:
        """Find volume imbalances (gaps in volume profile)"""
        if len(volume_nodes) < 3:
            return []
        
        # Sort nodes by price
        sorted_nodes = sorted(volume_nodes, key=lambda x: x.price)
        
        # Calculate average volume
        avg_volume = sum(node.volume for node in volume_nodes) / len(volume_nodes)
        imbalance_threshold = avg_volume * self.imbalance_threshold
        
        imbalances = []
        
        for i in range(1, len(sorted_nodes) - 1):
            current_node = sorted_nodes[i]
            
            if current_node.volume < imbalance_threshold:
                # Found potential imbalance
                imbalance_start = current_node.price
                imbalance_end = current_node.price
                imbalance_volume = current_node.volume
                
                # Extend imbalance if adjacent nodes are also low volume
                j = i + 1
                while (j < len(sorted_nodes) and 
                       sorted_nodes[j].volume < imbalance_threshold):
                    imbalance_end = sorted_nodes[j].price
                    imbalance_volume += sorted_nodes[j].volume
                    j += 1
                
                j = i - 1
                while (j >= 0 and 
                       sorted_nodes[j].volume < imbalance_threshold):
                    imbalance_start = sorted_nodes[j].price
                    imbalance_volume += sorted_nodes[j].volume
                    j -= 1
                
                if imbalance_end > imbalance_start:
                    imbalances.append({
                        'start_price': imbalance_start,
                        'end_price': imbalance_end,
                        'total_volume': imbalance_volume,
                        'range': imbalance_end - imbalance_start,
                        'severity': 1 - (imbalance_volume / (avg_volume * ((imbalance_end - imbalance_start) / (sorted_nodes[-1].price - sorted_nodes[0].price)) * len(sorted_nodes)))
                    })
        
        # Remove overlapping imbalances
        unique_imbalances = []
        for imbalance in imbalances:
            is_duplicate = False
            for existing in unique_imbalances:
                if (imbalance['start_price'] <= existing['end_price'] and 
                    imbalance['end_price'] >= existing['start_price']):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_imbalances.append(imbalance)
        
        return unique_imbalances
    
    def _identify_acceptance_levels(self, volume_nodes: List[VolumeNode], 
                                  data: pd.DataFrame) -> List[float]:
        """Identify price levels with sustained acceptance (high time at price)"""
        acceptance_levels = []
        
        if not volume_nodes:
            return acceptance_levels
        
        # Calculate average time spent at each price level
        for node in volume_nodes:
            time_spent = (node.last_time - node.first_time).total_seconds() / 3600 if node.last_time and node.first_time else 0
            volume_per_hour = node.volume / time_spent if time_spent > 0 else 0
            
            # High acceptance = high volume AND significant time spent
            if (node.percentage > 5.0 and  # At least 5% of total volume
                time_spent > 1 and        # At least 1 hour
                volume_per_hour > 0):     # Sustained activity
                acceptance_levels.append(node.price)
        
        return acceptance_levels
    
    def _identify_rejection_levels(self, volume_nodes: List[VolumeNode], 
                                 data: pd.DataFrame) -> List[float]:
        """Identify price levels with quick rejection (low time, high volume spikes)"""
        rejection_levels = []
        
        if not volume_nodes:
            return rejection_levels
        
        for node in volume_nodes:
            time_spent = (node.last_time - node.first_time).total_seconds() / 3600 if node.last_time and node.first_time else 0
            volume_per_hour = node.volume / time_spent if time_spent > 0 else float('inf')
            
            # High rejection = high volume BUT very short time
            if (node.percentage > 3.0 and      # At least 3% of volume
                time_spent < 0.5 and          # Less than 30 minutes
                volume_per_hour > 1000):      # Very high volume rate
                rejection_levels.append(node.price)
        
        return rejection_levels
    
    def _cleanup_old_profiles(self):
        """Remove old session profiles to manage memory"""
        max_profiles = 10
        if len(self.session_profiles) > max_profiles:
            self.session_profiles = self.session_profiles[-max_profiles:]
    
    def get_current_profile_context(self, current_price: float, 
                                  profile: VolumeProfile) -> Dict:
        """Get current market context relative to volume profile"""
        if not profile or not profile.volume_nodes:
            return {'context': 'no_profile_data'}
        
        context = {
            'current_price': current_price,
            'poc_price': profile.poc.price,
            'value_area_high': profile.value_area.value_area_high,
            'value_area_low': profile.value_area.value_area_low,
            'price_position': 'unknown',
            'volume_context': 'unknown',
            'nearest_high_volume_node': None,
            'nearest_imbalance': None
        }
        
        # Determine price position
        if current_price > profile.value_area.value_area_high:
            context['price_position'] = 'above_value_area'
        elif current_price < profile.value_area.value_area_low:
            context['price_position'] = 'below_value_area'
        else:
            context['price_position'] = 'inside_value_area'
        
        # Determine volume context
        poc_distance = abs(current_price - profile.poc.price)
        va_range = profile.value_area.value_area_range
        
        if poc_distance < va_range * 0.1:  # Within 10% of POC
            context['volume_context'] = 'near_poc'
        elif context['price_position'] == 'inside_value_area':
            context['volume_context'] = 'inside_value_area'
        else:
            context['volume_context'] = 'outside_value_area'
        
        # Find nearest high volume node
        if profile.high_volume_nodes:
            nearest_hv_node = min(profile.high_volume_nodes, 
                                key=lambda x: abs(x.price - current_price))
            context['nearest_high_volume_node'] = {
                'price': nearest_hv_node.price,
                'volume': nearest_hv_node.volume,
                'distance': abs(current_price - nearest_hv_node.price)
            }
        
        # Find nearest volume imbalance
        if profile.volume_imbalances:
            for imbalance in profile.volume_imbalances:
                if (imbalance['start_price'] <= current_price <= imbalance['end_price']):
                    context['nearest_imbalance'] = {
                        'inside_imbalance': True,
                        'imbalance': imbalance
                    }
                    break
            
            if 'nearest_imbalance' not in context:
                nearest_imbalance = min(profile.volume_imbalances,
                                      key=lambda x: min(
                                          abs(current_price - x['start_price']),
                                          abs(current_price - x['end_price'])
                                      ))
                context['nearest_imbalance'] = {
                    'inside_imbalance': False,
                    'imbalance': nearest_imbalance,
                    'distance': min(
                        abs(current_price - nearest_imbalance['start_price']),
                        abs(current_price - nearest_imbalance['end_price'])
                    )
                }
        
        return context
    
    def get_trading_levels(self, profile: VolumeProfile) -> Dict:
        """Get key trading levels from volume profile"""
        if not profile:
            return {}
        
        levels = {
            'poc': profile.poc.price,
            'value_area_high': profile.value_area.value_area_high,
            'value_area_low': profile.value_area.value_area_low,
            'high_volume_levels': [node.price for node in profile.high_volume_nodes],
            'acceptance_levels': profile.acceptance_levels,
            'rejection_levels': profile.rejection_levels,
            'imbalance_zones': [
                {
                    'start': imb['start_price'],
                    'end': imb['end_price'],
                    'mid': (imb['start_price'] + imb['end_price']) / 2
                }
                for imb in profile.volume_imbalances
            ]
        }
        
        return levels
    
    def analyze_session_profile(self, data: pd.DataFrame) -> Optional[VolumeProfile]:
        """Analyze session-based volume profile"""
        return self.analyze_volume_profile(data, VolumeProfileType.SESSION)
    
    def analyze_anchored_profile(self, data: pd.DataFrame, 
                               anchor_time: datetime) -> Optional[VolumeProfile]:
        """Analyze anchored volume profile from specific time"""
        return self.analyze_volume_profile(data, VolumeProfileType.ANCHORED, anchor_time)
    
    def get_profile_summary(self, profile: VolumeProfile) -> Dict:
        """Get summary of volume profile analysis"""
        if not profile:
            return {'error': 'No profile data'}
        
        summary = {
            'profile_type': profile.profile_type.value,
            'total_volume': profile.total_volume,
            'total_bars': profile.total_bars,
            'price_range': profile.price_range,
            'poc_price': profile.poc.price,
            'poc_volume': profile.poc.volume,
            'poc_percentage': profile.poc.percentage,
            'value_area_range': profile.value_area.value_area_range,
            'value_area_percentage': profile.value_area.value_area_percentage,
            'volume_distribution': profile.volume_distribution_balance,
            'high_volume_nodes_count': len(profile.high_volume_nodes),
            'low_volume_nodes_count': len(profile.low_volume_nodes),
            'volume_imbalances_count': len(profile.volume_imbalances),
            'acceptance_levels_count': len(profile.acceptance_levels),
            'rejection_levels_count': len(profile.rejection_levels)
        }
        
        return summary

# Export main classes
__all__ = [
    'VolumeProfileAnalyzer',
    'VolumeProfile',
    'VolumeNode',
    'ValueArea',
    'VolumeProfileType',
    'POCType'
]