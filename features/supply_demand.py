"""
Supply/Demand Zone Analysis
Identifies institutional supply and demand zones for SMC trading
Integrates with Premium/Discount, Liquidity, and FVG analysis
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
    """Supply/Demand zone types"""
    SUPPLY = "supply"
    DEMAND = "demand"
    
class ZoneStatus(Enum):
    """Zone status classification"""
    FRESH = "fresh"        # Never tested
    TESTED = "tested"      # Touched but held
    BROKEN = "broken"      # Price closed through
    MITIGATED = "mitigated" # Partially consumed
    
class ZoneQuality(Enum):
    """Zone quality assessment"""
    HIGH = "high"          # Strong institutional footprint
    MEDIUM = "medium"      # Good zone characteristics
    LOW = "low"           # Weak or unclear zone

class FormationType(Enum):
    """Zone formation types"""
    BASE_BREAKOUT = "base_breakout"       # Base formation + breakout
    IMPULSE_CORRECTION = "impulse_correction" # Strong move + pullback
    REJECTION_CANDLE = "rejection_candle"     # Strong rejection from level
    VOLUME_IMBALANCE = "volume_imbalance"     # High volume accumulation
    
@dataclass
class SupplyDemandZone:
    """Supply/Demand zone data structure"""
    timestamp: datetime
    zone_type: ZoneType
    formation_type: FormationType
    top: float
    bottom: float
    entry_price: float      # Best entry level within zone
    creation_candles: int   # Number of candles that formed the zone
    volume_profile: float   # Volume strength during formation
    strength: float         # Overall zone strength (0.0 to 1.0)
    quality: ZoneQuality
    status: ZoneStatus = ZoneStatus.FRESH
    test_count: int = 0
    last_test_time: Optional[datetime] = None
    mitigation_level: float = None  # Level where zone gets invalidated
    
    @property
    def zone_size(self) -> float:
        """Get zone size"""
        return self.top - self.bottom
    
    @property
    def zone_mid(self) -> float:
        """Get zone midpoint"""
        return (self.top + self.bottom) / 2
    
    @property
    def is_valid(self) -> bool:
        """Check if zone is still valid for trading"""
        return self.status in [ZoneStatus.FRESH, ZoneStatus.TESTED]
    
    @property
    def risk_reward_level(self) -> float:
        """Get optimal risk/reward entry level"""
        if self.zone_type == ZoneType.DEMAND:
            # For demand zones, enter near the top for better R:R
            return self.top - (self.zone_size * 0.2)  # 20% from top
        else:  # SUPPLY
            # For supply zones, enter near the bottom for better R:R
            return self.bottom + (self.zone_size * 0.2)  # 20% from bottom

class SupplyDemandAnalyzer:
    """
    Analyzes Supply and Demand zones using institutional methods
    """
    
    def __init__(self,
                 min_zone_size: float = 0.0005,      # 5 pips minimum
                 max_zone_size: float = 0.005,       # 50 pips maximum
                 min_impulse_size: float = 0.001,    # 10 pips minimum impulse
                 volume_threshold: float = 1.5,      # Volume multiplier
                 base_formation_bars: int = 5,       # Min bars for base formation
                 confirmation_bars: int = 3,         # Bars needed for confirmation
                 zone_validity_hours: int = 168):    # 1 week validity
        
        self.min_zone_size = min_zone_size
        self.max_zone_size = max_zone_size
        self.min_impulse_size = min_impulse_size
        self.volume_threshold = volume_threshold
        self.base_formation_bars = base_formation_bars
        self.confirmation_bars = confirmation_bars
        self.zone_validity_hours = zone_validity_hours
        
        # Storage for identified zones
        self.supply_zones = []
        self.demand_zones = []
        
    def analyze_supply_demand(self, data: pd.DataFrame) -> Dict:
        """
        Main function to analyze supply/demand zones
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            Dictionary with supply/demand analysis
        """
        if len(data) < self.base_formation_bars * 3:
            logger.warning("Insufficient data for supply/demand analysis")
            return self._empty_analysis()
        
        try:
            # Step 1: Identify potential supply zones
            supply_zones = self._identify_supply_zones(data)
            
            # Step 2: Identify potential demand zones
            demand_zones = self._identify_demand_zones(data)
            
            # Step 3: Validate zones with volume and structure analysis
            validated_supply = self._validate_zones(supply_zones, data, ZoneType.SUPPLY)
            validated_demand = self._validate_zones(demand_zones, data, ZoneType.DEMAND)
            
            # Step 4: Update existing zones status
            self._update_zone_status(data)
            
            # Step 5: Add new validated zones
            self._add_new_zones(validated_supply, validated_demand)
            
            # Step 6: Calculate zone metrics and confluence
            zone_metrics = self._calculate_zone_metrics(data)
            confluence_analysis = self._analyze_confluence(data)
            
            return {
                'timestamp': data.index[-1],
                'current_price': data['Close'].iloc[-1],
                'supply_zones': self.supply_zones,
                'demand_zones': self.demand_zones,
                'new_supply_zones': validated_supply,
                'new_demand_zones': validated_demand,
                'zone_metrics': zone_metrics,
                'confluence_analysis': confluence_analysis,
                'summary': self._generate_summary(data)
            }
            
        except Exception as e:
            logger.error(f"Error in supply/demand analysis: {e}")
            return self._empty_analysis()
    
    def _identify_supply_zones(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Identify potential supply zones"""
        supply_zones = []
        
        # Method 1: Base formation + bearish breakout
        supply_zones.extend(self._find_base_breakout_supply(data))
        
        # Method 2: Impulse + correction supply zones
        supply_zones.extend(self._find_impulse_correction_supply(data))
        
        # Method 3: Rejection candle supply zones
        supply_zones.extend(self._find_rejection_supply(data))
        
        return supply_zones
    
    def _identify_demand_zones(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Identify potential demand zones"""
        demand_zones = []
        
        # Method 1: Base formation + bullish breakout
        demand_zones.extend(self._find_base_breakout_demand(data))
        
        # Method 2: Impulse + correction demand zones
        demand_zones.extend(self._find_impulse_correction_demand(data))
        
        # Method 3: Rejection candle demand zones
        demand_zones.extend(self._find_rejection_demand(data))
        
        return demand_zones
    
    def _find_base_breakout_supply(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Find supply zones from base formation + bearish breakout"""
        zones = []
        
        for i in range(self.base_formation_bars, len(data) - self.confirmation_bars):
            # Look for base formation (consolidation)
            base_start = i - self.base_formation_bars
            base_end = i
            base_data = data.iloc[base_start:base_end]
            
            # Check if we have a base (low volatility consolidation)
            base_high = base_data['High'].max()
            base_low = base_data['Low'].min()
            base_size = base_high - base_low
            
            if base_size < self.min_zone_size or base_size > self.max_zone_size:
                continue
            
            # Check for bearish breakout after base
            breakout_data = data.iloc[i:i + self.confirmation_bars]
            if breakout_data.empty:
                continue
                
            # Strong bearish move required
            breakout_move = base_low - breakout_data['Low'].min()
            if breakout_move < self.min_impulse_size:
                continue
            
            # Calculate volume during base formation
            avg_volume = base_data['Volume'].mean() if 'Volume' in base_data.columns else 1.0
            
            # Create supply zone
            zone = SupplyDemandZone(
                timestamp=data.index[i],
                zone_type=ZoneType.SUPPLY,
                formation_type=FormationType.BASE_BREAKOUT,
                top=base_high,
                bottom=base_low,
                entry_price=base_low + (base_size * 0.3),  # 30% from bottom
                creation_candles=self.base_formation_bars,
                volume_profile=avg_volume,
                strength=self._calculate_base_strength(base_data, breakout_data),
                quality=ZoneQuality.MEDIUM,
                mitigation_level=base_high + (base_size * 0.1)  # 10% above zone
            )
            
            zones.append(zone)
        
        return zones
    
    def _find_base_breakout_demand(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Find demand zones from base formation + bullish breakout"""
        zones = []
        
        for i in range(self.base_formation_bars, len(data) - self.confirmation_bars):
            # Look for base formation
            base_start = i - self.base_formation_bars
            base_end = i
            base_data = data.iloc[base_start:base_end]
            
            base_high = base_data['High'].max()
            base_low = base_data['Low'].min()
            base_size = base_high - base_low
            
            if base_size < self.min_zone_size or base_size > self.max_zone_size:
                continue
            
            # Check for bullish breakout
            breakout_data = data.iloc[i:i + self.confirmation_bars]
            if breakout_data.empty:
                continue
                
            breakout_move = breakout_data['High'].max() - base_high
            if breakout_move < self.min_impulse_size:
                continue
            
            avg_volume = base_data['Volume'].mean() if 'Volume' in base_data.columns else 1.0
            
            zone = SupplyDemandZone(
                timestamp=data.index[i],
                zone_type=ZoneType.DEMAND,
                formation_type=FormationType.BASE_BREAKOUT,
                top=base_high,
                bottom=base_low,
                entry_price=base_high - (base_size * 0.3),  # 30% from top
                creation_candles=self.base_formation_bars,
                volume_profile=avg_volume,
                strength=self._calculate_base_strength(base_data, breakout_data),
                quality=ZoneQuality.MEDIUM,
                mitigation_level=base_low - (base_size * 0.1)  # 10% below zone
            )
            
            zones.append(zone)
        
        return zones
    
    def _find_impulse_correction_supply(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Find supply zones from impulse + correction pattern"""
        zones = []
        
        for i in range(10, len(data) - 5):
            # Look for strong bearish impulse
            impulse_start = i - 5
            impulse_end = i
            impulse_data = data.iloc[impulse_start:impulse_end]
            
            impulse_move = impulse_data['High'].max() - impulse_data['Low'].min()
            if impulse_move < self.min_impulse_size * 2:  # Stronger impulse required
                continue
            
            # Look for correction (pullback) after impulse
            correction_data = data.iloc[i:i + 3]
            if correction_data.empty:
                continue
            
            # Check if price pulled back into supply zone
            impulse_high = impulse_data['High'].max()
            correction_high = correction_data['High'].max()
            
            if correction_high < impulse_high * 0.8:  # Significant pullback
                continue
            
            # Define supply zone around the correction area
            zone_top = correction_data['High'].max()
            zone_bottom = correction_data['Low'].min()
            zone_size = zone_top - zone_bottom
            
            if zone_size < self.min_zone_size:
                zone_bottom = zone_top - self.min_zone_size
            
            avg_volume = correction_data['Volume'].mean() if 'Volume' in correction_data.columns else 1.0
            
            zone = SupplyDemandZone(
                timestamp=data.index[i],
                zone_type=ZoneType.SUPPLY,
                formation_type=FormationType.IMPULSE_CORRECTION,
                top=zone_top,
                bottom=zone_bottom,
                entry_price=zone_bottom + (zone_size * 0.2),
                creation_candles=3,
                volume_profile=avg_volume,
                strength=self._calculate_impulse_strength(impulse_data, correction_data),
                quality=ZoneQuality.HIGH,
                mitigation_level=zone_top + (zone_size * 0.2)
            )
            
            zones.append(zone)
        
        return zones
    
    def _find_impulse_correction_demand(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Find demand zones from impulse + correction pattern"""
        zones = []
        
        for i in range(10, len(data) - 5):
            # Look for strong bullish impulse
            impulse_start = i - 5
            impulse_end = i
            impulse_data = data.iloc[impulse_start:impulse_end]
            
            impulse_move = impulse_data['High'].max() - impulse_data['Low'].min()
            if impulse_move < self.min_impulse_size * 2:
                continue
            
            # Look for correction
            correction_data = data.iloc[i:i + 3]
            if correction_data.empty:
                continue
            
            impulse_low = impulse_data['Low'].min()
            correction_low = correction_data['Low'].min()
            
            if correction_low > impulse_low * 1.2:  # Significant pullback
                continue
            
            zone_top = correction_data['High'].max()
            zone_bottom = correction_data['Low'].min()
            zone_size = zone_top - zone_bottom
            
            if zone_size < self.min_zone_size:
                zone_top = zone_bottom + self.min_zone_size
            
            avg_volume = correction_data['Volume'].mean() if 'Volume' in correction_data.columns else 1.0
            
            zone = SupplyDemandZone(
                timestamp=data.index[i],
                zone_type=ZoneType.DEMAND,
                formation_type=FormationType.IMPULSE_CORRECTION,
                top=zone_top,
                bottom=zone_bottom,
                entry_price=zone_top - (zone_size * 0.2),
                creation_candles=3,
                volume_profile=avg_volume,
                strength=self._calculate_impulse_strength(impulse_data, correction_data),
                quality=ZoneQuality.HIGH,
                mitigation_level=zone_bottom - (zone_size * 0.2)
            )
            
            zones.append(zone)
        
        return zones
    
    def _find_rejection_supply(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Find supply zones from strong rejection candles"""
        zones = []
        
        for i in range(5, len(data) - 2):
            candle = data.iloc[i]
            
            # Look for bearish rejection candle
            body_size = abs(candle['Close'] - candle['Open'])
            total_range = candle['High'] - candle['Low']
            upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
            
            # Strong upper wick rejection
            if (upper_wick > body_size * 2 and 
                upper_wick > total_range * 0.6 and
                total_range > self.min_zone_size):
                
                # Create supply zone around the rejection area
                zone_top = candle['High']
                zone_bottom = max(candle['Open'], candle['Close'])
                
                zone = SupplyDemandZone(
                    timestamp=candle.name,
                    zone_type=ZoneType.SUPPLY,
                    formation_type=FormationType.REJECTION_CANDLE,
                    top=zone_top,
                    bottom=zone_bottom,
                    entry_price=zone_bottom + ((zone_top - zone_bottom) * 0.3),
                    creation_candles=1,
                    volume_profile=candle.get('Volume', 1.0),
                    strength=self._calculate_rejection_strength(candle),
                    quality=ZoneQuality.MEDIUM,
                    mitigation_level=zone_top + ((zone_top - zone_bottom) * 0.1)
                )
                
                zones.append(zone)
        
        return zones
    
    def _find_rejection_demand(self, data: pd.DataFrame) -> List[SupplyDemandZone]:
        """Find demand zones from strong rejection candles"""
        zones = []
        
        for i in range(5, len(data) - 2):
            candle = data.iloc[i]
            
            # Look for bullish rejection candle
            body_size = abs(candle['Close'] - candle['Open'])
            total_range = candle['High'] - candle['Low']
            lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
            
            # Strong lower wick rejection
            if (lower_wick > body_size * 2 and 
                lower_wick > total_range * 0.6 and
                total_range > self.min_zone_size):
                
                zone_top = min(candle['Open'], candle['Close'])
                zone_bottom = candle['Low']
                
                zone = SupplyDemandZone(
                    timestamp=candle.name,
                    zone_type=ZoneType.DEMAND,
                    formation_type=FormationType.REJECTION_CANDLE,
                    top=zone_top,
                    bottom=zone_bottom,
                    entry_price=zone_top - ((zone_top - zone_bottom) * 0.3),
                    creation_candles=1,
                    volume_profile=candle.get('Volume', 1.0),
                    strength=self._calculate_rejection_strength(candle),
                    quality=ZoneQuality.MEDIUM,
                    mitigation_level=zone_bottom - ((zone_top - zone_bottom) * 0.1)
                )
                
                zones.append(zone)
        
        return zones
    
    def _calculate_base_strength(self, base_data: pd.DataFrame, breakout_data: pd.DataFrame) -> float:
        """Calculate strength of base formation zone"""
        try:
            # Factor 1: Base tightness (smaller range = stronger)
            base_range = base_data['High'].max() - base_data['Low'].min()
            tightness_score = max(0, 1 - (base_range / self.max_zone_size))
            
            # Factor 2: Breakout strength
            breakout_range = breakout_data['High'].max() - breakout_data['Low'].min()
            breakout_score = min(breakout_range / self.min_impulse_size, 2.0) / 2.0
            
            # Factor 3: Volume (if available)
            volume_score = 0.5
            if 'Volume' in base_data.columns:
                avg_volume = base_data['Volume'].mean()
                if avg_volume > 0:
                    volume_score = min(avg_volume / (avg_volume * self.volume_threshold), 1.0)
            
            # Combined strength
            strength = (tightness_score * 0.4 + breakout_score * 0.4 + volume_score * 0.2)
            return min(max(strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating base strength: {e}")
            return 0.5
    
    def _calculate_impulse_strength(self, impulse_data: pd.DataFrame, correction_data: pd.DataFrame) -> float:
        """Calculate strength of impulse + correction zone"""
        try:
            # Factor 1: Impulse size
            impulse_range = impulse_data['High'].max() - impulse_data['Low'].min()
            impulse_score = min(impulse_range / (self.min_impulse_size * 3), 1.0)
            
            # Factor 2: Correction ratio (how much it pulled back)
            correction_range = correction_data['High'].max() - correction_data['Low'].min()
            correction_ratio = correction_range / impulse_range if impulse_range > 0 else 0
            correction_score = max(0, 1 - correction_ratio)  # Smaller correction = stronger
            
            # Factor 3: Speed of impulse (fewer candles = stronger)
            speed_score = max(0, 1 - (len(impulse_data) / 10))
            
            strength = (impulse_score * 0.5 + correction_score * 0.3 + speed_score * 0.2)
            return min(max(strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating impulse strength: {e}")
            return 0.6
    
    def _calculate_rejection_strength(self, candle: pd.Series) -> float:
        """Calculate strength of rejection candle zone"""
        try:
            total_range = candle['High'] - candle['Low']
            body_size = abs(candle['Close'] - candle['Open'])
            
            # Factor 1: Wick to body ratio
            wick_ratio = (total_range - body_size) / total_range if total_range > 0 else 0
            
            # Factor 2: Candle size
            size_score = min(total_range / self.min_impulse_size, 2.0) / 2.0
            
            # Factor 3: Body position (for rejection, body should be small)
            body_ratio = body_size / total_range if total_range > 0 else 1
            body_score = max(0, 1 - body_ratio)
            
            strength = (wick_ratio * 0.4 + size_score * 0.3 + body_score * 0.3)
            return min(max(strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating rejection strength: {e}")
            return 0.4
    
    def _validate_zones(self, zones: List[SupplyDemandZone], data: pd.DataFrame, zone_type: ZoneType) -> List[SupplyDemandZone]:
        """Validate zones with additional criteria"""
        validated = []
        
        for zone in zones:
            # Validation 1: Zone size within limits
            if zone.zone_size < self.min_zone_size or zone.zone_size > self.max_zone_size:
                continue
            
            # Validation 2: Minimum strength threshold
            if zone.strength < 0.3:
                continue
            
            # Validation 3: No overlapping with existing zones
            if self._overlaps_with_existing(zone, zone_type):
                continue
            
            # Validation 4: Zone age (not too old)
            latest_time = data.index[-1]
            zone_age_hours = (latest_time - zone.timestamp).total_seconds() / 3600
            if zone_age_hours > self.zone_validity_hours:
                continue
            
            # Assign quality based on formation type and strength
            zone.quality = self._assess_zone_quality(zone)
            
            validated.append(zone)
        
        return validated
    
    def _overlaps_with_existing(self, new_zone: SupplyDemandZone, zone_type: ZoneType) -> bool:
        """Check if new zone overlaps with existing zones"""
        existing_zones = self.supply_zones if zone_type == ZoneType.SUPPLY else self.demand_zones
        
        for existing in existing_zones:
            if existing.status == ZoneStatus.BROKEN:
                continue
            
            # Check for overlap
            overlap_top = min(new_zone.top, existing.top)
            overlap_bottom = max(new_zone.bottom, existing.bottom)
            
            if overlap_top > overlap_bottom:  # There is overlap
                overlap_size = overlap_top - overlap_bottom
                new_zone_size = new_zone.zone_size
                
                # If overlap is more than 50% of new zone, consider it duplicate
                if overlap_size > new_zone_size * 0.5:
                    return True
        
        return False
    
    def _assess_zone_quality(self, zone: SupplyDemandZone) -> ZoneQuality:
        """Assess zone quality based on multiple factors"""
        score = 0
        
        # Formation type scoring
        formation_scores = {
            FormationType.IMPULSE_CORRECTION: 0.8,
            FormationType.BASE_BREAKOUT: 0.6,
            FormationType.REJECTION_CANDLE: 0.4,
            FormationType.VOLUME_IMBALANCE: 0.7
        }
        score += formation_scores.get(zone.formation_type, 0.5)
        
        # Strength scoring
        score += zone.strength
        
        # Volume scoring (if available)
        if zone.volume_profile > 0:
            volume_score = min(zone.volume_profile / 100, 1.0)  # Normalize
            score += volume_score * 0.5
        
        # Normalize to 0-3 range
        normalized_score = score / 3.0
        
        if normalized_score >= 0.7:
            return ZoneQuality.HIGH
        elif normalized_score >= 0.5:
            return ZoneQuality.MEDIUM
        else:
            return ZoneQuality.LOW
    
    def _update_zone_status(self, data: pd.DataFrame):
        """Update status of existing zones"""
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1]
        
        all_zones = self.supply_zones + self.demand_zones
        
        for zone in all_zones:
            if zone.status == ZoneStatus.BROKEN:
                continue
            
            # Check if zone has been broken
            if self._is_zone_broken(zone, current_price):
                zone.status = ZoneStatus.BROKEN
                continue
            
            # Check if zone has been tested
            if self._is_zone_tested(zone, data):
                if zone.status == ZoneStatus.FRESH:
                    zone.status = ZoneStatus.TESTED
                    zone.test_count += 1
                    zone.last_test_time = current_time
    
    def _is_zone_broken(self, zone: SupplyDemandZone, current_price: float) -> bool:
        """Check if zone has been broken"""
        if zone.zone_type == ZoneType.SUPPLY:
            # Supply zone broken if price closes above top
            return current_price > zone.top
        else:  # DEMAND
            # Demand zone broken if price closes below bottom
            return current_price < zone.bottom
    
    def _is_zone_tested(self, zone: SupplyDemandZone, data: pd.DataFrame) -> bool:
        """Check if zone has been tested recently"""
        # Get recent data after zone creation
        zone_time = zone.timestamp
        recent_data = data[data.index > zone_time].tail(10)
        
        if recent_data.empty:
            return False
        
        for _, candle in recent_data.iterrows():
            # Check if price entered the zone
            if (candle['Low'] <= zone.top and candle['High'] >= zone.bottom):
                return True
        
        return False
    
    def _add_new_zones(self, supply_zones: List[SupplyDemandZone], demand_zones: List[SupplyDemandZone]):
        """Add new validated zones to storage"""
        self.supply_zones.extend(supply_zones)
        self.demand_zones.extend(demand_zones)
        
        # Clean up old/broken zones
        self._cleanup_zones()
    
    def _cleanup_zones(self):
        """Remove old or broken zones"""
        current_time = datetime.now()
        
        # Remove broken zones older than 24 hours
        self.supply_zones = [
            zone for zone in self.supply_zones
            if not (zone.status == ZoneStatus.BROKEN and 
                   (current_time - zone.timestamp).total_seconds() > 86400)
        ]
        
        self.demand_zones = [
            zone for zone in self.demand_zones  
            if not (zone.status == ZoneStatus.BROKEN and
                   (current_time - zone.timestamp).total_seconds() > 86400)
        ]
        
        # Keep only top zones by quality and strength
        max_zones = 15
        
        if len(self.supply_zones) > max_zones:
            self.supply_zones.sort(key=lambda x: (x.quality.value, x.strength), reverse=True)
            self.supply_zones = self.supply_zones[:max_zones]
        
        if len(self.demand_zones) > max_zones:
            self.demand_zones.sort(key=lambda x: (x.quality.value, x.strength), reverse=True)
            self.demand_zones = self.demand_zones[:max_zones]
    
    def _calculate_zone_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive zone metrics"""
        current_price = data['Close'].iloc[-1]
        
        # Active zones (fresh + tested)
        active_supply = [z for z in self.supply_zones if z.is_valid]
        active_demand = [z for z in self.demand_zones if z.is_valid]
        
        # Zone counts by quality
        supply_quality_counts = self._count_by_quality(active_supply)
        demand_quality_counts = self._count_by_quality(active_demand)
        
        # Zone counts by formation type
        supply_formation_counts = self._count_by_formation(active_supply)
        demand_formation_counts = self._count_by_formation(active_demand)
        
        # Strength statistics
        supply_strengths = [z.strength for z in active_supply]
        demand_strengths = [z.strength for z in active_demand]
        
        # Proximity analysis
        nearby_supply = self._get_zones_near_price(active_supply, current_price, 0.002)  # 20 pips
        nearby_demand = self._get_zones_near_price(active_demand, current_price, 0.002)
        
        return {
            'total_supply_zones': len(active_supply),
            'total_demand_zones': len(active_demand),
            'supply_quality_distribution': supply_quality_counts,
            'demand_quality_distribution': demand_quality_counts,
            'supply_formation_distribution': supply_formation_counts,
            'demand_formation_distribution': demand_formation_counts,
            'avg_supply_strength': np.mean(supply_strengths) if supply_strengths else 0.0,
            'avg_demand_strength': np.mean(demand_strengths) if demand_strengths else 0.0,
            'nearby_supply_zones': len(nearby_supply),
            'nearby_demand_zones': len(nearby_demand),
            'nearest_supply_distance': self._get_nearest_zone_distance(active_supply, current_price),
            'nearest_demand_distance': self._get_nearest_zone_distance(active_demand, current_price),
            'zone_bias': self._calculate_zone_bias(active_supply, active_demand, current_price)
        }
    
    def _analyze_confluence(self, data: pd.DataFrame) -> Dict:
        """Analyze confluence between supply/demand zones and price action"""
        current_price = data['Close'].iloc[-1]
        
        # Find zones near current price
        active_supply = [z for z in self.supply_zones if z.is_valid]
        active_demand = [z for z in self.demand_zones if z.is_valid]
        
        confluence_zones = []
        
        # Check supply zones for confluence
        for zone in active_supply:
            if self._is_price_near_zone(current_price, zone, 0.001):  # 10 pips
                confluence_score = self._calculate_confluence_score(zone, data)
                confluence_zones.append({
                    'zone': zone,
                    'confluence_score': confluence_score,
                    'distance': abs(current_price - zone.zone_mid),
                    'recommendation': self._get_zone_recommendation(zone, current_price)
                })
        
        # Check demand zones for confluence
        for zone in active_demand:
            if self._is_price_near_zone(current_price, zone, 0.001):
                confluence_score = self._calculate_confluence_score(zone, data)
                confluence_zones.append({
                    'zone': zone,
                    'confluence_score': confluence_score,
                    'distance': abs(current_price - zone.zone_mid),
                    'recommendation': self._get_zone_recommendation(zone, current_price)
                })
        
        # Sort by confluence score
        confluence_zones.sort(key=lambda x: x['confluence_score'], reverse=True)
        
        return {
            'confluence_zones': confluence_zones[:5],  # Top 5
            'highest_confluence': confluence_zones[0] if confluence_zones else None,
            'total_confluence_zones': len(confluence_zones)
        }
    
    def _calculate_confluence_score(self, zone: SupplyDemandZone, data: pd.DataFrame) -> float:
        """Calculate confluence score for a zone"""
        score = 0.0
        
        # Base score from zone strength and quality
        score += zone.strength * 0.4
        
        quality_multiplier = {
            ZoneQuality.HIGH: 1.0,
            ZoneQuality.MEDIUM: 0.8,
            ZoneQuality.LOW: 0.6
        }
        score += quality_multiplier[zone.quality] * 0.3
        
        # Formation type bonus
        formation_bonus = {
            FormationType.IMPULSE_CORRECTION: 0.2,
            FormationType.BASE_BREAKOUT: 0.15,
            FormationType.REJECTION_CANDLE: 0.1,
            FormationType.VOLUME_IMBALANCE: 0.15
        }
        score += formation_bonus.get(zone.formation_type, 0.1)
        
        # Fresh zone bonus
        if zone.status == ZoneStatus.FRESH:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_zone_recommendation(self, zone: SupplyDemandZone, current_price: float) -> str:
        """Get trading recommendation for a zone"""
        if zone.zone_type == ZoneType.SUPPLY:
            if current_price <= zone.risk_reward_level:
                return "SHORT_ENTRY"
            elif current_price <= zone.zone_mid:
                return "WATCH_FOR_SHORT"
            else:
                return "WAIT_FOR_PULLBACK"
        else:  # DEMAND
            if current_price >= zone.risk_reward_level:
                return "LONG_ENTRY"
            elif current_price >= zone.zone_mid:
                return "WATCH_FOR_LONG"
            else:
                return "WAIT_FOR_PULLBACK"
    
    def _generate_summary(self, data: pd.DataFrame) -> Dict:
        """Generate summary of supply/demand analysis"""
        current_price = data['Close'].iloc[-1]
        
        active_supply = [z for z in self.supply_zones if z.is_valid]
        active_demand = [z for z in self.demand_zones if z.is_valid]
        
        # Market bias based on zone proximity and strength
        zone_bias = self._calculate_zone_bias(active_supply, active_demand, current_price)
        
        # Find strongest zones
        strongest_supply = max(active_supply, key=lambda x: x.strength) if active_supply else None
        strongest_demand = max(active_demand, key=lambda x: x.strength) if active_demand else None
        
        # Current market position
        market_position = self._assess_market_position(current_price, active_supply, active_demand)
        
        return {
            'market_bias': zone_bias,
            'market_position': market_position,
            'strongest_supply_zone': strongest_supply,
            'strongest_demand_zone': strongest_demand,
            'active_zones_count': len(active_supply) + len(active_demand),
            'trading_opportunities': self._identify_trading_opportunities(current_price, active_supply, active_demand)
        }
    
    def _count_by_quality(self, zones: List[SupplyDemandZone]) -> Dict[str, int]:
        """Count zones by quality"""
        counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for zone in zones:
            counts[zone.quality.value.upper()] += 1
        return counts
    
    def _count_by_formation(self, zones: List[SupplyDemandZone]) -> Dict[str, int]:
        """Count zones by formation type"""
        counts = {}
        for zone in zones:
            formation = zone.formation_type.value
            counts[formation] = counts.get(formation, 0) + 1
        return counts
    
    def _get_zones_near_price(self, zones: List[SupplyDemandZone], price: float, distance: float) -> List[SupplyDemandZone]:
        """Get zones within specified distance of price"""
        nearby = []
        for zone in zones:
            zone_distance = min(
                abs(price - zone.top),
                abs(price - zone.bottom),
                abs(price - zone.zone_mid)
            )
            if zone_distance <= distance:
                nearby.append(zone)
        return nearby
    
    def _get_nearest_zone_distance(self, zones: List[SupplyDemandZone], price: float) -> float:
        """Get distance to nearest zone"""
        if not zones:
            return float('inf')
        
        distances = []
        for zone in zones:
            distance = min(
                abs(price - zone.top),
                abs(price - zone.bottom)
            )
            distances.append(distance)
        
        return min(distances) if distances else float('inf')
    
    def _calculate_zone_bias(self, supply_zones: List[SupplyDemandZone], 
                           demand_zones: List[SupplyDemandZone], current_price: float) -> str:
        """Calculate overall market bias from zones"""
        supply_weight = 0.0
        demand_weight = 0.0
        
        # Weight zones by strength and proximity to current price
        for zone in supply_zones:
            distance = abs(current_price - zone.zone_mid)
            proximity_factor = max(0, 1 - (distance / 0.005))  # 50 pips max influence
            supply_weight += zone.strength * proximity_factor
        
        for zone in demand_zones:
            distance = abs(current_price - zone.zone_mid)
            proximity_factor = max(0, 1 - (distance / 0.005))
            demand_weight += zone.strength * proximity_factor
        
        if supply_weight > demand_weight * 1.2:
            return "BEARISH"
        elif demand_weight > supply_weight * 1.2:
            return "BULLISH"
        else:
            return "NEUTRAL"
    
    def _is_price_near_zone(self, price: float, zone: SupplyDemandZone, tolerance: float) -> bool:
        """Check if price is near a zone"""
        return (zone.bottom - tolerance <= price <= zone.top + tolerance)
    
    def _assess_market_position(self, current_price: float, 
                               supply_zones: List[SupplyDemandZone],
                               demand_zones: List[SupplyDemandZone]) -> str:
        """Assess current market position relative to zones"""
        
        # Check if price is inside any zone
        for zone in supply_zones + demand_zones:
            if zone.bottom <= current_price <= zone.top:
                return f"INSIDE_{zone.zone_type.value.upper()}_ZONE"
        
        # Check proximity to zones
        nearby_supply = self._get_zones_near_price(supply_zones, current_price, 0.001)
        nearby_demand = self._get_zones_near_price(demand_zones, current_price, 0.001)
        
        if nearby_supply and nearby_demand:
            return "BETWEEN_ZONES"
        elif nearby_supply:
            return "NEAR_SUPPLY"
        elif nearby_demand:
            return "NEAR_DEMAND"
        else:
            return "NO_MAJOR_ZONES"
    
    def _identify_trading_opportunities(self, current_price: float,
                                      supply_zones: List[SupplyDemandZone],
                                      demand_zones: List[SupplyDemandZone]) -> List[Dict]:
        """Identify current trading opportunities"""
        opportunities = []
        
        # Check supply zones for short opportunities
        for zone in supply_zones:
            if (zone.quality in [ZoneQuality.HIGH, ZoneQuality.MEDIUM] and
                zone.bottom <= current_price <= zone.risk_reward_level):
                
                opportunities.append({
                    'type': 'SHORT',
                    'zone': zone,
                    'entry_level': zone.risk_reward_level,
                    'stop_loss': zone.mitigation_level,
                    'confidence': zone.strength,
                    'risk_reward': self._calculate_risk_reward(current_price, zone, 'SHORT')
                })
        
        # Check demand zones for long opportunities
        for zone in demand_zones:
            if (zone.quality in [ZoneQuality.HIGH, ZoneQuality.MEDIUM] and
                zone.risk_reward_level <= current_price <= zone.top):
                
                opportunities.append({
                    'type': 'LONG',
                    'zone': zone,
                    'entry_level': zone.risk_reward_level,
                    'stop_loss': zone.mitigation_level,
                    'confidence': zone.strength,
                    'risk_reward': self._calculate_risk_reward(current_price, zone, 'LONG')
                })
        
        # Sort by confidence and risk/reward
        opportunities.sort(key=lambda x: (x['confidence'], x['risk_reward']), reverse=True)
        
        return opportunities[:3]  # Top 3 opportunities
    
    def _calculate_risk_reward(self, current_price: float, zone: SupplyDemandZone, direction: str) -> float:
        """Calculate risk/reward ratio for a trade"""
        try:
            if direction == 'SHORT':
                entry = zone.risk_reward_level
                stop = zone.mitigation_level
                target = current_price - (zone.zone_size * 2)  # 2:1 target
                
                risk = abs(entry - stop)
                reward = abs(entry - target)
                
            else:  # LONG
                entry = zone.risk_reward_level
                stop = zone.mitigation_level
                target = current_price + (zone.zone_size * 2)  # 2:1 target
                
                risk = abs(entry - stop)
                reward = abs(target - entry)
            
            return reward / risk if risk > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward: {e}")
            return 0.0
    
    def get_zones_near_price(self, price: float, distance: float = 0.001) -> Dict[str, List[SupplyDemandZone]]:
        """Get supply and demand zones near specified price"""
        active_supply = [z for z in self.supply_zones if z.is_valid]
        active_demand = [z for z in self.demand_zones if z.is_valid]
        
        nearby_supply = self._get_zones_near_price(active_supply, price, distance)
        nearby_demand = self._get_zones_near_price(active_demand, price, distance)
        
        return {
            'supply_zones': nearby_supply,
            'demand_zones': nearby_demand
        }
    
    def get_strongest_zones(self, count: int = 5) -> Dict[str, List[SupplyDemandZone]]:
        """Get strongest supply and demand zones"""
        active_supply = [z for z in self.supply_zones if z.is_valid]
        active_demand = [z for z in self.demand_zones if z.is_valid]
        
        # Sort by strength
        active_supply.sort(key=lambda x: x.strength, reverse=True)
        active_demand.sort(key=lambda x: x.strength, reverse=True)
        
        return {
            'strongest_supply': active_supply[:count],
            'strongest_demand': active_demand[:count]
        }
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'timestamp': datetime.now(),
            'current_price': 0.0,
            'supply_zones': [],
            'demand_zones': [],
            'new_supply_zones': [],
            'new_demand_zones': [],
            'zone_metrics': {
                'total_supply_zones': 0,
                'total_demand_zones': 0,
                'avg_supply_strength': 0.0,
                'avg_demand_strength': 0.0,
                'zone_bias': 'NEUTRAL'
            },
            'confluence_analysis': {
                'confluence_zones': [],
                'highest_confluence': None,
                'total_confluence_zones': 0
            },
            'summary': {
                'market_bias': 'NEUTRAL',
                'market_position': 'NO_DATA',
                'active_zones_count': 0,
                'trading_opportunities': []
            }
        }

# Export main classes
__all__ = [
    'SupplyDemandAnalyzer',
    'SupplyDemandZone',
    'ZoneType',
    'ZoneStatus', 
    'ZoneQuality',
    'FormationType'
]