"""
Fair Value Gap (FVG) Detection and Analysis
Identifies imbalances and gap filling patterns for SMC trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FVGType(Enum):
    """Fair Value Gap types"""
    BULLISH = "bullish"
    BEARISH = "bearish"

class FVGStatus(Enum):
    """Fair Value Gap status"""
    OPEN = "open"
    PARTIAL_FILL = "partial_fill"
    FULL_FILL = "full_fill"
    EXPIRED = "expired"

class FVGQuality(Enum):
    """Fair Value Gap quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class FairValueGap:
    """Fair Value Gap data structure"""
    timestamp: datetime
    fvg_type: FVGType
    top: float
    bottom: float
    size: float
    creation_candles: Tuple[int, int, int]  # Indices of 3 candles
    volume: int
    strength: float  # 0.0 to 1.0
    quality: FVGQuality
    status: FVGStatus = FVGStatus.OPEN
    fill_percentage: float = 0.0
    fill_timestamp: Optional[datetime] = None
    rejection_count: int = 0  # Number of times price rejected from gap
    
    @property
    def mid_point(self) -> float:
        """Get gap midpoint"""
        return (self.top + self.bottom) / 2
    
    @property
    def age_hours(self) -> float:
        """Get age of FVG in hours"""
        return (datetime.now() - self.timestamp).total_seconds() / 3600
    
    @property
    def is_filled(self) -> bool:
        """Check if gap is completely filled"""
        return self.status == FVGStatus.FULL_FILL
    
    @property
    def remaining_size(self) -> float:
        """Get remaining unfilled gap size"""
        return self.size * (1 - self.fill_percentage)

class FVGAnalyzer:
    """
    Fair Value Gap detection and analysis
    """
    
    def __init__(self,
                 min_gap_size: float = 0.0002,  # 2 pips minimum
                 max_gap_age_hours: int = 72,   # 3 days max
                 partial_fill_threshold: float = 0.5,  # 50% fill threshold
                 quality_volume_threshold: float = 1.5):  # Volume multiplier for quality
        
        self.min_gap_size = min_gap_size
        self.max_gap_age_hours = max_gap_age_hours
        self.partial_fill_threshold = partial_fill_threshold
        self.quality_volume_threshold = quality_volume_threshold
        
        # Storage for identified FVGs
        self.fair_value_gaps = []
        
    def analyze_fair_value_gaps(self, data: pd.DataFrame) -> Dict:
        """
        Main function to analyze Fair Value Gaps
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            Dictionary with FVG analysis
        """
        if len(data) < 10:
            logger.warning("Insufficient data for FVG analysis")
            return self._empty_analysis()
        
        try:
            # Step 1: Identify new Fair Value Gaps
            new_fvgs = self._identify_fair_value_gaps(data)
            
            # Step 2: Update existing FVGs status
            self._update_fvg_status(data)
            
            # Step 3: Add new FVGs to collection
            for fvg in new_fvgs:
                if not self._is_duplicate_fvg(fvg):
                    self.fair_value_gaps.append(fvg)
            
            # Step 4: Clean up old/filled FVGs
            self._cleanup_old_fvgs()
            
            # Step 5: Calculate FVG metrics
            fvg_metrics = self._calculate_fvg_metrics(data)
            
            # Step 6: Assess current FVG zones
            current_zones = self._assess_current_fvg_zones(data)
            
            return {
                'timestamp': data.index[-1],
                'fair_value_gaps': self.fair_value_gaps,
                'new_fvgs': new_fvgs,
                'open_fvgs': [fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.OPEN],
                'partial_fvgs': [fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.PARTIAL_FILL],
                'filled_fvgs': [fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.FULL_FILL],
                'metrics': fvg_metrics,
                'current_zones': current_zones,
                'summary': self._generate_fvg_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in FVG analysis: {e}")
            return self._empty_analysis()
    
    def _identify_fair_value_gaps(self, data: pd.DataFrame) -> List[FairValueGap]:
        """Identify Fair Value Gaps in price action"""
        fvgs = []
        
        opens = data['Open'].values
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        volumes = data.get('Volume', pd.Series(0, index=data.index)).values
        times = data.index
        
        # Need at least 3 candles to form FVG
        for i in range(1, len(data) - 1):
            # Get three consecutive candles
            candle1_idx = i - 1  # Previous candle
            candle2_idx = i      # Current candle (gap candle)
            candle3_idx = i + 1  # Next candle
            
            # Check for bullish FVG
            bullish_fvg = self._check_bullish_fvg(
                candle1_idx, candle2_idx, candle3_idx,
                opens, highs, lows, closes, volumes, times
            )
            if bullish_fvg:
                fvgs.append(bullish_fvg)
            
            # Check for bearish FVG
            bearish_fvg = self._check_bearish_fvg(
                candle1_idx, candle2_idx, candle3_idx,
                opens, highs, lows, closes, volumes, times
            )
            if bearish_fvg:
                fvgs.append(bearish_fvg)
        
        logger.debug(f"Identified {len(fvgs)} new Fair Value Gaps")
        return fvgs
    
    def _check_bullish_fvg(self, idx1: int, idx2: int, idx3: int,
                          opens: np.ndarray, highs: np.ndarray, 
                          lows: np.ndarray, closes: np.ndarray,
                          volumes: np.ndarray, times: pd.DatetimeIndex) -> Optional[FairValueGap]:
        """Check for bullish Fair Value Gap pattern"""
        
        # Bullish FVG: Low of candle 3 > High of candle 1
        candle1_high = highs[idx1]
        candle3_low = lows[idx3]
        
        if candle3_low <= candle1_high:
            return None  # No gap
        
        # Calculate gap size
        gap_size = candle3_low - candle1_high
        
        if gap_size < self.min_gap_size:
            return None  # Gap too small
        
        # Additional validation: candle 2 should show momentum
        candle2_body = abs(closes[idx2] - opens[idx2])
        candle2_range = highs[idx2] - lows[idx2]
        
        # Body should be significant part of range
        if candle2_range > 0 and candle2_body / candle2_range < 0.4:
            return None  # Weak momentum candle
        
        # Calculate strength based on gap size and volume
        avg_volume = np.mean(volumes[max(0, idx1-10):idx3+1])
        volume_strength = min(volumes[idx2] / (avg_volume + 1), 3.0) / 3.0 if avg_volume > 0 else 0.5
        
        # Size strength (larger gaps = stronger)
        max_reasonable_gap = 0.002  # 20 pips
        size_strength = min(gap_size / max_reasonable_gap, 1.0)
        
        # Combined strength
        strength = (volume_strength + size_strength) / 2
        
        # Determine quality
        quality = self._assess_fvg_quality(gap_size, volume_strength, candle2_body)
        
        return FairValueGap(
            timestamp=times[idx2],
            fvg_type=FVGType.BULLISH,
            top=candle3_low,
            bottom=candle1_high,
            size=gap_size,
            creation_candles=(idx1, idx2, idx3),
            volume=int(volumes[idx2]),
            strength=strength,
            quality=quality
        )
    
    def _check_bearish_fvg(self, idx1: int, idx2: int, idx3: int,
                          opens: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, closes: np.ndarray,
                          volumes: np.ndarray, times: pd.DatetimeIndex) -> Optional[FairValueGap]:
        """Check for bearish Fair Value Gap pattern"""
        
        # Bearish FVG: High of candle 3 < Low of candle 1
        candle1_low = lows[idx1]
        candle3_high = highs[idx3]
        
        if candle3_high >= candle1_low:
            return None  # No gap
        
        # Calculate gap size
        gap_size = candle1_low - candle3_high
        
        if gap_size < self.min_gap_size:
            return None  # Gap too small
        
        # Additional validation
        candle2_body = abs(closes[idx2] - opens[idx2])
        candle2_range = highs[idx2] - lows[idx2]
        
        if candle2_range > 0 and candle2_body / candle2_range < 0.4:
            return None
        
        # Calculate strength
        avg_volume = np.mean(volumes[max(0, idx1-10):idx3+1])
        volume_strength = min(volumes[idx2] / (avg_volume + 1), 3.0) / 3.0 if avg_volume > 0 else 0.5
        
        max_reasonable_gap = 0.002
        size_strength = min(gap_size / max_reasonable_gap, 1.0)
        
        strength = (volume_strength + size_strength) / 2
        quality = self._assess_fvg_quality(gap_size, volume_strength, candle2_body)
        
        return FairValueGap(
            timestamp=times[idx2],
            fvg_type=FVGType.BEARISH,
            top=candle1_low,
            bottom=candle3_high,
            size=gap_size,
            creation_candles=(idx1, idx2, idx3),
            volume=int(volumes[idx2]),
            strength=strength,
            quality=quality
        )
    
    def _assess_fvg_quality(self, gap_size: float, volume_strength: float, body_size: float) -> FVGQuality:
        """Assess the quality of a Fair Value Gap"""
        
        # Size factor
        size_score = min(gap_size / 0.001, 1.0)  # Normalized to 10 pips
        
        # Volume factor
        volume_score = volume_strength
        
        # Body factor (momentum)
        body_score = min(body_size / 0.0005, 1.0)  # Normalized to 5 pips
        
        # Combined quality score
        quality_score = (size_score + volume_score + body_score) / 3
        
        if quality_score >= 0.7:
            return FVGQuality.HIGH
        elif quality_score >= 0.4:
            return FVGQuality.MEDIUM
        else:
            return FVGQuality.LOW
    
    def _update_fvg_status(self, data: pd.DataFrame):
        """Update status of existing Fair Value Gaps"""
        current_price = data['Close'].iloc[-1]
        current_high = data['High'].iloc[-1]
        current_low = data['Low'].iloc[-1]
        current_time = data.index[-1]
        
        for fvg in self.fair_value_gaps:
            if fvg.status == FVGStatus.FULL_FILL:
                continue  # Already filled
            
            # Check for gap filling
            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish gap filled when price goes back down into gap
                if current_low <= fvg.bottom:
                    # Full fill
                    fvg.status = FVGStatus.FULL_FILL
                    fvg.fill_percentage = 1.0
                    fvg.fill_timestamp = current_time
                elif current_low <= fvg.mid_point:
                    # Partial fill
                    fill_distance = fvg.top - current_low
                    fvg.fill_percentage = min(fill_distance / fvg.size, 1.0)
                    if fvg.fill_percentage >= self.partial_fill_threshold:
                        fvg.status = FVGStatus.PARTIAL_FILL
            
            else:  # BEARISH FVG
                # Bearish gap filled when price goes back up into gap
                if current_high >= fvg.top:
                    # Full fill
                    fvg.status = FVGStatus.FULL_FILL
                    fvg.fill_percentage = 1.0
                    fvg.fill_timestamp = current_time
                elif current_high >= fvg.mid_point:
                    # Partial fill
                    fill_distance = current_high - fvg.bottom
                    fvg.fill_percentage = min(fill_distance / fvg.size, 1.0)
                    if fvg.fill_percentage >= self.partial_fill_threshold:
                        fvg.status = FVGStatus.PARTIAL_FILL
            
            # Check for rejections
            if fvg.status == FVGStatus.OPEN:
                # Check if price approached but rejected from gap
                if fvg.fvg_type == FVGType.BULLISH:
                    if (current_low <= fvg.top * 1.0002 and  # Came close to gap
                        current_price > fvg.mid_point):  # But closed away from it
                        fvg.rejection_count += 1
                else:  # BEARISH FVG
                    if (current_high >= fvg.bottom * 0.9998 and  # Came close to gap
                        current_price < fvg.mid_point):  # But closed away from it
                        fvg.rejection_count += 1
            
            # Check for expiration
            if fvg.age_hours > self.max_gap_age_hours and fvg.status == FVGStatus.OPEN:
                fvg.status = FVGStatus.EXPIRED
    
    def _is_duplicate_fvg(self, new_fvg: FairValueGap) -> bool:
        """Check if FVG already exists"""
        for existing_fvg in self.fair_value_gaps:
            # Check if gaps overlap significantly
            overlap_top = min(new_fvg.top, existing_fvg.top)
            overlap_bottom = max(new_fvg.bottom, existing_fvg.bottom)
            
            if overlap_top > overlap_bottom:  # There is overlap
                overlap_size = overlap_top - overlap_bottom
                min_gap_size = min(new_fvg.size, existing_fvg.size)
                
                # If overlap is more than 70% of smaller gap, consider duplicate
                if overlap_size / min_gap_size > 0.7:
                    return True
        
        return False
    
    def _cleanup_old_fvgs(self):
        """Remove old or irrelevant Fair Value Gaps"""
        current_time = datetime.now()
        
        # Remove very old FVGs
        self.fair_value_gaps = [
            fvg for fvg in self.fair_value_gaps
            if (current_time - fvg.timestamp).total_seconds() < (self.max_gap_age_hours * 3600 * 2)  # Double the max age for cleanup
        ]
        
        # Keep only top 100 FVGs by strength and recency
        if len(self.fair_value_gaps) > 100:
            # Score combining strength and recency
            for fvg in self.fair_value_gaps:
                recency_score = max(0, 1 - (fvg.age_hours / (self.max_gap_age_hours * 2)))
                fvg._cleanup_score = (fvg.strength + recency_score) / 2
            
            self.fair_value_gaps.sort(key=lambda x: x._cleanup_score, reverse=True)
            self.fair_value_gaps = self.fair_value_gaps[:100]
    
    def _calculate_fvg_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate Fair Value Gap metrics"""
        if not self.fair_value_gaps:
            return {
                'total_fvgs': 0,
                'open_count': 0,
                'partial_fill_count': 0,
                'full_fill_count': 0,
                'expired_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_size': 0.0,
                'avg_strength': 0.0,
                'fill_rate': 0.0,
                'high_quality_count': 0
            }
        
        total_fvgs = len(self.fair_value_gaps)
        
        # Count by status
        open_count = len([fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.OPEN])
        partial_count = len([fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.PARTIAL_FILL])
        full_fill_count = len([fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.FULL_FILL])
        expired_count = len([fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.EXPIRED])
        
        # Count by type
        bullish_count = len([fvg for fvg in self.fair_value_gaps if fvg.fvg_type == FVGType.BULLISH])
        bearish_count = len([fvg for fvg in self.fair_value_gaps if fvg.fvg_type == FVGType.BEARISH])
        
        # Count by quality
        high_quality_count = len([fvg for fvg in self.fair_value_gaps if fvg.quality == FVGQuality.HIGH])
        
        # Calculate averages
        avg_size = sum(fvg.size for fvg in self.fair_value_gaps) / total_fvgs
        avg_strength = sum(fvg.strength for fvg in self.fair_value_gaps) / total_fvgs
        
        # Calculate fill rate
        filled_fvgs = full_fill_count + partial_count
        fill_rate = filled_fvgs / total_fvgs if total_fvgs > 0 else 0.0
        
        return {
            'total_fvgs': total_fvgs,
            'open_count': open_count,
            'partial_fill_count': partial_count,
            'full_fill_count': full_fill_count,
            'expired_count': expired_count,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'avg_size': avg_size,
            'avg_strength': avg_strength,
            'fill_rate': fill_rate,
            'high_quality_count': high_quality_count
        }
    
    def _assess_current_fvg_zones(self, data: pd.DataFrame) -> Dict:
        """Assess current FVG zones relative to price"""
        current_price = data['Close'].iloc[-1]
        
        # Find nearby FVGs
        nearby_fvgs = []
        for fvg in self.fair_value_gaps:
            if fvg.status not in [FVGStatus.OPEN, FVGStatus.PARTIAL_FILL]:
                continue
            
            # Check if current price is near the gap
            distance_to_gap = min(
                abs(current_price - fvg.top),
                abs(current_price - fvg.bottom),
                0 if fvg.bottom <= current_price <= fvg.top else float('inf')
            )
            
            if distance_to_gap <= 0.002:  # Within 20 pips
                nearby_fvgs.append({
                    'fvg': fvg,
                    'distance': distance_to_gap,
                    'in_gap': fvg.bottom <= current_price <= fvg.top,
                    'direction': 'above' if current_price < fvg.bottom else 'below' if current_price > fvg.top else 'inside'
                })
        
        # Sort by distance
        nearby_fvgs.sort(key=lambda x: x['distance'])
        
        # Find gaps above and below current price
        gaps_above = [g for g in nearby_fvgs if g['direction'] == 'above'][:3]
        gaps_below = [g for g in nearby_fvgs if g['direction'] == 'below'][:3]
        gaps_inside = [g for g in nearby_fvgs if g['direction'] == 'inside']
        
        return {
            'current_price': current_price,
            'nearby_fvgs': nearby_fvgs[:10],
            'gaps_above': gaps_above,
            'gaps_below': gaps_below,
            'inside_gap': len(gaps_inside) > 0,
            'nearest_gap': nearby_fvgs[0] if nearby_fvgs else None
        }
    
    def _generate_fvg_summary(self) -> Dict:
        """Generate summary of FVG state"""
        if not self.fair_value_gaps:
            return {
                'dominant_gap_type': None,
                'gap_activity': 'low',
                'fill_tendency': 'unknown',
                'overall_quality': 0.0
            }
        
        # Determine dominant gap type
        bullish_gaps = len([fvg for fvg in self.fair_value_gaps 
                           if fvg.fvg_type == FVGType.BULLISH and fvg.status == FVGStatus.OPEN])
        bearish_gaps = len([fvg for fvg in self.fair_value_gaps 
                           if fvg.fvg_type == FVGType.BEARISH and fvg.status == FVGStatus.OPEN])
        
        if bullish_gaps > bearish_gaps * 1.5:
            dominant_gap_type = FVGType.BULLISH
        elif bearish_gaps > bullish_gaps * 1.5:
            dominant_gap_type = FVGType.BEARISH
        else:
            dominant_gap_type = None
        
        # Assess gap activity
        recent_gaps = [fvg for fvg in self.fair_value_gaps if fvg.age_hours < 24]
        if len(recent_gaps) > 5:
            gap_activity = 'high'
        elif len(recent_gaps) > 2:
            gap_activity = 'medium'
        else:
            gap_activity = 'low'
        
        # Assess fill tendency
        filled_gaps = [fvg for fvg in self.fair_value_gaps if fvg.status == FVGStatus.FULL_FILL]
        total_testable = len([fvg for fvg in self.fair_value_gaps if fvg.age_hours > 12])  # Had time to be tested
        
        if total_testable > 0:
            fill_rate = len(filled_gaps) / total_testable
            if fill_rate > 0.7:
                fill_tendency = 'high_fill'
            elif fill_rate > 0.4:
                fill_tendency = 'medium_fill'
            else:
                fill_tendency = 'low_fill'
        else:
            fill_tendency = 'unknown'
        
        # Overall quality
        high_quality_gaps = [fvg for fvg in self.fair_value_gaps if fvg.quality == FVGQuality.HIGH]
        overall_quality = len(high_quality_gaps) / len(self.fair_value_gaps) if self.fair_value_gaps else 0.0
        
        return {
            'dominant_gap_type': dominant_gap_type,
            'gap_activity': gap_activity,
            'fill_tendency': fill_tendency,
            'overall_quality': overall_quality
        }
    
    def get_fvgs_near_price(self, price: float, distance: float = 0.001) -> List[FairValueGap]:
        """Get Fair Value Gaps near specified price"""
        nearby_fvgs = []
        
        for fvg in self.fair_value_gaps:
            if fvg.status not in [FVGStatus.OPEN, FVGStatus.PARTIAL_FILL]:
                continue
            
            # Calculate distance to gap
            if fvg.bottom <= price <= fvg.top:
                # Price is inside gap
                gap_distance = 0.0
            else:
                gap_distance = min(abs(price - fvg.top), abs(price - fvg.bottom))
            
            if gap_distance <= distance:
                nearby_fvgs.append(fvg)
        
        # Sort by distance and strength
        nearby_fvgs.sort(key=lambda x: (
            min(abs(price - x.top), abs(price - x.bottom)) if not (x.bottom <= price <= x.top) else 0,
            -x.strength
        ))
        
        return nearby_fvgs
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'timestamp': datetime.now(),
            'fair_value_gaps': [],
            'new_fvgs': [],
            'open_fvgs': [],
            'partial_fvgs': [],
            'filled_fvgs': [],
            'metrics': {
                'total_fvgs': 0,
                'open_count': 0,
                'partial_fill_count': 0,
                'full_fill_count': 0,
                'expired_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_size': 0.0,
                'avg_strength': 0.0,
                'fill_rate': 0.0,
                'high_quality_count': 0
            },
            'current_zones': {
                'current_price': 0.0,
                'nearby_fvgs': [],
                'gaps_above': [],
                'gaps_below': [],
                'inside_gap': False,
                'nearest_gap': None
            },
            'summary': {
                'dominant_gap_type': None,
                'gap_activity': 'low',
                'fill_tendency': 'unknown',
                'overall_quality': 0.0
            }
        }

# Export main classes
__all__ = [
    'FVGAnalyzer',
    'FairValueGap',
    'FVGType',
    'FVGStatus',
    'FVGQuality'
]