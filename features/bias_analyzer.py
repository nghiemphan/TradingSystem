"""
BIAS Analyzer - Smart Money Concepts Directional Bias Integration
Combines all SMC components to determine market directional bias
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

from features.smc_calculator import MarketStructureAnalyzer, TrendDirection
from features.order_blocks import OrderBlockAnalyzer, OrderBlockType, OrderBlockStatus
from features.fair_value_gaps import FVGAnalyzer, FVGType, FVGStatus
from features.liquidity_analyzer import LiquidityAnalyzer, LiquidityType
from features.premium_discount import PremiumDiscountAnalyzer, ZoneType
from features.supply_demand import SupplyDemandAnalyzer, ZoneType as SDZoneType

logger = logging.getLogger(__name__)

class BiasDirection(Enum):
    """BIAS direction enumeration"""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

class BiasStrength(Enum):
    """BIAS strength levels"""
    EXTREME = "extreme"      # 0.8 - 1.0
    STRONG = "strong"        # 0.6 - 0.8
    MODERATE = "moderate"    # 0.4 - 0.6
    WEAK = "weak"           # 0.2 - 0.4
    VERY_WEAK = "very_weak"  # 0.0 - 0.2

class BiasTimeframe(Enum):
    """BIAS timeframe context"""
    LONG_TERM = "long_term"      # D1, W1
    MEDIUM_TERM = "medium_term"  # H4, H1
    SHORT_TERM = "short_term"    # M15, M5, M1

class SessionType(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"

@dataclass
class BiasComponent:
    """Individual BIAS component from SMC analysis"""
    component_name: str
    direction: BiasDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    weight: float  # Component weight in overall BIAS
    timeframe: str
    timestamp: datetime
    details: Dict = None

@dataclass
class SessionBias:
    """Session-specific BIAS analysis"""
    session: SessionType
    direction: BiasDirection
    strength: float
    consistency: float  # How consistent across session
    volume_profile: float  # Volume-weighted direction
    price_action_score: float
    duration_hours: float

@dataclass
class MultiTimeframeBias:
    """Multi-timeframe BIAS alignment"""
    long_term_bias: BiasDirection
    medium_term_bias: BiasDirection
    short_term_bias: BiasDirection
    alignment_score: float  # 0.0 to 1.0 (higher = more aligned)
    dominant_timeframe: BiasTimeframe
    conflict_zones: List[str]  # Timeframes with conflicting bias

@dataclass
class OverallBias:
    """Complete BIAS assessment"""
    timestamp: datetime
    direction: BiasDirection
    strength: BiasStrength
    confidence: float  # 0.0 to 1.0
    score: float  # -1.0 to 1.0 (negative = bearish, positive = bullish)
    
    # Component contributions
    structural_bias: float
    institutional_bias: float
    liquidity_bias: float
    zone_bias: float
    session_bias: float
    
    # Multi-timeframe analysis
    mtf_bias: MultiTimeframeBias
    
    # Supporting data
    components: List[BiasComponent]
    session_analysis: List[SessionBias]
    
    # Trading context
    trading_recommendation: str
    risk_level: str
    invalidation_level: Optional[float] = None

class BiasAnalyzer:
    """
    Comprehensive BIAS analyzer integrating all SMC components
    """
    
    def __init__(self,
                 # Component weights (should sum to 1.0)
                 structural_weight: float = 0.25,
                 institutional_weight: float = 0.25,
                 liquidity_weight: float = 0.20,
                 zone_weight: float = 0.20,
                 session_weight: float = 0.10,
                 
                 # Timeframe analysis
                 analyze_mtf: bool = True,
                 mtf_alignment_threshold: float = 0.7,
                 
                 # Session analysis
                 session_analysis_enabled: bool = True,
                 session_history_hours: int = 24,
                 
                 # BIAS persistence
                 bias_memory_bars: int = 50):
        
        # Validate weights
        total_weight = (structural_weight + institutional_weight + 
                       liquidity_weight + zone_weight + session_weight)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight}")
        
        self.weights = {
            'structural': structural_weight,
            'institutional': institutional_weight,
            'liquidity': liquidity_weight,
            'zone': zone_weight,
            'session': session_weight
        }
        
        self.analyze_mtf = analyze_mtf
        self.mtf_alignment_threshold = mtf_alignment_threshold
        self.session_analysis_enabled = session_analysis_enabled
        self.session_history_hours = session_history_hours
        self.bias_memory_bars = bias_memory_bars
        
        # Initialize SMC analyzers
        self.ms_analyzer = MarketStructureAnalyzer()
        self.ob_analyzer = OrderBlockAnalyzer()
        self.fvg_analyzer = FVGAnalyzer()
        self.liq_analyzer = LiquidityAnalyzer()
        self.pd_analyzer = PremiumDiscountAnalyzer()
        self.sd_analyzer = SupplyDemandAnalyzer()
        
        # BIAS history for persistence analysis
        self.bias_history = []
        
    def analyze_bias(self, 
                    symbol: str,
                    timeframe_data: Dict[str, pd.DataFrame],
                    current_timeframe: str = "H1") -> OverallBias:
        """
        Main function to analyze comprehensive market BIAS
        
        Args:
            symbol: Trading symbol
            timeframe_data: Dictionary mapping timeframes to OHLCV data
            current_timeframe: Primary timeframe for analysis
            
        Returns:
            OverallBias object with complete analysis
        """
        if not timeframe_data or current_timeframe not in timeframe_data:
            logger.error(f"No data available for {current_timeframe}")
            return self._empty_bias_analysis()
        
        primary_data = timeframe_data[current_timeframe]
        if primary_data.empty:
            logger.error(f"Empty data for {current_timeframe}")
            return self._empty_bias_analysis()
        
        try:
            # Step 1: Analyze individual SMC components
            components = self._analyze_smc_components(symbol, primary_data, current_timeframe)
            
            # Step 2: Multi-timeframe analysis
            mtf_bias = None
            if self.analyze_mtf and len(timeframe_data) > 1:
                mtf_bias = self._analyze_multi_timeframe_bias(symbol, timeframe_data)
            
            # Step 3: Session analysis
            session_analysis = []
            if self.session_analysis_enabled:
                session_analysis = self._analyze_session_bias(primary_data)
            
            # Step 4: Calculate component-specific biases
            structural_bias = self._calculate_structural_bias(components)
            institutional_bias = self._calculate_institutional_bias(components)
            liquidity_bias = self._calculate_liquidity_bias(components)
            zone_bias = self._calculate_zone_bias(components)
            session_bias = self._calculate_session_bias(session_analysis)
            
            # Step 5: Calculate overall BIAS score
            overall_score = (
                structural_bias * self.weights['structural'] +
                institutional_bias * self.weights['institutional'] +
                liquidity_bias * self.weights['liquidity'] +
                zone_bias * self.weights['zone'] +
                session_bias * self.weights['session']
            )
            
            # Step 6: Determine direction and strength
            direction = self._score_to_direction(overall_score)
            strength = self._score_to_strength(abs(overall_score))
            confidence = self._calculate_confidence(components, mtf_bias)
            
            # Step 7: Generate trading recommendations
            recommendation, risk_level = self._generate_trading_recommendation(
                direction, strength, confidence, mtf_bias
            )
            
            # Step 8: Calculate invalidation level
            invalidation_level = self._calculate_invalidation_level(
                primary_data, direction, components
            )
            
            # Create overall BIAS result
            overall_bias = OverallBias(
                timestamp=primary_data.index[-1],
                direction=direction,
                strength=strength,
                confidence=confidence,
                score=overall_score,
                structural_bias=structural_bias,
                institutional_bias=institutional_bias,
                liquidity_bias=liquidity_bias,
                zone_bias=zone_bias,
                session_bias=session_bias,
                mtf_bias=mtf_bias,
                components=components,
                session_analysis=session_analysis,
                trading_recommendation=recommendation,
                risk_level=risk_level,
                invalidation_level=invalidation_level
            )
            
            # Add to history for persistence analysis
            self._update_bias_history(overall_bias)
            
            return overall_bias
            
        except Exception as e:
            logger.error(f"Error in BIAS analysis: {e}")
            return self._empty_bias_analysis()
    
    def _analyze_smc_components(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[BiasComponent]:
        """Analyze all SMC components for BIAS signals"""
        components = []
        current_time = data.index[-1]
        
        try:
            # 1. Market Structure Analysis
            ms_analysis = self.ms_analyzer.analyze_market_structure(data)
            if ms_analysis:
                trend = ms_analysis.get('trend', TrendDirection.SIDEWAYS)
                trend_strength = ms_analysis.get('trend_strength', 0.0)
                
                ms_direction = BiasDirection.NEUTRAL
                if trend == TrendDirection.BULLISH:
                    ms_direction = BiasDirection.BULLISH
                elif trend == TrendDirection.BEARISH:
                    ms_direction = BiasDirection.BEARISH
                
                components.append(BiasComponent(
                    component_name="market_structure",
                    direction=ms_direction,
                    strength=trend_strength,
                    confidence=ms_analysis.get('structure_quality', 0.5),
                    weight=0.4,  # High weight for trend
                    timeframe=timeframe,
                    timestamp=current_time,
                    details={'bos_count': len(ms_analysis.get('bos_events', [])),
                            'choch_count': len(ms_analysis.get('choch_events', []))}
                ))
            
            # 2. Order Blocks Analysis
            ob_analysis = self.ob_analyzer.analyze_order_blocks(data)
            if ob_analysis:
                metrics = ob_analysis.get('metrics', {})
                bullish_obs = metrics.get('bullish_count', 0)
                bearish_obs = metrics.get('bearish_count', 0)
                total_obs = bullish_obs + bearish_obs
                
                if total_obs > 0:
                    ob_bias_score = (bullish_obs - bearish_obs) / total_obs
                    ob_direction = self._score_to_direction(ob_bias_score)
                    
                    components.append(BiasComponent(
                        component_name="order_blocks",
                        direction=ob_direction,
                        strength=abs(ob_bias_score),
                        confidence=min(metrics.get('respect_rate', 0.5), 1.0),
                        weight=0.3,
                        timeframe=timeframe,
                        timestamp=current_time,
                        details={'bullish_obs': bullish_obs, 'bearish_obs': bearish_obs}
                    ))
            
            # 3. Fair Value Gaps Analysis
            fvg_analysis = self.fvg_analyzer.analyze_fair_value_gaps(data)
            if fvg_analysis:
                metrics = fvg_analysis.get('metrics', {})
                bullish_fvgs = metrics.get('bullish_count', 0)
                bearish_fvgs = metrics.get('bearish_count', 0)
                total_fvgs = bullish_fvgs + bearish_fvgs
                
                if total_fvgs > 0:
                    fvg_bias_score = (bullish_fvgs - bearish_fvgs) / total_fvgs
                    fvg_direction = self._score_to_direction(fvg_bias_score)
                    fill_rate = metrics.get('fill_rate', 0.5)
                    
                    components.append(BiasComponent(
                        component_name="fair_value_gaps",
                        direction=fvg_direction,
                        strength=abs(fvg_bias_score),
                        confidence=1.0 - fill_rate,  # Lower fill rate = higher confidence
                        weight=0.2,
                        timeframe=timeframe,
                        timestamp=current_time,
                        details={'bullish_fvgs': bullish_fvgs, 'bearish_fvgs': bearish_fvgs}
                    ))
            
            # 4. Liquidity Analysis
            liq_analysis = self.liq_analyzer.analyze_liquidity(data)
            if liq_analysis:
                metrics = liq_analysis.get('metrics', {})
                equal_highs = metrics.get('equal_highs_count', 0)
                equal_lows = metrics.get('equal_lows_count', 0)
                recent_sweeps = metrics.get('recent_sweeps', 0)
                
                # Liquidity above = bearish bias, liquidity below = bullish bias
                if equal_highs + equal_lows > 0:
                    liq_bias_score = (equal_lows - equal_highs) / (equal_highs + equal_lows)
                    liq_direction = self._score_to_direction(liq_bias_score)
                    
                    # Recent sweeps increase confidence
                    sweep_confidence = min(recent_sweeps * 0.2, 0.8)
                    
                    components.append(BiasComponent(
                        component_name="liquidity",
                        direction=liq_direction,
                        strength=abs(liq_bias_score),
                        confidence=0.5 + sweep_confidence,
                        weight=0.25,
                        timeframe=timeframe,
                        timestamp=current_time,
                        details={'equal_highs': equal_highs, 'equal_lows': equal_lows,
                                'recent_sweeps': recent_sweeps}
                    ))
            
            # 5. Premium/Discount Analysis
            pd_analysis = self.pd_analyzer.analyze_premium_discount(data)
            if pd_analysis and pd_analysis.get('current_assessment'):
                assessment = pd_analysis['current_assessment']
                zone_type = assessment.get('zone_type')
                zone_strength = assessment.get('zone_strength', 0.0)
                trading_bias = assessment.get('trading_bias', 'neutral')
                
                pd_direction = BiasDirection.NEUTRAL
                if trading_bias == 'bullish':
                    pd_direction = BiasDirection.BULLISH
                elif trading_bias == 'bearish':
                    pd_direction = BiasDirection.BEARISH
                
                components.append(BiasComponent(
                    component_name="premium_discount",
                    direction=pd_direction,
                    strength=zone_strength,
                    confidence=assessment.get('zone_strength', 0.5),
                    weight=0.2,
                    timeframe=timeframe,
                    timestamp=current_time,
                    details={'zone_type': zone_type.value if zone_type else 'none',
                            'trading_bias': trading_bias}
                ))
            
            # 6. Supply/Demand Analysis
            sd_analysis = self.sd_analyzer.analyze_supply_demand(data)
            if sd_analysis:
                metrics = sd_analysis.get('zone_metrics', {})
                summary = sd_analysis.get('summary', {})
                
                market_bias = summary.get('market_bias', 'NEUTRAL')
                supply_zones = metrics.get('total_supply_zones', 0)
                demand_zones = metrics.get('total_demand_zones', 0)
                
                sd_direction = BiasDirection.NEUTRAL
                if market_bias == 'BULLISH':
                    sd_direction = BiasDirection.BULLISH
                elif market_bias == 'BEARISH':
                    sd_direction = BiasDirection.BEARISH
                
                # Strength based on zone imbalance
                total_zones = supply_zones + demand_zones
                if total_zones > 0:
                    zone_imbalance = abs(supply_zones - demand_zones) / total_zones
                else:
                    zone_imbalance = 0.0
                
                components.append(BiasComponent(
                    component_name="supply_demand",
                    direction=sd_direction,
                    strength=zone_imbalance,
                    confidence=min(total_zones * 0.1, 1.0),  # More zones = higher confidence
                    weight=0.25,
                    timeframe=timeframe,
                    timestamp=current_time,
                    details={'supply_zones': supply_zones, 'demand_zones': demand_zones,
                            'market_bias': market_bias}
                ))
            
            logger.debug(f"Analyzed {len(components)} SMC components for BIAS")
            return components
            
        except Exception as e:
            logger.error(f"Error analyzing SMC components: {e}")
            return []
    
    def _analyze_multi_timeframe_bias(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> MultiTimeframeBias:
        """Analyze BIAS alignment across multiple timeframes"""
        try:
            # Categorize timeframes
            long_term_tfs = ['D1', 'W1']
            medium_term_tfs = ['H4', 'H1']
            short_term_tfs = ['M30', 'M15', 'M5', 'M1']
            
            # Get bias for each timeframe category
            long_term_bias = self._get_timeframe_category_bias(timeframe_data, long_term_tfs)
            medium_term_bias = self._get_timeframe_category_bias(timeframe_data, medium_term_tfs)
            short_term_bias = self._get_timeframe_category_bias(timeframe_data, short_term_tfs)
            
            # Calculate alignment score
            biases = [long_term_bias, medium_term_bias, short_term_bias]
            non_neutral_biases = [b for b in biases if b != BiasDirection.NEUTRAL]
            
            if len(non_neutral_biases) == 0:
                alignment_score = 0.0
                dominant_timeframe = BiasTimeframe.MEDIUM_TERM
            else:
                # Check how many agree
                bullish_count = sum(1 for b in non_neutral_biases if b == BiasDirection.BULLISH)
                bearish_count = sum(1 for b in non_neutral_biases if b == BiasDirection.BEARISH)
                
                max_agreement = max(bullish_count, bearish_count)
                alignment_score = max_agreement / len(non_neutral_biases)
                
                # Determine dominant timeframe
                if long_term_bias != BiasDirection.NEUTRAL:
                    dominant_timeframe = BiasTimeframe.LONG_TERM
                elif medium_term_bias != BiasDirection.NEUTRAL:
                    dominant_timeframe = BiasTimeframe.MEDIUM_TERM
                else:
                    dominant_timeframe = BiasTimeframe.SHORT_TERM
            
            # Identify conflict zones
            conflict_zones = []
            if long_term_bias != BiasDirection.NEUTRAL and medium_term_bias != BiasDirection.NEUTRAL:
                if long_term_bias != medium_term_bias:
                    conflict_zones.append("long_term_vs_medium_term")
            
            if medium_term_bias != BiasDirection.NEUTRAL and short_term_bias != BiasDirection.NEUTRAL:
                if medium_term_bias != short_term_bias:
                    conflict_zones.append("medium_term_vs_short_term")
            
            return MultiTimeframeBias(
                long_term_bias=long_term_bias,
                medium_term_bias=medium_term_bias,
                short_term_bias=short_term_bias,
                alignment_score=alignment_score,
                dominant_timeframe=dominant_timeframe,
                conflict_zones=conflict_zones
            )
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe BIAS analysis: {e}")
            return MultiTimeframeBias(
                long_term_bias=BiasDirection.NEUTRAL,
                medium_term_bias=BiasDirection.NEUTRAL,
                short_term_bias=BiasDirection.NEUTRAL,
                alignment_score=0.0,
                dominant_timeframe=BiasTimeframe.MEDIUM_TERM,
                conflict_zones=[]
            )
    
    def _get_timeframe_category_bias(self, timeframe_data: Dict[str, pd.DataFrame], 
                                   category_tfs: List[str]) -> BiasDirection:
        """Get aggregated bias for a timeframe category"""
        category_scores = []
        
        for tf in category_tfs:
            if tf in timeframe_data and not timeframe_data[tf].empty:
                # Quick bias calculation for this timeframe
                data = timeframe_data[tf]
                
                # Simple trend-based bias
                if len(data) >= 20:
                    recent_data = data.tail(20)
                    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
                    
                    # Normalize to -1 to 1 range
                    normalized_score = np.tanh(price_change * 100)  # Multiply by 100 to amplify
                    category_scores.append(normalized_score)
        
        if not category_scores:
            return BiasDirection.NEUTRAL
        
        # Average the scores
        avg_score = np.mean(category_scores)
        return self._score_to_direction(avg_score)
    
    def _analyze_session_bias(self, data: pd.DataFrame) -> List[SessionBias]:
        """Analyze BIAS by trading sessions"""
        session_analyses = []
        
        if len(data) < 24:  # Need at least 24 hours of data
            return session_analyses
        
        try:
            # Define session hours (UTC)
            sessions = {
                SessionType.ASIAN: (0, 9),
                SessionType.LONDON: (8, 17),
                SessionType.NEW_YORK: (13, 22),
                SessionType.OVERLAP: (13, 17)  # London-NY overlap
            }
            
            for session_type, (start_hour, end_hour) in sessions.items():
                session_data = self._filter_session_data(data, start_hour, end_hour)
                
                if len(session_data) > 0:
                    session_bias = self._calculate_session_bias_metrics(session_data, session_type)
                    session_analyses.append(session_bias)
            
            return session_analyses
            
        except Exception as e:
            logger.error(f"Error in session BIAS analysis: {e}")
            return []
    
    def _filter_session_data(self, data: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
        """Filter data by session hours"""
        # Get last 24 hours of data
        recent_data = data.tail(min(len(data), 24 * 12))  # Assuming 5-minute data
        
        # Filter by hour
        if start_hour <= end_hour:
            mask = (recent_data.index.hour >= start_hour) & (recent_data.index.hour < end_hour)
        else:  # Overnight session
            mask = (recent_data.index.hour >= start_hour) | (recent_data.index.hour < end_hour)
        
        return recent_data[mask]
    
    def _calculate_session_bias_metrics(self, session_data: pd.DataFrame, session_type: SessionType) -> SessionBias:
        """Calculate BIAS metrics for a specific session"""
        if session_data.empty:
            return SessionBias(
                session=session_type,
                direction=BiasDirection.NEUTRAL,
                strength=0.0,
                consistency=0.0,
                volume_profile=0.0,
                price_action_score=0.0,
                duration_hours=0.0
            )
        
        # Calculate price movement
        price_start = session_data['Open'].iloc[0]
        price_end = session_data['Close'].iloc[-1]
        price_change = (price_end - price_start) / price_start
        
        # Calculate volume-weighted direction
        volume_weighted_change = np.sum(
            (session_data['Close'] - session_data['Open']) * session_data.get('Volume', 1)
        ) / np.sum(session_data.get('Volume', 1))
        
        # Calculate consistency (how often price moved in same direction)
        bar_directions = np.sign(session_data['Close'] - session_data['Open'])
        dominant_direction = 1 if np.sum(bar_directions) > 0 else -1
        consistency = np.sum(bar_directions == dominant_direction) / len(bar_directions)
        
        # Price action score (range vs body ratio)
        ranges = session_data['High'] - session_data['Low']
        bodies = np.abs(session_data['Close'] - session_data['Open'])
        avg_body_to_range = np.mean(bodies / (ranges + 1e-8))
        
        # Duration
        duration_hours = (session_data.index[-1] - session_data.index[0]).total_seconds() / 3600
        
        return SessionBias(
            session=session_type,
            direction=self._score_to_direction(price_change),
            strength=abs(price_change) * 100,  # Convert to percentage-like
            consistency=consistency,
            volume_profile=volume_weighted_change,
            price_action_score=avg_body_to_range,
            duration_hours=duration_hours
        )
    
    def _calculate_structural_bias(self, components: List[BiasComponent]) -> float:
        """Calculate structural BIAS from market structure components"""
        structural_components = [c for c in components if c.component_name == "market_structure"]
        
        if not structural_components:
            return 0.0
        
        # Weight by confidence and strength
        weighted_scores = []
        for comp in structural_components:
            score = comp.direction.value * comp.strength * comp.confidence
            weighted_scores.append(score)
        
        return np.mean(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_institutional_bias(self, components: List[BiasComponent]) -> float:
        """Calculate institutional BIAS from order blocks and supply/demand"""
        institutional_components = [c for c in components 
                                  if c.component_name in ["order_blocks", "supply_demand"]]
        
        if not institutional_components:
            return 0.0
        
        weighted_scores = []
        for comp in institutional_components:
            score = comp.direction.value * comp.strength * comp.confidence * comp.weight
            weighted_scores.append(score)
        
        return np.sum(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_liquidity_bias(self, components: List[BiasComponent]) -> float:
        """Calculate liquidity BIAS from liquidity and FVG analysis"""
        liquidity_components = [c for c in components 
                               if c.component_name in ["liquidity", "fair_value_gaps"]]
        
        if not liquidity_components:
            return 0.0
        
        weighted_scores = []
        for comp in liquidity_components:
            score = comp.direction.value * comp.strength * comp.confidence * comp.weight
            weighted_scores.append(score)
        
        return np.sum(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_zone_bias(self, components: List[BiasComponent]) -> float:
        """Calculate zone BIAS from premium/discount analysis"""
        zone_components = [c for c in components if c.component_name == "premium_discount"]
        
        if not zone_components:
            return 0.0
        
        weighted_scores = []
        for comp in zone_components:
            score = comp.direction.value * comp.strength * comp.confidence
            weighted_scores.append(score)
        
        return np.mean(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_session_bias(self, session_analyses: List[SessionBias]) -> float:
        """Calculate session BIAS from session analysis"""
        if not session_analyses:
            return 0.0
        
        # Weight sessions by importance (overlap gets highest weight)
        session_weights = {
            SessionType.OVERLAP: 0.4,
            SessionType.LONDON: 0.3,
            SessionType.NEW_YORK: 0.2,
            SessionType.ASIAN: 0.1
        }
        
        weighted_scores = []
        for session in session_analyses:
            weight = session_weights.get(session.session, 0.1)
            score = session.direction.value * session.strength * session.consistency * weight
            weighted_scores.append(score)
        
        return np.sum(weighted_scores) if weighted_scores else 0.0
    
    def _score_to_direction(self, score: float) -> BiasDirection:
        """Convert numerical score to BIAS direction"""
        if score > 0.1:
            return BiasDirection.BULLISH
        elif score < -0.1:
            return BiasDirection.BEARISH
        else:
            return BiasDirection.NEUTRAL
    
    def _score_to_strength(self, abs_score: float) -> BiasStrength:
        """Convert absolute score to BIAS strength"""
        if abs_score >= 0.8:
            return BiasStrength.EXTREME
        elif abs_score >= 0.6:
            return BiasStrength.STRONG
        elif abs_score >= 0.4:
            return BiasStrength.MODERATE
        elif abs_score >= 0.2:
            return BiasStrength.WEAK
        else:
            return BiasStrength.VERY_WEAK
    
    def _calculate_confidence(self, components: List[BiasComponent], 
                            mtf_bias: Optional[MultiTimeframeBias]) -> float:
        """Calculate overall confidence in BIAS assessment"""
        if not components:
            return 0.0
        
        # Base confidence from component agreement
        directions = [c.direction for c in components if c.direction != BiasDirection.NEUTRAL]
        
        if not directions:
            return 0.0
        
        # Count agreement
        bullish_count = sum(1 for d in directions if d == BiasDirection.BULLISH)
        bearish_count = sum(1 for d in directions if d == BiasDirection.BEARISH)
        
        total_directional = bullish_count + bearish_count
        max_agreement = max(bullish_count, bearish_count)
        
        agreement_confidence = max_agreement / total_directional if total_directional > 0 else 0.0
        
        # Multi-timeframe alignment bonus
        mtf_bonus = 0.0
        if mtf_bias and mtf_bias.alignment_score > self.mtf_alignment_threshold:
            mtf_bonus = 0.2
        
        # Component quality bonus
        avg_component_confidence = np.mean([c.confidence for c in components])
        
        total_confidence = min(agreement_confidence + mtf_bonus + avg_component_confidence * 0.3, 1.0)
        
        return total_confidence
    
    def _generate_trading_recommendation(self, direction: BiasDirection, 
                                       strength: BiasStrength, 
                                       confidence: float,
                                       mtf_bias: Optional[MultiTimeframeBias]) -> Tuple[str, str]:
        """Generate trading recommendation and risk level"""
        
        # Base recommendation
        if direction == BiasDirection.NEUTRAL:
            return "STAY_SIDELINES", "HIGH"
        
        direction_str = "LONG" if direction == BiasDirection.BULLISH else "SHORT"
        
        # Strength and confidence modifiers
        if strength in [BiasStrength.EXTREME, BiasStrength.STRONG] and confidence >= 0.7:
            recommendation = f"STRONG_{direction_str}_BIAS"
            risk_level = "LOW"
        elif strength in [BiasStrength.MODERATE, BiasStrength.STRONG] and confidence >= 0.5:
            recommendation = f"MODERATE_{direction_str}_BIAS"
            risk_level = "MEDIUM"
        elif strength == BiasStrength.WEAK or confidence < 0.5:
            recommendation = f"WEAK_{direction_str}_BIAS"
            risk_level = "HIGH"
        else:
            recommendation = f"CAUTIOUS_{direction_str}_BIAS"
            risk_level = "HIGH"
        
        # Multi-timeframe conflict adjustment
        if mtf_bias and len(mtf_bias.conflict_zones) > 0:
            recommendation = recommendation.replace("STRONG", "CAUTIOUS")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
            elif risk_level == "MEDIUM":
                risk_level = "HIGH"
        
        return recommendation, risk_level
    
    def _calculate_invalidation_level(self, data: pd.DataFrame, 
                                    direction: BiasDirection,
                                    components: List[BiasComponent]) -> Optional[float]:
        """Calculate BIAS invalidation level"""
        if direction == BiasDirection.NEUTRAL:
            return None
        
        current_price = data['Close'].iloc[-1]
        
        # Get recent swing points for invalidation
        if len(data) >= 20:
            recent_data = data.tail(20)
            
            if direction == BiasDirection.BULLISH:
                # Bullish bias invalidated below recent significant low
                recent_low = recent_data['Low'].min()
                buffer = (current_price - recent_low) * 0.1  # 10% buffer
                invalidation_level = recent_low - buffer
            else:
                # Bearish bias invalidated above recent significant high
                recent_high = recent_data['High'].max()
                buffer = (recent_high - current_price) * 0.1  # 10% buffer
                invalidation_level = recent_high + buffer
            
            return invalidation_level
        
        return None
    
    def _update_bias_history(self, bias: OverallBias):
        """Update BIAS history for persistence analysis"""
        self.bias_history.append({
            'timestamp': bias.timestamp,
            'direction': bias.direction,
            'score': bias.score,
            'confidence': bias.confidence
        })
        
        # Keep only recent history
        if len(self.bias_history) > self.bias_memory_bars:
            self.bias_history = self.bias_history[-self.bias_memory_bars:]
    
    def get_bias_persistence(self) -> Dict:
        """Analyze BIAS persistence over time"""
        if len(self.bias_history) < 5:
            return {
                'persistence_score': 0.0,
                'avg_duration': 0.0,
                'direction_changes': 0,
                'confidence_trend': 'stable'
            }
        
        # Calculate direction changes
        directions = [b['direction'] for b in self.bias_history]
        direction_changes = sum(1 for i in range(1, len(directions)) 
                               if directions[i] != directions[i-1])
        
        # Calculate persistence score (lower changes = higher persistence)
        persistence_score = max(0, 1 - (direction_changes / len(directions)))
        
        # Average bias duration
        avg_duration = len(self.bias_history) / max(direction_changes, 1)
        
        # Confidence trend
        recent_confidences = [b['confidence'] for b in self.bias_history[-5:]]
        confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
        
        if confidence_trend > 0.05:
            confidence_trend_str = 'improving'
        elif confidence_trend < -0.05:
            confidence_trend_str = 'declining'
        else:
            confidence_trend_str = 'stable'
        
        return {
            'persistence_score': persistence_score,
            'avg_duration': avg_duration,
            'direction_changes': direction_changes,
            'confidence_trend': confidence_trend_str,
            'current_streak': self._calculate_current_streak()
        }
    
    def _calculate_current_streak(self) -> int:
        """Calculate current consecutive BIAS direction streak"""
        if not self.bias_history:
            return 0
        
        current_direction = self.bias_history[-1]['direction']
        streak = 1
        
        for i in range(len(self.bias_history) - 2, -1, -1):
            if self.bias_history[i]['direction'] == current_direction:
                streak += 1
            else:
                break
        
        return streak
    
    def get_component_breakdown(self, bias: OverallBias) -> Dict:
        """Get detailed breakdown of BIAS components"""
        breakdown = {
            'component_scores': {},
            'component_weights': self.weights.copy(),
            'component_contributions': {},
            'strongest_components': [],
            'weakest_components': [],
            'conflicting_components': []
        }
        
        # Component scores
        breakdown['component_scores'] = {
            'structural': bias.structural_bias,
            'institutional': bias.institutional_bias,
            'liquidity': bias.liquidity_bias,
            'zone': bias.zone_bias,
            'session': bias.session_bias
        }
        
        # Component contributions to final score
        for component, score in breakdown['component_scores'].items():
            weight = self.weights[component]
            contribution = score * weight
            breakdown['component_contributions'][component] = contribution
        
        # Sort components by absolute contribution
        sorted_contributions = sorted(
            breakdown['component_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        breakdown['strongest_components'] = [comp[0] for comp in sorted_contributions[:2]]
        breakdown['weakest_components'] = [comp[0] for comp in sorted_contributions[-2:]]
        
        # Find conflicting components
        positive_components = [comp for comp, score in breakdown['component_scores'].items() if score > 0.1]
        negative_components = [comp for comp, score in breakdown['component_scores'].items() if score < -0.1]
        
        if len(positive_components) > 0 and len(negative_components) > 0:
            breakdown['conflicting_components'] = positive_components + negative_components
        
        return breakdown
    
    def get_trading_zones(self, bias: OverallBias, data: pd.DataFrame) -> Dict:
        """Get key trading zones based on BIAS analysis"""
        current_price = data['Close'].iloc[-1]
        zones = {
            'entry_zones': [],
            'invalidation_zone': bias.invalidation_level,
            'target_zones': [],
            'risk_zones': []
        }
        
        if bias.direction == BiasDirection.NEUTRAL:
            return zones
        
        # Calculate key levels from recent price action
        recent_data = data.tail(50)
        highs = recent_data['High']
        lows = recent_data['Low']
        
        if bias.direction == BiasDirection.BULLISH:
            # Entry zones: Recent lows, support levels
            support_levels = []
            for i in range(10, len(recent_data) - 10):
                if (recent_data['Low'].iloc[i] == recent_data['Low'].iloc[i-10:i+10].min()):
                    support_levels.append(recent_data['Low'].iloc[i])
            
            zones['entry_zones'] = sorted(support_levels, reverse=True)[:3]
            
            # Target zones: Recent highs
            resistance_levels = []
            for i in range(10, len(recent_data) - 10):
                if (recent_data['High'].iloc[i] == recent_data['High'].iloc[i-10:i+10].max()):
                    resistance_levels.append(recent_data['High'].iloc[i])
            
            zones['target_zones'] = sorted(resistance_levels)[:3]
            
        else:  # BEARISH
            # Entry zones: Recent highs, resistance levels
            resistance_levels = []
            for i in range(10, len(recent_data) - 10):
                if (recent_data['High'].iloc[i] == recent_data['High'].iloc[i-10:i+10].max()):
                    resistance_levels.append(recent_data['High'].iloc[i])
            
            zones['entry_zones'] = sorted(resistance_levels)[:3]
            
            # Target zones: Recent lows
            support_levels = []
            for i in range(10, len(recent_data) - 10):
                if (recent_data['Low'].iloc[i] == recent_data['Low'].iloc[i-10:i+10].min()):
                    support_levels.append(recent_data['Low'].iloc[i])
            
            zones['target_zones'] = sorted(support_levels, reverse=True)[:3]
        
        return zones
    
    def _empty_bias_analysis(self) -> OverallBias:
        """Return empty BIAS analysis"""
        return OverallBias(
            timestamp=datetime.now(),
            direction=BiasDirection.NEUTRAL,
            strength=BiasStrength.VERY_WEAK,
            confidence=0.0,
            score=0.0,
            structural_bias=0.0,
            institutional_bias=0.0,
            liquidity_bias=0.0,
            zone_bias=0.0,
            session_bias=0.0,
            mtf_bias=MultiTimeframeBias(
                long_term_bias=BiasDirection.NEUTRAL,
                medium_term_bias=BiasDirection.NEUTRAL,
                short_term_bias=BiasDirection.NEUTRAL,
                alignment_score=0.0,
                dominant_timeframe=BiasTimeframe.MEDIUM_TERM,
                conflict_zones=[]
            ),
            components=[],
            session_analysis=[],
            trading_recommendation="NO_CLEAR_BIAS",
            risk_level="HIGH"
        )

# Utility functions for integration with other modules
def calculate_quick_bias(data: pd.DataFrame) -> Dict:
    """Quick BIAS calculation for single timeframe"""
    if data.empty or len(data) < 20:
        return {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.0}
    
    # Simple trend-based bias
    recent_data = data.tail(20)
    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
    
    # Momentum confirmation
    short_ma = recent_data['Close'].rolling(5).mean().iloc[-1]
    long_ma = recent_data['Close'].rolling(10).mean().iloc[-1]
    ma_signal = 1 if short_ma > long_ma else -1
    
    # Combine signals
    trend_signal = np.sign(price_change)
    combined_signal = (trend_signal + ma_signal) / 2
    
    direction = 'bullish' if combined_signal > 0.2 else 'bearish' if combined_signal < -0.2 else 'neutral'
    strength = min(abs(combined_signal), 1.0)
    confidence = min(abs(price_change) * 100, 1.0)  # Higher price change = higher confidence
    
    return {
        'direction': direction,
        'strength': strength,
        'confidence': confidence
    }

def get_session_bias_summary(bias_analyzer: BiasAnalyzer) -> Dict:
    """Get summary of session-based BIAS patterns"""
    if not bias_analyzer.bias_history:
        return {'error': 'No BIAS history available'}
    
    persistence = bias_analyzer.get_bias_persistence()
    
    return {
        'current_streak': persistence['current_streak'],
        'persistence_score': persistence['persistence_score'],
        'direction_changes': persistence['direction_changes'],
        'confidence_trend': persistence['confidence_trend'],
        'analysis_quality': 'high' if persistence['persistence_score'] > 0.7 else 'medium' if persistence['persistence_score'] > 0.4 else 'low'
    }

# Export main classes and functions
__all__ = [
    'BiasAnalyzer',
    'BiasDirection',
    'BiasStrength', 
    'BiasTimeframe',
    'SessionType',
    'BiasComponent',
    'SessionBias',
    'MultiTimeframeBias',
    'OverallBias',
    'calculate_quick_bias',
    'get_session_bias_summary'
]