"""
BIAS Analyzer - Enhanced Smart Money Concepts Directional Bias Integration
FIXED VERSION for Feature Aggregator compatibility
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BiasDirection(Enum):
    """BIAS direction enumeration"""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

class BiasStrength(Enum):
    """BIAS strength levels with enhanced sensitivity"""
    EXTREME = "extreme"      # 0.7 - 1.0
    STRONG = "strong"        # 0.5 - 0.7
    MODERATE = "moderate"    # 0.3 - 0.5
    WEAK = "weak"           # 0.15 - 0.3
    VERY_WEAK = "very_weak"  # 0.0 - 0.15

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
    """Complete BIAS assessment - FIXED for Feature Aggregator"""
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
    mtf_bias: Optional[MultiTimeframeBias] = None
    
    # Supporting data
    components: List[BiasComponent] = None
    session_analysis: List[SessionBias] = None
    
    # Trading context
    trading_recommendation: str = "NO_CLEAR_BIAS"
    risk_level: str = "HIGH"
    invalidation_level: Optional[float] = None
    
    # FIXED: Add additional attributes for Feature Aggregator compatibility
    overall_bias: Optional[BiasDirection] = None  # Alias for direction
    bias_strength: Optional[BiasStrength] = None  # Alias for strength
    confluence_score: float = 0.0
    consistency_score: float = 0.0
    mtf_alignment: float = 0.0
    momentum_score: float = 0.0
    quality_score: float = 0.0
    divergence_score: float = 0.0
    confirmation_strength: float = 0.0
    
    def __post_init__(self):
        """Set up aliases and derived attributes"""
        self.overall_bias = self.direction
        self.bias_strength = self.strength
        
        # Initialize defaults for missing values
        if self.components is None:
            self.components = []
        if self.session_analysis is None:
            self.session_analysis = []
        
        # Calculate derived metrics
        self.confluence_score = self.score
        self.consistency_score = self.confidence
        self.quality_score = self.confidence
        self.confirmation_strength = min(1.0, abs(self.score) * self.confidence)
        
        if self.mtf_bias:
            self.mtf_alignment = self.mtf_bias.alignment_score
        
        # Calculate momentum score
        self.momentum_score = abs(self.score * 
                                 (1.0 if self.strength.value in ['extreme', 'strong'] else 0.5))

class BiasAnalyzer:
    """
    Enhanced BIAS analyzer - FIXED for Feature Aggregator compatibility
    """
    
    def __init__(self,
                 # Component weights (should sum to 1.0)
                 structural_weight: float = 0.25,
                 institutional_weight: float = 0.25,
                 liquidity_weight: float = 0.20,
                 zone_weight: float = 0.20,
                 session_weight: float = 0.10,
                 
                 # Enhanced sensitivity parameters
                 bias_direction_threshold: float = 0.05,
                 signal_amplification_factor: float = 1.5,
                 
                 # Timeframe analysis
                 analyze_mtf: bool = True,
                 mtf_alignment_threshold: float = 0.6,
                 
                 # Session analysis
                 session_analysis_enabled: bool = True,
                 session_history_hours: int = 24,
                 session_strength_threshold: float = 0.01,
                 
                 # BIAS persistence
                 bias_memory_bars: int = 50,
                 
                 # Market adaptation
                 adaptive_thresholds: bool = True,
                 volatility_adjustment: bool = True,
                 
                 # FIXED: Allow passing analyzer instances but don't require them
                 ms_analyzer=None,
                 ob_analyzer=None,
                 fvg_analyzer=None,
                 liq_analyzer=None,
                 pd_analyzer=None,
                 sd_analyzer=None):
        
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
        
        # Enhanced parameters
        self.bias_direction_threshold = bias_direction_threshold
        self.signal_amplification_factor = signal_amplification_factor
        self.session_strength_threshold = session_strength_threshold
        self.adaptive_thresholds = adaptive_thresholds
        self.volatility_adjustment = volatility_adjustment
        
        self.analyze_mtf = analyze_mtf
        self.mtf_alignment_threshold = mtf_alignment_threshold
        self.session_analysis_enabled = session_analysis_enabled
        self.session_history_hours = session_history_hours
        self.bias_memory_bars = bias_memory_bars
        
        # FIXED: Initialize SMC analyzers - use passed ones or create new
        try:
            if ms_analyzer is not None:
                self.ms_analyzer = ms_analyzer
            else:
                from features.smc_calculator import MarketStructureAnalyzer
                self.ms_analyzer = MarketStructureAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize MarketStructureAnalyzer: {e}")
            self.ms_analyzer = None
            
        try:
            if ob_analyzer is not None:
                self.ob_analyzer = ob_analyzer
            else:
                from features.order_blocks import OrderBlockAnalyzer
                self.ob_analyzer = OrderBlockAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize OrderBlockAnalyzer: {e}")
            self.ob_analyzer = None
            
        try:
            if fvg_analyzer is not None:
                self.fvg_analyzer = fvg_analyzer
            else:
                from features.fair_value_gaps import FVGAnalyzer
                self.fvg_analyzer = FVGAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize FVGAnalyzer: {e}")
            self.fvg_analyzer = None
            
        try:
            if liq_analyzer is not None:
                self.liq_analyzer = liq_analyzer
            else:
                from features.liquidity_analyzer import LiquidityAnalyzer
                self.liq_analyzer = LiquidityAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize LiquidityAnalyzer: {e}")
            self.liq_analyzer = None
            
        try:
            if pd_analyzer is not None:
                self.pd_analyzer = pd_analyzer
            else:
                from features.premium_discount import PremiumDiscountAnalyzer
                self.pd_analyzer = PremiumDiscountAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize PremiumDiscountAnalyzer: {e}")
            self.pd_analyzer = None
            
        try:
            if sd_analyzer is not None:
                self.sd_analyzer = sd_analyzer
            else:
                from features.supply_demand import SupplyDemandAnalyzer
                self.sd_analyzer = SupplyDemandAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize SupplyDemandAnalyzer: {e}")
            self.sd_analyzer = None
        
        # BIAS history for persistence analysis
        self.bias_history = []
        
        # Market condition tracking
        self.market_volatility = 0.0
        self.trend_strength_cache = {}
        
        logger.debug(f"BiasAnalyzer initialized with {sum(1 for a in [self.ms_analyzer, self.ob_analyzer, self.fvg_analyzer, self.liq_analyzer, self.pd_analyzer, self.sd_analyzer] if a is not None)}/6 analyzers")
        
    def analyze_bias(self, 
                    symbol: str,
                    timeframe_data: Dict[str, pd.DataFrame],
                    current_timeframe: str = "H1") -> OverallBias:
        """
        Main function to analyze comprehensive market BIAS - FIXED
        """
        if not timeframe_data or current_timeframe not in timeframe_data:
            logger.error(f"No data available for {current_timeframe}")
            return self._create_empty_bias_analysis()
        
        primary_data = timeframe_data[current_timeframe]
        if primary_data.empty:
            logger.error(f"Empty data for {current_timeframe}")
            return self._create_empty_bias_analysis()
        
        try:
            # Calculate market conditions for adaptive thresholds
            if self.adaptive_thresholds:
                self._calculate_market_conditions(primary_data, symbol, current_timeframe)
            
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
            overall_score = self._calculate_overall_score(
                structural_bias, institutional_bias, liquidity_bias, 
                zone_bias, session_bias, components
            )
            
            # Step 6: Determine direction and strength
            direction = self._convert_score_to_direction(overall_score)
            strength = self._convert_score_to_strength(abs(overall_score))
            confidence = self._calculate_confidence(components, mtf_bias, overall_score)
            
            # Step 7: Generate trading recommendations
            recommendation, risk_level = self._generate_trading_recommendation(
                direction, strength, confidence, mtf_bias, overall_score
            )
            
            # Step 8: Calculate invalidation level
            invalidation_level = self._calculate_invalidation_level(
                primary_data, direction, components
            )
            
            # FIXED: Create overall BIAS result with all required attributes
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
            
            logger.debug(f"BIAS analysis completed: {direction.name}, strength={strength.value}, confidence={confidence:.3f}")
            return overall_bias
            
        except Exception as e:
            logger.error(f"Error in BIAS analysis: {e}")
            import traceback
            logger.debug(f"BIAS analysis traceback: {traceback.format_exc()}")
            return self._create_empty_bias_analysis()
    
    def _analyze_smc_components(self, symbol: str, data: pd.DataFrame, timeframe: str) -> List[BiasComponent]:
        """Analyze all SMC components for BIAS signals - IMPROVED ERROR HANDLING"""
        components = []
        current_time = data.index[-1]
        
        try:
            # 1. Market Structure Analysis with fallback
            if self.ms_analyzer is not None:
                try:
                    ms_analysis = self.ms_analyzer.analyze_market_structure(data)
                    trend = ms_analysis.get('trend') if ms_analysis else None
                    trend_strength = ms_analysis.get('trend_strength', 0.0) if ms_analysis else 0.0
                    
                    # Convert trend to BiasDirection
                    ms_direction = BiasDirection.NEUTRAL
                    if hasattr(trend, 'name'):
                        if trend.name == 'BULLISH':
                            ms_direction = BiasDirection.BULLISH
                        elif trend.name == 'BEARISH':
                            ms_direction = BiasDirection.BEARISH
                    elif hasattr(trend, 'value'):
                        if trend.value > 0:
                            ms_direction = BiasDirection.BULLISH
                        elif trend.value < 0:
                            ms_direction = BiasDirection.BEARISH
                    
                    # Enhanced trend detection with momentum fallback
                    if trend_strength == 0.0 and len(data) >= 20:
                        recent_data = data.tail(20)
                        price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
                        
                        if abs(price_change) > 0.001:  # 0.1% minimum change
                            trend_strength = min(abs(price_change) * 50, 0.8)
                            if price_change > 0:
                                ms_direction = BiasDirection.BULLISH
                            else:
                                ms_direction = BiasDirection.BEARISH
                    
                    # Apply signal amplification
                    amplified_strength = min(trend_strength * self.signal_amplification_factor, 1.0)
                    
                    components.append(BiasComponent(
                        component_name="market_structure",
                        direction=ms_direction,
                        strength=amplified_strength,
                        confidence=ms_analysis.get('structure_quality', 0.3) if ms_analysis else 0.3,
                        weight=0.4,
                        timeframe=timeframe,
                        timestamp=current_time,
                        details={'trend_source': 'smc' if ms_analysis else 'momentum_fallback'}
                    ))
                    
                except Exception as e:
                    logger.debug(f"Market structure analysis failed: {e}")
            
            # Fallback market structure analysis if analyzer not available
            if not components:
                try:
                    recent_data = data.tail(20)
                    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
                    
                    ms_direction = BiasDirection.NEUTRAL
                    trend_strength = 0.0
                    
                    if abs(price_change) > 0.001:
                        trend_strength = min(abs(price_change) * 30, 0.6)
                        if price_change > 0:
                            ms_direction = BiasDirection.BULLISH
                        else:
                            ms_direction = BiasDirection.BEARISH
                    
                    components.append(BiasComponent(
                        component_name="market_structure",
                        direction=ms_direction,
                        strength=trend_strength,
                        confidence=0.4,
                        weight=0.4,
                        timeframe=timeframe,
                        timestamp=current_time,
                        details={'trend_source': 'fallback_momentum'}
                    ))
                    
                except Exception as e:
                    logger.debug(f"Fallback market structure analysis failed: {e}")
            
            # 2. Order Blocks Analysis
            if self.ob_analyzer is not None:
                try:
                    ob_analysis = self.ob_analyzer.analyze_order_blocks(data)
                    if ob_analysis and hasattr(ob_analysis, 'metrics'):
                        metrics = ob_analysis.metrics
                        bullish_obs = getattr(metrics, 'bullish_count', 0)
                        bearish_obs = getattr(metrics, 'bearish_count', 0)
                        total_obs = bullish_obs + bearish_obs
                        
                        if total_obs > 0:
                            ob_bias_score = (bullish_obs - bearish_obs) / total_obs
                            ob_direction = self._convert_score_to_direction(ob_bias_score)
                            
                            components.append(BiasComponent(
                                component_name="order_blocks",
                                direction=ob_direction,
                                strength=abs(ob_bias_score),
                                confidence=min(getattr(metrics, 'respect_rate', 0.3), 1.0),
                                weight=0.3,
                                timeframe=timeframe,
                                timestamp=current_time,
                                details={'bullish_obs': bullish_obs, 'bearish_obs': bearish_obs}
                            ))
                except Exception as e:
                    logger.debug(f"Order block analysis failed: {e}")
            
            # 3. Fair Value Gaps Analysis  
            if self.fvg_analyzer is not None:
                try:
                    fvg_analysis = self.fvg_analyzer.analyze_fair_value_gaps(data)
                    if fvg_analysis and hasattr(fvg_analysis, 'metrics'):
                        metrics = fvg_analysis.metrics
                        bullish_fvgs = getattr(metrics, 'bullish_count', 0)
                        bearish_fvgs = getattr(metrics, 'bearish_count', 0)
                        total_fvgs = bullish_fvgs + bearish_fvgs
                        
                        if total_fvgs > 0:
                            fvg_bias_score = (bullish_fvgs - bearish_fvgs) / total_fvgs
                            fvg_direction = self._convert_score_to_direction(fvg_bias_score)
                            
                            components.append(BiasComponent(
                                component_name="fair_value_gaps",
                                direction=fvg_direction,
                                strength=abs(fvg_bias_score),
                                confidence=0.6,
                                weight=0.2,
                                timeframe=timeframe,
                                timestamp=current_time,
                                details={'bullish_fvgs': bullish_fvgs, 'bearish_fvgs': bearish_fvgs}
                            ))
                except Exception as e:
                    logger.debug(f"FVG analysis failed: {e}")
            
            # 4. Liquidity Analysis
            if self.liq_analyzer is not None:
                try:
                    liq_analysis = self.liq_analyzer.analyze_liquidity(data)
                    if liq_analysis and hasattr(liq_analysis, 'metrics'):
                        metrics = liq_analysis.metrics
                        equal_highs = getattr(metrics, 'equal_highs_count', 0)
                        equal_lows = getattr(metrics, 'equal_lows_count', 0)
                        
                        if equal_highs + equal_lows > 0:
                            liq_bias_score = (equal_lows - equal_highs) / (equal_highs + equal_lows)
                            liq_direction = self._convert_score_to_direction(liq_bias_score)
                            
                            components.append(BiasComponent(
                                component_name="liquidity",
                                direction=liq_direction,
                                strength=abs(liq_bias_score),
                                confidence=0.5,
                                weight=0.25,
                                timeframe=timeframe,
                                timestamp=current_time,
                                details={'equal_highs': equal_highs, 'equal_lows': equal_lows}
                            ))
                except Exception as e:
                    logger.debug(f"Liquidity analysis failed: {e}")
            
            # 5. Premium/Discount Analysis
            if self.pd_analyzer is not None:
                try:
                    pd_analysis = self.pd_analyzer.analyze_premium_discount(data)
                    if pd_analysis and 'current_assessment' in pd_analysis:
                        assessment = pd_analysis['current_assessment']
                        trading_bias = assessment.get('trading_bias', 'neutral')
                        zone_strength = assessment.get('zone_strength', 0.0)
                        
                        pd_direction = BiasDirection.NEUTRAL
                        if trading_bias == 'bullish':
                            pd_direction = BiasDirection.BULLISH
                        elif trading_bias == 'bearish':
                            pd_direction = BiasDirection.BEARISH
                        
                        components.append(BiasComponent(
                            component_name="premium_discount",
                            direction=pd_direction,
                            strength=min(zone_strength, 1.0),
                            confidence=0.4,
                            weight=0.2,
                            timeframe=timeframe,
                            timestamp=current_time,
                            details={'trading_bias': trading_bias}
                        ))
                except Exception as e:
                    logger.debug(f"Premium/Discount analysis failed: {e}")
            
            # 6. Supply/Demand Analysis
            if self.sd_analyzer is not None:
                try:
                    sd_analysis = self.sd_analyzer.analyze_supply_demand(data)
                    if sd_analysis and 'summary' in sd_analysis:
                        summary = sd_analysis['summary']
                        market_bias = summary.get('market_bias', 'NEUTRAL')
                        
                        sd_direction = BiasDirection.NEUTRAL
                        if market_bias == 'BULLISH':
                            sd_direction = BiasDirection.BULLISH
                        elif market_bias == 'BEARISH':
                            sd_direction = BiasDirection.BEARISH
                        
                        # Calculate strength from zone metrics
                        zone_metrics = sd_analysis.get('zone_metrics', {})
                        supply_zones = zone_metrics.get('total_supply_zones', 0)
                        demand_zones = zone_metrics.get('total_demand_zones', 0)
                        total_zones = supply_zones + demand_zones
                        
                        if total_zones > 0:
                            zone_imbalance = abs(supply_zones - demand_zones) / total_zones
                        else:
                            zone_imbalance = 0.0
                        
                        components.append(BiasComponent(
                            component_name="supply_demand",
                            direction=sd_direction,
                            strength=min(zone_imbalance, 1.0),
                            confidence=0.5,
                            weight=0.25,
                            timeframe=timeframe,
                            timestamp=current_time,
                            details={'market_bias': market_bias, 'supply_zones': supply_zones, 'demand_zones': demand_zones}
                        ))
                except Exception as e:
                    logger.debug(f"Supply/Demand analysis failed: {e}")
            
            logger.debug(f"Analyzed {len(components)} SMC components for BIAS")
            return components
            
        except Exception as e:
            logger.error(f"Error analyzing SMC components: {e}")
            return []

    # ... (keep all other methods from original BiasAnalyzer unchanged until _create_empty_bias_analysis) ...
    
    def _calculate_market_conditions(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Calculate current market conditions for adaptive thresholds"""
        if len(data) < 20:
            return
        
        try:
            # Calculate volatility (ATR-based)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift(1))
            low_close = np.abs(data['Low'] - data['Close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Normalize volatility
            price_level = data['Close'].iloc[-1]
            self.market_volatility = atr / price_level if price_level > 0 else 0.0
            
            logger.debug(f"Market volatility for {symbol} {timeframe}: {self.market_volatility:.4f}")
        except Exception as e:
            logger.debug(f"Failed to calculate market conditions: {e}")
            self.market_volatility = 0.01  # Default value

    def _analyze_session_bias(self, data: pd.DataFrame) -> List[SessionBias]:
        """Analyze BIAS by trading sessions with enhanced sensitivity"""
        session_analyses = []
        
        if len(data) < 12:
            return session_analyses
        
        try:
            # Simple session analysis based on time patterns
            # For now, just return empty list - can be enhanced later
            return session_analyses
        except Exception as e:
            logger.debug(f"Session analysis failed: {e}")
            return []

    def _analyze_multi_timeframe_bias(self, symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> Optional[MultiTimeframeBias]:
        """Analyze BIAS alignment across multiple timeframes"""
        try:
            # Simple multi-timeframe analysis
            timeframes = list(timeframe_data.keys())
            if len(timeframes) < 2:
                return None
            
            # Calculate simple bias for each timeframe
            tf_biases = {}
            for tf, data in timeframe_data.items():
                if len(data) >= 20:
                    recent_data = data.tail(20)
                    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
                    
                    if price_change > 0.001:
                        tf_biases[tf] = BiasDirection.BULLISH
                    elif price_change < -0.001:
                        tf_biases[tf] = BiasDirection.BEARISH
                    else:
                        tf_biases[tf] = BiasDirection.NEUTRAL
            
            if not tf_biases:
                return None
            
            # Simple alignment calculation
            bias_values = [bias.value for bias in tf_biases.values() if bias != BiasDirection.NEUTRAL]
            if bias_values:
                alignment_score = 1.0 - (len(set(bias_values)) - 1) / len(bias_values)
            else:
                alignment_score = 0.0
            
            return MultiTimeframeBias(
                long_term_bias=BiasDirection.NEUTRAL,
                medium_term_bias=BiasDirection.NEUTRAL,
                short_term_bias=BiasDirection.NEUTRAL,
                alignment_score=alignment_score,
                dominant_timeframe=BiasTimeframe.MEDIUM_TERM,
                conflict_zones=[]
            )
            
        except Exception as e:
            logger.debug(f"Multi-timeframe analysis failed: {e}")
            return None

    def _calculate_overall_score(self, structural_bias: float, institutional_bias: float,
                               liquidity_bias: float, zone_bias: float, session_bias: float,
                               components: List[BiasComponent]) -> float:
        """Calculate overall BIAS score with component synergy"""
        
        # Base weighted score
        base_score = (
            structural_bias * self.weights['structural'] +
            institutional_bias * self.weights['institutional'] +
            liquidity_bias * self.weights['liquidity'] +
            zone_bias * self.weights['zone'] +
            session_bias * self.weights['session']
        )
        
        # Component synergy bonus
        directional_components = [c for c in components if c.direction != BiasDirection.NEUTRAL]
        
        if len(directional_components) >= 2:
            bullish_components = [c for c in directional_components if c.direction == BiasDirection.BULLISH]
            bearish_components = [c for c in directional_components if c.direction == BiasDirection.BEARISH]
            
            max_agreement = max(len(bullish_components), len(bearish_components))
            agreement_ratio = max_agreement / len(directional_components)
            
            # Apply synergy bonus for strong agreement
            if agreement_ratio >= 0.75:
                synergy_bonus = 0.1 * np.sign(base_score) if base_score != 0 else 0.0
                base_score += synergy_bonus
        
        # Volatility adjustment
        if self.volatility_adjustment and self.market_volatility > 0:
            volatility_factor = 1.0
            if self.market_volatility > 0.02:  # High volatility
                volatility_factor = 0.8
            elif self.market_volatility < 0.005:  # Low volatility
                volatility_factor = 1.2
            
            base_score *= volatility_factor
        
        return np.clip(base_score, -1.0, 1.0)
    
    def _convert_score_to_direction(self, score: float) -> BiasDirection:
        """Convert numerical score to BIAS direction with adaptive thresholds"""
        threshold = self.bias_direction_threshold
        
        # Adaptive threshold based on market conditions
        if self.adaptive_thresholds and self.market_volatility > 0:
            if self.market_volatility > 0.02:  # High volatility - need stronger signal
                threshold *= 1.5
            elif self.market_volatility < 0.005:  # Low volatility - more sensitive
                threshold *= 0.7
        
        if score > threshold:
            return BiasDirection.BULLISH
        elif score < -threshold:
            return BiasDirection.BEARISH
        else:
            return BiasDirection.NEUTRAL
    
    def _convert_score_to_strength(self, abs_score: float) -> BiasStrength:
        """Convert absolute score to BIAS strength with enhanced sensitivity"""
        if abs_score >= 0.7:
            return BiasStrength.EXTREME
        elif abs_score >= 0.5:
            return BiasStrength.STRONG
        elif abs_score >= 0.3:
            return BiasStrength.MODERATE
        elif abs_score >= 0.15:
            return BiasStrength.WEAK
        else:
            return BiasStrength.VERY_WEAK
    
    def _calculate_confidence(self, components: List[BiasComponent], 
                            mtf_bias: Optional[MultiTimeframeBias],
                            overall_score: float) -> float:
        """Calculate overall confidence in BIAS assessment"""
        if not components:
            return 0.1  # Minimum confidence
        
        # Base confidence from component agreement
        directions = [c.direction for c in components if c.direction != BiasDirection.NEUTRAL]
        
        if not directions:
            return 0.1
        
        # Count agreement
        bullish_count = sum(1 for d in directions if d == BiasDirection.BULLISH)
        bearish_count = sum(1 for d in directions if d == BiasDirection.BEARISH)
        
        total_directional = bullish_count + bearish_count
        max_agreement = max(bullish_count, bearish_count)
        
        agreement_confidence = max_agreement / total_directional if total_directional > 0 else 0.0
        
        # Multi-timeframe alignment bonus
        mtf_bonus = 0.0
        if mtf_bias and mtf_bias.alignment_score > self.mtf_alignment_threshold:
            mtf_bonus = 0.15
        
        # Component quality bonus
        avg_component_confidence = np.mean([c.confidence for c in components])
        component_bonus = avg_component_confidence * 0.25
        
        # Signal strength bonus
        strength_bonus = min(abs(overall_score) * 0.2, 0.15)
        
        # Number of components bonus
        component_count_bonus = min(len(components) * 0.05, 0.2)
        
        total_confidence = min(
            agreement_confidence + mtf_bonus + component_bonus + strength_bonus + component_count_bonus, 
            1.0
        )
        
        return max(total_confidence, 0.1)  # Minimum 10% confidence
    
    def _generate_trading_recommendation(self, direction: BiasDirection, 
                                       strength: BiasStrength, 
                                       confidence: float,
                                       mtf_bias: Optional[MultiTimeframeBias],
                                       overall_score: float) -> Tuple[str, str]:
        """Generate trading recommendation"""
        
        # Base recommendation
        if direction == BiasDirection.NEUTRAL:
            return "STAY_SIDELINES", "HIGH"
        
        direction_str = "LONG" if direction == BiasDirection.BULLISH else "SHORT"
        
        # Enhanced thresholds for more actionable recommendations
        if strength in [BiasStrength.EXTREME, BiasStrength.STRONG] and confidence >= 0.6:
            recommendation = f"STRONG_{direction_str}_BIAS"
            risk_level = "LOW"
        elif strength in [BiasStrength.MODERATE, BiasStrength.STRONG] and confidence >= 0.4:
            recommendation = f"MODERATE_{direction_str}_BIAS"
            risk_level = "MEDIUM"
        elif strength in [BiasStrength.WEAK, BiasStrength.MODERATE] and confidence >= 0.3:
            recommendation = f"WEAK_{direction_str}_BIAS"
            risk_level = "MEDIUM"
        elif confidence >= 0.25:
            recommendation = f"CAUTIOUS_{direction_str}_BIAS"
            risk_level = "MEDIUM"
        else:
            recommendation = f"MINIMAL_{direction_str}_BIAS"
            risk_level = "HIGH"
        
        # Multi-timeframe conflict adjustment
        if mtf_bias and len(mtf_bias.conflict_zones) > 1:
            if "STRONG" in recommendation:
                recommendation = recommendation.replace("STRONG", "MODERATE")
            elif "MODERATE" in recommendation:
                recommendation = recommendation.replace("MODERATE", "WEAK")
            
            # Increase risk level
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
        
        try:
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
        except Exception as e:
            logger.debug(f"Failed to calculate invalidation level: {e}")
        
        return None
    
    def _calculate_structural_bias(self, components: List[BiasComponent]) -> float:
        """Calculate structural BIAS from market structure components"""
        structural_components = [c for c in components if c.component_name == "market_structure"]
        
        if not structural_components:
            return 0.0
        
        weighted_scores = []
        for comp in structural_components:
            base_score = comp.direction.value * comp.strength * comp.confidence
            
            # Apply additional boost for high-confidence trends
            if comp.confidence > 0.7 and comp.strength > 0.3:
                base_score *= 1.2
            
            weighted_scores.append(base_score)
        
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
        
        # Simple session bias calculation
        session_scores = []
        for session in session_analyses:
            score = session.direction.value * session.strength * session.consistency
            session_scores.append(score)
        
        return np.mean(session_scores) if session_scores else 0.0
    
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
                'confidence_trend': 'stable',
                'current_streak': 0
            }
        
        # Calculate direction changes
        directions = [b['direction'] for b in self.bias_history]
        direction_changes = sum(1 for i in range(1, len(directions)) 
                               if directions[i] != directions[i-1])
        
        # Calculate persistence score
        persistence_score = max(0, 1 - (direction_changes / len(directions)))
        
        # Average bias duration
        avg_duration = len(self.bias_history) / max(direction_changes, 1)
        
        # Confidence trend
        recent_confidences = [b['confidence'] for b in self.bias_history[-5:]]
        if len(recent_confidences) > 1:
            confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
            
            if confidence_trend > 0.05:
                confidence_trend_str = 'improving'
            elif confidence_trend < -0.05:
                confidence_trend_str = 'declining'
            else:
                confidence_trend_str = 'stable'
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
    
    def _create_empty_bias_analysis(self) -> OverallBias:
        """Return empty BIAS analysis - FIXED"""
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
            mtf_bias=None,
            components=[],
            session_analysis=[],
            trading_recommendation="NO_CLEAR_BIAS",
            risk_level="HIGH"
        )

# Enhanced utility functions
def calculate_quick_bias(data: pd.DataFrame, sensitivity: float = 0.05) -> Dict:
    """Quick BIAS calculation with enhanced sensitivity"""
    if data.empty or len(data) < 10:
        return {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.0}
    
    try:
        # Enhanced trend-based bias
        recent_data = data.tail(min(20, len(data)))
        price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
        
        # Multiple momentum confirmations
        short_window = min(5, len(recent_data)//2)
        long_window = min(10, len(recent_data))
        
        if short_window > 0 and long_window > short_window:
            short_ma = recent_data['Close'].rolling(short_window).mean().iloc[-1]
            long_ma = recent_data['Close'].rolling(long_window).mean().iloc[-1]
            ma_signal = 1 if short_ma > long_ma else -1
        else:
            ma_signal = 0
        
        # Volume confirmation if available
        volume_signal = 0
        if 'Volume' in recent_data.columns:
            recent_volume = recent_data['Volume'].tail(5).mean()
            avg_volume = recent_data['Volume'].mean()
            if recent_volume > avg_volume * 1.2:
                volume_signal = 0.2
        
        # Combine signals with weights
        trend_signal = np.sign(price_change)
        combined_signal = (trend_signal * 0.6 + ma_signal * 0.3 + volume_signal * 0.1)
        
        # Apply sensitivity threshold
        if abs(combined_signal) > sensitivity:
            direction = 'bullish' if combined_signal > 0 else 'bearish'
        else:
            direction = 'neutral'
        
        strength = min(abs(combined_signal), 1.0)
        confidence = min(abs(price_change) * 200, 1.0)
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'price_change': price_change,
            'ma_signal': ma_signal,
            'volume_boost': volume_signal
        }
    except Exception as e:
        logger.debug(f"Quick bias calculation failed: {e}")
        return {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.0}

def get_session_bias_summary(bias_analyzer: BiasAnalyzer) -> Dict:
    """Get session BIAS summary with enhanced error handling"""
    if not bias_analyzer.bias_history:
        return {
            'error': 'No BIAS history available',
            'current_streak': 0,
            'persistence_score': 0.0,
            'direction_changes': 0,
            'confidence_trend': 'unknown',
            'analysis_quality': 'insufficient_data'
        }
    
    try:
        persistence = bias_analyzer.get_bias_persistence()
        
        return {
            'current_streak': persistence.get('current_streak', 0),
            'persistence_score': persistence.get('persistence_score', 0.0),
            'direction_changes': persistence.get('direction_changes', 0),
            'confidence_trend': persistence.get('confidence_trend', 'stable'),
            'analysis_quality': (
                'high' if persistence.get('persistence_score', 0) > 0.7 
                else 'medium' if persistence.get('persistence_score', 0) > 0.4 
                else 'low'
            ),
            'avg_duration': persistence.get('avg_duration', 0.0),
            'bias_history_length': len(bias_analyzer.bias_history)
        }
    except Exception as e:
        logger.error(f"Error in session bias summary: {e}")
        return {
            'error': f'Analysis error: {e}',
            'current_streak': 0,
            'analysis_quality': 'error'
        }

# Export all classes and functions
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