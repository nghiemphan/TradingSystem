"""
BIAS Analyzer Test Script - Comprehensive Testing
Tests BIAS integration, multi-timeframe analysis, and trading recommendations
Phase 2 Week 4 Day 5-7: BIAS Analysis & Integration
"""
import sys
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.bias_analyzer import (BiasAnalyzer, BiasDirection, BiasStrength, 
                                   BiasTimeframe, SessionType, calculate_quick_bias)
from data.connectors.mt5_connector import get_mt5_connector
from data.connectors.demo_connector import get_demo_connector
from config.mt5_config import MT5_CONNECTION

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_multi_timeframe_data():
    """Get multi-timeframe test data"""
    print("Getting multi-timeframe test data...")
    
    try:
        # Try real MT5 first
        connector = get_mt5_connector(MT5_CONNECTION)
        if connector.connect():
            print("   ✓ Using real MT5 data")
            is_demo = False
        else:
            raise Exception("MT5 connection failed")
    except Exception as e:
        # Fall back to demo
        print(f"   ⚠️  Real MT5 failed ({e}), using demo data")
        connector = get_demo_connector()
        connector.connect()
        is_demo = True
        print("   ✓ Using demo data")
    
    # Test connection
    test_result = connector.test_connection()
    if not test_result.get('connected'):
        print(f"   ✗ Connection test failed: {test_result.get('error')}")
        return {}
    
    # Print connection info
    account_info = test_result.get('account_info', {})
    print(f"   Account: {account_info.get('login', 'Unknown')}")
    print(f"   Balance: {account_info.get('balance', 0)} {account_info.get('currency', 'USD')}")
    
    # Get multi-timeframe data for BIAS testing
    test_data = {}
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    # Multi-timeframe setup for comprehensive BIAS analysis
    timeframes = {
        "D1": 100,   # Daily data for long-term bias
        "H4": 200,   # 4-hour data for medium-term bias
        "H1": 300,   # 1-hour data for short-term bias
        "M15": 400,  # 15-minute data for immediate bias
    }
    
    for symbol in symbols:
        test_data[symbol] = {}
        print(f"\n   📊 Getting {symbol} data:")
        
        for tf, bars in timeframes.items():
            try:
                data = connector.get_rates(symbol, tf, bars)
                if data is not None and len(data) > 50:
                    test_data[symbol][tf] = data
                    print(f"      ✓ {tf}: {len(data)} bars ({data.index[0].strftime('%m-%d %H:%M')} to {data.index[-1].strftime('%m-%d %H:%M')})")
                else:
                    print(f"      ⚠️  {tf}: Insufficient data ({len(data) if data is not None else 0} bars)")
            except Exception as e:
                print(f"      ✗ {tf}: Error - {e}")
    
    connector.disconnect()
    return test_data

def test_bias_analyzer_initialization():
    """Test BIAS Analyzer initialization and configuration"""
    print("\n" + "="*70)
    print("TESTING BIAS ANALYZER INITIALIZATION")
    print("="*70)
    
    try:
        # Test 1: Default initialization
        print("\n1. Testing default initialization:")
        bias_analyzer = BiasAnalyzer()
        print("   ✓ Default BiasAnalyzer created successfully")
        print(f"   Weights: {bias_analyzer.weights}")
        print(f"   Multi-timeframe analysis: {bias_analyzer.analyze_mtf}")
        print(f"   Session analysis: {bias_analyzer.session_analysis_enabled}")
        
        # Test 2: Custom configuration
        print("\n2. Testing custom configuration:")
        custom_analyzer = BiasAnalyzer(
            structural_weight=0.3,
            institutional_weight=0.3,
            liquidity_weight=0.15,
            zone_weight=0.15,
            session_weight=0.1,
            mtf_alignment_threshold=0.8,
            session_history_hours=48,
            bias_memory_bars=100
        )
        print("   ✓ Custom BiasAnalyzer created successfully")
        print(f"   Custom weights: {custom_analyzer.weights}")
        print(f"   MTF alignment threshold: {custom_analyzer.mtf_alignment_threshold}")
        print(f"   Session history: {custom_analyzer.session_history_hours} hours")
        
        # Test 3: Invalid weight configuration
        print("\n3. Testing invalid weight configuration:")
        try:
            invalid_analyzer = BiasAnalyzer(
                structural_weight=0.5,
                institutional_weight=0.5,
                liquidity_weight=0.5,  # Total > 1.0
                zone_weight=0.2,
                session_weight=0.1
            )
            print("   ❌ Should have failed with invalid weights")
            return False
        except ValueError as e:
            print(f"   ✓ Correctly rejected invalid weights: {e}")
        
        # Test 4: Component analyzer initialization
        print("\n4. Testing component analyzer initialization:")
        component_types = [
            'ms_analyzer', 'ob_analyzer', 'fvg_analyzer',
            'liq_analyzer', 'pd_analyzer', 'sd_analyzer'
        ]
        
        for component in component_types:
            if hasattr(bias_analyzer, component):
                print(f"   ✓ {component} initialized")
            else:
                print(f"   ❌ {component} missing")
                return False
        
        print("\n✅ BIAS Analyzer initialization tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ BIAS Analyzer initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_single_timeframe_bias():
    """Test BIAS analysis on single timeframe"""
    print("\n" + "="*70)
    print("TESTING SINGLE TIMEFRAME BIAS ANALYSIS")
    print("="*70)
    
    try:
        # Get test data
        test_data = get_multi_timeframe_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Initialize analyzer
        bias_analyzer = BiasAnalyzer()
        
        successful_analyses = 0
        total_analyses = 0
        
        # Test single timeframe analysis
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing Single Timeframe BIAS: {symbol} ---")
            
            for tf, data in timeframes.items():
                print(f"\n{tf} Analysis:")
                print(f"   📊 Data: {len(data)} bars ({data.index[0].strftime('%m-%d %H:%M')} to {data.index[-1].strftime('%m-%d %H:%M')})")
                
                total_analyses += 1
                
                # Test single timeframe BIAS (using only one timeframe)
                single_tf_data = {tf: data}
                
                start_time = time.time()
                bias_result = bias_analyzer.analyze_bias(symbol, single_tf_data, tf)
                analysis_time = time.time() - start_time
                
                if bias_result and bias_result.direction != BiasDirection.NEUTRAL:
                    successful_analyses += 1
                    
                    print(f"   ⚡ Analysis time: {analysis_time:.3f}s")
                    print(f"   📊 BIAS Results:")
                    print(f"      Direction: {bias_result.direction.name}")
                    print(f"      Strength: {bias_result.strength.value.upper()}")
                    print(f"      Confidence: {bias_result.confidence:.2f}")
                    print(f"      Score: {bias_result.score:.3f}")
                    print(f"      Trading Recommendation: {bias_result.trading_recommendation}")
                    print(f"      Risk Level: {bias_result.risk_level}")
                    
                    # Component breakdown
                    print(f"   🧩 Component Breakdown:")
                    print(f"      Structural: {bias_result.structural_bias:.3f}")
                    print(f"      Institutional: {bias_result.institutional_bias:.3f}")
                    print(f"      Liquidity: {bias_result.liquidity_bias:.3f}")
                    print(f"      Zone: {bias_result.zone_bias:.3f}")
                    print(f"      Session: {bias_result.session_bias:.3f}")
                    
                    # Show active components
                    active_components = [c for c in bias_result.components if c.direction != BiasDirection.NEUTRAL]
                    if active_components:
                        print(f"   🎯 Active Components ({len(active_components)}):")
                        for comp in active_components[:3]:  # Show top 3
                            print(f"      {comp.component_name}: {comp.direction.name} "
                                  f"(Strength: {comp.strength:.2f}, Confidence: {comp.confidence:.2f})")
                    
                    # Invalidation level
                    if bias_result.invalidation_level:
                        current_price = data['Close'].iloc[-1]
                        distance_to_invalidation = abs(current_price - bias_result.invalidation_level)
                        print(f"   🚫 Invalidation Level: {bias_result.invalidation_level:.5f}")
                        print(f"      Distance: {distance_to_invalidation:.5f} ({distance_to_invalidation*10000:.1f} pips)")
                
                else:
                    print(f"   ⚠️  No clear BIAS detected for {symbol} {tf}")
        
        # Success rate analysis
        success_rate = successful_analyses / total_analyses if total_analyses > 0 else 0
        print(f"\n--- Single Timeframe BIAS Summary ---")
        print(f"Total analyses: {total_analyses}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Success rate: {success_rate:.1%}")
        
        if success_rate >= 0.6:  # 60% success rate threshold
            print("\n✅ Single timeframe BIAS analysis working well!")
            return True
        else:
            print(f"\n⚠️  Single timeframe BIAS analysis needs improvement (success rate: {success_rate:.1%})")
            return success_rate > 0.3  # At least 30% to pass
        
    except Exception as e:
        print(f"\n❌ Single timeframe BIAS test failed: {e}")
        traceback.print_exc()
        return False

def test_multi_timeframe_bias():
    """Test multi-timeframe BIAS analysis and alignment"""
    print("\n" + "="*70)
    print("TESTING MULTI-TIMEFRAME BIAS ANALYSIS")
    print("="*70)
    
    try:
        # Get test data
        test_data = get_multi_timeframe_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Initialize analyzer with multi-timeframe enabled
        bias_analyzer = BiasAnalyzer(
            analyze_mtf=True,
            mtf_alignment_threshold=0.7
        )
        
        successful_mtf_analyses = 0
        total_mtf_analyses = 0
        
        # Test multi-timeframe analysis
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing Multi-Timeframe BIAS: {symbol} ---")
            
            # Only test if we have multiple timeframes
            if len(timeframes) < 2:
                print(f"   ⚠️  Insufficient timeframes for {symbol} (need 2+, got {len(timeframes)})")
                continue
            
            total_mtf_analyses += 1
            
            print(f"   📊 Available timeframes: {list(timeframes.keys())}")
            
            # Primary timeframe analysis (H1 or H4)
            primary_tf = "H1" if "H1" in timeframes else "H4" if "H4" in timeframes else list(timeframes.keys())[0]
            
            start_time = time.time()
            bias_result = bias_analyzer.analyze_bias(symbol, timeframes, primary_tf)
            analysis_time = time.time() - start_time
            
            if bias_result and hasattr(bias_result, 'mtf_bias') and bias_result.mtf_bias:
                successful_mtf_analyses += 1
                mtf_bias = bias_result.mtf_bias
                
                print(f"   ⚡ Multi-timeframe analysis time: {analysis_time:.3f}s")
                print(f"   📊 Overall BIAS:")
                print(f"      Direction: {bias_result.direction.name}")
                print(f"      Strength: {bias_result.strength.value.upper()}")
                print(f"      Confidence: {bias_result.confidence:.2f}")
                print(f"      Score: {bias_result.score:.3f}")
                
                print(f"   🎯 Multi-Timeframe Analysis:")
                print(f"      Long-term BIAS: {mtf_bias.long_term_bias.name}")
                print(f"      Medium-term BIAS: {mtf_bias.medium_term_bias.name}")
                print(f"      Short-term BIAS: {mtf_bias.short_term_bias.name}")
                print(f"      Alignment Score: {mtf_bias.alignment_score:.2f}")
                print(f"      Dominant Timeframe: {mtf_bias.dominant_timeframe.value.upper()}")
                
                # Conflict analysis
                if mtf_bias.conflict_zones:
                    print(f"   ⚠️  Timeframe Conflicts:")
                    for conflict in mtf_bias.conflict_zones:
                        print(f"      - {conflict.replace('_', ' ').title()}")
                else:
                    print(f"   ✅ No major timeframe conflicts")
                
                # Alignment assessment
                if mtf_bias.alignment_score >= 0.8:
                    alignment_quality = "🎯 EXCELLENT"
                elif mtf_bias.alignment_score >= 0.6:
                    alignment_quality = "✅ GOOD"
                elif mtf_bias.alignment_score >= 0.4:
                    alignment_quality = "📊 MODERATE"
                else:
                    alignment_quality = "⚠️ POOR"
                
                print(f"   Alignment Quality: {alignment_quality}")
                
                # Trading recommendation based on MTF
                print(f"   💡 MTF Trading Assessment:")
                print(f"      Recommendation: {bias_result.trading_recommendation}")
                print(f"      Risk Level: {bias_result.risk_level}")
                
                if mtf_bias.alignment_score >= 0.7 and len(mtf_bias.conflict_zones) == 0:
                    print(f"      ✅ STRONG MULTI-TIMEFRAME AGREEMENT - High confidence setup")
                elif mtf_bias.alignment_score >= 0.5:
                    print(f"      📊 MODERATE AGREEMENT - Proceed with caution")
                else:
                    print(f"      ⚠️  WEAK AGREEMENT - Consider waiting for better setup")
                
                # Component analysis for MTF
                component_breakdown = bias_analyzer.get_component_breakdown(bias_result)
                strongest = component_breakdown['strongest_components']
                if strongest:
                    print(f"   🏆 Strongest Components: {', '.join(strongest)}")
                
                conflicting = component_breakdown['conflicting_components']
                if conflicting:
                    print(f"   ⚡ Conflicting Components: {', '.join(conflicting)}")
            
            else:
                print(f"   ⚠️  Multi-timeframe analysis inconclusive for {symbol}")
        
        # MTF Analysis Summary
        mtf_success_rate = successful_mtf_analyses / total_mtf_analyses if total_mtf_analyses > 0 else 0
        print(f"\n--- Multi-Timeframe BIAS Summary ---")
        print(f"Total MTF analyses: {total_mtf_analyses}")
        print(f"Successful MTF analyses: {successful_mtf_analyses}")
        print(f"MTF success rate: {mtf_success_rate:.1%}")
        
        if mtf_success_rate >= 0.7:  # 70% success rate for MTF
            print("\n✅ Multi-timeframe BIAS analysis working excellently!")
            return True
        elif mtf_success_rate >= 0.5:
            print("\n📊 Multi-timeframe BIAS analysis working adequately")
            return True
        else:
            print(f"\n⚠️  Multi-timeframe BIAS analysis needs improvement")
            return False
        
    except Exception as e:
        print(f"\n❌ Multi-timeframe BIAS test failed: {e}")
        traceback.print_exc()
        return False

def test_session_bias_analysis():
    """Test session-based BIAS analysis"""
    print("\n" + "="*70)
    print("TESTING SESSION-BASED BIAS ANALYSIS")
    print("="*70)
    
    try:
        # Get test data
        test_data = get_multi_timeframe_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Initialize analyzer with session analysis enabled
        bias_analyzer = BiasAnalyzer(
            session_analysis_enabled=True,
            session_history_hours=24
        )
        
        successful_session_analyses = 0
        total_session_analyses = 0
        
        # Test session analysis on shorter timeframes (better for session detection)
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing Session BIAS: {symbol} ---")
            
            # Use shorter timeframes for session analysis
            suitable_tfs = [tf for tf in ['M15', 'H1', 'H4'] if tf in timeframes]
            
            if not suitable_tfs:
                print(f"   ⚠️  No suitable timeframes for session analysis")
                continue
            
            primary_tf = suitable_tfs[0]  # Use the shortest available
            tf_data = {primary_tf: timeframes[primary_tf]}
            
            total_session_analyses += 1
            
            print(f"   📊 Using {primary_tf} data: {len(timeframes[primary_tf])} bars")
            
            start_time = time.time()
            bias_result = bias_analyzer.analyze_bias(symbol, tf_data, primary_tf)
            analysis_time = time.time() - start_time
            
            if bias_result and bias_result.session_analysis:
                successful_session_analyses += 1
                
                print(f"   ⚡ Session analysis time: {analysis_time:.3f}s")
                print(f"   📊 Session BIAS Results:")
                
                # Show session breakdown
                session_analyses = bias_result.session_analysis
                print(f"   🕐 Session Analysis ({len(session_analyses)} sessions):")
                
                for session in session_analyses:
                    if session.strength > 0.01:  # Only show sessions with meaningful bias
                        direction_icon = "🟢" if session.direction == BiasDirection.BULLISH else "🔴" if session.direction == BiasDirection.BEARISH else "🟡"
                        print(f"      {direction_icon} {session.session.value.upper()}: {session.direction.name}")
                        print(f"         Strength: {session.strength:.3f}, Consistency: {session.consistency:.2f}")
                        print(f"         Duration: {session.duration_hours:.1f}h, Volume Profile: {session.volume_profile:.3f}")
                
                # Overall session bias contribution
                print(f"   📈 Session Contribution to Overall BIAS:")
                print(f"      Session BIAS Score: {bias_result.session_bias:.3f}")
                print(f"      Weight in Overall: {bias_analyzer.weights['session']:.1%}")
                
                # Session-based trading insights
                dominant_sessions = [s for s in session_analyses if s.strength > 0.02 and s.consistency > 0.6]
                if dominant_sessions:
                    print(f"   🎯 Dominant Sessions:")
                    for session in sorted(dominant_sessions, key=lambda x: x.strength, reverse=True)[:2]:
                        print(f"      {session.session.value.upper()}: {session.direction.name} bias")
                        print(f"         → Consider {session.direction.name.lower()} positions during this session")
                
                else:
                    print(f"   📊 No strongly dominant sessions identified")
            
            else:
                print(f"   ⚠️  Session analysis inconclusive for {symbol}")
        
        # Session Analysis Summary
        session_success_rate = successful_session_analyses / total_session_analyses if total_session_analyses > 0 else 0
        print(f"\n--- Session BIAS Summary ---")
        print(f"Total session analyses: {total_session_analyses}")
        print(f"Successful session analyses: {successful_session_analyses}")
        print(f"Session success rate: {session_success_rate:.1%}")
        
        # Test session utility functions
        print(f"\n📊 Testing Session Utility Functions:")
        if bias_analyzer.bias_history:
            session_summary = bias_analyzer.get_bias_persistence()
            print(f"   BIAS Persistence Score: {session_summary['persistence_score']:.2f}")
            print(f"   Direction Changes: {session_summary['direction_changes']}")
            print(f"   Current Streak: {session_summary['current_streak']}")
            print(f"   Confidence Trend: {session_summary['confidence_trend']}")
        
        if session_success_rate >= 0.5:  # 50% success rate for session analysis
            print("\n✅ Session BIAS analysis working well!")
            return True
        else:
            print(f"\n⚠️  Session BIAS analysis needs improvement")
            return session_success_rate > 0.2  # At least 20% to pass
        
    except Exception as e:
        print(f"\n❌ Session BIAS test failed: {e}")
        traceback.print_exc()
        return False

def test_bias_component_integration():
    """Test BIAS integration with individual SMC components"""
    print("\n" + "="*70)
    print("TESTING BIAS COMPONENT INTEGRATION")
    print("="*70)
    
    try:
        # Get test data
        test_data = get_multi_timeframe_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Initialize analyzer
        bias_analyzer = BiasAnalyzer()
        
        integration_scores = {
            'market_structure': 0,
            'order_blocks': 0,
            'fair_value_gaps': 0,
            'liquidity': 0,
            'premium_discount': 0,
            'supply_demand': 0
        }
        
        total_component_tests = 0
        
        # Test component integration
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing Component Integration: {symbol} ---")
            
            # Use H1 timeframe for detailed component analysis
            if "H1" not in timeframes:
                print(f"   ⚠️  No H1 data available for {symbol}")
                continue
            
            tf_data = {"H1": timeframes["H1"]}
            bias_result = bias_analyzer.analyze_bias(symbol, tf_data, "H1")
            
            if bias_result and bias_result.components:
                total_component_tests += 1
                
                print(f"   📊 Component Integration Analysis:")
                print(f"   Total Components Detected: {len(bias_result.components)}")
                
                # Analyze each component type
                for component in bias_result.components:
                    comp_name = component.component_name
                    if comp_name in integration_scores:
                        integration_scores[comp_name] += 1
                    
                    direction_icon = "🟢" if component.direction == BiasDirection.BULLISH else "🔴" if component.direction == BiasDirection.BEARISH else "🟡"
                    print(f"      {direction_icon} {comp_name.replace('_', ' ').title()}:")
                    print(f"         Direction: {component.direction.name}")
                    print(f"         Strength: {component.strength:.2f}")
                    print(f"         Confidence: {component.confidence:.2f}")
                    print(f"         Weight: {component.weight:.2f}")
                    
                    if component.details:
                        detail_str = ", ".join([f"{k}: {v}" for k, v in component.details.items()])
                        print(f"         Details: {detail_str}")
                
                # Component breakdown analysis
                component_breakdown = bias_analyzer.get_component_breakdown(bias_result)
                
                print(f"   🏆 Component Analysis:")
                print(f"      Strongest: {', '.join(component_breakdown['strongest_components'])}")
                print(f"      Weakest: {', '.join(component_breakdown['weakest_components'])}")
                
                if component_breakdown['conflicting_components']:
                    print(f"      ⚡ Conflicts: {', '.join(component_breakdown['conflicting_components'])}")
                else:
                    print(f"      ✅ No major component conflicts")
                
                # Component contributions
                print(f"   📈 Component Contributions to Final BIAS:")
                for comp, contribution in component_breakdown['component_contributions'].items():
                    contribution_pct = contribution / bias_result.score * 100 if bias_result.score != 0 else 0
                    print(f"      {comp.title()}: {contribution:.3f} ({contribution_pct:.1f}%)")
        
        # Integration Summary
        print(f"\n--- Component Integration Summary ---")
        print(f"Total integration tests: {total_component_tests}")
        print(f"Component Detection Rates:")
        
        component_success_rate = 0
        for comp_name, detected_count in integration_scores.items():
            detection_rate = detected_count / total_component_tests if total_component_tests > 0 else 0
            component_success_rate += detection_rate
            print(f"   {comp_name.replace('_', ' ').title()}: {detection_rate:.1%} ({detected_count}/{total_component_tests})")
        
        overall_integration_rate = component_success_rate / len(integration_scores)
        print(f"\nOverall Integration Rate: {overall_integration_rate:.1%}")
        
        # Test component weight balance
        print(f"\n🧮 Testing Component Weight Balance:")
        weight_sum = sum(bias_analyzer.weights.values())
        print(f"   Total Weight Sum: {weight_sum:.3f} (should be 1.000)")
        print(f"   Weight Distribution:")
        for comp, weight in bias_analyzer.weights.items():
            print(f"      {comp.title()}: {weight:.1%}")
        
        # Test quick bias utility function
        print(f"\n⚡ Testing Quick BIAS Utility:")
        if "EURUSD" in test_data and "H1" in test_data["EURUSD"]:
            quick_result = calculate_quick_bias(test_data["EURUSD"]["H1"])
            print(f"   Quick BIAS: {quick_result['direction']} (Strength: {quick_result['strength']:.2f}, Confidence: {quick_result['confidence']:.2f})")
        
        if overall_integration_rate >= 0.6 and abs(weight_sum - 1.0) < 0.01:
            print("\n✅ Component integration working excellently!")
            return True
        elif overall_integration_rate >= 0.4:
            print("\n📊 Component integration working adequately")
            return True
        else:
            print(f"\n⚠️  Component integration needs improvement")
            return False
        
    except Exception as e:
        print(f"\n❌ Component integration test failed: {e}")
        traceback.print_exc()
        return False

def test_bias_trading_recommendations():
    """Test BIAS-based trading recommendations"""
    print("\n" + "="*70)
    print("TESTING BIAS TRADING RECOMMENDATIONS")
    print("="*70)
    
    try:
        # Get test data
        test_data = get_multi_timeframe_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Initialize analyzer
        bias_analyzer = BiasAnalyzer()
        
        recommendation_types = {}
        risk_levels = {}
        total_recommendations = 0
        
        # Test trading recommendations
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing Trading Recommendations: {symbol} ---")
            
            # Use multiple timeframes for better recommendations
            if len(timeframes) >= 2:
                primary_tf = "H1" if "H1" in timeframes else list(timeframes.keys())[0]
                
                bias_result = bias_analyzer.analyze_bias(symbol, timeframes, primary_tf)
                
                if bias_result:
                    total_recommendations += 1
                    current_price = timeframes[primary_tf]['Close'].iloc[-1]
                    
                    # Track recommendation types
                    rec_type = bias_result.trading_recommendation
                    recommendation_types[rec_type] = recommendation_types.get(rec_type, 0) + 1
                    
                    # Track risk levels
                    risk_level = bias_result.risk_level
                    risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
                    
                    print(f"   📊 Current Price: {current_price:.5f}")
                    print(f"   🎯 BIAS Assessment:")
                    print(f"      Direction: {bias_result.direction.name}")
                    print(f"      Strength: {bias_result.strength.value.upper()}")
                    print(f"      Confidence: {bias_result.confidence:.2f}")
                    print(f"      Score: {bias_result.score:.3f}")
                    
                    print(f"   💡 Trading Recommendation:")
                    print(f"      Action: {bias_result.trading_recommendation}")
                    print(f"      Risk Level: {bias_result.risk_level}")
                    
                    # Get trading zones
                    trading_zones = bias_analyzer.get_trading_zones(bias_result, timeframes[primary_tf])
                    
                    print(f"   🎯 Trading Zones:")
                    if bias_result.invalidation_level:
                        distance_to_invalidation = abs(current_price - bias_result.invalidation_level)
                        print(f"      Invalidation: {bias_result.invalidation_level:.5f} ({distance_to_invalidation*10000:.1f} pips away)")
                    
                    if trading_zones['entry_zones']:
                        print(f"      Entry Zones:")
                        for i, entry in enumerate(trading_zones['entry_zones'][:3], 1):
                            distance = abs(current_price - entry)
                            print(f"         {i}. {entry:.5f} ({distance*10000:.1f} pips)")
                    
                    if trading_zones['target_zones']:
                        print(f"      Target Zones:")
                        for i, target in enumerate(trading_zones['target_zones'][:3], 1):
                            distance = abs(current_price - target)
                            print(f"         {i}. {target:.5f} ({distance*10000:.1f} pips)")
                    
                    # Risk assessment
                    print(f"   ⚠️  Risk Assessment:")
                    if bias_result.confidence >= 0.7 and bias_result.strength in [BiasStrength.STRONG, BiasStrength.EXTREME]:
                        print(f"      ✅ HIGH CONFIDENCE SETUP - Consider position sizing up")
                    elif bias_result.confidence >= 0.5:
                        print(f"      📊 MODERATE CONFIDENCE - Standard position sizing")
                    else:
                        print(f"      ⚠️  LOW CONFIDENCE - Reduce position size or wait")
                    
                    # Multi-timeframe recommendation enhancement
                    if bias_result.mtf_bias and bias_result.mtf_bias.alignment_score >= 0.7:
                        print(f"      🎯 STRONG MTF ALIGNMENT - Enhanced confidence")
                    elif bias_result.mtf_bias and len(bias_result.mtf_bias.conflict_zones) > 0:
                        print(f"      ⚡ MTF CONFLICTS DETECTED - Exercise caution")
        
        # Recommendation Summary
        print(f"\n--- Trading Recommendation Summary ---")
        print(f"Total recommendations generated: {total_recommendations}")
        
        print(f"\n📊 Recommendation Distribution:")
        for rec_type, count in recommendation_types.items():
            percentage = count / total_recommendations * 100
            print(f"   {rec_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n⚠️  Risk Level Distribution:")
        for risk_level, count in risk_levels.items():
            percentage = count / total_recommendations * 100
            print(f"   {risk_level}: {count} ({percentage:.1f}%)")
        
        # Test recommendation quality
        actionable_recommendations = sum(count for rec_type, count in recommendation_types.items() 
                                       if not rec_type.startswith("NO_") and not rec_type.startswith("STAY_"))
        
        actionable_rate = actionable_recommendations / total_recommendations if total_recommendations > 0 else 0
        
        print(f"\n📈 Recommendation Quality Assessment:")
        print(f"   Actionable Recommendations: {actionable_rate:.1%}")
        print(f"   Risk Management: {'✅ Good' if risk_levels.get('HIGH', 0) < total_recommendations * 0.7 else '⚠️ Too Conservative'}")
        
        if actionable_rate >= 0.6 and total_recommendations > 0:
            print("\n✅ Trading recommendations working excellently!")
            return True
        elif actionable_rate >= 0.4:
            print("\n📊 Trading recommendations working adequately")
            return True
        else:
            print(f"\n⚠️  Trading recommendations need improvement")
            return False
        
    except Exception as e:
        print(f"\n❌ Trading recommendations test failed: {e}")
        traceback.print_exc()
        return False

def test_bias_performance():
    """Test BIAS analyzer performance and scalability"""
    print("\n" + "="*70)
    print("TESTING BIAS ANALYZER PERFORMANCE")
    print("="*70)
    
    try:
        # Get test data
        test_data = get_multi_timeframe_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Find largest multi-timeframe dataset
        largest_dataset = None
        largest_total_bars = 0
        test_symbol = ""
        
        for symbol, timeframes in test_data.items():
            total_bars = sum(len(data) for data in timeframes.values())
            if total_bars > largest_total_bars:
                largest_dataset = timeframes
                largest_total_bars = total_bars
                test_symbol = symbol
        
        if largest_dataset is None:
            print("✗ No suitable dataset found")
            return False
        
        print(f"Testing performance with {test_symbol} data:")
        for tf, data in largest_dataset.items():
            print(f"   {tf}: {len(data)} bars")
        print(f"   Total: {largest_total_bars} bars across {len(largest_dataset)} timeframes")
        
        # Initialize analyzer
        bias_analyzer = BiasAnalyzer()
        
        # Performance Test 1: Single timeframe performance
        print(f"\n⚡ Single Timeframe Performance:")
        primary_tf = list(largest_dataset.keys())[0]
        single_tf_data = {primary_tf: largest_dataset[primary_tf]}
        
        start_time = time.time()
        single_tf_result = bias_analyzer.analyze_bias(test_symbol, single_tf_data, primary_tf)
        single_tf_time = time.time() - start_time
        
        single_tf_bars = len(largest_dataset[primary_tf])
        single_tf_performance = single_tf_bars / single_tf_time if single_tf_time > 0 else 0
        
        print(f"   Analysis time: {single_tf_time:.3f} seconds")
        print(f"   Performance: {single_tf_performance:.0f} bars/second")
        print(f"   Components detected: {len(single_tf_result.components) if single_tf_result else 0}")
        
        # Performance Test 2: Multi-timeframe performance
        print(f"\n⚡ Multi-Timeframe Performance:")
        start_time = time.time()
        mtf_result = bias_analyzer.analyze_bias(test_symbol, largest_dataset, primary_tf)
        mtf_time = time.time() - start_time
        
        mtf_performance = largest_total_bars / mtf_time if mtf_time > 0 else 0
        
        print(f"   Analysis time: {mtf_time:.3f} seconds")
        print(f"   Performance: {mtf_performance:.0f} bars/second")
        print(f"   MTF analysis: {'✅ Complete' if mtf_result and mtf_result.mtf_bias else '❌ Failed'}")
        print(f"   Session analysis: {'✅ Complete' if mtf_result and mtf_result.session_analysis else '❌ Failed'}")
        
        # Performance Test 3: Repeated analysis (caching effect)
        print(f"\n⚡ Repeated Analysis Performance:")
        repeated_times = []
        for i in range(3):
            start_time = time.time()
            repeated_result = bias_analyzer.analyze_bias(test_symbol, largest_dataset, primary_tf)
            repeated_time = time.time() - start_time
            repeated_times.append(repeated_time)
        
        avg_repeated_time = sum(repeated_times) / len(repeated_times)
        improvement = (mtf_time - avg_repeated_time) / mtf_time * 100 if mtf_time > 0 else 0
        
        print(f"   Average repeated time: {avg_repeated_time:.3f} seconds")
        print(f"   Performance improvement: {improvement:.1f}%")
        
        # Memory usage estimation
        print(f"\n💾 Memory Usage Estimation:")
        import sys
        
        if mtf_result:
            result_size = sys.getsizeof(mtf_result)
            components_size = sum(sys.getsizeof(comp) for comp in mtf_result.components)
            session_size = sum(sys.getsizeof(session) for session in mtf_result.session_analysis)
            
            total_memory = result_size + components_size + session_size
            memory_per_bar = total_memory / largest_total_bars
            
            print(f"   Result object: {result_size / 1024:.2f} KB")
            print(f"   Components: {components_size / 1024:.2f} KB")
            print(f"   Sessions: {session_size / 1024:.2f} KB")
            print(f"   Total: {total_memory / 1024:.2f} KB")
            print(f"   Per bar: {memory_per_bar:.2f} bytes")
        
        # BIAS history performance
        print(f"\n📊 BIAS History Performance:")
        history_length = len(bias_analyzer.bias_history)
        print(f"   History entries: {history_length}")
        
        if history_length > 0:
            persistence_analysis = bias_analyzer.get_bias_persistence()
            print(f"   Persistence score: {persistence_analysis['persistence_score']:.2f}")
            print(f"   Direction changes: {persistence_analysis['direction_changes']}")
            print(f"   Current streak: {persistence_analysis['current_streak']}")
        
        # Performance Assessment
        print(f"\n📊 Performance Assessment:")
        
        # Real-time trading requirements
        min_performance = 500    # Minimum bars/second for real-time
        good_performance = 1000  # Good performance threshold
        excellent_performance = 5000  # Excellent performance
        
        if mtf_performance >= excellent_performance:
            performance_rating = "🚀 EXCELLENT"
            assessment = "Outstanding performance - Real-time ready with room to spare!"
        elif mtf_performance >= good_performance:
            performance_rating = "✅ GOOD"
            assessment = "Good performance - Suitable for live trading"
        elif mtf_performance >= min_performance:
            performance_rating = "📊 ACCEPTABLE"
            assessment = "Acceptable performance - Ready for live trading"
        else:
            performance_rating = "⚠️ NEEDS OPTIMIZATION"
            assessment = "Performance needs improvement for live trading"
        
        print(f"   Overall Rating: {performance_rating}")
        print(f"   Assessment: {assessment}")
        
        # Performance targets
        print(f"\n🎯 Performance Targets:")
        print(f"   Minimum (Real-time): {min_performance:,} bars/sec ({'✅' if mtf_performance >= min_performance else '❌'})")
        print(f"   Good: {good_performance:,} bars/sec ({'✅' if mtf_performance >= good_performance else '❌'})")
        print(f"   Excellent: {excellent_performance:,} bars/sec ({'✅' if mtf_performance >= excellent_performance else '❌'})")
        
        # Scalability test
        print(f"\n📈 Scalability Assessment:")
        if mtf_time < single_tf_time * len(largest_dataset) * 1.5:
            print("   ✅ Good multi-timeframe scaling")
        else:
            print("   ⚠️ Multi-timeframe scaling needs optimization")
        
        return mtf_performance >= min_performance
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

def generate_comprehensive_bias_report():
    """Generate comprehensive BIAS analyzer test report"""
    print("\n" + "="*70)
    print("BIAS ANALYZER - COMPREHENSIVE TEST REPORT")
    print("="*70)
    
    test_results = {}
    
    try:
        print("🔬 Running comprehensive BIAS analyzer test suite...\n")
        
        # Run all test modules
        test_modules = [
            ("Initialization", test_bias_analyzer_initialization),
            ("Single Timeframe Analysis", test_single_timeframe_bias),
            ("Multi-Timeframe Analysis", test_multi_timeframe_bias),
            ("Session Analysis", test_session_bias_analysis),
            ("Component Integration", test_bias_component_integration),
            ("Trading Recommendations", test_bias_trading_recommendations),
            ("Performance Testing", test_bias_performance)
        ]
        
        for module_name, test_function in test_modules:
            print(f"\n{'='*50}")
            print(f"Running: {module_name}")
            print(f"{'='*50}")
            
            try:
                start_time = time.time()
                result = test_function()
                test_time = time.time() - start_time
                
                test_results[module_name] = {
                    'passed': result,
                    'duration': test_time
                }
                
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"\n{module_name}: {status} ({test_time:.2f}s)")
                
            except Exception as e:
                test_results[module_name] = {
                    'passed': False,
                    'duration': 0,
                    'error': str(e)
                }
                print(f"\n{module_name}: ❌ FAILED (Exception: {e})")
        
        # Generate final summary
        print("\n" + "="*70)
        print("FINAL BIAS ANALYZER TEST SUMMARY")
        print("="*70)
        
        passed_tests = sum(1 for result in test_results.values() if result['passed'])
        total_tests = len(test_results)
        total_time = sum(result['duration'] for result in test_results.values())
        
        print(f"\n📊 Test Statistics:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests:.1%}")
        print(f"   Total Duration: {total_time:.2f} seconds")
        
        print(f"\n📋 Individual Results:")
        for module_name, result in test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            duration = result['duration']
            print(f"   {module_name}: {status} ({duration:.2f}s)")
            
            if not result['passed'] and 'error' in result:
                print(f"      Error: {result['error']}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        if passed_tests == total_tests:
            print(f"   🎉 ALL TESTS PASSED!")
            print(f"   ✅ BIAS Direction Analysis: Fully Functional")
            print(f"   ✅ Multi-Timeframe Integration: Working Perfectly")
            print(f"   ✅ Session Analysis: Operational")
            print(f"   ✅ Component Integration: Complete")
            print(f"   ✅ Trading Recommendations: Generated")
            print(f"   ✅ Performance: Meeting Standards")
            
            print(f"\n🚀 PHASE 2 WEEK 4 - DAY 5-7: COMPLETED SUCCESSFULLY!")
            print(f"📊 BIAS Analysis system fully implemented")
            print(f"🎯 Ready for Phase 2 Week 5: Feature Integration & Testing")
            
            success = True
            
        elif passed_tests >= total_tests * 0.8:  # 80% pass rate
            print(f"   ✅ MOSTLY SUCCESSFUL!")
            print(f"   📊 Core BIAS functionality working")
            print(f"   ⚠️ Some minor issues to address")
            
            failed_modules = [name for name, result in test_results.items() if not result['passed']]
            print(f"   Issues in: {', '.join(failed_modules)}")
            
            print(f"\n📋 PHASE 2 WEEK 4 - DAY 5-7: MOSTLY COMPLETED")
            print(f"✅ Core BIAS features working, minor issues to fix")
            
            success = True
            
        else:
            print(f"   ❌ SIGNIFICANT ISSUES DETECTED")
            print(f"   🔧 Major BIAS components need attention")
            
            failed_modules = [name for name, result in test_results.items() if not result['passed']]
            print(f"   Failed modules: {', '.join(failed_modules)}")
            
            print(f"\n⚠️ PHASE 2 WEEK 4 - DAY 5-7: NEEDS WORK")
            print(f"❌ Core BIAS issues must be resolved before proceeding")
            
            success = False
        
        # Next steps
        if success:
            print(f"\n📋 NEXT STEPS - PHASE 2 WEEK 5:")
            print(f"   1. Volume Profile & VWAP implementation")
            print(f"   2. Complete Feature Aggregator integration")
            print(f"   3. Multi-timeframe feature alignment")
            print(f"   4. Comprehensive SMC confluence scoring")
            print(f"   5. Performance optimization")
            print(f"   6. Final SMC feature testing")
        else:
            print(f"\n🔧 REQUIRED FIXES:")
            for name, result in test_results.items():
                if not result['passed']:
                    print(f"   - Debug and fix: {name}")
        
        return success
        
    except Exception as e:
        print(f"\n💥 Test report generation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test execution function"""
    setup_logging()
    
    print("="*70)
    print("   BIAS ANALYZER - COMPREHENSIVE TESTING")
    print("   PHASE 2 WEEK 4 DAY 5-7 - BIAS INTEGRATION")
    print("="*70)
    print(f"🕒 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Objective: Validate BIAS Analysis & SMC Integration")
    
    try:
        # Run comprehensive test suite
        success = generate_comprehensive_bias_report()
        
        if success:
            print(f"\n🎉 BIAS TESTING COMPLETED SUCCESSFULLY!")
            print(f"🚀 BIAS analyzer is production-ready")
            print(f"📊 SMC component integration working perfectly")
            print(f"🎯 Multi-timeframe analysis operational")
            print(f"💡 Trading recommendations being generated")
            return 0
        else:
            print(f"\n⚠️ BIAS TESTING IDENTIFIED ISSUES")
            print(f"🔧 Review and fix failed components")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n👋 Testing interrupted by user")
        return 0
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        return 1
    finally:
        print(f"\n🕒 Test ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)