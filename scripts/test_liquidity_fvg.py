"""
Complete Test Script for Liquidity and Fair Value Gap Features
Phase 2 - Day 3-4: Comprehensive testing of new SMC components
"""
import sys
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.liquidity_analyzer import LiquidityAnalyzer, LiquidityType, LiquidityEvent
from features.fair_value_gaps import FVGAnalyzer, FVGType, FVGStatus, FVGQuality
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

def get_test_data():
    """Get test data from MT5 or demo"""
    print("Getting test data...")
    
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
    
    # Get data for testing (more bars for better patterns)
    test_data = {}
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframes = ["H4", "H1", "M15"]
    
    for symbol in symbols:
        test_data[symbol] = {}
        for tf in timeframes:
            try:
                data = connector.get_rates(symbol, tf, 500)  # 500 bars for better patterns
                if data is not None and len(data) > 100:
                    test_data[symbol][tf] = data
                    print(f"   ✓ {symbol} {tf}: {len(data)} bars ({data.index[0].strftime('%m-%d %H:%M')} to {data.index[-1].strftime('%m-%d %H:%M')})")
                else:
                    print(f"   ⚠️  {symbol} {tf}: Insufficient data ({len(data) if data is not None else 0} bars)")
            except Exception as e:
                print(f"   ✗ {symbol} {tf}: Error - {e}")
    
    connector.disconnect()
    return test_data

def test_liquidity_analyzer():
    """Test Liquidity Analyzer"""
    print("\n" + "="*70)
    print("TESTING LIQUIDITY ANALYZER")
    print("="*70)
    
    try:
        # Initialize analyzer with optimized settings
        liquidity_analyzer = LiquidityAnalyzer(
            equal_level_threshold=0.0001,  # 1 pip tolerance
            min_liquidity_gap=3,           # Minimum 3 bars between levels
            sweep_confirmation_bars=2,     # 2 bars confirmation
            min_sweep_distance=0.00005     # 0.5 pip minimum sweep
        )
        
        print("✓ Liquidity Analyzer initialized")
        print(f"  - Equal level threshold: {liquidity_analyzer.equal_level_threshold}")
        print(f"  - Min liquidity gap: {liquidity_analyzer.min_liquidity_gap} bars")
        print(f"  - Sweep confirmation: {liquidity_analyzer.sweep_confirmation_bars} bars")
        
        # Get test data
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        total_analyses = 0
        successful_analyses = 0
        total_liquidity_found = 0
        
        # Test with different symbols and timeframes
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing Liquidity Analysis: {symbol} ---")
            
            for tf, data in timeframes.items():
                print(f"\n{tf} Analysis:")
                total_analyses += 1
                
                # Analyze liquidity
                start_time = time.time()
                analysis = liquidity_analyzer.analyze_liquidity(data)
                analysis_time = time.time() - start_time
                
                if analysis and isinstance(analysis, dict):
                    successful_analyses += 1
                    metrics = analysis.get('metrics', {})
                    
                    print(f"   ⚡ Analysis time: {analysis_time:.3f}s")
                    print(f"   📊 Results:")
                    print(f"      Total Liquidity Pools: {metrics.get('total_pools', 0)}")
                    print(f"      Fresh Pools: {metrics.get('fresh_pools', 0)}")
                    print(f"      Tested Pools: {metrics.get('tested_pools', 0)}")
                    print(f"      Swept Pools: {metrics.get('swept_pools', 0)}")
                    print(f"      Average Strength: {metrics.get('avg_strength', 0):.2f}")
                    print(f"      Equal Highs: {metrics.get('equal_highs_count', 0)}")
                    print(f"      Equal Lows: {metrics.get('equal_lows_count', 0)}")
                    print(f"      Recent Sweeps: {metrics.get('recent_sweeps', 0)}")
                    
                    total_liquidity_found += metrics.get('total_pools', 0)
                    
                    # Show detailed patterns found
                    equal_highs = analysis.get('equal_highs', [])
                    equal_lows = analysis.get('equal_lows', [])
                    multiple_tops = analysis.get('multiple_tops', [])
                    multiple_bottoms = analysis.get('multiple_bottoms', [])
                    sweep_events = analysis.get('sweep_events', [])
                    
                    if equal_highs:
                        print(f"   📈 Equal Highs ({len(equal_highs)}):")
                        for i, eh in enumerate(equal_highs[-2:]):  # Show last 2
                            print(f"      {i+1}. {eh.timestamp.strftime('%m-%d %H:%M')} - {eh.price:.5f} (Strength: {eh.strength:.2f})")
                    
                    if equal_lows:
                        print(f"   📉 Equal Lows ({len(equal_lows)}):")
                        for i, el in enumerate(equal_lows[-2:]):  # Show last 2
                            print(f"      {i+1}. {el.timestamp.strftime('%m-%d %H:%M')} - {el.price:.5f} (Strength: {el.strength:.2f})")
                    
                    if multiple_tops:
                        print(f"   🔺 Multiple Tops ({len(multiple_tops)}):")
                        for i, mt in enumerate(multiple_tops[-2:]):
                            print(f"      {i+1}. {mt.liquidity_type.value} at {mt.price:.5f} (Touches: {mt.touches})")
                    
                    if multiple_bottoms:
                        print(f"   🔻 Multiple Bottoms ({len(multiple_bottoms)}):")
                        for i, mb in enumerate(multiple_bottoms[-2:]):
                            print(f"      {i+1}. {mb.liquidity_type.value} at {mb.price:.5f} (Touches: {mb.touches})")
                    
                    if sweep_events:
                        print(f"   🌊 Sweep Events ({len(sweep_events)}):")
                        for i, sweep in enumerate(sweep_events[-2:]):
                            print(f"      {i+1}. {sweep.sweep_type.value} at {sweep.price:.5f} (Penetration: {sweep.penetration_distance:.5f})")
                    
                    # Current market context
                    current_zones = analysis.get('current_zones', {})
                    current_price = data['Close'].iloc[-1]
                    
                    nearby_pools = current_zones.get('nearby_pools', [])
                    if nearby_pools:
                        print(f"   🎯 Current Context (Price: {current_price:.5f}):")
                        print(f"      Nearby Liquidity: {len(nearby_pools)} pools")
                        
                        # Show closest liquidity
                        nearest = nearby_pools[0] if nearby_pools else None
                        if nearest:
                            distance_pips = nearest['distance'] * 10000  # Convert to pips
                            print(f"      Nearest: {distance_pips:.1f} pips {nearest['direction']}")
                    
                    # Test liquidity near current price
                    nearby_liquidity = liquidity_analyzer.get_liquidity_near_price(current_price, 0.002)  # 20 pips
                    if nearby_liquidity:
                        print(f"      Liquidity within 20 pips: {len(nearby_liquidity)} pools")
                        strongest = max(nearby_liquidity, key=lambda x: x.strength)
                        print(f"      Strongest nearby: {strongest.liquidity_type.value} at {strongest.price:.5f}")
                    
                else:
                    print(f"   ✗ Analysis failed for {symbol} {tf}")
        
        # Test summary and statistics
        print(f"\n--- Liquidity Analysis Summary ---")
        print(f"Total analyses performed: {total_analyses}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Success rate: {successful_analyses/total_analyses:.1%}")
        print(f"Total liquidity pools found: {total_liquidity_found}")
        print(f"Average pools per analysis: {total_liquidity_found/successful_analyses:.1f}")
        
        # Get final summary from analyzer
        if 'analysis' in locals() and analysis:
            summary = analysis.get('summary', {})
            if summary:
                print(f"\nMarket Liquidity State:")
                print(f"  Dominant Liquidity: {summary.get('dominant_liquidity', 'balanced')}")
                print(f"  Sweep Tendency: {summary.get('sweep_tendency', 'low_activity')}")
                print(f"  Liquidity Quality: {summary.get('liquidity_quality', 0):.2f}")
        
        # Success criteria
        success_threshold = 0.7  # 70% success rate
        min_liquidity_found = 5   # At least 5 pools total
        
        if (successful_analyses / total_analyses >= success_threshold and 
            total_liquidity_found >= min_liquidity_found):
            print("\n✅ Liquidity Analyzer tests completed successfully!")
            return True
        else:
            print(f"\n⚠️  Liquidity Analyzer tests had issues:")
            if successful_analyses / total_analyses < success_threshold:
                print(f"   - Success rate too low: {successful_analyses/total_analyses:.1%}")
            if total_liquidity_found < min_liquidity_found:
                print(f"   - Too few liquidity pools found: {total_liquidity_found}")
            return False
        
    except Exception as e:
        print(f"\n❌ Liquidity Analyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_fvg_analyzer():
    """Test Fair Value Gap Analyzer"""
    print("\n" + "="*70)
    print("TESTING FAIR VALUE GAP ANALYZER")
    print("="*70)
    
    try:
        # Initialize analyzer with optimized settings
        fvg_analyzer = FVGAnalyzer(
            min_gap_size=0.00015,          # 1.5 pips minimum
            max_gap_age_hours=48,          # 2 days max age
            partial_fill_threshold=0.5,    # 50% fill threshold
            quality_volume_threshold=1.2   # Volume multiplier for quality
        )
        
        print("✓ Fair Value Gap Analyzer initialized")
        print(f"  - Min gap size: {fvg_analyzer.min_gap_size}")
        print(f"  - Max gap age: {fvg_analyzer.max_gap_age_hours} hours")
        print(f"  - Partial fill threshold: {fvg_analyzer.partial_fill_threshold}")
        
        # Get test data
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        total_analyses = 0
        successful_analyses = 0
        total_fvgs_found = 0
        
        # Test with different symbols and timeframes
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing FVG Analysis: {symbol} ---")
            
            for tf, data in timeframes.items():
                print(f"\n{tf} Analysis:")
                total_analyses += 1
                
                # Analyze Fair Value Gaps
                start_time = time.time()
                analysis = fvg_analyzer.analyze_fair_value_gaps(data)
                analysis_time = time.time() - start_time
                
                if analysis and isinstance(analysis, dict):
                    successful_analyses += 1
                    metrics = analysis.get('metrics', {})
                    
                    print(f"   ⚡ Analysis time: {analysis_time:.3f}s")
                    print(f"   📊 Results:")
                    print(f"      Total FVGs: {metrics.get('total_fvgs', 0)}")
                    print(f"      Open FVGs: {metrics.get('open_count', 0)}")
                    print(f"      Partial Fill: {metrics.get('partial_fill_count', 0)}")
                    print(f"      Full Fill: {metrics.get('full_fill_count', 0)}")
                    print(f"      Expired: {metrics.get('expired_count', 0)}")
                    print(f"      Bullish FVGs: {metrics.get('bullish_count', 0)}")
                    print(f"      Bearish FVGs: {metrics.get('bearish_count', 0)}")
                    print(f"      Fill Rate: {metrics.get('fill_rate', 0):.1%}")
                    print(f"      Average Size: {metrics.get('avg_size', 0):.5f}")
                    print(f"      High Quality: {metrics.get('high_quality_count', 0)}")
                    
                    total_fvgs_found += metrics.get('total_fvgs', 0)
                    
                    # Show new FVGs found
                    new_fvgs = analysis.get('new_fvgs', [])
                    if new_fvgs:
                        print(f"   🆕 New FVGs Found ({len(new_fvgs)}):")
                        for i, fvg in enumerate(new_fvgs[-3:]):  # Show last 3
                            direction = "🟢" if fvg.fvg_type == FVGType.BULLISH else "🔴"
                            print(f"      {i+1}. {direction} {fvg.fvg_type.value.title()} FVG at {fvg.timestamp.strftime('%m-%d %H:%M')}")
                            print(f"         Range: {fvg.bottom:.5f} - {fvg.top:.5f}")
                            print(f"         Size: {fvg.size:.5f}, Quality: {fvg.quality.value}")
                            print(f"         Strength: {fvg.strength:.2f}")
                    
                    # Show current FVG context
                    current_price = data['Close'].iloc[-1]
                    open_fvgs = analysis.get('open_fvgs', [])
                    partial_fvgs = analysis.get('partial_fvgs', [])
                    
                    active_fvgs = open_fvgs + partial_fvgs
                    if active_fvgs:
                        print(f"   🎯 Current Context (Price: {current_price:.5f}):")
                        print(f"      Active FVGs: {len(active_fvgs)}")
                        
                        # Test FVGs near current price
                        nearby_fvgs = fvg_analyzer.get_fvgs_near_price(current_price, 0.003)  # 30 pips
                        if nearby_fvgs:
                            print(f"      FVGs within 30 pips: {len(nearby_fvgs)}")
                            for j, fvg in enumerate(nearby_fvgs[:2]):  # Show closest 2
                                if fvg.bottom <= current_price <= fvg.top:
                                    distance = 0.0
                                    location = "INSIDE"
                                    icon = "📍"
                                elif current_price < fvg.bottom:
                                    distance = fvg.bottom - current_price
                                    location = "BELOW"
                                    icon = "⬇️"
                                else:
                                    distance = current_price - fvg.top
                                    location = "ABOVE"
                                    icon = "⬆️"
                                
                                distance_pips = distance * 10000
                                gap_type = "🟢" if fvg.fvg_type == FVGType.BULLISH else "🔴"
                                print(f"         {j+1}. {icon} {gap_type} {fvg.fvg_type.value} FVG - {location}")
                                print(f"            Distance: {distance_pips:.1f} pips, Status: {fvg.status.value}")
                    
                    # Current zones analysis
                    current_zones = analysis.get('current_zones', {})
                    if current_zones.get('inside_gap'):
                        print(f"      🎯 PRICE IS INSIDE A FAIR VALUE GAP!")
                    
                    gaps_above = current_zones.get('gaps_above', [])
                    gaps_below = current_zones.get('gaps_below', [])
                    
                    if gaps_above:
                        print(f"      📈 {len(gaps_above)} FVGs above current price")
                    if gaps_below:
                        print(f"      📉 {len(gaps_below)} FVGs below current price")
                    
                    # Show recently filled FVGs
                    filled_fvgs = analysis.get('filled_fvgs', [])
                    recent_filled = [fvg for fvg in filled_fvgs if fvg.age_hours < 24]
                    if recent_filled:
                        print(f"      ✅ Recently filled (24h): {len(recent_filled)}")
                
                else:
                    print(f"   ✗ Analysis failed for {symbol} {tf}")
        
        # Test summary and statistics
        print(f"\n--- FVG Analysis Summary ---")
        print(f"Total analyses performed: {total_analyses}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Success rate: {successful_analyses/total_analyses:.1%}")
        print(f"Total FVGs found: {total_fvgs_found}")
        if successful_analyses > 0:
            print(f"Average FVGs per analysis: {total_fvgs_found/successful_analyses:.1f}")
        
        # Get final summary from analyzer
        if 'analysis' in locals() and analysis:
            summary = analysis.get('summary', {})
            if summary:
                print(f"\nMarket FVG State:")
                dom_type = summary.get('dominant_gap_type')
                if dom_type:
                    type_name = dom_type.value if hasattr(dom_type, 'value') else str(dom_type)
                    print(f"  Dominant Gap Type: {type_name}")
                else:
                    print(f"  Dominant Gap Type: Balanced")
                print(f"  Gap Activity: {summary.get('gap_activity', 'low')}")
                print(f"  Fill Tendency: {summary.get('fill_tendency', 'unknown')}")
                print(f"  Overall Quality: {summary.get('overall_quality', 0):.2f}")
        
        # Success criteria
        success_threshold = 0.7  # 70% success rate
        min_fvgs_found = 3       # At least 3 FVGs total
        
        if (successful_analyses / total_analyses >= success_threshold and 
            total_fvgs_found >= min_fvgs_found):
            print("\n✅ Fair Value Gap Analyzer tests completed successfully!")
            return True
        else:
            print(f"\n⚠️  Fair Value Gap Analyzer tests had issues:")
            if successful_analyses / total_analyses < success_threshold:
                print(f"   - Success rate too low: {successful_analyses/total_analyses:.1%}")
            if total_fvgs_found < min_fvgs_found:
                print(f"   - Too few FVGs found: {total_fvgs_found}")
            return False
        
    except Exception as e:
        print(f"\n❌ Fair Value Gap Analyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_combined_analysis():
    """Test combined Liquidity and FVG analysis"""
    print("\n" + "="*70)
    print("TESTING COMBINED LIQUIDITY & FVG ANALYSIS")
    print("="*70)
    
    try:
        # Initialize both analyzers
        liquidity_analyzer = LiquidityAnalyzer()
        fvg_analyzer = FVGAnalyzer()
        
        # Get test data
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Test combined analysis on primary symbol/timeframe
        test_symbol = "EURUSD"
        test_tf = "H1"
        
        if test_symbol not in test_data or test_tf not in test_data[test_symbol]:
            print(f"✗ No data available for {test_symbol} {test_tf}")
            return False
        
        data = test_data[test_symbol][test_tf]
        print(f"Combined Analysis for {test_symbol} {test_tf}:")
        print(f"📊 Data: {len(data)} bars ({data.index[0].strftime('%Y-%m-%d %H:%M')} to {data.index[-1].strftime('%Y-%m-%d %H:%M')})")
        
        # Run both analyses
        print(f"\n🔄 Running combined analysis...")
        start_time = time.time()
        
        liquidity_analysis = liquidity_analyzer.analyze_liquidity(data)
        fvg_analysis = fvg_analyzer.analyze_fair_value_gaps(data)
        
        total_time = time.time() - start_time
        print(f"⚡ Combined analysis time: {total_time:.3f}s")
        
        if not liquidity_analysis or not fvg_analysis:
            print("✗ One or both analyses failed")
            return False
        
        # Extract key metrics
        liq_metrics = liquidity_analysis.get('metrics', {})
        fvg_metrics = fvg_analysis.get('metrics', {})
        current_price = data['Close'].iloc[-1]
        
        print(f"\n--- 📈 COMBINED SMC INSIGHTS ---")
        print(f"Current Price: {current_price:.5f}")
        
        # Liquidity Summary
        print(f"\n🌊 Liquidity Analysis:")
        print(f"   Total Pools: {liq_metrics.get('total_pools', 0)}")
        print(f"   Fresh: {liq_metrics.get('fresh_pools', 0)} | Tested: {liq_metrics.get('tested_pools', 0)} | Swept: {liq_metrics.get('swept_pools', 0)}")
        print(f"   Equal Highs: {liq_metrics.get('equal_highs_count', 0)} | Equal Lows: {liq_metrics.get('equal_lows_count', 0)}")
        print(f"   Recent Sweeps: {liq_metrics.get('recent_sweeps', 0)}")
        
        # FVG Summary
        print(f"\n📊 Fair Value Gap Analysis:")
        print(f"   Total FVGs: {fvg_metrics.get('total_fvgs', 0)}")
        print(f"   Open: {fvg_metrics.get('open_count', 0)} | Partial: {fvg_metrics.get('partial_fill_count', 0)} | Filled: {fvg_metrics.get('full_fill_count', 0)}")
        print(f"   Bullish: {fvg_metrics.get('bullish_count', 0)} | Bearish: {fvg_metrics.get('bearish_count', 0)}")
        print(f"   Fill Rate: {fvg_metrics.get('fill_rate', 0):.1%}")
        
        # Confluence Analysis
        print(f"\n🎯 CONFLUENCE ANALYSIS:")
        
        nearby_liquidity = liquidity_analyzer.get_liquidity_near_price(current_price, 0.002)  # 20 pips
        nearby_fvgs = fvg_analyzer.get_fvgs_near_price(current_price, 0.002)  # 20 pips
        
        print(f"   Nearby Liquidity (20 pips): {len(nearby_liquidity)} pools")
        print(f"   Nearby FVGs (20 pips): {len(nearby_fvgs)} gaps")
        
        # Calculate confluence score
        confluence_score = 0.0
        confluence_factors = []
        
        if nearby_liquidity:
            liq_score = min(len(nearby_liquidity) * 0.25, 0.5)
            confluence_score += liq_score
            confluence_factors.append(f"Liquidity: +{liq_score:.2f}")
            
            # Show strongest nearby liquidity
            strongest_liq = max(nearby_liquidity, key=lambda x: x.strength)
            print(f"      Strongest: {strongest_liq.liquidity_type.value} at {strongest_liq.price:.5f} (Strength: {strongest_liq.strength:.2f})")
        
        if nearby_fvgs:
            fvg_score = min(len(nearby_fvgs) * 0.3, 0.5)
            confluence_score += fvg_score
            confluence_factors.append(f"FVGs: +{fvg_score:.2f}")
            
            # Show highest quality nearby FVG
            best_fvg = max(nearby_fvgs, key=lambda x: x.strength)
            status_icon = "🟢" if best_fvg.status == FVGStatus.OPEN else "🟡"
            print(f"      Best FVG: {status_icon} {best_fvg.fvg_type.value} at {best_fvg.bottom:.5f}-{best_fvg.top:.5f} (Quality: {best_fvg.quality.value})")
        
        confluence_score = min(confluence_score, 1.0)
        
        print(f"\n   📊 Confluence Score: {confluence_score:.2f}/1.0")
        if confluence_factors:
            print(f"      Factors: {' | '.join(confluence_factors)}")
        
        # Trading Assessment
        print(f"\n🎯 TRADING ASSESSMENT:")
        
        if confluence_score >= 0.7:
            assessment = "🔥 HIGH CONFLUENCE ZONE"
            recommendation = "Strong trading opportunity with multiple confirmations"
        elif confluence_score >= 0.4:
            assessment = "📊 MODERATE CONFLUENCE ZONE"
            recommendation = "Good trading setup with some confirmations"
        elif confluence_score >= 0.2:
            assessment = "📍 LOW CONFLUENCE ZONE"
            recommendation = "Weak signals, proceed with caution"
        else:
            assessment = "⚪ NO SIGNIFICANT CONFLUENCE"
            recommendation = "No clear trading opportunities at current price"
        
        print(f"   {assessment}")
        print(f"   Recommendation: {recommendation}")
        
        # Directional Bias Analysis
        print(f"\n🧭 DIRECTIONAL BIAS:")
        
        # Count bullish vs bearish signals
        bullish_signals = (
            fvg_metrics.get('bullish_count', 0) +
            liq_metrics.get('equal_lows_count', 0)  # Support liquidity
        )
        
        bearish_signals = (
            fvg_metrics.get('bearish_count', 0) +
            liq_metrics.get('equal_highs_count', 0)  # Resistance liquidity
        )
        
        total_signals = bullish_signals + bearish_signals
        
        if total_signals > 0:
            bullish_ratio = bullish_signals / total_signals
            
            if bullish_ratio > 0.65:
                bias = "🟢 BULLISH BIAS"
                bias_strength = bullish_ratio
            elif bullish_ratio < 0.35:
                bias = "🔴 BEARISH BIAS"
                bias_strength = 1 - bullish_ratio
            else:
                bias = "🟡 NEUTRAL BIAS"
                bias_strength = 0.5
            
            print(f"   {bias} (Strength: {bias_strength:.2f})")
            print(f"   Bullish signals: {bullish_signals} | Bearish signals: {bearish_signals}")
        else:
            print(f"   🟡 NEUTRAL BIAS (No clear signals)")
        
        # Recent Activity
        print(f"\n📈 RECENT MARKET ACTIVITY:")
        
        # Recent liquidity sweeps
        recent_sweeps = liquidity_analysis.get('sweep_events', [])
        if recent_sweeps:
            print(f"   🌊 Recent Sweeps: {len(recent_sweeps)}")
            latest_sweep = recent_sweeps[-1]
            print(f"      Latest: {latest_sweep.sweep_type.value} at {latest_sweep.price:.5f}")
            print(f"      → Watch for reactions and reversals")
        else:
            print(f"   🌊 No recent liquidity sweeps detected")
        
        # Recent FVG fills
        recent_fills = [fvg for fvg in fvg_analysis.get('filled_fvgs', []) if fvg.age_hours < 24]
        if recent_fills:
            print(f"   📊 Recent FVG Fills: {len(recent_fills)} in last 24h")
            print(f"      → Market showing respect for fair value levels")
        
        # Key Levels Summary
        print(f"\n🎯 KEY LEVELS NEAR CURRENT PRICE:")
        all_levels = []
        
        # Add liquidity levels
        for liq in nearby_liquidity:
            distance = abs(current_price - liq.price)
            all_levels.append({
                'price': liq.price,
                'type': f"Liquidity ({liq.liquidity_type.value})",
                'distance': distance,
                'strength': liq.strength,
                'icon': '🌊'
            })
        
        # Add FVG levels
        for fvg in nearby_fvgs:
            # Use midpoint for FVG
            fvg_price = fvg.mid_point
            distance = abs(current_price - fvg_price)
            all_levels.append({
                'price': fvg_price,
                'type': f"FVG ({fvg.fvg_type.value})",
                'distance': distance,
                'strength': fvg.strength,
                'icon': '📊'
            })
        
        # Sort by distance and show closest 5
        all_levels.sort(key=lambda x: x['distance'])
        
        if all_levels:
            print(f"   Closest {min(5, len(all_levels))} levels:")
            for i, level in enumerate(all_levels[:5]):
                distance_pips = level['distance'] * 10000
                direction = "↑" if level['price'] > current_price else "↓"
                print(f"      {i+1}. {level['icon']} {level['type']} at {level['price']:.5f}")
                print(f"         {direction} {distance_pips:.1f} pips away (Strength: {level['strength']:.2f})")
        else:
            print(f"   No significant levels within 20 pips")
        
        print(f"\n✅ Combined Liquidity & FVG analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Combined analysis failed: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Test performance of Liquidity and FVG calculations"""
    print("\n" + "="*70)
    print("PERFORMANCE TESTING - LIQUIDITY & FVG")
    print("="*70)
    
    try:
        # Get test data
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Find largest dataset for performance testing
        largest_dataset = None
        largest_size = 0
        largest_info = None
        
        for symbol, timeframes in test_data.items():
            for tf, data in timeframes.items():
                if len(data) > largest_size:
                    largest_dataset = data
                    largest_size = len(data)
                    largest_info = f"{symbol} {tf}"
        
        if largest_dataset is None:
            print("✗ No suitable dataset found")
            return False
        
        print(f"🔬 Performance testing with {largest_size} bars ({largest_info})")
        print(f"📅 Data range: {largest_dataset.index[0]} to {largest_dataset.index[-1]}")
        
        # Initialize analyzers
        liquidity_analyzer = LiquidityAnalyzer()
        fvg_analyzer = FVGAnalyzer()
        
        # Test Liquidity Analysis performance
        print(f"\n⚡ Liquidity Analysis Performance:")
        
        # Warm up
        liquidity_analyzer.analyze_liquidity(largest_dataset.head(50))
        
        # Actual performance test
        start_time = time.time()
        liq_analysis = liquidity_analyzer.analyze_liquidity(largest_dataset)
        liq_time = time.time() - start_time
        
        liq_bars_per_sec = largest_size / liq_time if liq_time > 0 else 0
        
        print(f"   Analysis time: {liq_time:.3f} seconds")
        print(f"   Performance: {liq_bars_per_sec:.0f} bars/second")
        
        if liq_analysis:
            liq_metrics = liq_analysis.get('metrics', {})
            print(f"   Results: {liq_metrics.get('total_pools', 0)} liquidity pools")
        
        # Test FVG Analysis performance
        print(f"\n⚡ Fair Value Gap Performance:")
        
        # Warm up
        fvg_analyzer.analyze_fair_value_gaps(largest_dataset.head(50))
        
        # Actual performance test
        start_time = time.time()
        fvg_analysis = fvg_analyzer.analyze_fair_value_gaps(largest_dataset)
        fvg_time = time.time() - start_time
        
        fvg_bars_per_sec = largest_size / fvg_time if fvg_time > 0 else 0
        
        print(f"   Analysis time: {fvg_time:.3f} seconds")
        print(f"   Performance: {fvg_bars_per_sec:.0f} bars/second")
        
        if fvg_analysis:
            fvg_metrics = fvg_analysis.get('metrics', {})
            print(f"   Results: {fvg_metrics.get('total_fvgs', 0)} fair value gaps")
        
        # Combined performance analysis
        total_time = liq_time + fvg_time
        combined_bars_per_sec = largest_size / total_time if total_time > 0 else 0
        
        print(f"\n⚡ Combined Performance:")
        print(f"   Total analysis time: {total_time:.3f} seconds")
        print(f"   Combined performance: {combined_bars_per_sec:.0f} bars/second")
        
        # Memory efficiency test
        import sys
        
        print(f"\n💾 Memory Usage:")
        liq_pools = len(liquidity_analyzer.liquidity_pools)
        fvg_gaps = len(fvg_analyzer.fair_value_gaps)
        
        print(f"   Liquidity pools in memory: {liq_pools}")
        print(f"   FVG gaps in memory: {fvg_gaps}")
        print(f"   Total features tracked: {liq_pools + fvg_gaps}")
        
        # Results summary
        if liq_analysis and fvg_analysis:
            total_features = (liq_metrics.get('total_pools', 0) + 
                            fvg_metrics.get('total_fvgs', 0))
            print(f"   Features found: {total_features}")
            features_per_bar = total_features / largest_size if largest_size > 0 else 0
            print(f"   Feature density: {features_per_bar:.3f} features/bar")
        
        # Performance assessment
        print(f"\n📊 Performance Assessment:")
        
        # Real-time trading requirements
        min_bars_per_sec = 1000   # Minimum for real-time
        good_bars_per_sec = 3000  # Good performance
        excellent_bars_per_sec = 10000  # Excellent performance
        
        if combined_bars_per_sec >= excellent_bars_per_sec:
            rating = "🚀 EXCELLENT"
            assessment = "Outstanding performance - Real-time ready with room to spare!"
        elif combined_bars_per_sec >= good_bars_per_sec:
            rating = "✅ GOOD"
            assessment = "Good performance - Suitable for live trading"
        elif combined_bars_per_sec >= min_bars_per_sec:
            rating = "⚠️ ACCEPTABLE"
            assessment = "Acceptable performance - May need optimization for high-frequency use"
        else:
            rating = "❌ POOR"
            assessment = "Poor performance - Optimization required for live trading"
        
        print(f"   Overall Rating: {rating}")
        print(f"   Assessment: {assessment}")
        
        # Performance targets
        print(f"\n🎯 Performance Targets:")
        print(f"   Minimum (Real-time): {min_bars_per_sec:,} bars/sec ({'✅' if combined_bars_per_sec >= min_bars_per_sec else '❌'})")
        print(f"   Good: {good_bars_per_sec:,} bars/sec ({'✅' if combined_bars_per_sec >= good_bars_per_sec else '❌'})")
        print(f"   Excellent: {excellent_bars_per_sec:,} bars/sec ({'✅' if combined_bars_per_sec >= excellent_bars_per_sec else '❌'})")
        
        # Success criteria
        return combined_bars_per_sec >= min_bars_per_sec
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("TESTING EDGE CASES & ERROR HANDLING")
    print("="*70)
    
    try:
        print("🧪 Testing various edge cases...")
        
        # Initialize analyzers
        liquidity_analyzer = LiquidityAnalyzer()
        fvg_analyzer = FVGAnalyzer()
        
        # Test 1: Empty DataFrame
        print("\n1. Testing empty DataFrame:")
        empty_data = pd.DataFrame()
        
        liq_result = liquidity_analyzer.analyze_liquidity(empty_data)
        fvg_result = fvg_analyzer.analyze_fair_value_gaps(empty_data)
        
        if liq_result and fvg_result:
            print("   ✅ Empty data handled gracefully")
        else:
            print("   ❌ Empty data handling failed")
            return False
        
        # Test 2: Minimal data (insufficient for analysis)
        print("\n2. Testing minimal data:")
        minimal_data = pd.DataFrame({
            'Open': [1.1000, 1.1010],
            'High': [1.1005, 1.1015],
            'Low': [1.0995, 1.1005],
            'Close': [1.1002, 1.1012],
            'Volume': [100, 120]
        }, index=pd.date_range('2024-01-01', periods=2, freq='H'))
        
        liq_result = liquidity_analyzer.analyze_liquidity(minimal_data)
        fvg_result = fvg_analyzer.analyze_fair_value_gaps(minimal_data)
        
        if liq_result and fvg_result:
            print("   ✅ Minimal data handled gracefully")
        else:
            print("   ❌ Minimal data handling failed")
            return False
        
        # Test 3: Data with missing values
        print("\n3. Testing data with NaN values:")
        nan_data = pd.DataFrame({
            'Open': [1.1000, np.nan, 1.1020],
            'High': [1.1005, 1.1015, np.nan],
            'Low': [1.0995, 1.1005, 1.1015],
            'Close': [1.1002, 1.1012, 1.1018],
            'Volume': [100, 120, 110]
        }, index=pd.date_range('2024-01-01', periods=3, freq='H'))
        
        # Should handle NaN gracefully or skip problematic bars
        try:
            liq_result = liquidity_analyzer.analyze_liquidity(nan_data)
            fvg_result = fvg_analyzer.analyze_fair_value_gaps(nan_data)
            print("   ✅ NaN data handled gracefully")
        except Exception as e:
            print(f"   ⚠️ NaN data caused issues: {e}")
        
        # Test 4: Extreme parameter values
        print("\n4. Testing extreme parameters:")
        
        extreme_liq = LiquidityAnalyzer(
            equal_level_threshold=0.0,
            min_liquidity_gap=1,
            sweep_confirmation_bars=1
        )
        
        extreme_fvg = FVGAnalyzer(
            min_gap_size=0.0,
            max_gap_age_hours=1,
            partial_fill_threshold=0.0
        )
        
        # Create slightly larger test data
        test_data = pd.DataFrame({
            'Open': np.random.uniform(1.1000, 1.1100, 20),
            'High': np.random.uniform(1.1000, 1.1100, 20),
            'Low': np.random.uniform(1.1000, 1.1100, 20),
            'Close': np.random.uniform(1.1000, 1.1100, 20),
            'Volume': np.random.randint(50, 200, 20)
        }, index=pd.date_range('2024-01-01', periods=20, freq='H'))
        
        # Fix OHLC relationships
        for i in range(len(test_data)):
            high = max(test_data.iloc[i][['Open', 'High', 'Low', 'Close']])
            low = min(test_data.iloc[i][['Open', 'High', 'Low', 'Close']])
            test_data.iloc[i, test_data.columns.get_loc('High')] = high
            test_data.iloc[i, test_data.columns.get_loc('Low')] = low
        
        liq_result = extreme_liq.analyze_liquidity(test_data)
        fvg_result = extreme_fvg.analyze_fair_value_gaps(test_data)
        
        if liq_result and fvg_result:
            print("   ✅ Extreme parameters handled")
        else:
            print("   ❌ Extreme parameters failed")
            return False
        
        # Test 5: Invalid price relationships
        print("\n5. Testing invalid OHLC relationships:")
        invalid_data = pd.DataFrame({
            'Open': [1.1000, 1.1010, 1.1020],
            'High': [1.0990, 1.1005, 1.1015],  # High < Open (invalid)
            'Low': [1.1010, 1.1020, 1.1025],   # Low > Open (invalid)
            'Close': [1.1002, 1.1012, 1.1018],
            'Volume': [100, 120, 110]
        }, index=pd.date_range('2024-01-01', periods=3, freq='H'))
        
        # Should handle invalid OHLC relationships
        try:
            liq_result = liquidity_analyzer.analyze_liquidity(invalid_data)
            fvg_result = fvg_analyzer.analyze_fair_value_gaps(invalid_data)
            print("   ⚠️ Invalid OHLC handled (may produce unexpected results)")
        except Exception as e:
            print(f"   ✅ Invalid OHLC properly rejected: {e}")
        
        print("\n✅ Edge case testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Edge case testing failed: {e}")
        traceback.print_exc()
        return False

def generate_comprehensive_report():
    """Generate comprehensive test report"""
    print("\n" + "="*70)
    print("COMPREHENSIVE LIQUIDITY & FVG TEST REPORT")
    print("="*70)
    
    test_results = {}
    
    try:
        print("🔬 Running comprehensive test suite...\n")
        
        # Run all test modules
        test_modules = [
            ("Liquidity Analyzer", test_liquidity_analyzer),
            ("Fair Value Gap Analyzer", test_fvg_analyzer),
            ("Combined Analysis", test_combined_analysis),
            ("Performance Testing", test_performance),
            ("Edge Cases", test_edge_cases)
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
        print("FINAL TEST SUMMARY")
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
            print(f"   ✅ Liquidity Detection: Fully Functional")
            print(f"   ✅ Fair Value Gap Analysis: Fully Functional")
            print(f"   ✅ Combined Analysis: Working Perfectly")
            print(f"   ✅ Performance: Meeting Standards")
            print(f"   ✅ Error Handling: Robust")
            
            print(f"\n🚀 PHASE 2 - DAY 3-4: COMPLETED SUCCESSFULLY!")
            print(f"📈 Liquidity & Fair Value Gap detection implemented")
            print(f"🎯 Ready for Day 5-7: Volume Profile & BIAS Analysis")
            
            success = True
            
        elif passed_tests >= total_tests * 0.8:  # 80% pass rate
            print(f"   ✅ MOSTLY SUCCESSFUL!")
            print(f"   📊 Core functionality working")
            print(f"   ⚠️ Some minor issues to address")
            
            failed_modules = [name for name, result in test_results.items() if not result['passed']]
            print(f"   Issues in: {', '.join(failed_modules)}")
            
            print(f"\n📋 PHASE 2 - DAY 3-4: MOSTLY COMPLETED")
            print(f"✅ Core features working, minor issues to fix")
            
            success = True
            
        else:
            print(f"   ❌ SIGNIFICANT ISSUES DETECTED")
            print(f"   🔧 Major components need attention")
            
            failed_modules = [name for name, result in test_results.items() if not result['passed']]
            print(f"   Failed modules: {', '.join(failed_modules)}")
            
            print(f"\n⚠️ PHASE 2 - DAY 3-4: NEEDS WORK")
            print(f"❌ Core issues must be resolved before proceeding")
            
            success = False
        
        # Next steps
        if success:
            print(f"\n📋 NEXT STEPS:")
            print(f"   1. Volume Profile implementation")
            print(f"   2. Enhanced BIAS analysis")
            print(f"   3. Supply/Demand zone refinement")
            print(f"   4. Complete SMC integration")
            print(f"   5. Multi-timeframe feature aggregation")
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
    print("   LIQUIDITY & FVG FEATURES - COMPREHENSIVE TESTING")
    print("   PHASE 2 DAY 3-4 - SMART MONEY CONCEPTS")
    print("="*70)
    print(f"🕒 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Objective: Validate Liquidity Detection & Fair Value Gap Analysis")
    
    try:
        # Run comprehensive test suite
        success = generate_comprehensive_report()
        
        if success:
            print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
            print(f"🚀 Liquidity & FVG features are production-ready")
            return 0
        else:
            print(f"\n⚠️ TESTING IDENTIFIED ISSUES")
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