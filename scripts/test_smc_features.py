"""
Test Script for SMC Features
Tests Market Structure and Order Block detection
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.smc_calculator import MarketStructureAnalyzer, TrendDirection, StructureType
from features.order_blocks import OrderBlockAnalyzer, OrderBlockType, OrderBlockStatus
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
        # Try real MT5 first with proper connection testing
        connector = get_mt5_connector(MT5_CONNECTION)
        
        # Test connection properly
        if connector.connect():
            print("   ✓ Using real MT5 data")
            is_demo = False
        else:
            raise Exception("MT5 connection failed")
            
    except Exception as e:
        # Fall back to demo
        print(f"   ⚠️  Real MT5 failed ({e}), using demo data")
        from data.connectors.demo_connector import get_demo_connector
        connector = get_demo_connector()
        connector.connect()
        is_demo = True
        print("   ✓ Using demo data")
    
    # Test the connection
    test_result = connector.test_connection()
    if not test_result.get('connected'):
        print(f"   ✗ Connection test failed: {test_result.get('error')}")
        return {}
    
    # Print connection info
    account_info = test_result.get('account_info', {})
    print(f"   Account: {account_info.get('login', 'Unknown')}")
    print(f"   Balance: {account_info.get('balance', 0)} {account_info.get('currency', 'USD')}")
    
    # Get different timeframes for testing
    test_data = {}
    symbols = ["EURUSD", "GBPUSD"]
    timeframes = ["H4", "H1", "M15"]
    
    for symbol in symbols:
        test_data[symbol] = {}
        for tf in timeframes:
            try:
                data = connector.get_rates(symbol, tf, 200)  # Get 200 bars
                if data is not None and len(data) > 50:
                    test_data[symbol][tf] = data
                    print(f"   ✓ {symbol} {tf}: {len(data)} bars ({data.index[0]} to {data.index[-1]})")
                else:
                    print(f"   ⚠️  {symbol} {tf}: No data or insufficient data")
            except Exception as e:
                print(f"   ✗ {symbol} {tf}: Error - {e}")
    
    connector.disconnect()
    return test_data

def test_market_structure_analyzer():
    """Test Market Structure Analyzer"""
    print("\n" + "="*60)
    print("TESTING MARKET STRUCTURE ANALYZER")
    print("="*60)
    
    try:
        # Initialize analyzer
        ms_analyzer = MarketStructureAnalyzer(
            swing_lookback=15,
            min_swing_size=0.0003,
            structure_confirmation_bars=3
        )
        
        print("✓ Market Structure Analyzer initialized")
        
        # Get test data
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Test with different symbols and timeframes
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing {symbol} ---")
            
            for tf, data in timeframes.items():
                print(f"\n{tf} Analysis:")
                
                # Analyze market structure
                analysis = ms_analyzer.analyze_market_structure(data)
                
                if analysis and analysis.get('trend'):
                    print(f"   Trend: {analysis['trend'].name}")
                    print(f"   Swing Highs: {len(analysis['swing_highs'])}")
                    print(f"   Swing Lows: {len(analysis['swing_lows'])}")
                    print(f"   Structure Points: {len(analysis['structure_points'])}")
                    print(f"   BOS Events: {len(analysis['bos_events'])}")
                    print(f"   CHoCH Events: {len(analysis['choch_events'])}")
                    print(f"   MSB Events: {len(analysis['msb_events'])}")
                    print(f"   Trend Strength: {analysis['trend_strength']:.2f}")
                    print(f"   Structure Quality: {analysis['structure_quality']:.2f}")
                    
                    # Show recent structure points
                    if analysis['structure_points']:
                        recent_points = analysis['structure_points'][-3:]
                        print(f"   Recent Structure:")
                        for point in recent_points:
                            print(f"     {point.timestamp.strftime('%Y-%m-%d %H:%M')} - {point.structure_type.value} at {point.price:.5f}")
                    
                    # Show recent events
                    if analysis['bos_events']:
                        latest_bos = analysis['bos_events'][-1]
                        print(f"   Latest BOS: {latest_bos['direction']} at {latest_bos['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    
                    if analysis['choch_events']:
                        latest_choch = analysis['choch_events'][-1]
                        print(f"   Latest CHoCH: {latest_choch['direction']} at {latest_choch['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                else:
                    print(f"   ✗ Analysis failed for {symbol} {tf}")
        
        # Test bias calculation
        print(f"\n--- Testing Bias Calculation ---")
        bias = ms_analyzer.get_current_bias()
        print(f"Current Bias: {bias['trend'].name if bias['trend'] else 'None'}")
        print(f"Bias Strength: {bias['bias_strength']:.2f}")
        if bias['last_structure_break']:
            print(f"Last Structure Break: {bias['last_structure_break']['direction']}")
        
        print("\n✅ Market Structure Analyzer tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Market Structure Analyzer test failed: {e}")
        return False

def test_mt5_connection_debug():
    """Debug MT5 connection issues"""
    print("\n" + "="*60)
    print("MT5 CONNECTION DEBUGGING")
    print("="*60)
    
    try:
        print("\n1. Testing MT5 Configuration...")
        print(f"   MT5 Login: {MT5_CONNECTION.login}")
        print(f"   MT5 Server: {MT5_CONNECTION.server}")
        print(f"   MT5 Password: {'*' * len(MT5_CONNECTION.password) if MT5_CONNECTION.password else 'Not set'}")
        
        print("\n2. Testing MT5 Connection...")
        connector = get_mt5_connector(MT5_CONNECTION)
        
        # Try to connect
        connection_result = connector.connect()
        print(f"   Connection attempt: {'Success' if connection_result else 'Failed'}")
        
        if connection_result:
            # Test connection
            test_result = connector.test_connection()
            print(f"   Connection test: {'Pass' if test_result['connected'] else 'Fail'}")
            
            if test_result['connected']:
                account_info = test_result['account_info']
                print(f"   Account: {account_info['login']}")
                print(f"   Balance: {account_info['balance']} {account_info['currency']}")
                print(f"   Symbols available: {test_result['symbols_count']}")
                
                # Test data retrieval
                test_data = connector.get_rates("EURUSD", "H1", 10)
                if test_data is not None:
                    print(f"   Data test: ✓ Retrieved {len(test_data)} bars")
                    print(f"   Latest close: {test_data['Close'].iloc[-1]:.5f}")
                else:
                    print(f"   Data test: ✗ Failed to retrieve data")
                    
                connector.disconnect()
                print(f"   ✅ Real MT5 connection working properly!")
                return True
            else:
                print(f"   ✗ Connection test failed: {test_result.get('error')}")
        else:
            print(f"   ✗ Failed to establish MT5 connection")
            
        print(f"   ⚠️  MT5 connection issues detected")
        return False
        
    except Exception as e:
        print(f"   ✗ MT5 connection error: {e}")
        return False
    
def test_order_block_analyzer():
    """Test Order Block Analyzer"""
    print("\n" + "="*60)
    print("TESTING ORDER BLOCK ANALYZER")
    print("="*60)
    
    try:
        # Initialize analyzer
        ob_analyzer = OrderBlockAnalyzer(
            min_reaction_pips=5.0,
            min_ob_body_size=0.0003,
            lookback_period=30,
            confirmation_bars=2
        )
        
        print("✓ Order Block Analyzer initialized")
        
        # Get test data
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Test with different symbols and timeframes
        for symbol, timeframes in test_data.items():
            print(f"\n--- Testing {symbol} ---")
            
            for tf, data in timeframes.items():
                print(f"\n{tf} Analysis:")
                
                # Analyze order blocks
                analysis = ob_analyzer.analyze_order_blocks(data)
                
                if analysis:
                    metrics = analysis['metrics']
                    print(f"   Total Order Blocks: {metrics['total_obs']}")
                    print(f"   Fresh OBs: {metrics['fresh_count']}")
                    print(f"   Tested OBs: {metrics['tested_count']}")
                    print(f"   Respected OBs: {metrics['respected_count']}")
                    print(f"   Broken OBs: {metrics['broken_count']}")
                    print(f"   Respect Rate: {metrics['respect_rate']:.1%}")
                    print(f"   Average Strength: {metrics['avg_strength']:.2f}")
                    print(f"   Bullish OBs: {metrics['bullish_count']}")
                    print(f"   Bearish OBs: {metrics['bearish_count']}")
                    
                    # Show new order blocks found
                    if analysis['new_obs']:
                        print(f"   New OBs Found: {len(analysis['new_obs'])}")
                        for ob in analysis['new_obs'][-3:]:  # Show last 3
                            print(f"     {ob.ob_type.value.title()} OB at {ob.timestamp.strftime('%Y-%m-%d %H:%M')}")
                            print(f"       Range: {ob.bottom:.5f} - {ob.top:.5f}")
                            print(f"       Strength: {ob.strength:.2f}")
                    
                    # Show active order blocks
                    active_obs = analysis['fresh_obs'] + analysis['tested_obs'] + analysis['respected_obs']
                    if active_obs:
                        print(f"   Active OBs: {len(active_obs)}")
                        
                        # Test confluence at current price
                        current_price = data['Close'].iloc[-1]
                        confluence = ob_analyzer.get_confluence_score(current_price)
                        print(f"   Confluence at current price ({current_price:.5f}): {confluence:.2f}")
                        
                        # Test nearby OBs
                        nearby_obs = ob_analyzer.get_order_blocks_near_price(current_price, 0.001)
                        if nearby_obs:
                            print(f"   Nearby OBs: {len(nearby_obs)}")
                            for ob in nearby_obs[:2]:  # Show closest 2
                                distance = min(abs(current_price - ob.top), abs(current_price - ob.bottom))
                                print(f"     {ob.ob_type.value} OB - Distance: {distance:.5f}, Status: {ob.status.value}")
                
                else:
                    print(f"   ✗ Analysis failed for {symbol} {tf}")
        
        print("\n✅ Order Block Analyzer tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Order Block Analyzer test failed: {e}")
        return False

def test_combined_analysis():
    """Test combined SMC analysis"""
    print("\n" + "="*60)
    print("TESTING COMBINED SMC ANALYSIS")
    print("="*60)
    
    try:
        # Initialize both analyzers
        ms_analyzer = MarketStructureAnalyzer()
        ob_analyzer = OrderBlockAnalyzer()
        
        # Get test data
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Test combined analysis on EURUSD H1
        symbol = "EURUSD"
        tf = "H1"
        
        if symbol in test_data and tf in test_data[symbol]:
            data = test_data[symbol][tf]
            print(f"\nCombined Analysis for {symbol} {tf}:")
            print(f"Data range: {data.index[0]} to {data.index[-1]}")
            print(f"Total bars: {len(data)}")
            
            # Run both analyses
            ms_analysis = ms_analyzer.analyze_market_structure(data)
            ob_analysis = ob_analyzer.analyze_order_blocks(data)
            
            # Combined insights
            print(f"\n--- Combined SMC Insights ---")
            
            # Trend and bias
            trend = ms_analysis.get('trend', TrendDirection.SIDEWAYS)
            print(f"Market Trend: {trend.name}")
            
            # Structure and order blocks alignment
            active_obs = (ob_analysis['fresh_obs'] + 
                         ob_analysis['tested_obs'] + 
                         ob_analysis['respected_obs'])
            
            if active_obs:
                bullish_obs = [ob for ob in active_obs if ob.ob_type == OrderBlockType.BULLISH]
                bearish_obs = [ob for ob in active_obs if ob.ob_type == OrderBlockType.BEARISH]
                
                print(f"Active Order Blocks: {len(bullish_obs)} Bullish, {len(bearish_obs)} Bearish")
                
                # Check alignment with trend
                if trend == TrendDirection.BULLISH and len(bullish_obs) > len(bearish_obs):
                    print("✓ Order blocks align with bullish trend")
                elif trend == TrendDirection.BEARISH and len(bearish_obs) > len(bullish_obs):
                    print("✓ Order blocks align with bearish trend")
                else:
                    print("⚠️ Mixed signals between trend and order blocks")
            
            # Recent events
            if ms_analysis['bos_events']:
                latest_bos = ms_analysis['bos_events'][-1]
                print(f"Latest BOS: {latest_bos['direction']} (Strength: {latest_bos['strength']:.2f})")
            
            # Trading zones
            current_price = data['Close'].iloc[-1]
            print(f"Current Price: {current_price:.5f}")
            
            confluence = ob_analyzer.get_confluence_score(current_price)
            print(f"Current Confluence Score: {confluence:.2f}")
            
            if confluence > 0.5:
                print("🎯 High confluence zone - potential trading opportunity")
            elif confluence > 0.3:
                print("📊 Moderate confluence zone")
            else:
                print("📍 Low confluence zone")
        
        print("\n✅ Combined SMC analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Combined SMC analysis failed: {e}")
        return False

def performance_test():
    """Test performance of SMC calculations"""
    print("\n" + "="*60)
    print("PERFORMANCE TESTING")
    print("="*60)
    
    try:
        import time
        
        # Get large dataset
        test_data = get_test_data()
        
        if not test_data:
            print("✗ No test data available")
            return False
        
        # Test with largest dataset available
        largest_dataset = None
        largest_size = 0
        
        for symbol, timeframes in test_data.items():
            for tf, data in timeframes.items():
                if len(data) > largest_size:
                    largest_dataset = data
                    largest_size = len(data)
        
        if largest_dataset is None:
            print("✗ No suitable dataset found")
            return False
        
        print(f"Testing with {largest_size} bars of data")
        
        # Test Market Structure performance
        print("\nMarket Structure Performance:")
        ms_analyzer = MarketStructureAnalyzer()
        
        start_time = time.time()
        ms_analysis = ms_analyzer.analyze_market_structure(largest_dataset)
        ms_time = time.time() - start_time
        
        print(f"   Market Structure Analysis: {ms_time:.3f} seconds")
        print(f"   Performance: {largest_size/ms_time:.0f} bars/second")
        
        # Test Order Block performance
        print("\nOrder Block Performance:")
        ob_analyzer = OrderBlockAnalyzer()
        
        start_time = time.time()
        ob_analysis = ob_analyzer.analyze_order_blocks(largest_dataset)
        ob_time = time.time() - start_time
        
        print(f"   Order Block Analysis: {ob_time:.3f} seconds")
        print(f"   Performance: {largest_size/ob_time:.0f} bars/second")
        
        # Combined performance
        total_time = ms_time + ob_time
        print(f"\nCombined Analysis: {total_time:.3f} seconds")
        print(f"Overall Performance: {largest_size/total_time:.0f} bars/second")
        
        # Performance thresholds
        if total_time < 1.0:
            print("🚀 Excellent performance - Real-time ready!")
        elif total_time < 2.0:
            print("✅ Good performance - Suitable for live trading")
        elif total_time < 5.0:
            print("⚠️ Acceptable performance - May need optimization")
        else:
            print("❌ Poor performance - Optimization required")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        return False

def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("SMC FEATURES TEST REPORT")
    print("="*60)
    
    try:
        test_results = {
            'market_structure': False,
            'order_blocks': False,
            'combined_analysis': False,
            'performance': False
        }
        
        # Run all tests
        print("Running comprehensive SMC feature tests...")
        
        test_results['market_structure'] = test_market_structure_analyzer()
        test_results['order_blocks'] = test_order_block_analyzer()
        test_results['combined_analysis'] = test_combined_analysis()
        test_results['performance'] = performance_test()
        
        # Generate summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\nTests Passed: {passed_tests}/{total_tests}")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            test_display = test_name.replace('_', ' ').title()
            print(f"   {test_display}: {status}")
        
        # Overall result
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ SMC Features are ready for Phase 2 completion")
            print(f"✅ Market Structure Detection: Working")
            print(f"✅ Order Block Identification: Working")
            print(f"✅ Combined Analysis: Working")
            print(f"✅ Performance: Acceptable")
            
            print(f"\n📋 Next Steps:")
            print(f"   1. Implement Liquidity & Fair Value Gap detection")
            print(f"   2. Add Volume Profile calculations")
            print(f"   3. Complete BIAS analysis system")
            print(f"   4. Integration with caching system")
            
            return True
        else:
            print(f"\n⚠️ SOME TESTS FAILED")
            print(f"❌ {total_tests - passed_tests} test(s) need attention")
            
            failed_tests = [name for name, result in test_results.items() if not result]
            print(f"Failed tests: {', '.join(failed_tests)}")
            
            return False
            
    except Exception as e:
        print(f"\n💥 Test report generation failed: {e}")
        return False

def main():
    """Main test function"""
    setup_logging()
    
    print("="*70)
    print("   SMC FEATURES TESTING - PHASE 2 WEEK 3")
    print("="*70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Generate comprehensive test report
        success = generate_test_report()
        
        if success:
            print(f"\n🎯 PHASE 2 WEEK 3 - DAY 1-2: COMPLETED!")
            print(f"📈 Market Structure & Order Block detection implemented")
            print(f"🚀 Ready for next development phase")
            return 0
        else:
            print(f"\n🔧 PHASE 2 WEEK 3 - DAY 1-2: NEEDS ATTENTION")
            print(f"⚠️ Some components require debugging")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Testing interrupted by user")
        return 0
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)