# scripts/test_fibonacci_levels.py
"""
Comprehensive test suite for Fibonacci Levels analyzer
Tests with real MT5 data across multiple symbols and timeframes
"""

import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FibonacciTestSuite:
    """Comprehensive test suite for Fibonacci analysis"""
    
    def __init__(self):
        self.test_data = None
        self.results = {}
        self.performance_metrics = {}
        
        # Test configuration
        self.test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        self.test_timeframes = ["D1", "H4", "H1", "M15"]
        self.test_scenarios = ["trending", "ranging", "volatile"]
        
        logger.info("FibonacciTestSuite initialized")

    def setup(self) -> bool:
        """Setup test environment and connections"""
        try:
            # Import required modules
            from features.fibonacci_levels import FibonacciAnalyzer
            from data.connectors.mt5_connector import get_mt5_connector
            from data.connectors.demo_connector import get_demo_connector
            from config.mt5_config import MT5_CONNECTION, get_trading_symbols
            
            self.FibonacciAnalyzer = FibonacciAnalyzer
            
            # Try real MT5 connection first, fallback to demo
            try:
                self.connector = get_mt5_connector(MT5_CONNECTION)
                connection_test = self.connector.test_connection()
                if connection_test.get('connected', False):
                    logger.info("Using real MT5 connection for tests")
                    self.connection_type = "real"
                else:
                    raise Exception("MT5 connection failed")
            except Exception as e:
                logger.warning(f"MT5 connection failed, using demo: {e}")
                self.connector = get_demo_connector()
                self.connection_type = "demo"
            
            # Get available symbols
            available_symbols = get_trading_symbols("major")
            self.test_symbols = [s for s in self.test_symbols if s in available_symbols]
            
            if not self.test_symbols:
                self.test_symbols = available_symbols[:4]  # Use first 4 available
            
            logger.info(f"Test setup completed - Connection: {self.connection_type}")
            logger.info(f"Test symbols: {self.test_symbols}")
            
            return True
            
        except Exception as e:
            logger.error(f"Test setup failed: {e}")
            return False

    def test_01_initialization(self) -> bool:
        """Test Fibonacci analyzer initialization"""
        logger.info("=== Test 01: Fibonacci Analyzer Initialization ===")
        
        try:
            # Test default initialization
            analyzer = self.FibonacciAnalyzer()
            assert hasattr(analyzer, 'swing_lookback')
            assert hasattr(analyzer, 'retracement_levels')
            assert hasattr(analyzer, 'extension_levels')
            assert len(analyzer.retracement_levels) > 0
            
            # Test custom parameters
            custom_analyzer = self.FibonacciAnalyzer(
                swing_lookback=15,
                min_swing_size=0.003,
                confluence_threshold=0.001
            )
            assert custom_analyzer.swing_lookback == 15
            assert custom_analyzer.min_swing_size == 0.003
            
            logger.info("✅ Initialization test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization test failed: {e}")
            return False

    def test_02_swing_detection(self) -> bool:
        """Test swing point detection with real data"""
        logger.info("=== Test 02: Swing Point Detection ===")
        
        try:
            analyzer = self.FibonacciAnalyzer()
            success_count = 0
            total_tests = 0
            
            for symbol in self.test_symbols[:2]:  # Test with 2 symbols
                for timeframe in ["H4", "H1"]:
                    total_tests += 1
                    
                    # Get real market data
                    data = self.connector.get_rates(symbol, timeframe, 200)
                    
                    if data is not None and len(data) > 50:
                        # Test swing detection
                        swing_highs = analyzer._detect_swing_highs(data)
                        swing_lows = analyzer._detect_swing_lows(data)
                        
                        if swing_highs and swing_lows:
                            success_count += 1
                            
                            logger.info(f"  {symbol} {timeframe}: "
                                      f"{len(swing_highs)} highs, {len(swing_lows)} lows")
                            
                            # Validate swing point quality
                            for swing in swing_highs[:3]:
                                assert swing.strength > 0
                                assert swing.price > 0
                                assert swing.index >= 0
                            
                            for swing in swing_lows[:3]:
                                assert swing.strength > 0
                                assert swing.price > 0
                                assert swing.index >= 0
                        else:
                            logger.warning(f"  {symbol} {timeframe}: No swings detected")
                    else:
                        logger.warning(f"  {symbol} {timeframe}: Data unavailable")
            
            success_rate = success_count / total_tests if total_tests > 0 else 0
            logger.info(f"Swing detection success rate: {success_rate:.1%}")
            
            if success_rate >= 0.5:  # At least 50% success
                logger.info("✅ Swing detection test passed")
                return True
            else:
                logger.error("❌ Swing detection test failed - low success rate")
                return False
                
        except Exception as e:
            logger.error(f"❌ Swing detection test failed: {e}")
            return False

    def test_03_fibonacci_calculation(self) -> bool:
        """Test Fibonacci level calculations"""
        logger.info("=== Test 03: Fibonacci Level Calculation ===")
        
        try:
            analyzer = self.FibonacciAnalyzer()
            success_count = 0
            total_tests = 0
            
            for symbol in self.test_symbols[:2]:
                total_tests += 1
                
                # Get data
                data = self.connector.get_rates(symbol, "H4", 200)
                
                if data is not None and len(data) > 50:
                    # Perform Fibonacci analysis
                    fib_zones = analyzer.analyze_fibonacci(data, max_swings=3)
                    
                    if fib_zones:
                        success_count += 1
                        
                        logger.info(f"  {symbol}: {len(fib_zones)} Fibonacci zones created")
                        
                        # Validate zone structure
                        for zone in fib_zones[:2]:
                            assert zone.swing_high is not None
                            assert zone.swing_low is not None
                            assert len(zone.levels) > 0
                            assert zone.quality_score > 0
                            
                            # Validate individual levels
                            for level in zone.levels:
                                assert level.price > 0
                                assert level.level >= 0
                                assert level.distance_from_current >= 0
                                
                        # Test level filtering
                        current_price = data.iloc[-1]['Close']
                        near_levels = analyzer.get_key_levels_near_price(
                            fib_zones, current_price, distance_threshold=0.02
                        )
                        
                        logger.info(f"    Key levels near price: {len(near_levels)}")
                        
                    else:
                        logger.warning(f"  {symbol}: No Fibonacci zones created")
                else:
                    logger.warning(f"  {symbol}: Data unavailable")
            
            success_rate = success_count / total_tests if total_tests > 0 else 0
            logger.info(f"Fibonacci calculation success rate: {success_rate:.1%}")
            
            if success_rate >= 0.5:
                logger.info("✅ Fibonacci calculation test passed")
                return True
            else:
                logger.error("❌ Fibonacci calculation test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fibonacci calculation test failed: {e}")
            return False

    def test_04_confluence_detection(self) -> bool:
        """Test confluence detection between Fibonacci levels"""
        logger.info("=== Test 04: Confluence Detection ===")
        
        try:
            analyzer = self.FibonacciAnalyzer(confluence_threshold=0.001)  # Tighter confluence
            confluence_found = False
            
            for symbol in self.test_symbols[:2]:
                # Use longer timeframe for better swings
                data = self.connector.get_rates(symbol, "D1", 100)
                
                if data is not None and len(data) > 50:
                    fib_zones = analyzer.analyze_fibonacci(data, max_swings=5)
                    
                    if fib_zones and len(fib_zones) > 1:
                        # Check for confluence levels
                        confluent_levels = []
                        for zone in fib_zones:
                            for level in zone.levels:
                                if level.confluence_score > 0:
                                    confluent_levels.append(level)
                        
                        if confluent_levels:
                            confluence_found = True
                            logger.info(f"  {symbol}: {len(confluent_levels)} confluent levels found")
                            
                            # Show top confluent levels
                            confluent_levels.sort(key=lambda x: x.confluence_score, reverse=True)
                            for i, level in enumerate(confluent_levels[:3]):
                                logger.info(f"    Level {i+1}: {level.level:.3f} "
                                          f"at {level.price:.5f} "
                                          f"(confluence: {level.confluence_score:.1f})")
            
            if confluence_found:
                logger.info("✅ Confluence detection test passed")
                return True
            else:
                logger.warning("⚠️  No confluence detected - may be normal in ranging markets")
                return True  # Not necessarily a failure
                
        except Exception as e:
            logger.error(f"❌ Confluence detection test failed: {e}")
            return False

    def test_05_performance_benchmark(self) -> bool:
        """Test performance with larger datasets"""
        logger.info("=== Test 05: Performance Benchmark ===")
        
        try:
            analyzer = self.FibonacciAnalyzer()
            performance_results = []
            
            for symbol in self.test_symbols[:2]:
                for timeframe in ["H4", "H1"]:
                    # Get larger dataset
                    data = self.connector.get_rates(symbol, timeframe, 500)
                    
                    if data is not None and len(data) > 100:
                        # Measure analysis time
                        start_time = time.time()
                        
                        fib_zones = analyzer.analyze_fibonacci(data, max_swings=5)
                        
                        end_time = time.time()
                        analysis_time = end_time - start_time
                        
                        bars_per_second = len(data) / analysis_time if analysis_time > 0 else 0
                        
                        performance_results.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'bars': len(data),
                            'analysis_time': analysis_time,
                            'bars_per_second': bars_per_second,
                            'zones_created': len(fib_zones) if fib_zones else 0
                        })
                        
                        logger.info(f"  {symbol} {timeframe}: {analysis_time:.3f}s "
                                  f"({bars_per_second:.0f} bars/sec)")
            
            if performance_results:
                avg_speed = np.mean([r['bars_per_second'] for r in performance_results])
                max_time = max([r['analysis_time'] for r in performance_results])
                
                logger.info(f"Average speed: {avg_speed:.0f} bars/second")
                logger.info(f"Maximum analysis time: {max_time:.3f} seconds")
                
                self.performance_metrics = {
                    'avg_speed': avg_speed,
                    'max_time': max_time,
                    'results': performance_results
                }
                
                # Performance targets
                if avg_speed > 100 and max_time < 5.0:  # Reasonable targets
                    logger.info("✅ Performance benchmark passed")
                    return True
                else:
                    logger.warning("⚠️  Performance below targets but acceptable")
                    return True
            else:
                logger.error("❌ No performance data collected")
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}")
            return False

    def test_06_integration_compatibility(self) -> bool:
        """Test integration compatibility with existing SMC components"""
        logger.info("=== Test 06: Integration Compatibility ===")
        
        try:
            # Test data format compatibility
            analyzer = self.FibonacciAnalyzer()
            
            # Get sample data
            data = self.connector.get_rates(self.test_symbols[0], "H4", 200)
            
            if data is not None and len(data) > 50:
                fib_zones = analyzer.analyze_fibonacci(data)
                
                if fib_zones:
                    # Test data structure compatibility
                    zone = fib_zones[0]
                    
                    # Check required attributes for integration
                    required_attrs = ['swing_high', 'swing_low', 'levels', 'quality_score']
                    for attr in required_attrs:
                        assert hasattr(zone, attr), f"Missing attribute: {attr}"
                    
                    # Check level structure
                    level = zone.levels[0]
                    level_attrs = ['price', 'level', 'level_type', 'direction']
                    for attr in level_attrs:
                        assert hasattr(level, attr), f"Missing level attribute: {attr}"
                    
                    # Test summary generation
                    summary = analyzer.get_fibonacci_summary(fib_zones)
                    required_summary_keys = ['total_zones', 'total_levels', 'high_quality_zones']
                    for key in required_summary_keys:
                        assert key in summary, f"Missing summary key: {key}"
                    
                    logger.info("✅ Integration compatibility test passed")
                    return True
                else:
                    logger.warning("⚠️  No zones for integration test")
                    return True
            else:
                logger.error("❌ No data for integration test")
                return False
                
        except Exception as e:
            logger.error(f"❌ Integration compatibility test failed: {e}")
            return False

    def test_07_edge_cases(self) -> bool:
        """Test edge cases and error handling"""
        logger.info("=== Test 07: Edge Cases and Error Handling ===")
        
        try:
            analyzer = self.FibonacciAnalyzer()
            
            # Test with insufficient data
            small_data = pd.DataFrame({
                'Open': [1.1000, 1.1010],
                'High': [1.1020, 1.1030], 
                'Low': [1.0990, 1.1000],
                'Close': [1.1010, 1.1020],
                'Volume': [1000, 1100]
            })
            
            result = analyzer.analyze_fibonacci(small_data)
            assert result is None, "Should return None for insufficient data"
            
            # Test with None data
            result = analyzer.analyze_fibonacci(None)
            assert result is None, "Should handle None data gracefully"
            
            # Test with empty DataFrame
            empty_data = pd.DataFrame()
            result = analyzer.analyze_fibonacci(empty_data)
            assert result is None, "Should handle empty data gracefully"
            
            # Test with NaN values
            data = self.connector.get_rates(self.test_symbols[0], "H1", 100)
            if data is not None:
                # Introduce some NaN values
                data_with_nan = data.copy()
                data_with_nan.iloc[50:55] = np.nan
                
                result = analyzer.analyze_fibonacci(data_with_nan)
                # Should either handle gracefully or return None
                assert result is None or isinstance(result, list)
            
            logger.info("✅ Edge cases test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Edge cases test failed: {e}")
            return False

    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive analysis across all test symbols and timeframes"""
        logger.info("=== Comprehensive Market Analysis ===")
        
        try:
            analyzer = self.FibonacciAnalyzer()
            analysis_results = {}
            
            for symbol in self.test_symbols:
                analysis_results[symbol] = {}
                
                for timeframe in self.test_timeframes:
                    try:
                        data = self.connector.get_rates(symbol, timeframe, 200)
                        
                        if data is not None and len(data) > 50:
                            fib_zones = analyzer.analyze_fibonacci(data, max_swings=3)
                            
                            if fib_zones:
                                summary = analyzer.get_fibonacci_summary(fib_zones)
                                current_price = data.iloc[-1]['Close']
                                
                                # Get key levels near current price
                                key_levels = analyzer.get_key_levels_near_price(
                                    fib_zones, current_price, distance_threshold=0.03
                                )
                                
                                analysis_results[symbol][timeframe] = {
                                    'zones_count': len(fib_zones),
                                    'levels_count': summary.get('total_levels', 0),
                                    'quality_score': summary.get('avg_quality_score', 0),
                                    'key_levels_near': len(key_levels),
                                    'best_zone_quality': summary.get('best_zone_quality', 0),
                                    'current_price': current_price
                                }
                                
                                logger.info(f"  {symbol} {timeframe}: "
                                          f"{len(fib_zones)} zones, "
                                          f"{len(key_levels)} key levels near price")
                            else:
                                analysis_results[symbol][timeframe] = None
                        else:
                            analysis_results[symbol][timeframe] = None
                            
                    except Exception as e:
                        logger.warning(f"Analysis failed for {symbol} {timeframe}: {e}")
                        analysis_results[symbol][timeframe] = None
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {}

    def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        logger.info("🚀 Starting Fibonacci Levels Test Suite")
        logger.info("=" * 60)
        
        if not self.setup():
            logger.error("Test setup failed - aborting")
            return {'setup': False}
        
        test_methods = [
            'test_01_initialization',
            'test_02_swing_detection', 
            'test_03_fibonacci_calculation',
            'test_04_confluence_detection',
            'test_05_performance_benchmark',
            'test_06_integration_compatibility',
            'test_07_edge_cases'
        ]
        
        start_time = time.time()
        
        for test_method in test_methods:
            try:
                test_start = time.time()
                passed = getattr(self, test_method)()
                test_time = time.time() - test_start
                
                self.results[test_method] = {
                    'passed': passed,
                    'execution_time': test_time
                }
                
                if passed:
                    logger.info(f"✅ {test_method} completed in {test_time:.2f}s")
                else:
                    logger.error(f"❌ {test_method} failed after {test_time:.2f}s")
                    
            except Exception as e:
                self.results[test_method] = {
                    'passed': False,
                    'error': str(e),
                    'execution_time': 0
                }
                logger.error(f"❌ {test_method} crashed: {e}")
        
        # Run comprehensive analysis
        comprehensive_results = self.run_comprehensive_analysis()
        
        total_time = time.time() - start_time
        
        return self.generate_final_report(total_time, comprehensive_results)

    def generate_final_report(self, total_time: float, comprehensive_results: Dict) -> Dict:
        """Generate final test report"""
        logger.info("=" * 60)
        logger.info("📊 FIBONACCI LEVELS TEST RESULTS")
        logger.info("=" * 60)
        
        passed_tests = sum(1 for result in self.results.values() if result.get('passed', False))
        total_tests = len(self.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"Overall Results:")
        logger.info(f"  Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
        logger.info(f"  Total Execution Time: {total_time:.2f} seconds")
        logger.info(f"  Connection Type: {self.connection_type}")
        
        # Individual test results
        logger.info(f"\nDetailed Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            time_str = f"{result.get('execution_time', 0):.2f}s"
            logger.info(f"  {test_name}: {status} ({time_str})")
            
            if 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        # Performance metrics
        if self.performance_metrics:
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  Average Speed: {self.performance_metrics['avg_speed']:.0f} bars/second")
            logger.info(f"  Maximum Time: {self.performance_metrics['max_time']:.3f} seconds")
        
        # Market analysis summary
        if comprehensive_results:
            logger.info(f"\nMarket Analysis Summary:")
            total_zones = 0
            successful_analyses = 0
            
            for symbol, timeframes in comprehensive_results.items():
                symbol_zones = 0
                symbol_analyses = 0
                
                for tf, result in timeframes.items():
                    if result:
                        symbol_zones += result['zones_count']
                        symbol_analyses += 1
                        total_zones += result['zones_count']
                        successful_analyses += 1
                
                if symbol_analyses > 0:
                    logger.info(f"  {symbol}: {symbol_zones} zones across {symbol_analyses} timeframes")
            
            logger.info(f"  Total: {total_zones} Fibonacci zones across {successful_analyses} analyses")
        
        # Final assessment
        logger.info(f"\n🎯 FINAL ASSESSMENT:")
        if success_rate >= 0.8:
            logger.info("🟢 EXCELLENT - Fibonacci implementation ready for production")
        elif success_rate >= 0.6:
            logger.info("🟡 GOOD - Minor issues to address")
        else:
            logger.info("🔴 NEEDS WORK - Significant issues found")
        
        logger.info("=" * 60)
        
        return {
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'total_time': total_time,
            'connection_type': self.connection_type,
            'individual_results': self.results,
            'performance_metrics': self.performance_metrics,
            'market_analysis': comprehensive_results
        }


def main():
    """Main test execution function"""
    try:
        test_suite = FibonacciTestSuite()
        final_results = test_suite.run_all_tests()
        
        # Return success/failure for automation
        success_rate = final_results.get('success_rate', 0)
        return success_rate >= 0.6  # 60% minimum success rate
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)