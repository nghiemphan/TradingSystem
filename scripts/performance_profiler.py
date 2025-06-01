# scripts/performance_profiler.py
"""
Performance Profiler for SMC Feature Aggregator
Identifies bottlenecks and optimization opportunities
"""

import time
import cProfile
import pstats
import io
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_aggregator import SMCFeatureAggregator
from data.connectors.mt5_connector import get_mt5_connector
from data.connectors.demo_connector import get_demo_connector
from config.mt5_config import MT5_CONNECTION, get_trading_symbols

class PerformanceProfiler:
    """
    Comprehensive performance profiler for Feature Aggregator
    """
    
    def __init__(self):
        self.aggregator = SMCFeatureAggregator()
        self.profiling_results = {}
        self.component_timings = {}
        
        # Setup connection
        try:
            self.connector = get_mt5_connector(MT5_CONNECTION)
            connection_test = self.connector.test_connection()
            if not connection_test.get('connected', False):
                raise Exception("MT5 not connected")
            print("✅ Using real MT5 connection for profiling")
        except:
            self.connector = get_demo_connector()
            print("⚠️ Using demo connector for profiling")
    
    def profile_component_performance(self, symbol: str = "EURUSD", 
                                    timeframe: str = "H1", 
                                    bars: int = 200) -> Dict:
        """
        Profile individual component performance
        """
        print(f"\n🔍 Profiling Component Performance: {symbol} {timeframe} ({bars} bars)")
        print("=" * 70)
        
        # Get data
        data = self.connector.get_rates(symbol, timeframe, bars)
        if data is None or len(data) < 50:
            print(f"❌ Insufficient data for {symbol} {timeframe}")
            return {}
        
        component_times = {}
        
        # 1. Market Structure Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.ms_analyzer:
                ms_result = self.aggregator.ms_analyzer.analyze_market_structure(data)
                component_times['market_structure'] = time.perf_counter() - start_time
                print(f"  📊 Market Structure: {component_times['market_structure']:.4f}s")
            else:
                component_times['market_structure'] = 0.0
        except Exception as e:
            component_times['market_structure'] = 0.0
            print(f"  ❌ Market Structure failed: {e}")
        
        # 2. Order Blocks Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.ob_analyzer:
                ob_result = self.aggregator.ob_analyzer.analyze_order_blocks(data)
                component_times['order_blocks'] = time.perf_counter() - start_time
                print(f"  📦 Order Blocks: {component_times['order_blocks']:.4f}s")
            else:
                component_times['order_blocks'] = 0.0
        except Exception as e:
            component_times['order_blocks'] = 0.0
            print(f"  ❌ Order Blocks failed: {e}")
        
        # 3. Fair Value Gaps Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.fvg_analyzer:
                fvg_result = self.aggregator.fvg_analyzer.analyze_fair_value_gaps(data)
                component_times['fair_value_gaps'] = time.perf_counter() - start_time
                print(f"  🕳️ Fair Value Gaps: {component_times['fair_value_gaps']:.4f}s")
            else:
                component_times['fair_value_gaps'] = 0.0
        except Exception as e:
            component_times['fair_value_gaps'] = 0.0
            print(f"  ❌ Fair Value Gaps failed: {e}")
        
        # 4. Liquidity Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.liq_analyzer:
                liq_result = self.aggregator.liq_analyzer.analyze_liquidity(data)
                component_times['liquidity'] = time.perf_counter() - start_time
                print(f"  💧 Liquidity: {component_times['liquidity']:.4f}s")
            else:
                component_times['liquidity'] = 0.0
        except Exception as e:
            component_times['liquidity'] = 0.0
            print(f"  ❌ Liquidity failed: {e}")
        
        # 5. Premium/Discount Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.pd_analyzer:
                pd_result = self.aggregator.pd_analyzer.analyze_premium_discount(data)
                component_times['premium_discount'] = time.perf_counter() - start_time
                print(f"  ⚖️ Premium/Discount: {component_times['premium_discount']:.4f}s")
            else:
                component_times['premium_discount'] = 0.0
        except Exception as e:
            component_times['premium_discount'] = 0.0
            print(f"  ❌ Premium/Discount failed: {e}")
        
        # 6. Supply/Demand Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.sd_analyzer:
                sd_result = self.aggregator.sd_analyzer.analyze_supply_demand(data)
                component_times['supply_demand'] = time.perf_counter() - start_time
                print(f"  📈 Supply/Demand: {component_times['supply_demand']:.4f}s")
            else:
                component_times['supply_demand'] = 0.0
        except Exception as e:
            component_times['supply_demand'] = 0.0
            print(f"  ❌ Supply/Demand failed: {e}")
        
        # 7. Volume Profile Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.vp_analyzer:
                vp_result = self.aggregator.vp_analyzer.analyze_volume_profile(data)
                component_times['volume_profile'] = time.perf_counter() - start_time
                print(f"  📊 Volume Profile: {component_times['volume_profile']:.4f}s")
            else:
                component_times['volume_profile'] = 0.0
        except Exception as e:
            component_times['volume_profile'] = 0.0
            print(f"  ❌ Volume Profile failed: {e}")
        
        # 8. VWAP Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.vwap_calculator:
                vwap_result = self.aggregator.vwap_calculator.calculate_vwap(data)
                component_times['vwap'] = time.perf_counter() - start_time
                print(f"  📈 VWAP: {component_times['vwap']:.4f}s")
            else:
                component_times['vwap'] = 0.0
        except Exception as e:
            component_times['vwap'] = 0.0
            print(f"  ❌ VWAP failed: {e}")
        
        # 9. Fibonacci Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.fib_analyzer:
                fib_result = self.aggregator.fib_analyzer.analyze_fibonacci(data, max_swings=3)
                component_times['fibonacci'] = time.perf_counter() - start_time
                print(f"  🌀 Fibonacci: {component_times['fibonacci']:.4f}s")
            else:
                component_times['fibonacci'] = 0.0
        except Exception as e:
            component_times['fibonacci'] = 0.0
            print(f"  ❌ Fibonacci failed: {e}")
        
        # 10. Enhanced BIAS Timing
        start_time = time.perf_counter()
        try:
            if self.aggregator.bias_analyzer:
                bias_result = self.aggregator.bias_analyzer.analyze_bias(
                    symbol, {timeframe: data}, timeframe
                )
                component_times['enhanced_bias'] = time.perf_counter() - start_time
                print(f"  🧠 Enhanced BIAS: {component_times['enhanced_bias']:.4f}s")
            else:
                component_times['enhanced_bias'] = 0.0
        except Exception as e:
            component_times['enhanced_bias'] = 0.0
            print(f"  ❌ Enhanced BIAS failed: {e}")
        
        # Calculate totals and percentages
        total_component_time = sum(component_times.values())
        
        print(f"\n📊 Component Performance Summary:")
        print(f"  Total Component Time: {total_component_time:.4f}s")
        
        # Sort by slowest components
        sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
        
        print(f"  🐌 Slowest Components:")
        for i, (component, timing) in enumerate(sorted_components[:5], 1):
            percentage = (timing / total_component_time * 100) if total_component_time > 0 else 0
            print(f"    {i}. {component}: {timing:.4f}s ({percentage:.1f}%)")
        
        return component_times
    
    def profile_full_extraction(self, symbol: str = "EURUSD", 
                              timeframe: str = "H1", 
                              bars: int = 200) -> Dict:
        """
        Profile full feature extraction with detailed timing
        """
        print(f"\n🔍 Profiling Full Feature Extraction: {symbol} {timeframe} ({bars} bars)")
        print("=" * 70)
        
        # Get data
        data = self.connector.get_rates(symbol, timeframe, bars)
        if data is None or len(data) < 50:
            print(f"❌ Insufficient data for {symbol} {timeframe}")
            return {}
        
        # Profile using cProfile
        profiler = cProfile.Profile()
        
        # Time the full extraction
        start_time = time.perf_counter()
        
        profiler.enable()
        feature_vector = self.aggregator.extract_features(symbol, timeframe, data)
        profiler.disable()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        if feature_vector:
            features_count = feature_vector.get_feature_count()
            bars_per_second = bars / total_time
            
            print(f"  ✅ Extraction successful:")
            print(f"    Total Time: {total_time:.4f}s")
            print(f"    Features: {features_count}")
            print(f"    Bars: {bars}")
            print(f"    Speed: {bars_per_second:.1f} bars/second")
            print(f"    Target: 1000+ bars/second")
            print(f"    Performance Gap: {1000/bars_per_second:.1f}x improvement needed")
        else:
            print(f"  ❌ Feature extraction failed")
            return {}
        
        # Analyze profiler results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 slowest functions
        
        profiler_output = s.getvalue()
        
        print(f"\n🔍 Top Performance Bottlenecks:")
        print("=" * 50)
        
        # Parse and display top bottlenecks
        lines = profiler_output.split('\n')
        in_stats = False
        bottleneck_count = 0
        
        for line in lines:
            if 'ncalls' in line and 'tottime' in line:
                in_stats = True
                continue
            
            if in_stats and line.strip() and bottleneck_count < 10:
                parts = line.split()
                if len(parts) >= 6:
                    cumtime = parts[3]
                    filename_func = ' '.join(parts[5:])
                    
                    # Filter relevant functions
                    if any(keyword in filename_func.lower() for keyword in 
                          ['analyze', 'calculate', 'extract', 'smc', 'features']):
                        print(f"  {bottleneck_count + 1}. {cumtime}s - {filename_func}")
                        bottleneck_count += 1
        
        return {
            'total_time': total_time,
            'bars_per_second': bars_per_second,
            'features_count': features_count,
            'profiler_output': profiler_output
        }
    
    def performance_comparison_test(self, bar_sizes: List[int] = [100, 200, 500, 1000]) -> Dict:
        """
        Test performance across different data sizes
        """
        print(f"\n🔍 Performance Comparison Test")
        print("=" * 70)
        
        results = {}
        symbol = "EURUSD"
        timeframe = "H1"
        
        for bars in bar_sizes:
            print(f"\n  Testing with {bars} bars...")
            
            # Get data
            data = self.connector.get_rates(symbol, timeframe, bars)
            if data is None or len(data) < 50:
                print(f"    ❌ Insufficient data for {bars} bars")
                continue
            
            # Time extraction
            start_time = time.perf_counter()
            feature_vector = self.aggregator.extract_features(symbol, timeframe, data)
            end_time = time.perf_counter()
            
            extraction_time = end_time - start_time
            
            if feature_vector:
                bars_per_second = bars / extraction_time
                features_count = feature_vector.get_feature_count()
                
                results[bars] = {
                    'time': extraction_time,
                    'bars_per_second': bars_per_second,
                    'features': features_count
                }
                
                print(f"    ✅ {bars} bars: {extraction_time:.3f}s ({bars_per_second:.1f} bars/sec)")
            else:
                print(f"    ❌ Failed extraction for {bars} bars")
        
        # Performance scaling analysis
        if len(results) >= 2:
            print(f"\n📊 Performance Scaling Analysis:")
            bar_counts = sorted(results.keys())
            
            for i, bars in enumerate(bar_counts):
                if i == 0:
                    print(f"  Baseline ({bars} bars): {results[bars]['bars_per_second']:.1f} bars/sec")
                else:
                    baseline_speed = results[bar_counts[0]]['bars_per_second']
                    current_speed = results[bars]['bars_per_second']
                    scaling_factor = current_speed / baseline_speed
                    
                    print(f"  {bars} bars: {current_speed:.1f} bars/sec (scaling: {scaling_factor:.2f}x)")
            
            # Linear complexity check
            times = [results[bars]['time'] for bars in bar_counts]
            bars_list = list(bar_counts)
            
            if len(times) >= 2:
                complexity = (times[-1] / times[0]) / (bars_list[-1] / bars_list[0])
                print(f"  Time Complexity: O(n^{complexity:.2f}) - {'Linear' if complexity < 1.2 else 'Superlinear'}")
        
        return results
    
    def identify_optimization_opportunities(self) -> Dict:
        """
        Identify key optimization opportunities
        """
        print(f"\n🎯 Optimization Opportunities Analysis")
        print("=" * 70)
        
        opportunities = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # Profile component performance
        component_times = self.profile_component_performance()
        
        if component_times:
            total_time = sum(component_times.values())
            
            # Identify high-impact optimizations
            for component, time_taken in component_times.items():
                if time_taken == 0:
                    continue
                    
                percentage = (time_taken / total_time) * 100
                
                if percentage > 20:
                    opportunities['high_priority'].append({
                        'component': component,
                        'time': time_taken,
                        'percentage': percentage,
                        'recommendation': f'Critical optimization needed - {percentage:.1f}% of total time'
                    })
                elif percentage > 10:
                    opportunities['medium_priority'].append({
                        'component': component,
                        'time': time_taken,
                        'percentage': percentage,
                        'recommendation': f'Moderate optimization opportunity - {percentage:.1f}% of total time'
                    })
                else:
                    opportunities['low_priority'].append({
                        'component': component,
                        'time': time_taken,
                        'percentage': percentage,
                        'recommendation': f'Low priority - {percentage:.1f}% of total time'
                    })
        
        # Display recommendations
        print(f"\n🚨 HIGH PRIORITY OPTIMIZATIONS:")
        for opp in opportunities['high_priority']:
            print(f"  • {opp['component']}: {opp['time']:.4f}s ({opp['percentage']:.1f}%)")
            print(f"    → {opp['recommendation']}")
        
        print(f"\n⚠️ MEDIUM PRIORITY OPTIMIZATIONS:")
        for opp in opportunities['medium_priority']:
            print(f"  • {opp['component']}: {opp['time']:.4f}s ({opp['percentage']:.1f}%)")
            print(f"    → {opp['recommendation']}")
        
        print(f"\n✅ LOW PRIORITY OPTIMIZATIONS:")
        for opp in opportunities['low_priority']:
            print(f"  • {opp['component']}: {opp['time']:.4f}s ({opp['percentage']:.1f}%)")
        
        return opportunities
    
    def run_comprehensive_profiling(self):
        """
        Run comprehensive performance profiling
        """
        print("🚀 COMPREHENSIVE PERFORMANCE PROFILING")
        print("=" * 80)
        print(f"Timestamp: {datetime.now()}")
        print(f"Target: 1000+ bars/second")
        print(f"Current baseline: ~69 bars/second")
        print(f"Improvement needed: 14.5x")
        
        # 1. Component Performance Profiling
        component_results = self.profile_component_performance()
        
        # 2. Full Extraction Profiling
        extraction_results = self.profile_full_extraction()
        
        # 3. Performance Scaling Test
        scaling_results = self.performance_comparison_test()
        
        # 4. Optimization Opportunities
        optimization_opportunities = self.identify_optimization_opportunities()
        
        # 5. Summary and Recommendations
        print(f"\n🎯 PROFILING SUMMARY & RECOMMENDATIONS")
        print("=" * 80)
        
        if extraction_results:
            current_speed = extraction_results.get('bars_per_second', 0)
            target_speed = 1000
            gap = target_speed / current_speed if current_speed > 0 else float('inf')
            
            print(f"📊 Performance Gap Analysis:")
            print(f"  Current Speed: {current_speed:.1f} bars/second")
            print(f"  Target Speed: {target_speed} bars/second")
            print(f"  Improvement Needed: {gap:.1f}x")
            
            if gap > 10:
                print(f"  Status: 🔴 CRITICAL - Major optimization required")
            elif gap > 5:
                print(f"  Status: 🟡 MODERATE - Significant optimization needed")
            else:
                print(f"  Status: 🟢 GOOD - Minor optimization sufficient")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Focus on HIGH PRIORITY components first")
        print(f"  2. Implement algorithmic optimizations")
        print(f"  3. Consider parallel processing")
        print(f"  4. Optimize data structures and operations")
        print(f"  5. Re-test after each optimization")

def main():
    """Main profiling execution"""
    profiler = PerformanceProfiler()
    profiler.run_comprehensive_profiling()

if __name__ == "__main__":
    main()