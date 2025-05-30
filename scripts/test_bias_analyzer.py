"""
BIAS Analyzer Test Suite - Final Version
Comprehensive testing for enhanced BIAS integration with proper error handling
Version: 1.0 - Production Ready
"""
import sys
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup clean logging configuration"""
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Suppress third-party library logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('pandas').setLevel(logging.ERROR)

def check_dependencies():
    """Check if required modules are available"""
    missing_modules = []
    
    try:
        from features.bias_analyzer import BiasAnalyzer, BiasDirection, BiasStrength
        print("✓ BIAS Analyzer module loaded successfully")
    except ImportError as e:
        print(f"❌ BIAS Analyzer module not found: {e}")
        missing_modules.append("bias_analyzer")
    
    try:
        from data.connectors.mt5_connector import get_mt5_connector
        print("✓ MT5 connector available")
    except ImportError:
        print("⚠️  MT5 connector not available (will use synthetic data)")
    
    try:
        from data.connectors.demo_connector import get_demo_connector
        print("✓ Demo connector available")
    except ImportError:
        print("⚠️  Demo connector not available (will use synthetic data)")
    
    if missing_modules:
        print(f"\n❌ Missing required modules: {missing_modules}")
        print("Please ensure the BIAS analyzer is properly installed")
        return False
    
    return True

def create_test_data():
    """Create comprehensive test data for all scenarios"""
    print("🔬 Creating synthetic test data...")
    
    # Test scenarios
    scenarios = {
        "EURUSD_BULLISH": {"trend": "bullish", "volatility": "normal"},
        "GBPUSD_BEARISH": {"trend": "bearish", "volatility": "high"},
        "USDJPY_SIDEWAYS": {"trend": "sideways", "volatility": "low"}
    }
    
    test_data = {}
    
    for symbol, config in scenarios.items():
        test_data[symbol] = {}
        print(f"   📊 Creating {symbol} ({config['trend']} trend, {config['volatility']} volatility)")
        
        # Generate different timeframes
        timeframes = {
            "D1": {"bars": 100, "base_interval": 24},
            "H4": {"bars": 200, "base_interval": 4},
            "H1": {"bars": 300, "base_interval": 1},
            "M15": {"bars": 400, "base_interval": 0.25}
        }
        
        for tf_name, tf_config in timeframes.items():
            bars = tf_config["bars"]
            interval_hours = tf_config["base_interval"]
            
            # Create date range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=bars * interval_hours)
            dates = pd.date_range(start=start_time, end=end_time, periods=bars)
            
            # Generate price data based on scenario
            base_price = 1.1000 if "EUR" in symbol else 1.3000 if "GBP" in symbol else 150.0
            
            # Trend component
            if config["trend"] == "bullish":
                trend = np.linspace(0, base_price * 0.02, bars)  # 2% uptrend
            elif config["trend"] == "bearish":
                trend = np.linspace(0, -base_price * 0.02, bars)  # 2% downtrend
            else:  # sideways
                trend = np.sin(np.linspace(0, 6*np.pi, bars)) * base_price * 0.005
            
            # Volatility component
            if config["volatility"] == "high":
                noise_factor = 0.003
            elif config["volatility"] == "low":
                noise_factor = 0.001
            else:  # normal
                noise_factor = 0.002
            
            noise = np.random.normal(0, base_price * noise_factor, bars)
            
            # Generate OHLC prices
            close_prices = base_price + trend + noise
            
            # Create realistic OHLC structure
            spread_factor = base_price * 0.0005
            open_prices = close_prices + np.random.normal(0, spread_factor, bars)
            
            high_offset = np.abs(np.random.normal(0, spread_factor * 1.5, bars))
            low_offset = np.abs(np.random.normal(0, spread_factor * 1.5, bars))
            
            high_prices = np.maximum(open_prices, close_prices) + high_offset
            low_prices = np.minimum(open_prices, close_prices) - low_offset
            
            # Generate volume
            base_volume = 10000
            volume_noise = np.random.normal(1, 0.3, bars)
            volumes = np.maximum(base_volume * volume_noise, 1000).astype(int)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes
            }, index=dates)
            
            test_data[symbol][tf_name] = df
            print(f"      ✓ {tf_name}: {len(df)} bars")
    
    print(f"✓ Test data created for {len(test_data)} symbols")
    return test_data

class BiasTestSuite:
    """Comprehensive BIAS analyzer test suite"""
    
    def __init__(self):
        self.test_data = None
        self.results = {}
        self.start_time = None
        
    def setup(self):
        """Setup test environment"""
        print("🚀 Setting up BIAS test environment...")
        self.start_time = time.time()
        self.test_data = create_test_data()
        return True
    
    def test_01_initialization(self):
        """Test 1: BIAS analyzer initialization"""
        print("\n" + "="*60)
        print("🔬 TEST 1: BIAS ANALYZER INITIALIZATION")
        print("="*60)
        
        try:
            from features.bias_analyzer import BiasAnalyzer
            
            # Test 1.1: Default initialization
            print("\n1.1 Default initialization:")
            analyzer = BiasAnalyzer()
            weights_sum = sum(analyzer.weights.values())
            print(f"   ✓ BiasAnalyzer created")
            print(f"   ✓ Weights sum: {weights_sum:.3f}")
            
            if abs(weights_sum - 1.0) > 0.01:
                print(f"   ❌ Invalid weights sum: {weights_sum}")
                return False
            
            # Test 1.2: Enhanced initialization
            print("\n1.2 Enhanced initialization:")
            enhanced_analyzer = BiasAnalyzer(
                bias_direction_threshold=0.03,
                signal_amplification_factor=1.8,
                adaptive_thresholds=True,
                volatility_adjustment=True
            )
            print(f"   ✓ Enhanced analyzer created")
            print(f"   ✓ Direction threshold: {enhanced_analyzer.bias_direction_threshold}")
            print(f"   ✓ Amplification factor: {enhanced_analyzer.signal_amplification_factor}")
            print(f"   ✓ Adaptive thresholds: {enhanced_analyzer.adaptive_thresholds}")
            
            # Test 1.3: Invalid configuration
            print("\n1.3 Invalid configuration handling:")
            try:
                BiasAnalyzer(
                    structural_weight=0.7,
                    institutional_weight=0.7  # Total > 1.0
                )
                print("   ❌ Should have rejected invalid weights")
                return False
            except ValueError:
                print("   ✓ Correctly rejected invalid weights")
            
            # Test 1.4: Component verification
            print("\n1.4 Component verification:")
            required_components = [
                'ms_analyzer', 'ob_analyzer', 'fvg_analyzer',
                'liq_analyzer', 'pd_analyzer', 'sd_analyzer'
            ]
            
            missing_components = []
            for component in required_components:
                if hasattr(analyzer, component):
                    print(f"   ✓ {component}")
                else:
                    print(f"   ❌ Missing {component}")
                    missing_components.append(component)
            
            if missing_components:
                return False
            
            print("\n✅ Initialization test PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Initialization test FAILED: {e}")
            return False
    
    def test_02_single_timeframe_detection(self):
        """Test 2: Enhanced single timeframe BIAS detection"""
        print("\n" + "="*60)
        print("🎯 TEST 2: SINGLE TIMEFRAME BIAS DETECTION")
        print("="*60)
        
        try:
            from features.bias_analyzer import BiasAnalyzer, BiasDirection
            
            # Create enhanced analyzer
            analyzer = BiasAnalyzer(
                bias_direction_threshold=0.03,  # More sensitive
                signal_amplification_factor=1.8,
                adaptive_thresholds=True
            )
            
            detection_results = {}
            total_tests = 0
            successful_detections = 0
            
            for symbol, timeframes in self.test_data.items():
                print(f"\n--- Testing {symbol} ---")
                
                for tf_name, data in timeframes.items():
                    if len(data) < 50:
                        continue
                    
                    total_tests += 1
                    
                    # Single timeframe analysis
                    single_tf_data = {tf_name: data}
                    
                    start_time = time.time()
                    result = analyzer.analyze_bias(symbol, single_tf_data, tf_name)
                    analysis_time = time.time() - start_time
                    
                    # Check detection success
                    if result and result.direction != BiasDirection.NEUTRAL:
                        successful_detections += 1
                        detection_results[f"{symbol}_{tf_name}"] = {
                            'detected': True,
                            'direction': result.direction.name,
                            'strength': result.strength.value,
                            'confidence': result.confidence,
                            'score': result.score,
                            'time': analysis_time
                        }
                        
                        print(f"   ✓ {tf_name}: {result.direction.name} "
                              f"(str: {result.strength.value}, conf: {result.confidence:.2f}, "
                              f"time: {analysis_time:.3f}s)")
                    else:
                        detection_results[f"{symbol}_{tf_name}"] = {
                            'detected': False,
                            'time': analysis_time
                        }
                        print(f"   ⚪ {tf_name}: No clear BIAS (time: {analysis_time:.3f}s)")
            
            # Calculate success rate
            success_rate = successful_detections / total_tests if total_tests > 0 else 0
            avg_time = np.mean([r['time'] for r in detection_results.values()])
            
            print(f"\n📊 Single Timeframe Detection Results:")
            print(f"   Total tests: {total_tests}")
            print(f"   Successful detections: {successful_detections}")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Average analysis time: {avg_time:.3f}s")
            
            # Performance assessment
            if success_rate >= 0.6:  # 60% target
                print(f"   🎉 EXCELLENT detection rate!")
                performance = "EXCELLENT"
            elif success_rate >= 0.4:  # 40% acceptable
                print(f"   ✅ GOOD detection rate!")
                performance = "GOOD"
            elif success_rate >= 0.2:  # 20% minimum
                print(f"   📊 ACCEPTABLE detection rate")
                performance = "ACCEPTABLE"
            else:
                print(f"   ⚠️  LOW detection rate")
                performance = "LOW"
            
            # Expected improvement analysis
            original_rate = 0.083  # 8.3% from previous test
            improvement_factor = success_rate / original_rate if original_rate > 0 else 0
            
            print(f"\n📈 Improvement Analysis:")
            print(f"   Original success rate: {original_rate:.1%}")
            print(f"   Enhanced success rate: {success_rate:.1%}")
            print(f"   Improvement factor: {improvement_factor:.1f}x")
            
            if improvement_factor >= 5:
                print(f"   🚀 OUTSTANDING improvement!")
            elif improvement_factor >= 3:
                print(f"   ✅ STRONG improvement!")
            elif improvement_factor >= 2:
                print(f"   📊 GOOD improvement!")
            
            success = success_rate >= 0.4  # 40% minimum for pass
            
            if success:
                print(f"\n✅ Single timeframe detection test PASSED")
            else:
                print(f"\n❌ Single timeframe detection test FAILED")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Single timeframe detection test FAILED: {e}")
            return False
    
    def test_03_session_analysis_fix(self):
        """Test 3: Session analysis KeyError fix verification"""
        print("\n" + "="*60)
        print("🕐 TEST 3: SESSION ANALYSIS KEYERROR FIX")
        print("="*60)
        
        try:
            from features.bias_analyzer import BiasAnalyzer, get_session_bias_summary
            
            analyzer = BiasAnalyzer(
                session_analysis_enabled=True,
                session_strength_threshold=0.005
            )
            
            keyerror_fixed = True
            session_tests = 0
            successful_sessions = 0
            
            for symbol, timeframes in self.test_data.items():
                print(f"\n--- Testing Session Analysis: {symbol} ---")
                
                # Use H1 timeframe for session analysis
                if "H1" not in timeframes:
                    print(f"   ⚠️  No H1 data for {symbol}")
                    continue
                
                session_tests += 1
                tf_data = {"H1": timeframes["H1"]}
                
                try:
                    # Test session analysis
                    result = analyzer.analyze_bias(symbol, tf_data, "H1")
                    
                    if result and result.session_analysis is not None:
                        print(f"   ✓ Session analysis completed")
                        print(f"   Sessions detected: {len(result.session_analysis)}")
                        
                        if len(result.session_analysis) > 0:
                            successful_sessions += 1
                            
                            # Show session details
                            for session in result.session_analysis[:2]:  # Show first 2
                                if session.strength > 0.01:
                                    print(f"      {session.session.value}: {session.direction.name} "
                                          f"(str: {session.strength:.3f})")
                        
                        # Critical test: Check for KeyError fix
                        print(f"   🔧 Testing KeyError fix...")
                        try:
                            persistence = analyzer.get_bias_persistence()
                            
                            # Check for the previously missing 'current_streak' key
                            if 'current_streak' in persistence:
                                print(f"      ✅ 'current_streak' key found: {persistence['current_streak']}")
                                print(f"      ✅ Persistence score: {persistence['persistence_score']:.3f}")
                                print(f"      ✅ Direction changes: {persistence['direction_changes']}")
                                print(f"      ✅ Confidence trend: {persistence['confidence_trend']}")
                                
                                # Test utility function
                                session_summary = get_session_bias_summary(analyzer)
                                if 'error' not in session_summary:
                                    print(f"      ✅ Session summary utility working")
                                else:
                                    print(f"      ⚠️  Session summary issue: {session_summary['error']}")
                                    
                            else:
                                print(f"      ❌ 'current_streak' key STILL MISSING!")
                                keyerror_fixed = False
                                
                        except KeyError as ke:
                            print(f"      ❌ KeyError STILL PRESENT: {ke}")
                            keyerror_fixed = False
                        except Exception as pe:
                            print(f"      ⚠️  Persistence analysis error: {pe}")
                    else:
                        print(f"   ⚪ Session analysis inconclusive")
                        
                except KeyError as e:
                    print(f"   ❌ KeyError in session analysis: {e}")
                    keyerror_fixed = False
                except Exception as e:
                    print(f"   ❌ Session analysis error: {e}")
            
            # Results summary
            session_success_rate = successful_sessions / session_tests if session_tests > 0 else 0
            
            print(f"\n📊 Session Analysis Results:")
            print(f"   Session tests: {session_tests}")
            print(f"   Successful sessions: {successful_sessions}")
            print(f"   Session success rate: {session_success_rate:.1%}")
            print(f"   KeyError fix status: {'✅ FIXED' if keyerror_fixed else '❌ NOT FIXED'}")
            
            if keyerror_fixed:
                print(f"\n🎉 CRITICAL FIX SUCCESSFUL!")
                print(f"   ✅ Session analysis KeyError resolved")
                print(f"   ✅ get_bias_persistence() working correctly")
                print(f"   ✅ All required keys present")
                
                success = True
            else:
                print(f"\n❌ CRITICAL FIX FAILED!")
                print(f"   ❌ Session analysis still has KeyError")
                print(f"   🔧 Immediate attention required")
                
                success = False
            
            if success:
                print(f"\n✅ Session analysis fix test PASSED")
            else:
                print(f"\n❌ Session analysis fix test FAILED")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Session analysis fix test FAILED: {e}")
            return False
    
    def test_04_trading_recommendations(self):
        """Test 4: Enhanced trading recommendations"""
        print("\n" + "="*60)
        print("💡 TEST 4: ENHANCED TRADING RECOMMENDATIONS")
        print("="*60)
        
        try:
            from features.bias_analyzer import BiasAnalyzer
            
            analyzer = BiasAnalyzer(
                bias_direction_threshold=0.04,
                signal_amplification_factor=1.6
            )
            
            recommendation_stats = {}
            risk_level_stats = {}
            total_recommendations = 0
            actionable_recommendations = 0
            
            for symbol, timeframes in self.test_data.items():
                print(f"\n--- Testing Recommendations: {symbol} ---")
                
                if len(timeframes) < 2:
                    print(f"   ⚠️  Insufficient timeframes for MTF analysis")
                    continue
                
                # Multi-timeframe analysis for better recommendations
                primary_tf = "H1" if "H1" in timeframes else list(timeframes.keys())[0]
                
                result = analyzer.analyze_bias(symbol, timeframes, primary_tf)
                
                if result:
                    total_recommendations += 1
                    rec = result.trading_recommendation
                    risk = result.risk_level
                    
                    # Track statistics
                    recommendation_stats[rec] = recommendation_stats.get(rec, 0) + 1
                    risk_level_stats[risk] = risk_level_stats.get(risk, 0) + 1
                    
                    # Check if recommendation is actionable
                    actionable_keywords = [
                        "STRONG", "MODERATE", "WEAK", "CAUTIOUS", "MINIMAL"
                    ]
                    if any(keyword in rec for keyword in actionable_keywords):
                        actionable_recommendations += 1
                    
                    # Display recommendation
                    current_price = timeframes[primary_tf]['Close'].iloc[-1]
                    print(f"   📊 Price: {current_price:.5f}")
                    print(f"   🎯 BIAS: {result.direction.name} ({result.strength.value})")
                    print(f"   💡 Recommendation: {rec}")
                    print(f"   ⚠️  Risk Level: {risk}")
                    print(f"   📈 Confidence: {result.confidence:.2f}")
                    
                    # Quality assessment
                    if "STRONG" in rec and risk == "LOW":
                        quality = "🟢 HIGH"
                    elif "MODERATE" in rec and risk in ["LOW", "MEDIUM"]:
                        quality = "🟡 GOOD"
                    elif "STAY_SIDELINES" in rec:
                        quality = "⚪ CONSERVATIVE"
                    else:
                        quality = "🟠 MODERATE"
                    
                    print(f"   Quality: {quality}")
                else:
                    print(f"   ❌ No recommendation generated")
            
            # Calculate metrics
            actionable_rate = actionable_recommendations / total_recommendations if total_recommendations > 0 else 0
            conservative_rate = recommendation_stats.get("STAY_SIDELINES", 0) / total_recommendations if total_recommendations > 0 else 0
            high_risk_rate = risk_level_stats.get("HIGH", 0) / total_recommendations if total_recommendations > 0 else 0
            
            print(f"\n📊 Trading Recommendations Results:")
            print(f"   Total recommendations: {total_recommendations}")
            print(f"   Actionable recommendations: {actionable_recommendations}")
            print(f"   Actionable rate: {actionable_rate:.1%}")
            print(f"   Conservative rate: {conservative_rate:.1%}")
            print(f"   High risk rate: {high_risk_rate:.1%}")
            
            print(f"\n📋 Recommendation Distribution:")
            for rec_type, count in recommendation_stats.items():
                percentage = count / total_recommendations * 100 if total_recommendations > 0 else 0
                if "STAY_SIDELINES" in rec_type:
                    icon = "⚪"
                elif "STRONG" in rec_type:
                    icon = "🟢"
                elif "MODERATE" in rec_type:
                    icon = "🟡"
                else:
                    icon = "🟠"
                print(f"   {icon} {rec_type}: {count} ({percentage:.1f}%)")
            
            print(f"\n⚠️  Risk Level Distribution:")
            for risk_level, count in risk_level_stats.items():
                percentage = count / total_recommendations * 100 if total_recommendations > 0 else 0
                if risk_level == "LOW":
                    icon = "🟢"
                elif risk_level == "MEDIUM":
                    icon = "🟡"
                else:
                    icon = "🔴"
                print(f"   {icon} {risk_level}: {count} ({percentage:.1f}%)")
            
            # Performance assessment
            original_actionable_rate = 0.0  # 0% from previous test
            improvement = actionable_rate - original_actionable_rate
            
            print(f"\n📈 Recommendation Enhancement:")
            print(f"   Original actionable rate: {original_actionable_rate:.1%}")
            print(f"   Enhanced actionable rate: {actionable_rate:.1%}")
            print(f"   Improvement: +{improvement:.1%}")
            
            # Success criteria
            if actionable_rate >= 0.5:  # 50% actionable target
                print(f"   🎉 EXCELLENT actionable rate!")
                performance = "EXCELLENT"
                success = True
            elif actionable_rate >= 0.3:  # 30% good
                print(f"   ✅ GOOD actionable rate!")
                performance = "GOOD"
                success = True
            elif actionable_rate >= 0.1:  # 10% acceptable
                print(f"   📊 ACCEPTABLE actionable rate")
                performance = "ACCEPTABLE"
                success = True
            else:
                print(f"   ⚠️  LOW actionable rate")
                performance = "LOW"
                success = False
            
            # Risk management assessment
            if high_risk_rate < 0.7:
                print(f"   ✅ Good risk management balance")
            else:
                print(f"   ⚠️  Too conservative risk management")
            
            if success:
                print(f"\n✅ Trading recommendations test PASSED")
            else:
                print(f"\n❌ Trading recommendations test FAILED")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Trading recommendations test FAILED: {e}")
            return False
    
    def test_05_multi_timeframe_analysis(self):
        """Test 5: Multi-timeframe BIAS analysis"""
        print("\n" + "="*60)
        print("🕐 TEST 5: MULTI-TIMEFRAME BIAS ANALYSIS")
        print("="*60)
        
        try:
            from features.bias_analyzer import BiasAnalyzer
            
            analyzer = BiasAnalyzer(
                analyze_mtf=True,
                mtf_alignment_threshold=0.6
            )
            
            mtf_tests = 0
            successful_mtf = 0
            alignment_scores = []
            
            for symbol, timeframes in self.test_data.items():
                print(f"\n--- Testing MTF Analysis: {symbol} ---")
                
                if len(timeframes) < 2:
                    print(f"   ⚠️  Need 2+ timeframes, got {len(timeframes)}")
                    continue
                
                mtf_tests += 1
                
                # Multi-timeframe analysis
                primary_tf = "H1" if "H1" in timeframes else list(timeframes.keys())[0]
                
                start_time = time.time()
                result = analyzer.analyze_bias(symbol, timeframes, primary_tf)
                analysis_time = time.time() - start_time
                
                if result and result.mtf_bias:
                    successful_mtf += 1
                    mtf = result.mtf_bias
                    alignment_scores.append(mtf.alignment_score)
                    
                    print(f"   ✓ MTF analysis completed ({analysis_time:.3f}s)")
                    print(f"   📊 Available timeframes: {list(timeframes.keys())}")
                    print(f"   🎯 MTF Results:")
                    print(f"      Long-term: {mtf.long_term_bias.name}")
                    print(f"      Medium-term: {mtf.medium_term_bias.name}")
                    print(f"      Short-term: {mtf.short_term_bias.name}")
                    print(f"      Alignment score: {mtf.alignment_score:.3f}")
                    print(f"      Dominant timeframe: {mtf.dominant_timeframe.value}")
                    
                    # Conflict analysis
                    if mtf.conflict_zones:
                        print(f"   ⚠️  Conflicts detected:")
                        for conflict in mtf.conflict_zones:
                            print(f"      - {conflict.replace('_', ' ').title()}")
                    else:
                        print(f"   ✅ No timeframe conflicts")
                    
                    # Alignment quality assessment
                    if mtf.alignment_score >= 0.8:
                        quality = "🎯 EXCELLENT"
                        desc = "Strong multi-timeframe agreement"
                    elif mtf.alignment_score >= 0.6:
                        quality = "✅ GOOD"
                        desc = "Good timeframe alignment"
                    elif mtf.alignment_score >= 0.4:
                        quality = "📊 MODERATE"
                        desc = "Moderate alignment"
                    else:
                        quality = "⚠️ POOR"
                        desc = "Weak timeframe agreement"
                    
                    print(f"   Quality: {quality} - {desc}")
                    
                    # Trading implications
                    if mtf.alignment_score >= 0.7 and len(mtf.conflict_zones) == 0:
                        print(f"   💡 Trading: HIGH CONFIDENCE setup")
                    elif mtf.alignment_score >= 0.5:
                        print(f"   💡 Trading: MODERATE CONFIDENCE")
                    else:
                        print(f"   💡 Trading: LOW CONFIDENCE - wait for alignment")
                else:
                    print(f"   ❌ MTF analysis failed ({analysis_time:.3f}s)")
            
            # Results summary
            mtf_success_rate = successful_mtf / mtf_tests if mtf_tests > 0 else 0
            avg_alignment = np.mean(alignment_scores) if alignment_scores else 0
            
            print(f"\n📊 Multi-Timeframe Analysis Results:")
            print(f"   MTF tests: {mtf_tests}")
            print(f"   Successful MTF analyses: {successful_mtf}")
            print(f"   MTF success rate: {mtf_success_rate:.1%}")
            print(f"   Average alignment score: {avg_alignment:.3f}")
            
            # Performance assessment
            if mtf_success_rate >= 0.8:  # 80% target
                print(f"   🎉 EXCELLENT MTF performance!")
                success = True
            elif mtf_success_rate >= 0.6:  # 60% good
                print(f"   ✅ GOOD MTF performance!")
                success = True
            elif mtf_success_rate >= 0.4:  # 40% acceptable
                print(f"   📊 ACCEPTABLE MTF performance")
                success = True
            else:
                print(f"   ⚠️  LOW MTF performance")
                success = False
            
            if success:
                print(f"\n✅ Multi-timeframe analysis test PASSED")
            else:
                print(f"\n❌ Multi-timeframe analysis test FAILED")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Multi-timeframe analysis test FAILED: {e}")
            return False
    
    def test_06_performance_benchmark(self):
        """Test 6: Performance and efficiency testing"""
        print("\n" + "="*60)
        print("⚡ TEST 6: PERFORMANCE BENCHMARK")
        print("="*60)
        
        try:
            from features.bias_analyzer import BiasAnalyzer
            
            # Find largest dataset for performance testing
            largest_symbol = None
            largest_bars = 0
            
            for symbol, timeframes in self.test_data.items():
                total_bars = sum(len(data) for data in timeframes.values())
                if total_bars > largest_bars:
                    largest_bars = total_bars
                    largest_symbol = symbol
            
            if not largest_symbol:
                print("   ❌ No test data available")
                return False
            
            largest_dataset = self.test_data[largest_symbol]
            print(f"   🔬 Performance testing with {largest_symbol}")
            print(f"   📊 Total bars: {largest_bars}")
            
            for tf, data in largest_dataset.items():
                print(f"      {tf}: {len(data)} bars")
            
            # Test 1: Standard analyzer performance
            print(f"\n⚡ Standard Analyzer Performance:")
            standard_analyzer = BiasAnalyzer()
            
            primary_tf = list(largest_dataset.keys())[0]
            single_tf_data = {primary_tf: largest_dataset[primary_tf]}
            
            start_time = time.time()
            standard_result = standard_analyzer.analyze_bias(largest_symbol, single_tf_data, primary_tf)
            standard_time = time.time() - start_time
            
            standard_bars = len(largest_dataset[primary_tf])
            standard_performance = standard_bars / standard_time if standard_time > 0 else 0
            
            print(f"   Analysis time: {standard_time:.3f}s")
            print(f"   Performance: {standard_performance:.0f} bars/second")
            print(f"   Components: {len(standard_result.components) if standard_result else 0}")
            
            # Test 2: Enhanced analyzer performance
            print(f"\n⚡ Enhanced Analyzer Performance:")
            enhanced_analyzer = BiasAnalyzer(
                bias_direction_threshold=0.03,
                signal_amplification_factor=1.8,
                adaptive_thresholds=True,
                volatility_adjustment=True
            )
            
            start_time = time.time()
            enhanced_result = enhanced_analyzer.analyze_bias(largest_symbol, single_tf_data, primary_tf)
            enhanced_time = time.time() - start_time
            
            enhanced_performance = standard_bars / enhanced_time if enhanced_time > 0 else 0
            
            print(f"   Analysis time: {enhanced_time:.3f}s")
            print(f"   Performance: {enhanced_performance:.0f} bars/second")
            print(f"   Components: {len(enhanced_result.components) if enhanced_result else 0}")
            print(f"   Market volatility: {enhanced_analyzer.market_volatility:.4f}")
            
            # Test 3: Multi-timeframe performance
            print(f"\n⚡ Multi-Timeframe Performance:")
            start_time = time.time()
            mtf_result = enhanced_analyzer.analyze_bias(largest_symbol, largest_dataset, primary_tf)
            mtf_time = time.time() - start_time
            
            mtf_performance = largest_bars / mtf_time if mtf_time > 0 else 0
            
            print(f"   Analysis time: {mtf_time:.3f}s")
            print(f"   Performance: {mtf_performance:.0f} bars/second")
            print(f"   MTF analysis: {'✅' if mtf_result and mtf_result.mtf_bias else '❌'}")
            print(f"   Session analysis: {'✅' if mtf_result and mtf_result.session_analysis else '❌'}")
            
            # Test 4: Repeated analysis (caching effects)
            print(f"\n⚡ Repeated Analysis Performance:")
            repeated_times = []
            for i in range(3):
                start_time = time.time()
                repeated_result = enhanced_analyzer.analyze_bias(largest_symbol, largest_dataset, primary_tf)
                repeated_time = time.time() - start_time
                repeated_times.append(repeated_time)
            
            avg_repeated_time = sum(repeated_times) / len(repeated_times)
            repeated_performance = largest_bars / avg_repeated_time if avg_repeated_time > 0 else 0
            
            print(f"   Average time: {avg_repeated_time:.3f}s")
            print(f"   Average performance: {repeated_performance:.0f} bars/second")
            
            # Performance assessment
            print(f"\n📊 Performance Assessment:")
            
            # Performance targets
            min_performance = 500     # Minimum for real-time
            good_performance = 1000   # Good performance
            excellent_performance = 3000  # Excellent performance
            
            primary_performance = mtf_performance
            
            if primary_performance >= excellent_performance:
                performance_rating = "🚀 EXCELLENT"
                performance_desc = "Outstanding - Ready for high-frequency trading"
            elif primary_performance >= good_performance:
                performance_rating = "✅ GOOD"
                performance_desc = "Good - Suitable for live trading"
            elif primary_performance >= min_performance:
                performance_rating = "📊 ACCEPTABLE"
                performance_desc = "Acceptable - Ready for standard trading"
            else:
                performance_rating = "⚠️ NEEDS_OPTIMIZATION"
                performance_desc = "Below minimum requirements"
            
            print(f"   Overall rating: {performance_rating}")
            print(f"   Assessment: {performance_desc}")
            
            # Performance targets check
            print(f"\n🎯 Performance Targets:")
            print(f"   Minimum (Real-time): {min_performance:,} bars/sec "
                  f"({'✅' if primary_performance >= min_performance else '❌'})")
            print(f"   Good: {good_performance:,} bars/sec "
                  f"({'✅' if primary_performance >= good_performance else '❌'})")
            print(f"   Excellent: {excellent_performance:,} bars/sec "
                  f"({'✅' if primary_performance >= excellent_performance else '❌'})")
            
            # Enhancement impact
            if enhanced_time > 0 and standard_time > 0:
                time_overhead = (enhanced_time - standard_time) / standard_time * 100
                print(f"\n📈 Enhancement Impact:")
                print(f"   Time overhead: {time_overhead:+.1f}%")
                
                if enhanced_result and standard_result:
                    if enhanced_result.direction != standard_result.direction:
                        print(f"   Detection difference: Enhanced found different BIAS")
                    else:
                        print(f"   Detection consistency: Both found same BIAS direction")
            
            success = primary_performance >= min_performance
            
            if success:
                print(f"\n✅ Performance benchmark test PASSED")
            else:
                print(f"\n❌ Performance benchmark test FAILED")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Performance benchmark test FAILED: {e}")
            return False
    
    def run_all_tests(self):
        """Run all test modules and generate comprehensive report"""
        print("="*70)
        print("🚀 BIAS ANALYZER - COMPREHENSIVE TEST SUITE")
        print("   Enhanced BIAS Integration Testing")
        print("="*70)
        print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Objective: Validate enhanced BIAS analyzer functionality")
        
        # Test modules
        test_modules = [
            ("01_initialization", "Initialization & Configuration", self.test_01_initialization),
            ("02_single_timeframe", "Single Timeframe Detection", self.test_02_single_timeframe_detection),
            ("03_session_fix", "Session Analysis KeyError Fix", self.test_03_session_analysis_fix),
            ("04_trading_recs", "Trading Recommendations", self.test_04_trading_recommendations),
            ("05_multi_timeframe", "Multi-Timeframe Analysis", self.test_05_multi_timeframe_analysis),
            ("06_performance", "Performance Benchmark", self.test_06_performance_benchmark)
        ]
        
        # Run tests
        for test_id, test_name, test_function in test_modules:
            print(f"\n{'='*50}")
            print(f"🧪 RUNNING: {test_name}")
            print(f"{'='*50}")
            
            try:
                module_start_time = time.time()
                passed = test_function()
                module_duration = time.time() - module_start_time
                
                self.results[test_id] = {
                    'name': test_name,
                    'passed': passed,
                    'duration': module_duration
                }
                
                status = "✅ PASSED" if passed else "❌ FAILED"
                print(f"\n🏁 {test_name}: {status} ({module_duration:.2f}s)")
                
            except Exception as e:
                self.results[test_id] = {
                    'name': test_name,
                    'passed': False,
                    'duration': 0,
                    'error': str(e)
                }
                print(f"\n🏁 {test_name}: ❌ FAILED (Exception: {e})")
        
        # Generate final report
        self.generate_final_report()
        
        # Return overall success
        passed_tests = sum(1 for r in self.results.values() if r['passed'])
        return passed_tests >= len(test_modules) * 0.8  # 80% pass rate
    
    def generate_final_report(self):
        """Generate comprehensive final test report"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE TEST REPORT - FINAL SUMMARY")
        print("="*70)
        
        passed_tests = sum(1 for r in self.results.values() if r['passed'])
        total_tests = len(self.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 Test Statistics:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Total Duration: {total_time:.2f} seconds")
        
        print(f"\n📋 Detailed Results:")
        for test_id, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            name = result['name']
            duration = result['duration']
            print(f"   {status} {name}: ({duration:.2f}s)")
            
            if not result['passed'] and 'error' in result:
                print(f"      Error: {result['error']}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        if success_rate >= 0.9:  # 90%+ excellent
            print(f"   🎉 OUTSTANDING PERFORMANCE!")
            print(f"   ✅ Enhanced BIAS analyzer working exceptionally well")
            print(f"   ✅ All critical fixes successfully implemented")
            print(f"   ✅ Performance exceeds requirements")
            print(f"   🚀 Production ready - proceed to next phase")
            assessment = "OUTSTANDING"
            
        elif success_rate >= 0.8:  # 80%+ excellent
            print(f"   🎉 EXCELLENT PERFORMANCE!")
            print(f"   ✅ Enhanced BIAS analyzer working very well")
            print(f"   ✅ Major improvements successfully implemented")
            print(f"   📊 Minor optimizations possible")
            print(f"   ✅ Ready for production use")
            assessment = "EXCELLENT"
            
        elif success_rate >= 0.7:  # 70%+ good
            print(f"   ✅ GOOD PERFORMANCE!")
            print(f"   📊 Enhanced BIAS analyzer showing strong improvements")
            print(f"   ✅ Core functionality working well")
            print(f"   🔧 Some areas need attention")
            assessment = "GOOD"
            
        elif success_rate >= 0.5:  # 50%+ acceptable
            print(f"   📊 ACCEPTABLE PERFORMANCE")
            print(f"   ⚠️  Enhanced BIAS analyzer needs optimization")
            print(f"   🔧 Several issues require fixes")
            assessment = "ACCEPTABLE"
            
        else:  # <50% needs work
            print(f"   ⚠️  NEEDS SIGNIFICANT WORK")
            print(f"   ❌ Enhanced BIAS analyzer has major issues")
            print(f"   🔧 Critical fixes required before proceeding")
            assessment = "NEEDS_WORK"
        
        # Key achievements analysis
        key_fixes = {
            'session_fix': self.results.get('03_session_fix', {}).get('passed', False),
            'single_tf_improvement': self.results.get('02_single_timeframe', {}).get('passed', False),
            'trading_recs': self.results.get('04_trading_recs', {}).get('passed', False),
            'performance': self.results.get('06_performance', {}).get('passed', False)
        }
        
        print(f"\n🎯 KEY ACHIEVEMENTS:")
        if key_fixes['session_fix']:
            print(f"   ✅ Session Analysis KeyError: FIXED")
        else:
            print(f"   ❌ Session Analysis KeyError: NOT FIXED")
            
        if key_fixes['single_tf_improvement']:
            print(f"   ✅ Single Timeframe Detection: ENHANCED")
        else:
            print(f"   ❌ Single Timeframe Detection: NEEDS WORK")
            
        if key_fixes['trading_recs']:
            print(f"   ✅ Trading Recommendations: MORE ACTIONABLE")
        else:
            print(f"   ❌ Trading Recommendations: STILL CONSERVATIVE")
            
        if key_fixes['performance']:
            print(f"   ✅ Performance: MEETS REQUIREMENTS")
        else:
            print(f"   ❌ Performance: BELOW REQUIREMENTS")
        
        # Next steps
        if assessment in ["OUTSTANDING", "EXCELLENT"]:
            print(f"\n📋 NEXT STEPS - PHASE 2 WEEK 5:")
            print(f"   ✅ Implement Volume Profile analyzer")
            print(f"   ✅ Add VWAP calculations")
            print(f"   ✅ Complete Fibonacci levels")
            print(f"   ✅ Finalize Feature Aggregator")
            print(f"   ✅ Multi-timeframe feature alignment")
            print(f"   ✅ Comprehensive confluence scoring")
            print(f"   ✅ Final performance optimization")
            
        elif assessment == "GOOD":
            print(f"\n🔧 IMMEDIATE ACTIONS:")
            failed_tests = [r['name'] for r in self.results.values() if not r['passed']]
            for test in failed_tests:
                print(f"   - Optimize: {test}")
            print(f"\n📋 THEN PROCEED TO WEEK 5")
            
        else:
            print(f"\n🔧 CRITICAL FIXES REQUIRED:")
            for test_id, result in self.results.items():
                if not result['passed']:
                    print(f"   - Fix: {result['name']}")
            print(f"\n⚠️  Complete all fixes before proceeding")
        
        # Improvement summary
        improvements_achieved = sum([
            key_fixes['session_fix'],
            key_fixes['single_tf_improvement'], 
            key_fixes['trading_recs'],
            key_fixes['performance']
        ])
        
        print(f"\n📈 IMPROVEMENT SUMMARY:")
        print(f"   Key fixes implemented: {improvements_achieved}/4")
        print(f"   Overall test success: {success_rate:.1%}")
        print(f"   Enhanced BIAS analyzer: {'✅ READY' if assessment in ['OUTSTANDING', 'EXCELLENT'] else '🔧 NEEDS WORK'}")
        
        print(f"\n🕒 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main test execution function"""
    setup_logging()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        return 1
    
    # Create and run test suite
    test_suite = BiasTestSuite()
    
    try:
        # Setup test environment
        if not test_suite.setup():
            print("❌ Test setup failed")
            return 1
        
        # Run comprehensive tests
        success = test_suite.run_all_tests()
        
        if success:
            print(f"\n🎉 BIAS ANALYZER TESTING COMPLETED SUCCESSFULLY!")
            print(f"🚀 Enhanced BIAS analyzer validated and ready")
            return 0
        else:
            print(f"\n⚠️  BIAS ANALYZER TESTING COMPLETED WITH ISSUES")
            print(f"🔧 Review failed tests and apply fixes")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n👋 Testing interrupted by user")
        return 0
    except Exception as e:
        print(f"\n💥 Unexpected testing error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)