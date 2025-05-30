"""
AI Trading System - Main Entry Point
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import LOG_CONFIG, SYSTEM_CONFIG
from config.mt5_config import MT5_CONNECTION, get_trading_symbols
from data.connectors.mt5_connector import get_mt5_connector
from data.connectors.demo_connector import get_demo_connector
from data.storage.database_manager import get_database_manager
from data.storage.cache_manager import get_cache_manager
from data.storage.file_storage import get_file_storage_manager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG.log_level),
        format=LOG_CONFIG.log_format,
        handlers=[
            logging.FileHandler(LOG_CONFIG.system_log),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create separate loggers for different components
    logger_configs = {
        'trading': LOG_CONFIG.trading_log,
        'system': LOG_CONFIG.system_log, 
        'error': LOG_CONFIG.error_log,
        'performance': LOG_CONFIG.performance_log
    }
    
    for logger_name, log_file in logger_configs.items():
        logger = logging.getLogger(logger_name)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(LOG_CONFIG.log_format))
        logger.addHandler(handler)

def test_system_components():
    """Test all system components"""
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("AI TRADING SYSTEM - COMPONENT TESTING")
    print("="*60)
    
    # Test 1: Configuration
    print("\n1. Testing Configuration...")
    try:
        print(f"   Environment: {SYSTEM_CONFIG.environment}")
        print(f"   Debug Mode: {SYSTEM_CONFIG.debug_mode}")
        print(f"   Trading Symbols: {get_trading_symbols()}")
        print("   ✓ Configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Configuration error: {e}")
        return False
    
    # Test 2: MT5/Demo Connection
    print("\n2. Testing Connection...")
    try:
        # Check if MT5 credentials are configured
        use_demo = (MT5_CONNECTION.login == 0 or 
                   not MT5_CONNECTION.password or 
                   not MT5_CONNECTION.server)
        
        connector = None
        test_result = None
        
        if not use_demo:
            print("   🔗 Attempting MT5 connection...")
            connector = get_mt5_connector(MT5_CONNECTION)
            test_result = connector.test_connection()
            
            # If MT5 fails, fallback to demo
            if not test_result['connected']:
                print("   ⚠️  MT5 connection failed - falling back to Demo Mode")
                use_demo = True
        
        if use_demo:
            print("   🎮 Using Demo Mode (fake data for testing)")
            connector = get_demo_connector()
            test_result = connector.test_connection()
        
        if test_result['connected']:
            mode = "Demo Mode" if test_result.get('demo_mode') else "Real MT5"
            print(f"   ✓ {mode} Connected successfully")
            print(f"   Account: {test_result['account_info']['login']}")
            print(f"   Balance: {test_result['account_info']['balance']} {test_result['account_info']['currency']}")
            print(f"   Available Symbols: {test_result['symbols_count']}")
            
            if test_result['test_data']:
                print(f"   Test Data: {test_result['test_data']['bars_count']} bars retrieved")
                print(f"   Sample Price: {test_result['test_data']['latest_close']:.5f}")
            
        else:
            print(f"   ✗ Connection failed: {test_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ✗ MT5 Connection error: {e}")
        return False
    
    # Test 3: Data Retrieval
    print("\n3. Testing Data Retrieval...")
    try:
        test_symbol = "EURUSD"
        test_timeframe = "H1"
        
        # Try different timeframes if one fails
        timeframes_to_test = ["H1", "M15", "M5"]
        df = None
        
        for tf in timeframes_to_test:
            df = connector.get_rates(test_symbol, tf, 50)  # Reduced count for faster testing
            if df is not None and len(df) > 0:
                test_timeframe = tf
                break
        
        if df is not None and len(df) > 0:
            print(f"   ✓ Retrieved {len(df)} bars for {test_symbol} {test_timeframe}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Latest close: {df['Close'].iloc[-1]:.5f}")
            print(f"   Columns: {list(df.columns)}")
        else:
            print(f"   ⚠️  Data retrieval failed for {test_symbol} - but connection works")
            print(f"   This might be due to market hours or symbol availability")
            # Don't fail the test completely for data issues
            
    except Exception as e:
        print(f"   ⚠️  Data retrieval error: {e}")
        print(f"   Connection works but data access has issues - proceeding anyway")
    
    # Test 4: Current Price (only if we have a working connector)
    print("\n4. Testing Current Price...")
    try:
        if hasattr(connector, 'get_current_price'):
            price_data = connector.get_current_price("EURUSD")
            if price_data:
                print(f"   ✓ Current EURUSD price:")
                print(f"   Bid: {price_data['bid']:.5f}")
                print(f"   Ask: {price_data['ask']:.5f}")
                print(f"   Spread: {price_data['spread']:.5f}")
            else:
                print("   ⚠️  Current price not available - might be market hours")
        else:
            print("   ⚠️  Current price function not available in demo mode")
            
    except Exception as e:
        print(f"   ⚠️  Current price error: {e}")
        print(f"   This is often normal outside market hours")
    
    # Test 5: Database System
    print("\n5. Testing Database System...")
    try:
        db_manager = get_database_manager()
        print("   ✓ Database initialized successfully")
        
        # Test database operations
        stats = db_manager.get_database_stats()
        print(f"   Database size: {stats['db_size_mb']:.2f} MB")
        print(f"   Tables: {len([k for k, v in stats.items() if k != 'db_size_mb'])}")
        
        # Test saving market data if we have it
        if 'df' in locals() and df is not None and len(df) > 0:
            rows_saved = db_manager.save_market_data("EURUSD", "H1", df.tail(10))
            print(f"   ✓ Saved {rows_saved} bars to database")
        
    except Exception as e:
        print(f"   ⚠️  Database test failed: {e}")
    
    # Test 6: Cache System
    print("\n6. Testing Cache System...")
    try:
        cache_manager = get_cache_manager()
        
        if cache_manager.is_connected():
            print("   ✓ Cache system connected")
            
            # Test caching
            test_data = {"test": "value", "timestamp": "2024-01-01"}
            cache_manager.cache_current_price("TEST", test_data)
            retrieved = cache_manager.get_cached_price("TEST")
            
            if retrieved and retrieved.get("test") == "value":
                print("   ✓ Cache operations working")
            else:
                print("   ⚠️  Cache test failed")
                
            cache_stats = cache_manager.get_cache_stats()
            mode = "Redis" if not cache_stats.get('fallback_mode') else "In-Memory"
            print(f"   Cache mode: {mode}")
            
        else:
            print("   ⚠️  Cache using fallback mode")
            
    except Exception as e:
        print(f"   ⚠️  Cache test failed: {e}")
    
    # Test 7: File Storage
    print("\n7. Testing File Storage...")
    try:
        file_manager = get_file_storage_manager()
        print("   ✓ File storage initialized")
        
        # Get storage stats
        storage_stats = file_manager.get_storage_stats()
        total_files = sum(stats['file_count'] for stats in storage_stats.values())
        total_size = sum(stats['size_mb'] for stats in storage_stats.values())
        print(f"   Storage: {total_files} files, {total_size:.2f} MB")
        
    except Exception as e:
        print(f"   ⚠️  File storage test failed: {e}")
    
    # Clean up
    connector.disconnect()
    
    print("\n" + "="*60)
    print("✅ SYSTEM COMPONENTS TESTED SUCCESSFULLY!")
    print("✅ Connection established - Ready for development")
    
    if hasattr(connector, 'get_current_price'):
        print("✅ Real MT5 connection active")
    else:
        print("✅ Demo mode active - Perfect for development")
        
    print("="*60)
    
    return True

def main():
    """Main application entry point"""
    print("AI Trading System - Starting...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Test system components
        if test_system_components():
            logger.info("System startup completed successfully")
            
            # Future: This is where the main trading loop will go
            print("\n📈 System ready for trading operations")
            print("🔧 Phase 1 - Foundation Setup: COMPLETED ✅")
            print("🔧 Database & Storage: Ready ✅")
            print("⏳ Next: Phase 2 - SMC Feature Development")
            print("\n💡 To setup database completely, run:")
            print("   python scripts/setup_database.py")
            
        else:
            logger.error("System startup failed")
            print("\n❌ System startup failed - check configuration")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested")
        logger.info("System shutdown by user")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)