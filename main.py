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
    
    # Test 2: MT5 Connection
    print("\n2. Testing MT5 Connection...")
    try:
        connector = get_mt5_connector(MT5_CONNECTION)
        test_result = connector.test_connection()
        
        if test_result['connected']:
            print("   ✓ MT5 Connected successfully")
            print(f"   Account: {test_result['account_info']['login']}")
            print(f"   Balance: {test_result['account_info']['balance']} {test_result['account_info']['currency']}")
            print(f"   Available Symbols: {test_result['symbols_count']}")
            
            if test_result['test_data']:
                print(f"   Test Data: {test_result['test_data']['bars_count']} bars retrieved")
            
        else:
            print(f"   ✗ MT5 Connection failed: {test_result.get('error', 'Unknown error')}")
            print("\n   Please check your MT5 configuration in config/mt5_config.py")
            print("   Make sure MT5 terminal is running and credentials are correct")
            return False
            
    except Exception as e:
        print(f"   ✗ MT5 Connection error: {e}")
        return False
    
    # Test 3: Data Retrieval
    print("\n3. Testing Data Retrieval...")
    try:
        test_symbol = "EURUSD"
        test_timeframe = "H1"
        
        df = connector.get_rates(test_symbol, test_timeframe, 100)
        if df is not None and len(df) > 0:
            print(f"   ✓ Retrieved {len(df)} bars for {test_symbol} {test_timeframe}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Latest close: {df['Close'].iloc[-1]:.5f}")
        else:
            print(f"   ✗ Failed to retrieve data for {test_symbol}")
            return False
            
    except Exception as e:
        print(f"   ✗ Data retrieval error: {e}")
        return False
    
    # Test 4: Current Price
    print("\n4. Testing Current Price...")
    try:
        price_data = connector.get_current_price("EURUSD")
        if price_data:
            print(f"   ✓ Current EURUSD price:")
            print(f"   Bid: {price_data['bid']:.5f}")
            print(f"   Ask: {price_data['ask']:.5f}")
            print(f"   Spread: {price_data['spread']:.5f}")
        else:
            print("   ✗ Failed to get current price")
            return False
            
    except Exception as e:
        print(f"   ✗ Current price error: {e}")
        return False
    
    # Clean up
    connector.disconnect()
    
    print("\n" + "="*60)
    print("✓ ALL SYSTEM COMPONENTS TESTED SUCCESSFULLY!")
    print("✓ Ready to proceed with Phase 1 development")
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
            print("🔧 Currently in Phase 1 - Foundation Setup")
            print("⏳ Next: Implement database and feature calculations")
            
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