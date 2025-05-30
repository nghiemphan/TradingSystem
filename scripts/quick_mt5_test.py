"""
Quick MT5 Connection Test
Simple script to test MT5 connection without full system
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.connectors.mt5_connector import get_mt5_connector
from config.mt5_config import MT5_CONNECTION

def main():
    """Quick MT5 test"""
    print("="*50)
    print("   QUICK MT5 CONNECTION TEST")
    print("="*50)
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n1. Configuration Check:")
    print(f"   Login: {MT5_CONNECTION.login}")
    print(f"   Server: {MT5_CONNECTION.server}")
    print(f"   Password: {'Set' if MT5_CONNECTION.password else 'Not Set'}")
    
    print(f"\n2. Connection Test:")
    try:
        # Get connector
        connector = get_mt5_connector(MT5_CONNECTION)
        
        # Try to connect
        print("   Attempting connection...")
        success = connector.connect()
        
        if success:
            print("   ✅ Connected successfully!")
            
            # Get account info
            account_info = connector.account_info
            if account_info:
                print(f"   Account: {account_info.login}")
                print(f"   Balance: {account_info.balance} {account_info.currency}")
                print(f"   Leverage: {account_info.leverage}")
            
            # Test data retrieval
            print(f"\n3. Data Test:")
            data = connector.get_rates("EURUSD", "H1", 5)
            if data is not None and len(data) > 0:
                print(f"   ✅ Data retrieval successful!")
                print(f"   Retrieved: {len(data)} bars")
                print(f"   Latest: {data.index[-1]} - Close: {data['Close'].iloc[-1]:.5f}")
            else:
                print(f"   ❌ Data retrieval failed!")
            
            # Test current price
            print(f"\n4. Current Price Test:")
            current_price = connector.get_current_price("EURUSD")
            if current_price:
                print(f"   ✅ Current price retrieval successful!")
                print(f"   Bid: {current_price['bid']:.5f}")
                print(f"   Ask: {current_price['ask']:.5f}")
            else:
                print(f"   ❌ Current price retrieval failed!")
            
            # Disconnect
            connector.disconnect()
            print(f"\n✅ MT5 CONNECTION WORKING PERFECTLY!")
            return True
            
        else:
            print("   ❌ Connection failed!")
            
            # Check what went wrong
            import MetaTrader5 as mt5
            error = mt5.last_error()
            print(f"   Error: {error}")
            
            # Common issues
            print(f"\n🔧 Troubleshooting:")
            print(f"   1. Check if MT5 terminal is running")
            print(f"   2. Verify login credentials in .env file")
            print(f"   3. Ensure account is not locked")
            print(f"   4. Check server name spelling")
            
            return False
            
    except Exception as e:
        print(f"   ❌ Exception occurred: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 MT5 is ready for SMC feature testing!")
        print(f"   Run: python scripts/test_smc_features.py")
    else:
        print(f"\n⚠️  MT5 connection needs attention")
        print(f"   SMC features will use demo data")
    
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)