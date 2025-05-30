"""
MetaTrader 5 Connection and Data Management
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
from dataclasses import dataclass
import time

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class MT5Config:
    """MT5 Connection Configuration"""
    login: int = 0  # Your MT5 account number
    password: str = ""  # Your MT5 password
    server: str = ""  # Your broker server
    path: str = ""  # Path to MT5 terminal (if needed)
    timeout: int = 60000  # Connection timeout in ms
    
class MT5Connector:
    """
    MetaTrader 5 API Connector for data retrieval and trading
    """
    
    def __init__(self, config: MT5Config = None):
        self.config = config or MT5Config()
        self.connected = False
        self.account_info = None
        
    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal
        
        Returns:
            bool: True if connection successful
        """
        try:
            # First try simple initialization (most common case)
            if not mt5.initialize():
                logger.error(f"MT5 simple initialization failed: {mt5.last_error()}")
                
                # Try with login credentials if provided
                if (self.config.login and self.config.password and self.config.server):
                    logger.info("Trying initialization with credentials...")
                    if not mt5.initialize(
                        login=self.config.login,
                        password=self.config.password,
                        server=self.config.server
                    ):
                        logger.error(f"MT5 credential initialization failed: {mt5.last_error()}")
                        return False
                else:
                    logger.error("MT5 initialization failed and no credentials provided")
                    return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account info")
                return False
            
            self.connected = True
            logger.info(f"Connected to MT5 - Account: {self.account_info.login}")
            logger.info(f"Balance: {self.account_info.balance} {self.account_info.currency}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def is_connected(self) -> bool:
        """Check if MT5 is connected"""
        if not self.connected:
            return False
            
        # Test connection with a simple call
        try:
            account = mt5.account_info()
            return account is not None
        except:
            self.connected = False
            return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5"""
        logger.info("Attempting to reconnect to MT5...")
        self.disconnect()
        time.sleep(5)  # Wait before reconnecting
        return self.connect()
    
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols
        
        Returns:
            List of available trading symbols
        """
        if not self.is_connected():
            logger.error("Not connected to MT5")
            return []
        
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                logger.error("Failed to get symbols")
                return []
            
            return [symbol.name for symbol in symbols if symbol.visible]
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol information
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol information
        """
        if not self.is_connected():
            logger.error("Not connected to MT5")
            return None
        
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            return {
                'name': info.name,
                'digits': info.digits,
                'point': info.point,
                'spread': info.spread,
                'min_lot': info.volume_min,
                'max_lot': info.volume_max,
                'lot_step': info.volume_step,
                'contract_size': info.trade_contract_size,
                'margin_required': info.margin_initial
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_rates(self, 
                  symbol: str, 
                  timeframe: str, 
                  count: int = 1000,
                  from_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Get historical price data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            count: Number of bars to retrieve
            from_date: Start date for data retrieval
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected():
            logger.error("Not connected to MT5")
            return None
        
        # Convert timeframe string to MT5 constant
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        if timeframe not in tf_map:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None
        
        try:
            if from_date:
                rates = mt5.copy_rates_from(symbol, tf_map[timeframe], from_date, count)
            else:
                rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No data received for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Check actual columns in the data
            original_columns = df.columns.tolist()
            logger.debug(f"Original columns: {original_columns}")
            
            # Standard MT5 columns mapping
            expected_columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            column_mapping = {}
            
            # Map available columns to standard names
            if 'open' in df.columns:
                column_mapping['open'] = 'Open'
            if 'high' in df.columns:
                column_mapping['high'] = 'High'  
            if 'low' in df.columns:
                column_mapping['low'] = 'Low'
            if 'close' in df.columns:
                column_mapping['close'] = 'Close'
            
            # Handle volume columns (tick_volume is more common)
            if 'tick_volume' in df.columns:
                column_mapping['tick_volume'] = 'Volume'
            elif 'real_volume' in df.columns:
                column_mapping['real_volume'] = 'Volume'
            
            # Handle spread
            if 'spread' in df.columns:
                column_mapping['spread'] = 'Spread'
                
            # Rename only existing columns
            df = df.rename(columns=column_mapping)
            
            # Ensure we have at least OHLC
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Add default Volume if missing
            if 'Volume' not in df.columns:
                df['Volume'] = 0
                logger.warning("Volume column missing, using default values")
            
            # Add default Spread if missing  
            if 'Spread' not in df.columns:
                df['Spread'] = 0
                logger.warning("Spread column missing, using default values")
            
            # Ensure correct column order
            final_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
            df = df[final_columns]
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting rates for {symbol} {timeframe}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current bid/ask prices
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with current prices
        """
        if not self.is_connected():
            logger.error("Not connected to MT5")
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"No tick data for {symbol}")
                return None
            
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def test_connection(self) -> Dict:
        """
        Test MT5 connection and return status
        
        Returns:
            Dictionary with connection test results
        """
        result = {
            'connected': False,
            'account_info': None,
            'symbols_count': 0,
            'test_data': None,
            'error': None
        }
        
        try:
            # Test connection
            if not self.is_connected():
                if not self.connect():
                    result['error'] = "Failed to connect to MT5"
                    return result
            
            result['connected'] = True
            result['account_info'] = {
                'login': self.account_info.login,
                'balance': self.account_info.balance,
                'currency': self.account_info.currency,
                'leverage': self.account_info.leverage
            }
            
            # Test symbol retrieval
            symbols = self.get_symbols()
            result['symbols_count'] = len(symbols)
            
            # Test data retrieval with EURUSD
            if 'EURUSD' in symbols:
                test_data = self.get_rates('EURUSD', 'M1', 10)
                if test_data is not None:
                    result['test_data'] = {
                        'symbol': 'EURUSD',
                        'bars_count': len(test_data),
                        'latest_close': float(test_data['Close'].iloc[-1])
                    }
            
            logger.info("MT5 connection test completed successfully")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"MT5 connection test failed: {e}")
        
        return result

# Singleton instance
_mt5_connector = None

def get_mt5_connector(config: MT5Config = None) -> MT5Connector:
    """
    Get singleton MT5 connector instance
    
    Args:
        config: MT5 configuration
        
    Returns:
        MT5Connector instance
    """
    global _mt5_connector
    
    if _mt5_connector is None:
        _mt5_connector = MT5Connector(config)
    
    return _mt5_connector

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test MT5 connection
    connector = get_mt5_connector()
    test_result = connector.test_connection()
    
    print("MT5 Connection Test Results:")
    for key, value in test_result.items():
        print(f"{key}: {value}")
    
    # Clean up
    connector.disconnect()