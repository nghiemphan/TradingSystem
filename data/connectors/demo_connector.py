"""
Demo connector for testing without real MT5 connection
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class DemoConnector:
    """
    Demo connector that generates fake data for testing
    """
    
    def __init__(self, config=None):
        self.connected = False
        self.account_info = None
        
    def connect(self) -> bool:
        """Simulate MT5 connection"""
        try:
            self.connected = True
            self.account_info = {
                'login': 99999999,
                'balance': 10000.0,
                'currency': 'USD',
                'leverage': 100
            }
            logger.info("Connected to Demo Mode - No real MT5 connection")
            return True
        except Exception as e:
            logger.error(f"Demo connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Simulate disconnect"""
        self.connected = False
        logger.info("Disconnected from Demo Mode")
    
    def is_connected(self) -> bool:
        """Check demo connection"""
        return self.connected
    
    def reconnect(self) -> bool:
        """Simulate reconnection"""
        return self.connect()
    
    def get_symbols(self) -> List[str]:
        """Return demo symbols"""
        return [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
            'USDCAD', 'AUDUSD', 'NZDUSD', 'EURGBP'
        ]
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Return demo symbol info"""
        return {
            'name': symbol,
            'digits': 5,
            'point': 0.00001,
            'spread': 2,
            'min_lot': 0.01,
            'max_lot': 100.0,
            'lot_step': 0.01,
            'contract_size': 100000,
            'margin_required': 1000.0
        }
    
    def generate_fake_data(self, symbol: str, timeframe: str, count: int = 1000) -> pd.DataFrame:
        """Generate fake OHLCV data"""
        
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.3000,
            'USDJPY': 110.00,
            'USDCHF': 0.9200,
            'USDCAD': 1.2500,
            'AUDUSD': 0.7500,
            'NZDUSD': 0.7000,
            'EURGBP': 0.8500
        }
        
        base_price = base_prices.get(symbol, 1.1000)
        
        # Generate time index
        timeframe_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 5)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes * count)
        
        time_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{minutes}min'
        )[:count]
        
        # Generate price data using random walk
        np.random.seed(42)  # For reproducible results
        
        # Random walk for prices
        returns = np.random.normal(0, 0.0001, count)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from prices
        opens = prices
        
        # Generate high/low with some spread
        highs = opens + np.random.uniform(0, 0.002, count)
        lows = opens - np.random.uniform(0, 0.002, count)
        
        # Ensure high >= open >= low and high >= close >= low
        closes = opens + np.random.normal(0, 0.0005, count)
        
        # Fix any inconsistencies
        for i in range(count):
            high = max(opens[i], highs[i], closes[i])
            low = min(opens[i], lows[i], closes[i])
            highs[i] = high
            lows[i] = low
        
        # Generate volume
        volumes = np.random.randint(50, 500, count)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes,
            'Spread': np.random.randint(1, 5, count)
        }, index=time_index)
        
        return df
    
    def get_rates(self, symbol: str, timeframe: str, count: int = 1000, 
                  from_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get fake historical data"""
        if not self.is_connected():
            logger.error("Not connected to Demo Mode")
            return None
        
        try:
            df = self.generate_fake_data(symbol, timeframe, count)
            logger.info(f"Generated {len(df)} fake bars for {symbol} {timeframe}")
            return df
        except Exception as e:
            logger.error(f"Error generating fake data: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get fake current price"""
        if not self.is_connected():
            return None
        
        try:
            # Get last price from fake data
            df = self.generate_fake_data(symbol, 'M1', 1)
            if df is None or len(df) == 0:
                return None
            
            last_close = float(df['Close'].iloc[-1])
            spread = 0.00002  # 2 pips
            
            return {
                'symbol': symbol,
                'bid': last_close - spread/2,
                'ask': last_close + spread/2,
                'spread': spread,
                'time': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting fake price: {e}")
            return None
    
    def test_connection(self) -> Dict:
        """Test demo connection"""
        result = {
            'connected': False,
            'account_info': None,
            'symbols_count': 0,
            'test_data': None,
            'error': None,
            'demo_mode': True
        }
        
        try:
            if not self.is_connected():
                if not self.connect():
                    result['error'] = "Failed to connect to Demo Mode"
                    return result
            
            result['connected'] = True
            result['account_info'] = self.account_info
            
            symbols = self.get_symbols()
            result['symbols_count'] = len(symbols)
            
            # Test data generation
            test_data = self.get_rates('EURUSD', 'M1', 10)
            if test_data is not None:
                result['test_data'] = {
                    'symbol': 'EURUSD',
                    'bars_count': len(test_data),
                    'latest_close': float(test_data['Close'].iloc[-1])
                }
            
            logger.info("Demo connection test completed successfully")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Demo connection test failed: {e}")
        
        return result

# Global demo connector instance
_demo_connector = None

def get_demo_connector():
    """Get singleton demo connector"""
    global _demo_connector
    
    if _demo_connector is None:
        _demo_connector = DemoConnector()
    
    return _demo_connector