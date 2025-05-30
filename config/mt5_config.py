"""
MetaTrader 5 Configuration Settings
"""
import os
from dataclasses import dataclass
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class MT5ConnectionConfig:
    """MT5 Connection Configuration"""
    # Account credentials (fill these with your actual values)
    login: int = 0  # Your MT5 account login
    password: str = ""  # Your MT5 account password  
    server: str = ""  # Your broker server name
    
    # Optional path to MT5 terminal
    path: str = ""  # Usually auto-detected
    
    # Connection settings
    timeout: int = 60000  # Connection timeout in milliseconds
    reconnect_attempts: int = 5
    reconnect_delay: int = 5  # seconds between reconnect attempts

@dataclass  
class TradingSymbolsConfig:
    """Trading symbols configuration"""
    # Major forex pairs
    major_pairs: List[str] = None
    
    # Minor and exotic pairs
    minor_pairs: List[str] = None
    
    # Default trading symbols
    default_symbols: List[str] = None
    
    def __post_init__(self):
        if self.major_pairs is None:
            self.major_pairs = [
                "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
                "USDCAD", "AUDUSD", "NZDUSD"
            ]
        
        if self.minor_pairs is None:
            self.minor_pairs = [
                "EURGBP", "EURJPY", "EURCHF", "EURCAD",
                "EURAUD", "EURNZD", "GBPJPY", "GBPCHF",
                "GBPCAD", "GBPAUD", "GBPNZD"
            ]
        
        if self.default_symbols is None:
            self.default_symbols = ["EURUSD", "GBPUSD", "USDJPY"]

@dataclass
class DataConfig:
    """Market data configuration"""
    # Timeframes for analysis (in order of importance)
    analysis_timeframes: List[str] = None
    
    # Historical data settings
    default_history_days: int = 365  # 1 year of history
    max_bars_per_request: int = 5000
    
    # Real-time data settings
    tick_data_enabled: bool = True
    price_update_interval: int = 1  # seconds
    
    # Data validation settings
    max_spread_threshold: float = 5.0  # Maximum spread in pips
    min_volume_threshold: int = 1
    
    def __post_init__(self):
        if self.analysis_timeframes is None:
            self.analysis_timeframes = ["D1", "H4", "H1", "M15", "M5", "M1"]

@dataclass
class TradingHoursConfig:
    """Trading hours and session configuration"""
    # Trading sessions (UTC time)
    sessions: Dict[str, Dict[str, int]] = None
    
    # Days to avoid trading  
    avoid_trading_days: List[str] = None
    
    # News events to avoid
    high_impact_news_buffer: int = 30  # minutes before/after news
    
    def __post_init__(self):
        if self.sessions is None:
            self.sessions = {
                "asian": {"start": 0, "end": 9},      # 00:00 - 09:00 UTC
                "london": {"start": 8, "end": 17},    # 08:00 - 17:00 UTC  
                "new_york": {"start": 13, "end": 22}, # 13:00 - 22:00 UTC
                "overlap": {"start": 13, "end": 17}   # London-NY overlap
            }
        
        if self.avoid_trading_days is None:
            self.avoid_trading_days = ["Saturday", "Sunday"]

@dataclass
class BackupConfig:
    """Data backup and recovery configuration"""
    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    max_backup_files: int = 10
    
    # Backup locations
    local_backup_path: str = "data_storage/backups"
    cloud_backup_enabled: bool = False
    cloud_backup_path: str = ""

# Environment-based configuration loading
def load_mt5_config():
    """Load MT5 configuration from environment variables or defaults"""
    
    # Try to load from environment variables (more secure)
    config = MT5ConnectionConfig(
        login=int(os.getenv("MT5_LOGIN", "0")),
        password=os.getenv("MT5_PASSWORD", ""),
        server=os.getenv("MT5_SERVER", ""),
        path=os.getenv("MT5_PATH", ""),
    )
    
    # Validate configuration
    if config.login == 0 or not config.password or not config.server:
        print("WARNING: MT5 credentials not configured!")
        print("Please set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER environment variables")
        print("Or modify the configuration directly in config/mt5_config.py")
    
    return config

# Create configuration instances
MT5_CONNECTION = load_mt5_config()
TRADING_SYMBOLS = TradingSymbolsConfig()
DATA_CONFIG = DataConfig()
TRADING_HOURS = TradingHoursConfig()
BACKUP_CONFIG = BackupConfig()

# Utility functions
def get_trading_symbols(category: str = "default") -> List[str]:
    """
    Get trading symbols by category
    
    Args:
        category: Symbol category (default, major, minor, all)
        
    Returns:
        List of trading symbols
    """
    if category == "major":
        return TRADING_SYMBOLS.major_pairs
    elif category == "minor":
        return TRADING_SYMBOLS.minor_pairs
    elif category == "all":
        return TRADING_SYMBOLS.major_pairs + TRADING_SYMBOLS.minor_pairs
    else:
        return TRADING_SYMBOLS.default_symbols

def is_trading_hours(current_hour: int, session: str = "london") -> bool:
    """
    Check if current hour is within trading session
    
    Args:
        current_hour: Current hour in UTC
        session: Trading session name
        
    Returns:
        True if within trading hours
    """
    if session not in TRADING_HOURS.sessions:
        return False
    
    session_config = TRADING_HOURS.sessions[session]
    start_hour = session_config["start"]
    end_hour = session_config["end"]
    
    if start_hour <= end_hour:
        return start_hour <= current_hour < end_hour
    else:  # Handles overnight sessions
        return current_hour >= start_hour or current_hour < end_hour

def get_timeframe_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes
    
    Args:
        timeframe: Timeframe string (M1, M5, H1, etc.)
        
    Returns:
        Number of minutes
    """
    timeframe_map = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440,
        "W1": 10080, "MN1": 43200
    }
    
    return timeframe_map.get(timeframe, 1)

# Export main configurations
__all__ = [
    "MT5_CONNECTION",
    "TRADING_SYMBOLS", 
    "DATA_CONFIG",
    "TRADING_HOURS",
    "BACKUP_CONFIG",
    "get_trading_symbols",
    "is_trading_hours", 
    "get_timeframe_minutes"
]