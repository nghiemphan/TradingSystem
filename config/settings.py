"""
Main Configuration File for AI Trading System
"""
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_storage"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "data_storage" / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    sqlite_path: str = str(DATA_DIR / "trading_system.db")
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    backup_interval_hours: int = 6

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Log files
    trading_log: str = str(LOGS_DIR / "trading.log")
    system_log: str = str(LOGS_DIR / "system.log")
    error_log: str = str(LOGS_DIR / "errors.log")
    performance_log: str = str(LOGS_DIR / "performance.log")

@dataclass
class SystemConfig:
    """System configuration"""
    environment: str = "development"  # development, testing, production
    debug_mode: bool = True
    max_workers: int = 4
    
    # Performance settings
    data_buffer_size: int = 1000
    feature_cache_size: int = 500
    model_inference_timeout: int = 30  # seconds
    
    # Monitoring
    health_check_interval: int = 60  # seconds
    performance_report_interval: int = 300  # 5 minutes

@dataclass
class TradingConfig:
    """Basic trading configuration"""
    demo_mode: bool = True  # Start with demo trading
    max_concurrent_positions: int = 3
    
    # Timeframes for analysis
    analysis_timeframes: List[str] = None
    execution_timeframe: str = "M5"
    
    def __post_init__(self):
        if self.analysis_timeframes is None:
            self.analysis_timeframes = ["H4", "H1", "M15", "M5"]

# Global configuration instances
DB_CONFIG = DatabaseConfig()
LOG_CONFIG = LoggingConfig()
SYSTEM_CONFIG = SystemConfig()
TRADING_CONFIG = TradingConfig()

# Environment-specific overrides
def load_environment_config():
    """Load environment-specific configurations"""
    env = os.getenv("TRADING_ENV", "development")
    
    if env == "production":
        SYSTEM_CONFIG.debug_mode = False
        SYSTEM_CONFIG.environment = "production"
        LOG_CONFIG.log_level = "WARNING"
        TRADING_CONFIG.demo_mode = False
    
    elif env == "testing":
        SYSTEM_CONFIG.environment = "testing"
        TRADING_CONFIG.demo_mode = True
        LOG_CONFIG.log_level = "DEBUG"

# Load environment config on import
load_environment_config()

# Export main configs
__all__ = [
    "DB_CONFIG",
    "LOG_CONFIG", 
    "SYSTEM_CONFIG",
    "TRADING_CONFIG",
    "PROJECT_ROOT",
    "DATA_DIR",
    "LOGS_DIR",
    "MODELS_DIR"
]