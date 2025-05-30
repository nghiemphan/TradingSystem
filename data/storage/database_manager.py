"""
Database Management System for Trading Data
Handles SQLite database operations, table creation, and data management
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
from contextlib import contextmanager
import threading

from config.settings import DB_CONFIG, DATA_DIR

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Centralized database management for trading system
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_CONFIG.sqlite_path
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Initialize database
        self._initialize_database()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys=ON")
            
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database operations"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
    
    def _initialize_database(self):
        """Create all necessary tables"""
        logger.info("Initializing database...")
        
        with self.get_cursor() as cursor:
            # Market data tables
            self._create_market_data_tables(cursor)
            
            # Trading tables
            self._create_trading_tables(cursor)
            
            # AI model tables
            self._create_model_tables(cursor)
            
            # System tables
            self._create_system_tables(cursor)
            
        logger.info("Database initialized successfully")
    
    def _create_market_data_tables(self, cursor):
        """Create market data related tables"""
        
        # OHLCV data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER DEFAULT 0,
                spread REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)
        
        # Indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_tf_time 
            ON market_data(symbol, timeframe, timestamp DESC)
        """)
        
        # Current prices table (latest tick data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_prices (
                symbol VARCHAR(20) PRIMARY KEY,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                spread REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Symbol information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol VARCHAR(20) PRIMARY KEY,
                digits INTEGER NOT NULL,
                point REAL NOT NULL,
                min_lot REAL NOT NULL,
                max_lot REAL NOT NULL,
                lot_step REAL NOT NULL,
                contract_size REAL NOT NULL,
                margin_required REAL NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_trading_tables(self, cursor):
        """Create trading related tables"""
        
        # Trading positions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                position_type VARCHAR(10) NOT NULL, -- 'BUY' or 'SELL'
                lot_size REAL NOT NULL,
                open_price REAL NOT NULL,
                close_price REAL,
                stop_loss REAL,
                take_profit REAL,
                open_time DATETIME NOT NULL,
                close_time DATETIME,
                profit REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                swap REAL DEFAULT 0,
                status VARCHAR(20) DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'CANCELLED'
                entry_reason TEXT,
                exit_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_positions_symbol_status 
            ON positions(symbol, status, open_time DESC)
        """)
        
        # Trading signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'CLOSE'
                confidence REAL NOT NULL, -- 0.0 to 1.0
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_reward_ratio REAL,
                signal_data TEXT, -- JSON with signal details
                timestamp DATETIME NOT NULL,
                executed BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk management logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20),
                position_id INTEGER,
                risk_amount REAL,
                portfolio_risk REAL,
                drawdown REAL,
                message TEXT,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (position_id) REFERENCES positions (id)
            )
        """)
    
    def _create_model_tables(self, cursor):
        """Create AI model related tables"""
        
        # Model training logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type VARCHAR(50) NOT NULL,
                model_version VARCHAR(20) NOT NULL,
                training_data_start DATETIME NOT NULL,
                training_data_end DATETIME NOT NULL,
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                validation_loss REAL,
                training_time_seconds INTEGER,
                model_path TEXT,
                hyperparameters TEXT, -- JSON
                performance_metrics TEXT, -- JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                prediction_type VARCHAR(20) NOT NULL, -- 'price', 'direction', 'action'
                predicted_value REAL,
                confidence REAL,
                actual_value REAL,
                prediction_timestamp DATETIME NOT NULL,
                validation_timestamp DATETIME,
                accuracy REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES model_training (id)
            )
        """)
        
        # Feature importance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                importance_score REAL NOT NULL,
                feature_type VARCHAR(50), -- 'SMC', 'Technical', 'Volume', etc.
                timeframe VARCHAR(10),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES model_training (id)
            )
        """)
    
    def _create_system_tables(self, cursor):
        """Create system monitoring tables"""
        
        # Performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type VARCHAR(50) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value REAL NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                details TEXT, -- JSON with additional data
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System events and errors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type VARCHAR(50) NOT NULL, -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                component VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                details TEXT, -- JSON with additional data
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_events_type_time 
            ON system_events(event_type, timestamp DESC)
        """)
        
        # Configuration history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_type VARCHAR(50) NOT NULL,
                config_key VARCHAR(100) NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_by VARCHAR(50),
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    # Market Data Methods
    def save_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> int:
        """
        Save market data to database
        
        Args:
            symbol: Trading symbol
            timeframe: Time frame
            data: DataFrame with OHLCV data
            
        Returns:
            Number of rows inserted
        """
        if data.empty:
            return 0
        
        # Prepare data for insertion
        records = []
        for timestamp, row in data.iterrows():
            records.append((
                symbol,
                timeframe,
                timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row.get('Volume', 0)),
                float(row.get('Spread', 0))
            ))
        
        with self.get_cursor() as cursor:
            cursor.executemany("""
                INSERT OR REPLACE INTO market_data 
                (symbol, timeframe, timestamp, open, high, low, close, volume, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
        
        logger.info(f"Saved {len(records)} bars for {symbol} {timeframe}")
        return len(records)
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       start_date: datetime = None, 
                       end_date: datetime = None,
                       limit: int = None) -> pd.DataFrame:
        """
        Retrieve market data from database
        
        Args:
            symbol: Trading symbol
            timeframe: Time frame
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
            
        Returns:
            DataFrame with market data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume, spread
            FROM market_data 
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        return df
    
    def save_current_price(self, symbol: str, bid: float, ask: float, 
                          timestamp: datetime = None):
        """Save current market price"""
        timestamp = timestamp or datetime.now()
        spread = ask - bid
        
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO current_prices
                (symbol, bid, ask, spread, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, bid, ask, spread, timestamp.strftime('%Y-%m-%d %H:%M:%S')))
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get latest price for symbol"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT bid, ask, spread, timestamp
                FROM current_prices
                WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'symbol': symbol,
                    'bid': row[0],
                    'ask': row[1], 
                    'spread': row[2],
                    'timestamp': datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S')
                }
        return None
    
    # Trading Methods
    def save_position(self, position_data: Dict) -> int:
        """Save trading position"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO positions 
                (symbol, position_type, lot_size, open_price, stop_loss, take_profit,
                 open_time, entry_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_data['symbol'],
                position_data['position_type'],
                position_data['lot_size'],
                position_data['open_price'],
                position_data.get('stop_loss'),
                position_data.get('take_profit'),
                position_data['open_time'].strftime('%Y-%m-%d %H:%M:%S'),
                position_data.get('entry_reason')
            ))
            
            return cursor.lastrowid
    
    def update_position(self, position_id: int, update_data: Dict):
        """Update existing position"""
        set_clauses = []
        params = []
        
        for key, value in update_data.items():
            if key in ['close_price', 'close_time', 'profit', 'commission', 
                      'swap', 'status', 'exit_reason']:
                set_clauses.append(f"{key} = ?")
                if key == 'close_time' and isinstance(value, datetime):
                    params.append(value.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    params.append(value)
        
        if set_clauses:
            params.append(position_id)
            query = f"UPDATE positions SET {', '.join(set_clauses)} WHERE id = ?"
            
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
    
    def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """Get all open positions"""
        query = "SELECT * FROM positions WHERE status = 'OPEN'"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY open_time DESC"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    # System Methods
    def log_system_event(self, event_type: str, component: str, 
                        message: str, details: Dict = None):
        """Log system event"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO system_events
                (event_type, component, message, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event_type,
                component,
                message,
                json.dumps(details) if details else None,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        tables = [
            'market_data', 'current_prices', 'symbols',
            'positions', 'trading_signals', 'risk_logs',
            'model_training', 'model_predictions', 'feature_importance',
            'performance_metrics', 'system_events', 'config_history'
        ]
        
        with self.get_cursor() as cursor:
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[table] = count
        
        # Database file size
        stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage database size"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
        
        with self.get_cursor() as cursor:
            # Clean old system events
            cursor.execute("""
                DELETE FROM system_events 
                WHERE created_at < ? AND event_type NOT IN ('ERROR', 'CRITICAL')
            """, (cutoff_str,))
            
            # Clean old model predictions
            cursor.execute("""
                DELETE FROM model_predictions 
                WHERE created_at < ?
            """, (cutoff_str,))
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
        
        logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()

# Singleton instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get singleton database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
    
    return _db_manager

# Export main functions
__all__ = [
    'DatabaseManager',
    'get_database_manager'
]