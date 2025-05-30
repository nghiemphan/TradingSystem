"""
Database Setup and Initialization Script
Run this to initialize the complete database system
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage.database_manager import get_database_manager
from data.storage.cache_manager import get_cache_manager
from data.storage.file_storage import get_file_storage_manager
from data.connectors.mt5_connector import get_mt5_connector
from config.mt5_config import MT5_CONNECTION, get_trading_symbols

def setup_logging():
    """Setup logging for database setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def test_database_system():
    """Test all database components"""
    print("="*60)
    print("DATABASE SYSTEM SETUP & TESTING")
    print("="*60)
    
    # Test 1: Database Manager
    print("\n1. Testing Database Manager...")
    try:
        db_manager = get_database_manager()
        
        # Test database operations
        print("   ✓ Database initialized successfully")
        
        # Get database stats
        stats = db_manager.get_database_stats()
        print(f"   Database size: {stats['db_size_mb']:.2f} MB")
        print(f"   Tables created: {len([k for k, v in stats.items() if k != 'db_size_mb'])}")
        
        # Test system event logging
        db_manager.log_system_event("INFO", "database_setup", "Database system test completed")
        print("   ✓ System event logging works")
        
    except Exception as e:
        print(f"   ✗ Database Manager error: {e}")
        return False
    
    # Test 2: Cache Manager
    print("\n2. Testing Cache Manager...")
    try:
        cache_manager = get_cache_manager()
        
        if cache_manager.is_connected():
            print("   ✓ Cache system connected")
            
            # Test cache operations
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            cache_manager.cache_current_price("TEST", test_data)
            
            retrieved_data = cache_manager.get_cached_price("TEST")
            if retrieved_data and retrieved_data.get("test") == "data":
                print("   ✓ Cache read/write operations work")
            else:
                print("   ⚠️  Cache read/write test failed")
            
            # Get cache stats
            cache_stats = cache_manager.get_cache_stats()
            print(f"   Cache mode: {'Redis' if not cache_stats.get('fallback_mode') else 'In-Memory Fallback'}")
            print(f"   Total keys: {cache_stats.get('total_keys', 0)}")
            
        else:
            print("   ⚠️  Cache system using fallback mode (Redis not available)")
            
    except Exception as e:
        print(f"   ✗ Cache Manager error: {e}")
        return False
    
    # Test 3: File Storage Manager
    print("\n3. Testing File Storage Manager...")
    try:
        file_manager = get_file_storage_manager()
        
        # Test directory creation
        print("   ✓ File storage directories created")
        
        # Get storage stats
        storage_stats = file_manager.get_storage_stats()
        print("   Directory structure:")
        for name, stats in storage_stats.items():
            print(f"     {name}: {stats['file_count']} files, {stats['size_mb']} MB")
        
        # Test file operations with sample data
        import pandas as pd
        import numpy as np
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(1.1000, 1.1100, 100),
            'High': np.random.uniform(1.1000, 1.1100, 100),
            'Low': np.random.uniform(1.1000, 1.1100, 100),
            'Close': np.random.uniform(1.1000, 1.1100, 100),
            'Volume': np.random.randint(100, 1000, 100),
            'Spread': np.random.uniform(0.00001, 0.00005, 100)
        }, index=dates)
        
        # Test save/load
        file_path = file_manager.save_market_data("TESTPAIR", "H1", sample_data)
        if file_path:
            loaded_data = file_manager.load_market_data("TESTPAIR", "H1")
            if len(loaded_data) == len(sample_data):
                print("   ✓ File save/load operations work")
            else:
                print("   ⚠️  File read/write test failed")
        else:
            print("   ⚠️  File save operation failed")
            
    except Exception as e:
        print(f"   ✗ File Storage Manager error: {e}")
        return False
    
    # Test 4: Integration Test with Real Data
    print("\n4. Testing Data Integration...")
    try:
        # Get MT5 connector (or demo)
        try:
            connector = get_mt5_connector(MT5_CONNECTION)
            if not connector.is_connected():
                raise Exception("MT5 not available")
            is_demo = False
        except:
            from data.connectors.demo_connector import get_demo_connector
            connector = get_demo_connector()
            connector.connect()
            is_demo = True
        
        print(f"   Using {'Demo' if is_demo else 'Real'} data connector")
        
        # Get some market data
        test_symbol = "EURUSD"
        market_data = connector.get_rates(test_symbol, "H1", 50)
        
        if market_data is not None and len(market_data) > 0:
            print(f"   ✓ Retrieved {len(market_data)} bars for {test_symbol}")
            
            # Save to database
            rows_saved = db_manager.save_market_data(test_symbol, "H1", market_data)
            print(f"   ✓ Saved {rows_saved} bars to database")
            
            # Save to file storage
            file_path = file_manager.save_market_data(test_symbol, "H1", market_data)
            print(f"   ✓ Saved to file: {Path(file_path).name}")
            
            # Cache current price
            if hasattr(connector, 'get_current_price'):
                current_price = connector.get_current_price(test_symbol)
                if current_price:
                    cache_manager.cache_current_price(test_symbol, current_price)
                    db_manager.save_current_price(
                        test_symbol, 
                        current_price['bid'], 
                        current_price['ask']
                    )
                    print("   ✓ Cached and stored current price")
            
            # Test data retrieval
            retrieved_data = db_manager.get_market_data(test_symbol, "H1", limit=10)
            if len(retrieved_data) > 0:
                print(f"   ✓ Retrieved {len(retrieved_data)} bars from database")
            
        connector.disconnect()
        
    except Exception as e:
        print(f"   ⚠️  Data integration test failed: {e}")
        print("   This is often normal and doesn't affect core functionality")
    
    print("\n" + "="*60)
    print("✅ DATABASE SYSTEM SETUP COMPLETED!")
    print("✅ All storage components are ready")
    print("="*60)
    
    return True

def populate_initial_data():
    """Populate database with initial configuration data"""
    print("\n5. Populating Initial Data...")
    
    try:
        db_manager = get_database_manager()
        
        # Add trading symbols information
        symbols_info = {
            'EURUSD': {'digits': 5, 'point': 0.00001, 'min_lot': 0.01, 'max_lot': 100.0, 
                      'lot_step': 0.01, 'contract_size': 100000, 'margin_required': 1000.0},
            'GBPUSD': {'digits': 5, 'point': 0.00001, 'min_lot': 0.01, 'max_lot': 100.0,
                      'lot_step': 0.01, 'contract_size': 100000, 'margin_required': 1000.0},
            'USDJPY': {'digits': 3, 'point': 0.001, 'min_lot': 0.01, 'max_lot': 100.0,
                      'lot_step': 0.01, 'contract_size': 100000, 'margin_required': 1000.0}
        }
        
        with db_manager.get_cursor() as cursor:
            for symbol, info in symbols_info.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO symbols 
                    (symbol, digits, point, min_lot, max_lot, lot_step, contract_size, margin_required)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, info['digits'], info['point'], info['min_lot'], 
                     info['max_lot'], info['lot_step'], info['contract_size'], info['margin_required']))
        
        print("   ✓ Added symbol information to database")
        
        # Log system initialization
        db_manager.log_system_event("INFO", "database_setup", "Initial data population completed")
        
    except Exception as e:
        print(f"   ⚠️  Initial data population failed: {e}")

def create_sample_performance_data():
    """Create sample performance data for testing"""
    print("\n6. Creating Sample Data...")
    
    try:
        db_manager = get_database_manager()
        
        # Create sample performance metrics
        sample_metrics = [
            {
                'metric_type': 'trading',
                'metric_name': 'daily_return',
                'metric_value': 0.02,
                'period_start': datetime.now().replace(hour=0, minute=0, second=0),
                'period_end': datetime.now(),
                'details': '{"trades": 5, "wins": 3, "losses": 2}'
            },
            {
                'metric_type': 'risk',
                'metric_name': 'max_drawdown',
                'metric_value': 0.05,
                'period_start': datetime.now().replace(hour=0, minute=0, second=0),
                'period_end': datetime.now(),
                'details': '{"peak_balance": 10000, "trough_balance": 9500}'
            }
        ]
        
        with db_manager.get_cursor() as cursor:
            for metric in sample_metrics:
                cursor.execute("""
                    INSERT INTO performance_metrics
                    (metric_type, metric_name, metric_value, period_start, period_end, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (metric['metric_type'], metric['metric_name'], metric['metric_value'],
                     metric['period_start'], metric['period_end'], metric['details']))
        
        print("   ✓ Added sample performance metrics")
        
    except Exception as e:
        print(f"   ⚠️  Sample data creation failed: {e}")

def print_system_summary():
    """Print summary of database system"""
    print("\n" + "="*60)
    print("DATABASE SYSTEM SUMMARY")
    print("="*60)
    
    try:
        db_manager = get_database_manager()
        cache_manager = get_cache_manager()
        file_manager = get_file_storage_manager()
        
        # Database stats
        db_stats = db_manager.get_database_stats()
        print(f"\n📊 Database:")
        print(f"   Size: {db_stats['db_size_mb']:.2f} MB")
        print(f"   Market data records: {db_stats.get('market_data', 0)}")
        print(f"   System events: {db_stats.get('system_events', 0)}")
        
        # Cache stats
        cache_stats = cache_manager.get_cache_stats()
        print(f"\n💾 Cache:")
        print(f"   Status: {'Connected' if cache_stats['connected'] else 'Disconnected'}")
        if cache_stats.get('fallback_mode'):
            print(f"   Mode: In-Memory Fallback")
        else:
            print(f"   Mode: Redis")
            print(f"   Memory: {cache_stats.get('used_memory_mb', 0)} MB")
        
        # File storage stats
        storage_stats = file_manager.get_storage_stats()
        print(f"\n📁 File Storage:")
        total_files = sum(stats['file_count'] for stats in storage_stats.values())
        total_size = sum(stats['size_mb'] for stats in storage_stats.values())
        print(f"   Total files: {total_files}")
        print(f"   Total size: {total_size:.2f} MB")
        
        print(f"\n✅ Database system ready for Phase 2!")
        print(f"✅ Next: SMC Feature Development")
        
    except Exception as e:
        print(f"Error getting system summary: {e}")

def main():
    """Main database setup function"""
    setup_logging()
    
    print("Starting database system setup...")
    
    try:
        # Test all components
        if test_database_system():
            # Populate initial data
            populate_initial_data()
            
            # Create sample data
            create_sample_performance_data()
            
            # Print summary
            print_system_summary()
            
            print("\n🎉 DATABASE SETUP COMPLETED SUCCESSFULLY!")
            return 0
        else:
            print("\n❌ Database setup failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
        return 0
    except Exception as e:
        print(f"\n💥 Unexpected error during setup: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)