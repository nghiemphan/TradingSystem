"""
File Storage Manager for Large Datasets
Handles Parquet files, CSV exports, and data archival
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import gzip
import shutil
import os
from dataclasses import dataclass

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """File storage configuration"""
    # Base directories
    historical_dir: Path = DATA_DIR / "historical"
    features_dir: Path = DATA_DIR / "features" 
    models_dir: Path = DATA_DIR / "models"
    backups_dir: Path = DATA_DIR / "backups"
    exports_dir: Path = DATA_DIR / "exports"
    
    # File formats
    market_data_format: str = "parquet"  # parquet, csv, hdf5
    feature_format: str = "parquet"
    compression: str = "snappy"  # snappy, gzip, brotli
    
    # Archival settings
    archive_after_days: int = 90
    compress_archives: bool = True
    max_file_size_mb: int = 100

class FileStorageManager:
    """
    File-based storage manager for large datasets
    """
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.historical_dir,
            self.config.features_dir,
            self.config.models_dir,
            self.config.backups_dir,
            self.config.exports_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("File storage directories initialized")
    
    def _get_file_path(self, data_type: str, symbol: str, timeframe: str = None, 
                      date: datetime = None) -> Path:
        """Generate standardized file path"""
        
        if data_type == "market_data":
            base_dir = self.config.historical_dir
            if date:
                # Organize by year/month for historical data
                subdir = base_dir / str(date.year) / f"{date.month:02d}"
                subdir.mkdir(parents=True, exist_ok=True)
                filename = f"{symbol}_{timeframe}_{date.strftime('%Y%m%d')}.{self.config.market_data_format}"
            else:
                subdir = base_dir / "current"
                subdir.mkdir(parents=True, exist_ok=True)
                filename = f"{symbol}_{timeframe}.{self.config.market_data_format}"
                
        elif data_type == "features":
            base_dir = self.config.features_dir
            subdir = base_dir / symbol
            subdir.mkdir(parents=True, exist_ok=True)
            filename = f"{symbol}_{timeframe}_features.{self.config.feature_format}"
            
        elif data_type == "model":
            base_dir = self.config.models_dir
            subdir = base_dir
            filename = f"{symbol}.pkl"  # Model files are typically pickle
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return subdir / filename
    
    # Market Data Storage
    def save_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame,
                        date: datetime = None, append: bool = False) -> str:
        """
        Save market data to file
        
        Args:
            symbol: Trading symbol
            timeframe: Time frame
            data: Market data DataFrame
            date: Specific date for historical data
            append: Whether to append to existing file
            
        Returns:
            File path where data was saved
        """
        if data.empty:
            logger.warning(f"No data to save for {symbol} {timeframe}")
            return ""
        
        file_path = self._get_file_path("market_data", symbol, timeframe, date)
        
        try:
            if self.config.market_data_format == "parquet":
                if append and file_path.exists():
                    # Read existing data and combine
                    existing_data = pd.read_parquet(file_path)
                    combined_data = pd.concat([existing_data, data]).drop_duplicates()
                    combined_data.sort_index(inplace=True)
                    combined_data.to_parquet(
                        file_path, 
                        compression=self.config.compression,
                        index=True
                    )
                else:
                    data.to_parquet(
                        file_path,
                        compression=self.config.compression,
                        index=True
                    )
                    
            elif self.config.market_data_format == "csv":
                mode = 'a' if append and file_path.exists() else 'w'
                header = not (append and file_path.exists())
                data.to_csv(file_path, mode=mode, header=header)
                
            else:
                raise ValueError(f"Unsupported format: {self.config.market_data_format}")
            
            logger.info(f"Saved {len(data)} bars to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving market data to {file_path}: {e}")
            return ""
    
    def load_market_data(self, symbol: str, timeframe: str, 
                        start_date: datetime = None, end_date: datetime = None,
                        date: datetime = None) -> pd.DataFrame:
        """
        Load market data from file
        
        Args:
            symbol: Trading symbol
            timeframe: Time frame
            start_date: Filter start date
            end_date: Filter end date
            date: Specific date for historical data
            
        Returns:
            Market data DataFrame
        """
        file_path = self._get_file_path("market_data", symbol, timeframe, date)
        
        if not file_path.exists():
            logger.warning(f"Market data file not found: {file_path}")
            return pd.DataFrame()
        
        try:
            if self.config.market_data_format == "parquet":
                data = pd.read_parquet(file_path)
            elif self.config.market_data_format == "csv":
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            else:
                raise ValueError(f"Unsupported format: {self.config.market_data_format}")
            
            # Apply date filters
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            logger.info(f"Loaded {len(data)} bars from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading market data from {file_path}: {e}")
            return pd.DataFrame()
    
    def get_available_data_range(self, symbol: str, timeframe: str) -> Tuple[datetime, datetime]:
        """Get date range of available data for symbol/timeframe"""
        pattern = f"{symbol}_{timeframe}_*.{self.config.market_data_format}"
        files = list(self.config.historical_dir.rglob(pattern))
        
        if not files:
            # Check current data file
            current_file = self.config.historical_dir / "current" / f"{symbol}_{timeframe}.{self.config.market_data_format}"
            if current_file.exists():
                try:
                    data = pd.read_parquet(current_file) if self.config.market_data_format == "parquet" else pd.read_csv(current_file, index_col=0, parse_dates=True)
                    return data.index.min(), data.index.max()
                except:
                    pass
            return None, None
        
        # Find min/max dates from historical files
        dates = []
        for file_path in files:
            try:
                # Extract date from filename
                date_str = file_path.stem.split('_')[-1]
                dates.append(datetime.strptime(date_str, '%Y%m%d'))
            except:
                continue
        
        if dates:
            return min(dates), max(dates)
        return None, None
    
    # Feature Storage
    def save_features(self, symbol: str, timeframe: str, features: Dict,
                     timestamp: datetime = None) -> str:
        """Save calculated features to file"""
        file_path = self._get_file_path("features", symbol, timeframe)
        
        try:
            # Convert features dict to DataFrame if needed
            if isinstance(features, dict):
                if timestamp:
                    features_df = pd.DataFrame([features], index=[timestamp])
                else:
                    features_df = pd.DataFrame([features])
            else:
                features_df = features
            
            # Append to existing features if file exists
            if file_path.exists():
                existing_features = pd.read_parquet(file_path)
                combined_features = pd.concat([existing_features, features_df])
                combined_features = combined_features[~combined_features.index.duplicated(keep='last')]
                combined_features.sort_index(inplace=True)
                combined_features.to_parquet(file_path, compression=self.config.compression)
            else:
                features_df.to_parquet(file_path, compression=self.config.compression)
            
            logger.info(f"Saved features to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving features to {file_path}: {e}")
            return ""
    
    def load_features(self, symbol: str, timeframe: str,
                     start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Load calculated features from file"""
        file_path = self._get_file_path("features", symbol, timeframe)
        
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            features = pd.read_parquet(file_path)
            
            # Apply date filters
            if start_date:
                features = features[features.index >= start_date]
            if end_date:
                features = features[features.index <= end_date]
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading features from {file_path}: {e}")
            return pd.DataFrame()
    
    # Data Export
    def export_data(self, data: pd.DataFrame, filename: str, 
                   format: str = "csv") -> str:
        """Export data to specified format"""
        export_path = self.config.exports_dir / f"{filename}.{format}"
        
        try:
            if format == "csv":
                data.to_csv(export_path)
            elif format == "parquet":
                data.to_parquet(export_path, compression=self.config.compression)
            elif format == "xlsx":
                data.to_excel(export_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported data to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return ""
    
    # Backup and Archival
    def create_backup(self, backup_name: str = None) -> str:
        """Create backup of all data"""
        backup_name = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.config.backups_dir / backup_name
        
        try:
            # Create backup directory
            backup_path.mkdir(exist_ok=True)
            
            # Copy historical data
            historical_backup = backup_path / "historical"
            if self.config.historical_dir.exists():
                shutil.copytree(self.config.historical_dir, historical_backup, dirs_exist_ok=True)
            
            # Copy features
            features_backup = backup_path / "features"
            if self.config.features_dir.exists():
                shutil.copytree(self.config.features_dir, features_backup, dirs_exist_ok=True)
            
            # Copy models
            models_backup = backup_path / "models"
            if self.config.models_dir.exists():
                shutil.copytree(self.config.models_dir, models_backup, dirs_exist_ok=True)
            
            # Compress if enabled
            if self.config.compress_archives:
                archive_path = f"{backup_path}.tar.gz"
                shutil.make_archive(str(backup_path), 'gztar', str(backup_path))
                shutil.rmtree(backup_path)  # Remove uncompressed version
                backup_path = Path(archive_path)
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""
    
    def archive_old_data(self, days_to_keep: int = None):
        """Archive old data files"""
        days_to_keep = days_to_keep or self.config.archive_after_days
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        archive_dir = self.config.historical_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        archived_count = 0
        
        # Archive old historical data files
        for file_path in self.config.historical_dir.rglob("*.parquet"):
            if file_path.parent.name == "archive":
                continue
                
            # Check file modification time
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                try:
                    # Move to archive
                    relative_path = file_path.relative_to(self.config.historical_dir)
                    archive_file_path = archive_dir / relative_path
                    archive_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if self.config.compress_archives:
                        # Compress before archiving
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(f"{archive_file_path}.gz", 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        file_path.unlink()  # Remove original
                    else:
                        shutil.move(str(file_path), str(archive_file_path))
                    
                    archived_count += 1
                    
                except Exception as e:
                    logger.error(f"Error archiving {file_path}: {e}")
        
        logger.info(f"Archived {archived_count} old data files")
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        stats = {}
        
        directories = {
            'historical': self.config.historical_dir,
            'features': self.config.features_dir,
            'models': self.config.models_dir,
            'backups': self.config.backups_dir,
            'exports': self.config.exports_dir
        }
        
        for name, directory in directories.items():
            if directory.exists():
                # Count files and calculate size
                files = list(directory.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                stats[name] = {
                    'file_count': file_count,
                    'size_mb': round(total_size / 1024 / 1024, 2)
                }
            else:
                stats[name] = {'file_count': 0, 'size_mb': 0}
        
        return stats
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_patterns = ["*.tmp", "*.temp", "*~"]
        cleaned_count = 0
        
        for directory in [self.config.historical_dir, self.config.features_dir]:
            for pattern in temp_patterns:
                for temp_file in directory.rglob(pattern):
                    try:
                        temp_file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"Error removing temp file {temp_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")

# Singleton instance
_file_storage_manager = None

def get_file_storage_manager() -> FileStorageManager:
    """Get singleton file storage manager instance"""
    global _file_storage_manager
    
    if _file_storage_manager is None:
        _file_storage_manager = FileStorageManager()
    
    return _file_storage_manager

# Export main functions
__all__ = [
    'FileStorageManager',
    'StorageConfig',
    'get_file_storage_manager'
]