"""
Redis Cache Manager for High-Performance Data Access
Handles caching of real-time data, features, and predictions
"""
import redis
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

from config.settings import DB_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    # Default TTL (Time To Live) in seconds
    price_ttl: int = 60          # 1 minute for current prices
    feature_ttl: int = 300       # 5 minutes for calculated features
    prediction_ttl: int = 900    # 15 minutes for AI predictions
    market_data_ttl: int = 3600  # 1 hour for historical data
    
    # Key prefixes
    price_prefix: str = "price:"
    feature_prefix: str = "feature:"
    prediction_prefix: str = "prediction:"
    market_data_prefix: str = "market:"
    model_prefix: str = "model:"

class CacheManager:
    """
    Redis-based cache manager for trading system
    """
    
    def __init__(self, redis_host: str = None, redis_port: int = None, 
                 redis_db: int = None):
        self.host = redis_host or DB_CONFIG.redis_host
        self.port = redis_port or DB_CONFIG.redis_port
        self.db = redis_db or DB_CONFIG.redis_db
        
        self.config = CacheConfig()
        self.redis_client = None
        self.connected = False
        
        # Try to connect
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to Redis server"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
            
        except redis.ConnectionError:
            logger.warning(f"Could not connect to Redis at {self.host}:{self.port}")
            logger.warning("Cache functionality will be disabled")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if Redis is connected and available"""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            self.connected = False
            return False
    
    def _ensure_connection(self) -> bool:
        """Ensure Redis connection is active"""
        if not self.is_connected():
            return self._connect()
        return True
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        if isinstance(data, pd.DataFrame):
            # For DataFrames, use pickle for full fidelity
            return pickle.dumps(data)
        elif isinstance(data, (dict, list, tuple)):
            # For JSON-serializable data, use JSON
            return json.dumps(data, default=str).encode('utf-8')
        else:
            # For other objects, use pickle
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes, data_type: str = 'auto') -> Any:
        """Deserialize data from Redis"""
        if not data:
            return None
        
        try:
            if data_type == 'json':
                return json.loads(data.decode('utf-8'))
            elif data_type == 'pickle':
                return pickle.loads(data)
            else:  # auto-detect
                # Try JSON first (more readable)
                try:
                    return json.loads(data.decode('utf-8'))
                except:
                    # Fall back to pickle
                    return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None
    
    # Current Price Caching
    def cache_current_price(self, symbol: str, price_data: Dict, ttl: int = None):
        """Cache current market price"""
        if not self._ensure_connection():
            return False
        
        key = f"{self.config.price_prefix}{symbol}"
        ttl = ttl or self.config.price_ttl
        
        try:
            serialized_data = self._serialize_data(price_data)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Error caching price for {symbol}: {e}")
            return False
    
    def get_cached_price(self, symbol: str) -> Optional[Dict]:
        """Get cached current price"""
        if not self._ensure_connection():
            return None
        
        key = f"{self.config.price_prefix}{symbol}"
        
        try:
            data = self.redis_client.get(key)
            return self._deserialize_data(data)
        except Exception as e:
            logger.error(f"Error getting cached price for {symbol}: {e}")
            return None
    
    # Market Data Caching  
    def cache_market_data(self, symbol: str, timeframe: str, 
                         data: pd.DataFrame, ttl: int = None):
        """Cache historical market data"""
        if not self._ensure_connection() or data.empty:
            return False
        
        key = f"{self.config.market_data_prefix}{symbol}:{timeframe}"
        ttl = ttl or self.config.market_data_ttl
        
        try:
            serialized_data = self._serialize_data(data)
            self.redis_client.setex(key, ttl, serialized_data)
            
            # Also cache metadata
            metadata = {
                'rows': len(data),
                'start_time': str(data.index[0]),
                'end_time': str(data.index[-1]),
                'cached_at': datetime.now().isoformat()
            }
            meta_key = f"{key}:meta"
            self.redis_client.setex(meta_key, ttl, json.dumps(metadata))
            
            return True
        except Exception as e:
            logger.error(f"Error caching market data for {symbol} {timeframe}: {e}")
            return False
    
    def get_cached_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get cached market data"""
        if not self._ensure_connection():
            return None
        
        key = f"{self.config.market_data_prefix}{symbol}:{timeframe}"
        
        try:
            data = self.redis_client.get(key)
            return self._deserialize_data(data)
        except Exception as e:
            logger.error(f"Error getting cached market data for {symbol} {timeframe}: {e}")
            return None
    
    # Feature Caching
    def cache_features(self, symbol: str, timeframe: str, 
                      features: Dict, ttl: int = None):
        """Cache calculated features"""
        if not self._ensure_connection():
            return False
        
        key = f"{self.config.feature_prefix}{symbol}:{timeframe}"
        ttl = ttl or self.config.feature_ttl
        
        try:
            # Add timestamp to features
            features_with_timestamp = {
                **features,
                '_cached_at': datetime.now().isoformat(),
                '_symbol': symbol,
                '_timeframe': timeframe
            }
            
            serialized_data = self._serialize_data(features_with_timestamp)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Error caching features for {symbol} {timeframe}: {e}")
            return False
    
    def get_cached_features(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get cached features"""
        if not self._ensure_connection():
            return None
        
        key = f"{self.config.feature_prefix}{symbol}:{timeframe}"
        
        try:
            data = self.redis_client.get(key)
            features = self._deserialize_data(data)
            
            if features and '_cached_at' in features:
                # Check if features are still fresh enough
                cached_time = datetime.fromisoformat(features['_cached_at'])
                age_seconds = (datetime.now() - cached_time).total_seconds()
                
                if age_seconds > self.config.feature_ttl:
                    # Features are too old, remove them
                    self.redis_client.delete(key)
                    return None
            
            return features
        except Exception as e:
            logger.error(f"Error getting cached features for {symbol} {timeframe}: {e}")
            return None
    
    # AI Prediction Caching
    def cache_prediction(self, symbol: str, timeframe: str, 
                        prediction_data: Dict, ttl: int = None):
        """Cache AI model predictions"""
        if not self._ensure_connection():
            return False
        
        key = f"{self.config.prediction_prefix}{symbol}:{timeframe}"
        ttl = ttl or self.config.prediction_ttl
        
        try:
            prediction_with_metadata = {
                **prediction_data,
                '_predicted_at': datetime.now().isoformat(),
                '_symbol': symbol,
                '_timeframe': timeframe
            }
            
            serialized_data = self._serialize_data(prediction_with_metadata)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Error caching prediction for {symbol} {timeframe}: {e}")
            return False
    
    def get_cached_prediction(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get cached prediction"""
        if not self._ensure_connection():
            return None
        
        key = f"{self.config.prediction_prefix}{symbol}:{timeframe}"
        
        try:
            data = self.redis_client.get(key)
            return self._deserialize_data(data)
        except Exception as e:
            logger.error(f"Error getting cached prediction for {symbol} {timeframe}: {e}")
            return None
    
    # Model Caching
    def cache_model_state(self, model_name: str, model_data: Any, ttl: int = None):
        """Cache AI model state or weights"""
        if not self._ensure_connection():
            return False
        
        key = f"{self.config.model_prefix}{model_name}"
        ttl = ttl or 86400  # 24 hours default for models
        
        try:
            serialized_data = self._serialize_data(model_data)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Error caching model state for {model_name}: {e}")
            return False
    
    def get_cached_model_state(self, model_name: str) -> Optional[Any]:
        """Get cached model state"""
        if not self._ensure_connection():
            return None
        
        key = f"{self.config.model_prefix}{model_name}"
        
        try:
            data = self.redis_client.get(key)
            return self._deserialize_data(data)
        except Exception as e:
            logger.error(f"Error getting cached model state for {model_name}: {e}")
            return None
    
    # Batch Operations
    def cache_multiple_prices(self, price_data: Dict[str, Dict], ttl: int = None):
        """Cache multiple symbol prices in one operation"""
        if not self._ensure_connection():
            return False
        
        ttl = ttl or self.config.price_ttl
        pipe = self.redis_client.pipeline()
        
        try:
            for symbol, price_info in price_data.items():
                key = f"{self.config.price_prefix}{symbol}"
                serialized_data = self._serialize_data(price_info)
                pipe.setex(key, ttl, serialized_data)
            
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Error batch caching prices: {e}")
            return False
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        """Get multiple cached prices in one operation"""
        if not self._ensure_connection():
            return {symbol: None for symbol in symbols}
        
        keys = [f"{self.config.price_prefix}{symbol}" for symbol in symbols]
        
        try:
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            
            results = pipe.execute()
            
            return {
                symbol: self._deserialize_data(data) if data else None
                for symbol, data in zip(symbols, results)
            }
        except Exception as e:
            logger.error(f"Error getting multiple cached prices: {e}")
            return {symbol: None for symbol in symbols}
    
    # Cache Management
    def clear_cache(self, pattern: str = None):
        """Clear cache data matching pattern"""
        if not self._ensure_connection():
            return False
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cache entries matching '{pattern}'")
            else:
                self.redis_client.flushdb()
                logger.info("Cleared entire cache database")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self._ensure_connection():
            return {'connected': False}
        
        try:
            info = self.redis_client.info()
            
            # Get key counts by prefix
            key_counts = {}
            for prefix in [self.config.price_prefix, self.config.feature_prefix, 
                          self.config.prediction_prefix, self.config.market_data_prefix]:
                pattern = f"{prefix}*"
                keys = self.redis_client.keys(pattern)
                key_counts[prefix.rstrip(':')] = len(keys)
            
            return {
                'connected': True,
                'redis_version': info.get('redis_version'),
                'used_memory_mb': round(info.get('used_memory', 0) / 1024 / 1024, 2),
                'total_keys': info.get('db0', {}).get('keys', 0),
                'key_counts': key_counts,
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'connected': False, 'error': str(e)}
    
    def cleanup_expired_keys(self):
        """Clean up expired keys manually if needed"""
        if not self._ensure_connection():
            return
        
        try:
            # Redis handles TTL automatically, but we can check for orphaned metadata
            meta_pattern = "*:meta"
            meta_keys = self.redis_client.keys(meta_pattern)
            
            for meta_key in meta_keys:
                base_key = meta_key[:-5]  # Remove ':meta'
                if not self.redis_client.exists(base_key):
                    # Metadata exists but base key doesn't, clean it up
                    self.redis_client.delete(meta_key)
                    
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def set_cache_ttl(self, cache_type: str, ttl: int):
        """Update TTL for specific cache type"""
        if cache_type == 'price':
            self.config.price_ttl = ttl
        elif cache_type == 'feature':
            self.config.feature_ttl = ttl
        elif cache_type == 'prediction':
            self.config.prediction_ttl = ttl
        elif cache_type == 'market_data':
            self.config.market_data_ttl = ttl
        else:
            logger.warning(f"Unknown cache type: {cache_type}")
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                self.redis_client.close()
                self.connected = False
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

# Fallback Cache (In-Memory when Redis unavailable)
class FallbackCache:
    """
    Simple in-memory cache when Redis is not available
    """
    
    def __init__(self):
        self._cache = {}
        self._expiry = {}
        self.config = CacheConfig()
        logger.info("Using fallback in-memory cache (Redis not available)")
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self._expiry:
            return False
        return datetime.now() > self._expiry[key]
    
    def _clean_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [k for k, exp_time in self._expiry.items() if now > exp_time]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
    
    def cache_current_price(self, symbol: str, price_data: Dict, ttl: int = None):
        self._clean_expired()
        key = f"price:{symbol}"
        self._cache[key] = price_data
        self._expiry[key] = datetime.now() + timedelta(seconds=ttl or self.config.price_ttl)
        return True
    
    def get_cached_price(self, symbol: str) -> Optional[Dict]:
        key = f"price:{symbol}"
        if self._is_expired(key):
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
            return None
        return self._cache.get(key)
    
    def cache_features(self, symbol: str, timeframe: str, features: Dict, ttl: int = None):
        self._clean_expired()
        key = f"feature:{symbol}:{timeframe}"
        self._cache[key] = features
        self._expiry[key] = datetime.now() + timedelta(seconds=ttl or self.config.feature_ttl)
        return True
    
    def get_cached_features(self, symbol: str, timeframe: str) -> Optional[Dict]:
        key = f"feature:{symbol}:{timeframe}"
        if self._is_expired(key):
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
            return None
        return self._cache.get(key)
    
    def is_connected(self) -> bool:
        return True  # Always "connected" for fallback
    
    def get_cache_stats(self) -> Dict:
        self._clean_expired()
        return {
            'connected': True,
            'fallback_mode': True,
            'total_keys': len(self._cache),
            'memory_usage': 'N/A (in-memory)'
        }
    
    def clear_cache(self, pattern: str = None):
        if pattern:
            # Simple pattern matching for fallback
            keys_to_remove = [k for k in self._cache.keys() if pattern.replace('*', '') in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._expiry.pop(key, None)
        else:
            self._cache.clear()
            self._expiry.clear()
        return True

# Singleton instances
_cache_manager = None

def get_cache_manager() -> Union[CacheManager, FallbackCache]:
    """Get singleton cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        # Try Redis first
        redis_cache = CacheManager()
        if redis_cache.is_connected():
            _cache_manager = redis_cache
        else:
            # Fall back to in-memory cache
            _cache_manager = FallbackCache()
    
    return _cache_manager

# Export main functions
__all__ = [
    'CacheManager',
    'FallbackCache',
    'get_cache_manager'
]