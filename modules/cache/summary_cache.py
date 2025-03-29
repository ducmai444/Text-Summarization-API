# modules/cache/summary_cache.py
import hashlib
import json
from config import CACHE_TTL
from redis_client import redis_client

def get_summary_cache_key(text, method='extractive', params=None):
    """Tạo cache key cho kết quả tóm tắt"""
    # Kết hợp nội dung và tham số để tạo key
    params_str = json.dumps(params or {}, sort_keys=True)
    key = f"summary:{method}:{params_str}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
    return key

def get_cached_summary(text, method='extractive', params=None):
    """Lấy bản tóm tắt từ cache nếu có"""
    cache_key = get_summary_cache_key(text, method, params)
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    return None

def cache_summary(text, result, method='extractive', params=None):
    """Lưu kết quả tóm tắt vào cache"""
    cache_key = get_summary_cache_key(text, method, params)
    redis_client.setex(
        cache_key, 
        CACHE_TTL['summarizations'],
        json.dumps(result, ensure_ascii=False)
    )