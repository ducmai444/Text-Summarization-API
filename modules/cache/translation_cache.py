# modules/cache/translation_cache.py
import hashlib
import json
from config import CACHE_TTL
from redis_client import redis_client

def get_cache_key(text, target_lang):
    """Tạo cache key duy nhất dựa trên nội dung và ngôn ngữ đích"""
    # Sử dụng md5 để tạo key ngắn và không có ký tự đặc biệt
    key = f"translate:{target_lang}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
    return key

def get_cached_translation(text, target_lang):
    """Lấy bản dịch từ cache nếu có"""
    cache_key = get_cache_key(text, target_lang)
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    return None

def cache_translation(text, target_lang, result):
    """Lưu kết quả dịch vào cache"""
    cache_key = get_cache_key(text, target_lang)
    redis_client.setex(
        cache_key, 
        CACHE_TTL['translations'],  
        json.dumps(result, ensure_ascii=False)
    )