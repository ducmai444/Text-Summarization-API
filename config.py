import os

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': 0,
    'password': os.getenv('REDIS_PASSWORD', 12345),
    'decode_responses': False,  # Giữ dữ liệu dạng bytes cho hiệu suất
    'socket_timeout': 5,        # Timeout kết nối
}

# Thời gian cache cho từng loại dữ liệu (giây)
CACHE_TTL = {
    'translations': 3600,      # 1 giờ
    'summarizations': 7200,    # 2 giờ 
    'language_detection': 86400,  # 1 ngày
    'supported_languages': 604800,  # 1 tuần
}