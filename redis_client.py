import redis
from config import REDIS_CONFIG

# Connection pooling để tối ưu kết nối
pool = redis.ConnectionPool(
    host=REDIS_CONFIG['host'],
    port=REDIS_CONFIG['port'],
    db=REDIS_CONFIG['db'],
    password=REDIS_CONFIG['password'],
    max_connections=100,  # Tối đa 100 kết nối đồng thời
    decode_responses=REDIS_CONFIG['decode_responses']
)

redis_client = redis.Redis(connection_pool=pool)
