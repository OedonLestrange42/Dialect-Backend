import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # 简单的 API 密钥
    API_KEY: str = "your-secret-api-key"

    # FunASR 模型配置
    FUNASR_MODEL_ID: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    MODEL_CACHE_DIR: str = "./model_cache"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()

# 确保模型缓存目录存在
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)