import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API服务器配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # 简单的 API 密钥
    API_KEY: str = "your-secret-api-key"

    # FunASR 模型配置
    FUNASR_MODEL_ID: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    
    # VAD 模型配置
    VAD_MODEL_ID: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    
    # 标点符号模型配置
    PUNC_MODEL_ID: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    
    # 说话人识别模型配置
    SPK_MODEL_ID: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    
    # 模型下载缓存目录
    MODEL_CACHE_DIR: str = "./model_cache"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()

# 确保模型缓存目录存在
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)