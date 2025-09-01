import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from app.core.config import settings
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ASRService:
    def __init__(self, model_revision: str = "master"):
        """
        初始化并加载完整的语音识别pipeline，包括VAD、ASR、PUNC、SPK模型。
        """
        # 确定设备 (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ASR Service: Initializing models on device: {self.device}")

        self.asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=settings.FUNASR_MODEL_ID,
            model_revision=model_revision,
            cache_dir=settings.MODEL_CACHE_DIR,
            vad_model=settings.VAD_MODEL_ID,
            punc_model=settings.PUNC_MODEL_ID,
            spk_model=settings.SPK_MODEL_ID,
            device=self.device
        )
        print("ASR Service: All models loaded successfully.")
    

    def transcribe(self, audio_file_path: str, hotword: str = None) -> Dict[str, Any]:
        """
        对给定的音频文件执行完整的语音识别pipeline。

        Args:
            audio_file_path (str): 音频文件的路径
            hotword (str, optional): 用于提高特定词汇识别准确率的热词

        Returns:
            dict: 包含完整处理结果的字典
        """
        print(f"ASR Service: Processing audio file: {audio_file_path}")
        
        result = None
        
        try:
            print("ASR Service: Running ASR...")
            asr_result = self.asr_pipeline(
                                            input=audio_file_path,
                                            hotword=hotword
                                        )
            if not asr_result:
                return result
            print("ASR Service: Complete pipeline processing finished.")
            print("Complete asr_result:", asr_result)
            return asr_result[0]

        except Exception as e:
            logger.error(f"Pipeline processing error: {str(e)}", exc_info=True)
            return None
    

# 在 main.py 中实例化
asr_service_instance: ASRService = None

# 便捷函数用于初始化服务
def initialize_asr_service(model_revision: str = "master") -> ASRService:
    """
    初始化ASR服务实例
    
    Args:
        model_revision (str): 模型版本，默认为"master"
    
    Returns:
        ASRService: 初始化后的ASR服务实例
    """
    global asr_service_instance
    if asr_service_instance is None:
        asr_service_instance = ASRService(model_revision=model_revision)
    return asr_service_instance

def get_asr_service() -> ASRService:
    """
    获取ASR服务实例
    
    Returns:
        ASRService: ASR服务实例
    
    Raises:
        RuntimeError: 如果服务未初始化
    """
    if asr_service_instance is None:
        raise RuntimeError("ASR service not initialized. Call initialize_asr_service() first.")
    return asr_service_instance