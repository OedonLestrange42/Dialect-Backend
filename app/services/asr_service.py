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

        # 初始化各个模型pipeline
        self._init_vad_pipeline(model_revision)
        self._init_asr_pipeline(model_revision)
        self._init_punc_pipeline(model_revision)
        self._init_spk_pipeline(model_revision)
        
        print("ASR Service: All models loaded successfully.")
    
    def _init_vad_pipeline(self, model_revision: str):
        """初始化VAD（语音活动检测）模型"""
        try:
            self.vad_pipeline = pipeline(
                task=Tasks.voice_activity_detection,
                model=settings.VAD_MODEL_ID,
                model_revision=model_revision,
                cache_dir=settings.MODEL_CACHE_DIR,
                device=self.device
            )
            print(f"VAD Model '{settings.VAD_MODEL_ID}' loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load VAD model: {e}. VAD will be disabled.")
            self.vad_pipeline = None
    
    def _init_asr_pipeline(self, model_revision: str):
        """初始化ASR（自动语音识别）模型"""
        self.asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=settings.FUNASR_MODEL_ID,
            model_revision=model_revision,
            cache_dir=settings.MODEL_CACHE_DIR,
            device=self.device
        )
        print(f"ASR Model '{settings.FUNASR_MODEL_ID}' loaded successfully.")
    
    def _init_punc_pipeline(self, model_revision: str):
        """初始化PUNC（标点符号）模型"""
        try:
            self.punc_pipeline = pipeline(
                task=Tasks.punctuation,
                model=settings.PUNC_MODEL_ID,
                model_revision=model_revision,
                cache_dir=settings.MODEL_CACHE_DIR,
                device=self.device
            )
            print(f"PUNC Model '{settings.PUNC_MODEL_ID}' loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load PUNC model: {e}. Punctuation will be disabled.")
            self.punc_pipeline = None
    
    def _init_spk_pipeline(self, model_revision: str):
        """初始化SPK（说话人识别）模型"""
        try:
            self.spk_pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=settings.SPK_MODEL_ID,
                model_revision=model_revision,
                cache_dir=settings.MODEL_CACHE_DIR,
                device=self.device
            )
            print(f"SPK Model '{settings.SPK_MODEL_ID}' loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load SPK model: {e}. Speaker recognition will be disabled.")
            self.spk_pipeline = None

    def transcribe(self, audio_file_path: str, hotword: str = None, 
                  enable_vad: bool = True, enable_punc: bool = True, 
                  enable_spk: bool = False) -> Dict[str, Any]:
        """
        对给定的音频文件执行完整的语音识别pipeline。

        Args:
            audio_file_path (str): 音频文件的路径
            hotword (str, optional): 用于提高特定词汇识别准确率的热词
            enable_vad (bool): 是否启用语音活动检测
            enable_punc (bool): 是否启用标点符号添加
            enable_spk (bool): 是否启用说话人识别

        Returns:
            dict: 包含完整处理结果的字典
        """
        print(f"ASR Service: Processing audio file: {audio_file_path}")
        
        result = {
            "text": "",
            "vad_segments": None,
            "punctuated_text": "",
            "speaker_info": None,
            "processing_info": {
                "vad_enabled": enable_vad and self.vad_pipeline is not None,
                "punc_enabled": enable_punc and self.punc_pipeline is not None,
                "spk_enabled": enable_spk and self.spk_pipeline is not None
            }
        }
        
        try:
            # 1. VAD处理（可选）
            if enable_vad and self.vad_pipeline is not None:
                print("ASR Service: Running VAD...")
                vad_result = self._run_vad(audio_file_path)
                result["vad_segments"] = vad_result
            
            # 2. ASR语音识别（核心）
            print("ASR Service: Running ASR...")
            asr_result = self._run_asr(audio_file_path, hotword)
            if not asr_result:
                return result
            
            result["text"] = asr_result
            
            # 3. 标点符号处理（可选）
            if enable_punc and self.punc_pipeline is not None and asr_result:
                print("ASR Service: Adding punctuation...")
                punctuated_text = self._run_punctuation(asr_result)
                result["punctuated_text"] = punctuated_text
            else:
                result["punctuated_text"] = asr_result
            
            # 4. 说话人识别（可选）
            if enable_spk and self.spk_pipeline is not None:
                print("ASR Service: Running speaker recognition...")
                speaker_info = self._run_speaker_recognition(audio_file_path)
                result["speaker_info"] = speaker_info
            
            print("ASR Service: Complete pipeline processing finished.")
            return result

        except Exception as e:
            logger.error(f"Pipeline processing error: {str(e)}", exc_info=True)
            return None
    
    def _run_vad(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """运行VAD语音活动检测"""
        try:
            vad_result = self.vad_pipeline(input=audio_file_path)
            return vad_result
        except Exception as e:
            logger.error(f"VAD processing error: {str(e)}")
            return None
    
    def _run_asr(self, audio_file_path: str, hotword: str = None) -> Optional[str]:
        """运行ASR语音识别"""
        try:
            rec_result_list = self.asr_pipeline(
                input=audio_file_path,
                hotword=hotword
            )
            
            if not rec_result_list:
                return ""
            
            # 将列表中所有字典的 'text' 键的值连接起来
            full_text = " ".join(item.get("text", "") for item in rec_result_list)
            return full_text
            
        except Exception as e:
            logger.error(f"ASR processing error: {str(e)}")
            return None
    
    def _run_punctuation(self, text: str) -> Optional[str]:
        """运行标点符号添加"""
        try:
            punc_result = self.punc_pipeline(input=text)
            if isinstance(punc_result, dict) and "text" in punc_result:
                return punc_result["text"]
            elif isinstance(punc_result, str):
                return punc_result
            else:
                return text
        except Exception as e:
            logger.error(f"Punctuation processing error: {str(e)}")
            return text
    
    def _run_speaker_recognition(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """运行说话人识别"""
        try:
            spk_result = self.spk_pipeline(input=audio_file_path)
            return spk_result
        except Exception as e:
            logger.error(f"Speaker recognition error: {str(e)}")
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