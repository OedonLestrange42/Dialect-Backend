import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from app.core.config import settings
import logging
logger = logging.getLogger(__name__)


class ASRService:
    def __init__(self, model_id: str, model_revision: str = "master"):
        """
        初始化并加载 FunASR 模型到指定设备。
        """
        self.model_id = model_id

        # 确定设备 (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ASR Service: Initializing model on device: {self.device}")

        # 加载模型
        # model_revision 用于指定模型版本，对于某些模型可能需要
        self.pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=self.model_id,
            model_revision=model_revision,
            cache_dir=settings.MODEL_CACHE_DIR,
            device=self.device
        )
        print(f"ASR Service: Model '{self.model_id}' loaded successfully.")

    def transcribe(self, audio_file_path: str, hotword: str = None) -> dict:
        """
        对给定的音频文件执行语音识别。

        Args:
            audio_file_path (str): 音频文件的路径.
            hotword (str, optional): 用于提高特定词汇识别准确率的热词. Defaults to None.

        Returns:
            dict: FunASR 模型返回的原始识别结果.
        """
        print(f"ASR Service: Transcribing audio file: {audio_file_path}")
        try:
            rec_result = self.pipeline(
                input=audio_file_path,
                hotword=hotword
            )
            print("ASR Service: Transcription completed.")
            return rec_result
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            return None



# 在 main.py 中实例化
asr_service_instance: ASRService = None