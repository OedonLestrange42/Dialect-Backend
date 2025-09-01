import shutil
import tempfile
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from typing import Annotated
from fastapi import Request

from app.api import deps
from app.services.asr_service import ASRService
from app.services import formatters

import logging
logger = logging.getLogger(__name__)
router = APIRouter()


class AudioResponseFormat(str, Enum):
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"


@router.post(
    "/v1/audio/transcriptions",
    dependencies=[Depends(deps.verify_api_key)],
    summary="Transcribes audio into the input language.",
    tags=["Audio"]
)
async def create_transcription(
        file: Annotated[UploadFile, File(...)],
        model: Annotated[str, Form(...)],  
        response_format: Annotated[AudioResponseFormat, Form()] = AudioResponseFormat.JSON,
        prompt: Annotated[str, Form()] = None,  # 对应 FunASR 的 hotword
        asr_service: ASRService = Depends(deps.get_asr_service)
):
    """
    将音频文件转录为文本。

    - **file**: 要转录的音频文件 (mp3, mp4, mpeg, mpga, m4a, wav, or webm)。
    - **model**: 使用的模型ID。当前服务器实例只加载一个模型，此参数仅为兼容性保留。
    - **response_format**: 返回结果的格式。
    - **prompt**: 可选的提示词/热词，以提高特定词汇的识别准确率。
    """
    # 使用临时文件处理上传的音频
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_audio_file:
            shutil.copyfileobj(file.file, tmp_audio_file)
            tmp_audio_path = tmp_audio_file.name
    finally:
        file.file.close()

    try:
        # 执行语音识别
        result = asr_service.transcribe(
            tmp_audio_path, 
            hotword=prompt,
        )

        # 根据请求的格式返回结果
        if response_format == AudioResponseFormat.JSON:
            return JSONResponse(content=formatters.to_simple_json(result))
        elif response_format == AudioResponseFormat.VERBOSE_JSON:
            return JSONResponse(content=formatters.to_verbose_json(result))
        elif response_format == AudioResponseFormat.TEXT:
            return PlainTextResponse(content=formatters.to_text(result))
        elif response_format == AudioResponseFormat.SRT:
            return PlainTextResponse(content=formatters.to_srt(result), media_type="text/plain")
        elif response_format == AudioResponseFormat.VTT:
            return PlainTextResponse(content=formatters.to_vtt(result), media_type="text/plain")

    except Exception as e:
        # 处理可能的识别错误
        logger.error(f"An error occurred during transcription: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {str(e)}")
    finally:
        # 清理临时文件
        import os
        os.remove(tmp_audio_path)




@router.post("/v1/audio/chunk", tags=["Audio"])
async def upload_chunk(request: Request):
    form = await request.form()
    chunk = form["file"]
    file_md5 = form["fileMd5"]
    chunk_index = int(form["chunkIndex"])
    total_chunks = int(form["totalChunks"])
    filename = form["filename"]
    chunk_dir = f"/tmp/chunks/{file_md5}"
    import os
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index}")
    with open(chunk_path, "wb") as f:
        f.write(await chunk.read())
    return {"status": "ok", "chunk_index": chunk_index}

@router.post("/v1/audio/merge", tags=["Audio"])
async def merge_chunks(request: Request, asr_service: ASRService = Depends(deps.get_asr_service)):
    data = await request.json()
    file_md5 = data["fileMd5"]
    chunk_dir = f"/tmp/chunks/{file_md5}"
    import os
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith("chunk_")], key=lambda x: int(x.split("_")[1]))
    merged_path = os.path.join(chunk_dir, f"{file_md5}.wav")
    with open(merged_path, "wb") as outfile:
        for chunk_file in chunk_files:
            with open(os.path.join(chunk_dir, chunk_file), "rb") as infile:
                outfile.write(infile.read())
    # 语音识别
    result = asr_service.transcribe(merged_path)
    return JSONResponse(content=formatters.to_simple_json(result))



