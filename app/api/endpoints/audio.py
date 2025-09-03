import shutil
import tempfile
import os
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from typing import Annotated
from fastapi import Request

from app.api import deps
from app.services.asr_service import ASRService
from app.services import formatters

import logging
# 新增导入：用于从 URL 流式下载
import urllib.request
import urllib.parse
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




@router.post("/v1/audio/chunk", dependencies=[Depends(deps.verify_api_key)], tags=["Audio"])
async def upload_chunk(request: Request):
    """
    分块上传接口：
    - 当 Content-Type 为 application/offset+octet-stream 或存在 upload-* 头时，从请求头读取分块元信息，原样写入临时目录；
    - 否则回退到 multipart/form-data 读取 form 字段。
    临时目录使用 tempfile.gettempdir()，兼容不同操作系统。
    """
    content_type = request.headers.get("content-type", "")

    # 通用：确定基础临时目录
    def _chunk_dir(md5: str) -> str:
        return os.path.join(tempfile.gettempdir(), "chunks", md5)

    # 路径确保存在
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    # 方案一：原始字节流（tus-js-client 透传）
    upload_hdr_present = (
        request.headers.get("upload-file-md5") is not None and
        request.headers.get("upload-chunk-index") is not None and
        request.headers.get("upload-total-chunks") is not None
    )
    if content_type.startswith("application/offset+octet-stream") or upload_hdr_present:
        file_md5 = request.headers.get("upload-file-md5")
        chunk_index = request.headers.get("upload-chunk-index")
        total_chunks = request.headers.get("upload-total-chunks")
        filename = request.headers.get("upload-filename", "unknown")

        if not file_md5 or chunk_index is None or total_chunks is None:
            raise HTTPException(status_code=400, detail="Missing required upload headers")

        # 转为整数并校验
        try:
            chunk_index_int = int(chunk_index)
            total_chunks_int = int(total_chunks)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chunk index or total chunks")

        base_dir = _chunk_dir(file_md5)
        _ensure_dir(base_dir)
        # 为了合并时排序稳定，建议零填充序号
        chunk_path = os.path.join(base_dir, f"chunk_{chunk_index_int:06d}")

        body = await request.body()
        with open(chunk_path, "wb") as f:
            f.write(body)

        return {"status": "ok", "chunk_index": chunk_index_int, "filename": filename, "bytes": len(body)}

    # 方案二：multipart/form-data 回退兼容
    try:
        form = await request.form()
        chunk = form.get("file")
        file_md5 = form.get("fileMd5")
        chunk_index = form.get("chunkIndex")
        total_chunks = form.get("totalChunks")
        filename = form.get("filename", "unknown")
        if chunk is None or not file_md5 or chunk_index is None or total_chunks is None:
            raise HTTPException(status_code=400, detail="Invalid multipart form fields")

        try:
            chunk_index_int = int(chunk_index)
            total_chunks_int = int(total_chunks)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chunk index or total chunks")

        base_dir = _chunk_dir(file_md5)
        _ensure_dir(base_dir)
        chunk_path = os.path.join(base_dir, f"chunk_{chunk_index_int:06d}")
        with open(chunk_path, "wb") as f:
            f.write(await chunk.read())
        return {"status": "ok", "chunk_index": chunk_index_int, "filename": filename}
    except Exception as e:
        # 如果不是 multipart 或其他解析错误
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Failed to process chunk: {str(e)}")


@router.post("/v1/audio/merge", dependencies=[Depends(deps.verify_api_key)], tags=["Audio"])
async def merge_chunks(request: Request, asr_service: ASRService = Depends(deps.get_asr_service)):
    """
    合并分块并进行识别：
    期待 JSON: { "fileMd5": "...", "filename": "...", "cleanup": true }
    """
    data = await request.json()
    file_md5 = data.get("fileMd5")
    filename = data.get("filename", f"{file_md5}.wav")
    cleanup = bool(data.get("cleanup", True))
    if not file_md5:
        raise HTTPException(status_code=400, detail="fileMd5 is required")

    base_dir = os.path.join(tempfile.gettempdir(), "chunks", file_md5)
    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=400, detail="Chunk directory not found")

    # 收集并排序所有分块
    chunk_files = sorted(
        [f for f in os.listdir(base_dir) if f.startswith("chunk_")],
        key=lambda x: int(x.split("_")[1])
    )
    if not chunk_files:
        raise HTTPException(status_code=400, detail="No chunks found to merge")

    merged_path = os.path.join(base_dir, filename)
    with open(merged_path, "wb") as outfile:
        for chunk_file in chunk_files:
            with open(os.path.join(base_dir, chunk_file), "rb") as infile:
                outfile.write(infile.read())

    # 语音识别
    result = asr_service.transcribe(merged_path)

    # 可选清理
    if cleanup:
        try:
            # 删除分块文件和合并后的文件
            for f in chunk_files:
                os.remove(os.path.join(base_dir, f))
            os.remove(merged_path)
            # 删除目录
            os.rmdir(base_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed for {base_dir}: {e}")

    return JSONResponse(content=formatters.to_verbose_json(result))


# 新增：直接从 URL 拉取音频并识别
@router.post("/v1/audio/from_url", dependencies=[Depends(deps.verify_api_key)], tags=["Audio"])
async def transcribe_from_url(request: Request, asr_service: ASRService = Depends(deps.get_asr_service)):
    """
    从远程 URL（例如 MinIO 预签名链接）下载音频并进行识别，避免前端先下载。
    期待 JSON: {
      "url": "https://...",
      "filename": "optional_filename.wav",
      "response_format": "json|verbose_json|text|srt|vtt",
      "prompt": "可选热词",
      "headers": {"Authorization": "Bearer ..."}  # 可选，若需要鉴权
    }
    """
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    # 可选参数
    filename = data.get("filename")
    prompt = data.get("prompt")
    response_format_str = (data.get("response_format") or AudioResponseFormat.JSON.value).lower()
    headers = data.get("headers") or {}

    # 推断文件名
    if not filename:
        parsed = urllib.parse.urlparse(url)
        base = os.path.basename(parsed.path) or "audio_from_url"
        filename = base

    # 以临时文件保存。
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=f"_{filename}")
        os.close(fd)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)

        # 执行识别
        result = asr_service.transcribe(tmp_path, hotword=prompt)

        # 返回格式处理
        if response_format_str == AudioResponseFormat.JSON.value:
            return JSONResponse(content=formatters.to_simple_json(result))
        elif response_format_str == AudioResponseFormat.VERBOSE_JSON.value:
            return JSONResponse(content=formatters.to_verbose_json(result))
        elif response_format_str == AudioResponseFormat.TEXT.value:
            return PlainTextResponse(content=formatters.to_text(result))
        elif response_format_str == AudioResponseFormat.SRT.value:
            return PlainTextResponse(content=formatters.to_srt(result), media_type="text/plain")
        elif response_format_str == AudioResponseFormat.VTT.value:
            return PlainTextResponse(content=formatters.to_vtt(result), media_type="text/plain")
        else:
            # 未知格式，默认 verbose_json
            return JSONResponse(content=formatters.to_verbose_json(result))
    except Exception as e:
        logger.error(f"Failed to transcribe from URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe from URL: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass



