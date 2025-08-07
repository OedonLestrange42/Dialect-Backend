from datetime import timedelta


def _format_timestamp(milliseconds: int) -> str:
    """将毫秒转换为 SRT/VTT 时间戳格式 (HH:MM:SS,ms)"""
    td = timedelta(milliseconds=milliseconds)
    total_seconds = int(td.total_seconds())
    ms = td.microseconds // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def to_text(result: dict) -> str:
    """转换为纯文本格式"""
    return result.get("text", "")


def to_simple_json(result: dict) -> dict:
    """转换为 OpenAI 的简单 JSON 格式"""
    return {"text": result.get("text", "")}


def to_verbose_json(result: dict) -> dict:
    """转换为 OpenAI 的详细 JSON 格式"""
    full_text = result.get("text", "")
    sentences = result.get("sentence_info", [])

    segments = []
    for sent in sentences:
        segments.append({
            "id": len(segments),
            "start": sent['start'] / 1000.0,
            "end": sent['end'] / 1000.0,
            "text": sent['text'].strip()
        })

    return {
        "text": full_text,
        "segments": segments,
        "language": "zh"
    }


def to_srt(result: dict) -> str:
    """转换为 SRT 字幕格式"""
    sentences = result.get("sentence_info", [])
    srt_content = []
    for i, sent in enumerate(sentences):
        start_time = _format_timestamp(sent['start'])
        end_time = _format_timestamp(sent['end'])
        text = sent['text'].strip()
        srt_content.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)


def to_vtt(result: dict) -> str:
    """转换为 VTT 字幕格式"""
    sentences = result.get("sentence_info", [])
    vtt_content = ["WEBVTT\n"]
    for sent in sentences:
        start_time = _format_timestamp(sent['start']).replace(",", ".")
        end_time = _format_timestamp(sent['end']).replace(",", ".")
        text = sent['text'].strip()
        vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
    return "\n".join(vtt_content)