# FunASR OpenAI-Compatible API

这是一个使用 FastAPI 搭建的生产级 API 服务器，它将阿里巴巴达摩院的 FunASR 语音识别模型封装成一个与 OpenAI `/v1/audio/transcriptions` 端点兼容的 API。

这使得开发者可以轻松地将现有的、为 OpenAI API 编写的客户端代码，通过修改 API 地址和密钥，无缝切换到使用 FunASR 模型进行语音识别。

## ✨ 主要特性

*   **生产级架构**: 清晰的多文件结构，易于维护和扩展。
*   **高性能**: FunASR 模型在服务启动时仅加载一次，作为单例存在，避免了重复加载的开销。
*   **OpenAI 兼容**: 模仿 OpenAI 的 API 路径、请求参数和响应结构，降低迁移成本。
*   **多种输出格式**: 支持 `json`, `verbose_json`, `text`, `srt`, `vtt` 多种响应格式。
*   **容器化部署**: 提供 `Dockerfile` 和 `docker-compose.yml`，实现一键部署。

## 🚀 快速开始

本项目推荐使用 Docker 进行部署，可以免去繁琐的环境配置。

### 1. 环境准备

*   确保你的系统已经安装了 [Docker](https://www.docker.com/) 和 [Docker Compose](https://docs.docker.com/compose/install/)。
*   如果你拥有支持 CUDA 的 NVIDIA 显卡，请确保已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 以启用 GPU 加速。

### 2. 配置

1.  克隆或下载本代码仓库到你的服务器。

2.  在项目根目录下，创建一个名为 `.env` 的文件，内容如下。你可以根据需要修改配置。

    ```env
    # API服务器监听的地址和端口
    API_HOST=0.0.0.0
    API_PORT=8000

    # 用于鉴权的API（用户API等于该API才能调用成功，后续考虑使用supabase维护API）
    API_KEY="your-secret-api-key"

    # 使用的 FunASR 模型 ID，可从 ModelScope 官网查找
    FUNASR_MODEL_ID="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    # 模型文件的缓存目录
    MODEL_CACHE_DIR="./model_cache"
    ```

### 3. 启动服务

在项目根目录打开终端，执行以下命令：

```bash
docker-compose up --build
```

非docker启动：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

*   **首次启动**: Docker 会构建镜像并下载指定的 FunASR 模型。根据你的网络状况，这可能需要几分钟到几十分钟。模型文件会保存在 `./model_cache` 目录下，下次启动时将直接加载，无需重新下载。
*   看到类似 `Uvicorn running on http://0.0.0.0:8000` 的日志输出时，表示服务已成功启动。

## 🛠️ API 调用指南

### 端点信息

*   **URL**: `http://<your-server-ip>:8000/v1/audio/transcriptions`
*   **Method**: `POST`
*   **Auth**: `Bearer` Token in `Authorization` Header.

### 请求参数 (`multipart/form-data`)

| 参数              | 类型       | 是否必须 | 描述                                                                                                      |
| ----------------- | ---------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `file`            | `File`     | **是**   | 需要识别的音频文件。支持 `mp3`, `wav`, `m4a` 等常见格式。                                                      |
| `model`           | `string`   | **是**   | 模型标识符，例如 `paraformer-large`。此参数仅为兼容 OpenAI API 格式，实际使用的模型由服务器配置决定。 |
| `response_format` | `string`   | 否       | 返回内容的格式。可选值：`json`(默认), `verbose_json`, `text`, `srt`, `vtt`。                             |
| `prompt`          | `string`   | 否       | 提供一组热词（用逗号分隔），可以显著提升这些词语的识别准确率。例如 `"阿里巴巴,通义千问"`。                  |

---

### 调用示例 1: 使用 cURL

这是一个基础的 HTTP 请求示例，适用于在命令行中快速测试。

请将 `/path/to/your/audio.wav` 替换为你的音频文件路径，并将 `your-secret-api-key` 替换为你在 `.env` 文件中设置的 `API_KEY`。

```bash
curl --location 'http://localhost:8000/v1/audio/transcriptions' \
--header 'Authorization: Bearer your-secret-api-key' \
--form 'file=@"/path/to/your/audio.wav"' \
--form 'model="paraformer-large"' \
--form 'response_format="verbose_json"' \
--form 'prompt="达摩院,模型服务"'
```

**预期响应 (`verbose_json` 格式):**

```json
{
    "text": "欢迎使用达摩院的模型服务。",
    "segments": [
        {
            "id": 0,
            "start": 0.5,
            "end": 3.2,
            "text": "欢迎使用达摩院的模型服务。"
        }
    ],
    "language": "zh"
}
```

### 调用示例 2: 使用 Python (`requests`)

这是一个在 Python 程序中调用 API 的完整示例。

```python
import requests
import os

# --- 配置 ---
# API 服务器地址
API_URL = "http://localhost:8000/v1/audio/transcriptions"
# 你的 API 密钥
API_KEY = "your-secret-api-key"
# 本地音频文件路径
FILE_PATH = "tests/test_audio.wav" # 请替换为你的文件路径

# 确保文件存在
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"音频文件未找到: {FILE_PATH}")

# --- 准备请求 ---
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

files = {
    "file": (os.path.basename(FILE_PATH), open(FILE_PATH, "rb")),
    # 'model', 'response_format' 等参数也作为 form data 发送
    "model": (None, "paraformer-large"),
    "response_format": (None, "verbose_json"),
    "prompt": (None, "阿里巴巴,通义千问")
}

# --- 发送请求 ---
try:
    response = requests.post(API_URL, headers=headers, files=files)
    
    # 检查响应状态码
    response.raise_for_status() 

    # --- 处理响应 ---
    result = response.json()
    print("识别成功!")
    print("完整文本:", result.get("text"))
    print("分段信息:")
    for segment in result.get("segments", []):
        start = segment.get('start')
        end = segment.get('end')
        text = segment.get('text')
        print(f"  [{start:.2f}s -> {end:.2f}s] {text}")

except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
    # 如果有响应内容，打印出来帮助调试
    if e.response is not None:
        print("错误详情:", e.response.text)
```
