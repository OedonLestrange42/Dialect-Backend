from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.services.asr_service import initialize_asr_service
from app.api.endpoints import audio


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 应用启动时执行 ---
    print("Application startup...")
    # 初始化 ASR 服务并将其存储在 app.state 中
    # 这是创建和管理单例的推荐方式
    print(f"Loading ASR models: ASR={settings.FUNASR_MODEL_ID}, VAD={settings.VAD_MODEL_ID}, PUNC={settings.PUNC_MODEL_ID}, SPK={settings.SPK_MODEL_ID}")
    app.state.asr_service = initialize_asr_service()
    print("Complete ASR pipeline has been initialized.")

    yield

    # --- 应用关闭时执行 ---
    print("Application shutdown...")
    # 清理资源 (如果需要)
    app.state.asr_service = None
    print("ASR Service has been shut down.")


app = FastAPI(
    title="FunASR OpenAI-Compatible API",
    description="An API for FunASR that mimics the OpenAI audio transcription API.",
    version="1.0.0",
    lifespan=lifespan
)

# 包含 API 路由
app.include_router(audio.router)


@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the FunASR API!"}