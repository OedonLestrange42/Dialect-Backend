from fastapi import Request, Depends, HTTPException, status, Header
from typing import Annotated
from app.core.config import settings
from app.services.asr_service import ASRService


def get_asr_service(request: Request) -> ASRService:
    """从应用状态中获取 ASR 服务单例"""
    return request.app.state.asr_service


async def verify_api_key(authorization: Annotated[str, Header()] = None):
    """验证 Bearer Token"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )

    scheme, _, token = authorization.partition(' ')
    if scheme.lower() != 'bearer' or token != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )