"""
SignGlove API 게이트웨이

외부 시스템들을 통합하는 API 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.integration_manager import integration_manager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="SignGlove API Gateway",
    description="SignGlove 통합 시스템 API 게이트웨이",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    message: str
    external_modules: Dict[str, Any]

class ModuleInfo(BaseModel):
    name: str
    path: str
    description: str
    health: Dict[str, Any]

@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트 - 시스템 상태 확인"""
    external_modules = {}
    for module_name in integration_manager.list_available_modules():
        external_modules[module_name] = integration_manager.check_module_health(module_name)
    
    return HealthResponse(
        status="healthy",
        message="SignGlove 통합 시스템이 정상적으로 실행 중입니다.",
        external_modules=external_modules
    )

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "message": "SignGlove API Gateway is running"}

@app.get("/modules", response_model=List[ModuleInfo])
async def list_modules():
    """사용 가능한 외부 모듈 목록 반환"""
    modules = []
    for module_name in integration_manager.list_available_modules():
        info = integration_manager.get_module_info(module_name)
        health = integration_manager.check_module_health(module_name)
        
        modules.append(ModuleInfo(
            name=info.get('name', module_name),
            path=info.get('path', ''),
            description=info.get('description', ''),
            health=health
        ))
    
    return modules

@app.get("/modules/{module_name}")
async def get_module_info(module_name: str):
    """특정 외부 모듈 정보 반환"""
    info = integration_manager.get_module_info(module_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    health = integration_manager.check_module_health(module_name)
    info['health'] = health
    
    return info

@app.get("/modules/{module_name}/health")
async def get_module_health(module_name: str):
    """특정 외부 모듈 상태 확인"""
    health = integration_manager.check_module_health(module_name)
    if health.get('status') == 'not_found':
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    return health

@app.get("/sync/status")
async def get_sync_status():
    """외부 저장소 동기화 상태 확인"""
    # GitHub Actions 동기화 상태를 확인하는 로직
    # 실제 구현에서는 GitHub API를 사용하여 워크플로우 상태를 확인
    return {
        "last_sync": "2024-08-28T15:00:00Z",
        "next_sync": "2024-09-02T09:00:00Z",
        "status": "scheduled",
        "external_repos": {
            "KLP-SignGlove": "https://github.com/Kyle-Riss/KLP-SignGlove",
            "SignGlove_HW": "https://github.com/KNDG01001/SignGlove_HW"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
