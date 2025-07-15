"""
SignGlove 데이터 수집 서버
아두이노에서 전송되는 센서 데이터를 수집하고 CSV 형태로 저장하는 FastAPI 서버
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import logging
from pathlib import Path

from .data_storage import DataStorage
from .data_validation import DataValidator
from .models.sensor_data import SensorData, SignGestureData

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="SignGlove 데이터 수집 서버",
    description="수어 인식을 위한 센서 데이터 수집 및 저장 서버",
    version="1.0.0"
)

# CORS 설정 (개발 환경용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 저장소 및 검증기 초기화
data_storage = DataStorage()
data_validator = DataValidator()

# 응답 모델 정의
class ServerStatus(BaseModel):
    """서버 상태 응답 모델"""
    status: str
    timestamp: datetime
    total_records: int
    server_version: str

class DataResponse(BaseModel):
    """데이터 수집 응답 모델"""
    success: bool
    message: str
    record_id: Optional[str] = None
    timestamp: datetime

@app.get("/", response_model=ServerStatus)
async def root():
    """서버 상태 확인 엔드포인트"""
    total_records = await data_storage.get_total_records()
    return ServerStatus(
        status="running",
        timestamp=datetime.now(),
        total_records=total_records,
        server_version="1.0.0"
    )

@app.post("/data/sensor", response_model=DataResponse)
async def collect_sensor_data(
    sensor_data: SensorData,
    background_tasks: BackgroundTasks
):
    """
    센서 데이터 수집 엔드포인트
    아두이노에서 전송되는 플렉스 센서 및 자이로 센서 데이터 수집
    """
    try:
        # 데이터 검증
        validation_result = data_validator.validate_sensor_data(sensor_data)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"데이터 검증 실패: {validation_result.error_message}"
            )
        
        # 백그라운드에서 데이터 저장
        background_tasks.add_task(
            data_storage.save_sensor_data,
            sensor_data
        )
        
        logger.info(f"센서 데이터 수집 완료: {sensor_data.device_id}")
        
        return DataResponse(
            success=True,
            message="센서 데이터 수집 성공",
            record_id=sensor_data.device_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"센서 데이터 수집 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )

@app.post("/data/gesture", response_model=DataResponse)
async def collect_gesture_data(
    gesture_data: SignGestureData,
    background_tasks: BackgroundTasks
):
    """
    수어 제스처 데이터 수집 엔드포인트
    라벨링된 수어 동작 데이터 수집
    """
    try:
        # 데이터 검증
        validation_result = data_validator.validate_gesture_data(gesture_data)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"데이터 검증 실패: {validation_result.error_message}"
            )
        
        # 백그라운드에서 데이터 저장
        background_tasks.add_task(
            data_storage.save_gesture_data,
            gesture_data
        )
        
        logger.info(f"제스처 데이터 수집 완료: {gesture_data.gesture_label}")
        
        return DataResponse(
            success=True,
            message="제스처 데이터 수집 성공",
            record_id=f"{gesture_data.gesture_label}_{gesture_data.timestamp}",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"제스처 데이터 수집 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )

@app.get("/data/stats")
async def get_data_stats():
    """데이터 수집 통계 조회"""
    try:
        stats = await data_storage.get_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"통계 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"통계 조회 실패: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """서버 헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "SignGlove Data Collection Server"
    }

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    logger.info("SignGlove 데이터 수집 서버 시작")
    await data_storage.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("SignGlove 데이터 수집 서버 종료")
    await data_storage.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 