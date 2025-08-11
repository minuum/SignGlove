#!/usr/bin/env python3
"""
SignGlove 통합 데이터 수집 서버
베스트 프랙티스 구조와 연동된 실험 관리 및 데이터 수집 서버

역할 분담:
- 이민우: 서버 구축, 데이터 저장, 실험 관리
- 양동건: 아두이노 펌웨어, UART/WiFi 통신, 클라이언트
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import logging
import asyncio
import json
from pathlib import Path
import pandas as pd

from .models.sensor_data import SensorData, SignGestureData
from .data_validation import DataValidator
from .ksl_classes import ksl_manager, KSLCategory
from .preprocessing import SensorPreprocessor
from .inference_engine import get_inference_engine, InferenceResult
from .tts_engine import get_tts_engine, speak_ksl_async
from .performance_monitor import get_performance_monitor, capture_inference_metrics

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="SignGlove 통합 데이터 수집 서버",
    description="베스트 프랙티스 적용된 수어 데이터 수집 및 실험 관리 서버",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (모니터링 대시보드용)
app.mount("/static", StaticFiles(directory="static"), name="static")


class UnifiedDataManager:
    """통합 데이터 관리자"""
    
    def __init__(self):
        """초기화"""
        self.setup_directories()
        self.load_config()
        self.current_session = None
        self.current_experiment = None
        self.collected_samples = []
        
    def setup_directories(self):
        """베스트 프랙티스 디렉토리 구조 생성"""
        self.data_root = Path("data")
        
        self.directories = {
            'raw': self.data_root / "raw",
            'processed': self.data_root / "processed", 
            'interim': self.data_root / "interim",
            'unified': self.data_root / "unified",
            'splits': self.data_root / "splits",
            'metadata': self.data_root / "metadata",
            'stats': self.data_root / "stats"
        }
        
        # 디렉토리 생성
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 클래스별 하위 디렉토리
        categories = ['consonant', 'vowel', 'number']
        for category in categories:
            (self.directories['raw'] / category).mkdir(exist_ok=True)
            (self.directories['processed'] / category).mkdir(exist_ok=True)
    
    def load_config(self):
        """실험 설정 로드"""
        config_file = self.directories['metadata'] / "experiment_config.json"
        
        default_config = {
            'target_classes': {
                'consonant': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'],
                'vowel': ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'],
                'number': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            },
            'target_samples_per_class': 60,
            'measurement_duration': 5,
            'sampling_rate': 20,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        }
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """설정 저장"""
        config_file = self.directories['metadata'] / "experiment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def get_class_category(self, class_label: str) -> str:
        """클래스 카테고리 반환"""
        for category, classes in self.config['target_classes'].items():
            if class_label in classes:
                return category
        return "unknown"


# 전역 데이터 매니저
data_manager = UnifiedDataManager()
data_validator = DataValidator()
preprocessor = SensorPreprocessor()
inference_engine = get_inference_engine()
tts_engine = get_tts_engine()
performance_monitor = get_performance_monitor()


# API 모델 정의
class ExperimentSession(BaseModel):
    """실험 세션 모델"""
    session_id: str
    performer_id: str = "default_performer"
    class_label: str
    category: str
    target_samples: int = 60
    duration_per_sample: int = 5
    sampling_rate: int = 20
    storage_strategy: str = "both"  # individual, unified, both
    notes: Optional[str] = None


class ExperimentStatus(BaseModel):
    """실험 상태 모델"""
    session_id: Optional[str]
    class_label: Optional[str]
    current_sample: int = 0
    target_samples: int = 0
    is_collecting: bool = False
    samples_collected: int = 0
    timestamp: datetime


class DataResponse(BaseModel):
    """데이터 응답 모델"""
    success: bool
    message: str
    session_id: Optional[str] = None
    sample_id: Optional[int] = None
    timestamp: datetime


class ServerStats(BaseModel):
    """서버 통계 모델"""
    total_sessions: int
    total_samples: int
    classes_collected: List[str]
    disk_usage: Dict[str, int]
    uptime: str


# ===== API 엔드포인트 =====

@app.get("/", response_class=JSONResponse)
async def root():
    """서버 루트"""
    return {
        "message": "SignGlove 통합 데이터 수집 서버",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status", response_model=ExperimentStatus)
async def get_server_status():
    """서버 상태 조회"""
    return ExperimentStatus(
        session_id=data_manager.current_session.session_id if data_manager.current_session else None,
        class_label=data_manager.current_experiment.get('class_label') if data_manager.current_experiment else None,
        current_sample=len(data_manager.collected_samples),
        target_samples=data_manager.current_experiment.get('target_samples', 0) if data_manager.current_experiment else 0,
        is_collecting=data_manager.current_session is not None,
        samples_collected=len(data_manager.collected_samples),
        timestamp=datetime.now()
    )


@app.post("/experiment/start", response_model=DataResponse)
async def start_experiment(session: ExperimentSession):
    """실험 세션 시작"""
    try:
        # 세션 ID 생성
        if not session.session_id:
            session.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 클래스 카테고리 확인
        category = data_manager.get_class_category(session.class_label)
        if category == "unknown":
            raise HTTPException(
                status_code=400,
                detail=f"알 수 없는 클래스: {session.class_label}"
            )
        
        # 현재 세션 설정
        data_manager.current_session = session
        data_manager.current_experiment = {
            'session_id': session.session_id,
            'class_label': session.class_label,
            'category': category,
            'target_samples': session.target_samples,
            'duration_per_sample': session.duration_per_sample,
            'sampling_rate': session.sampling_rate,
            'storage_strategy': session.storage_strategy,
            'start_time': datetime.now(),
            'performer_id': session.performer_id
        }
        data_manager.collected_samples = []
        
        logger.info(f"실험 세션 시작: {session.session_id} - 클래스: {session.class_label}")
        
        return DataResponse(
            success=True,
            message=f"실험 세션 시작됨: {session.class_label}",
            session_id=session.session_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"실험 시작 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/experiment/stop", response_model=DataResponse)
async def stop_experiment():
    """실험 세션 종료"""
    try:
        if not data_manager.current_session:
            raise HTTPException(status_code=400, detail="진행 중인 실험이 없습니다")
        
        session_id = data_manager.current_session.session_id
        
        # 수집된 데이터 최종 저장
        if data_manager.collected_samples:
            await save_experiment_data()
        
        # 세션 종료
        data_manager.current_session = None
        data_manager.current_experiment = None
        data_manager.collected_samples = []
        
        logger.info(f"실험 세션 종료: {session_id}")
        
        return DataResponse(
            success=True,
            message="실험 세션 종료됨",
            session_id=session_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"실험 종료 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/sensor", response_model=DataResponse)
async def collect_sensor_data(sensor_data: SensorData, background_tasks: BackgroundTasks):
    """
    센서 데이터 수집 (양동건 팀원이 보내는 데이터를 받음)
    """
    try:
        # 진행 중인 실험 확인
        if not data_manager.current_session:
            raise HTTPException(status_code=400, detail="진행 중인 실험이 없습니다. /experiment/start를 먼저 호출하세요.")
        
        # 데이터 검증
        validation_result = data_validator.validate_sensor_data(sensor_data)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"데이터 검증 실패: {validation_result.error_message}"
            )
        
        # 현재 세션에 데이터 추가
        data_manager.collected_samples.append(sensor_data)
        
        sample_count = len(data_manager.collected_samples)
        logger.info(f"센서 데이터 수집: {sensor_data.device_id} - 샘플 {sample_count}")
        
        return DataResponse(
            success=True,
            message="센서 데이터 수집됨",
            session_id=data_manager.current_session.session_id,
            sample_id=sample_count,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"센서 데이터 수집 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sample/complete", response_model=DataResponse)
async def complete_sample():
    """현재 샘플 완료 처리"""
    try:
        if not data_manager.current_session:
            raise HTTPException(status_code=400, detail="진행 중인 실험이 없습니다")
        
        if not data_manager.collected_samples:
            raise HTTPException(status_code=400, detail="수집된 데이터가 없습니다")
        
        # 개별 샘플 저장
        await save_current_sample()
        
        sample_count = len(data_manager.collected_samples)
        target_samples = data_manager.current_experiment['target_samples']
        
        return DataResponse(
            success=True,
            message=f"샘플 완료 ({sample_count}/{target_samples})",
            session_id=data_manager.current_session.session_id,
            sample_id=sample_count,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"샘플 완료 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def save_current_sample():
    """현재 수집된 샘플 저장"""
    if not data_manager.current_experiment or not data_manager.collected_samples:
        return
    
    try:
        experiment = data_manager.current_experiment
        samples = data_manager.collected_samples
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_idx = len(samples) // experiment['sampling_rate'] // experiment['duration_per_sample']
        filename = f"{experiment['class_label']}_sample_{sample_idx:03d}_{timestamp}"
        
        # 카테고리별 저장 경로
        category = experiment['category']
        raw_dir = data_manager.directories['raw'] / category
        
        # CSV 저장
        csv_file = raw_dir / f"{filename}.csv"
        
        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'timestamp', 'device_id', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                'battery_level', 'signal_strength'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for sensor_data in samples:
                row = {
                    'timestamp': sensor_data.timestamp.isoformat(),
                    'device_id': sensor_data.device_id,
                    'flex_1': sensor_data.flex_sensors.flex_1,
                    'flex_2': sensor_data.flex_sensors.flex_2,
                    'flex_3': sensor_data.flex_sensors.flex_3,
                    'flex_4': sensor_data.flex_sensors.flex_4,
                    'flex_5': sensor_data.flex_sensors.flex_5,
                    'gyro_x': sensor_data.gyro_data.gyro_x,
                    'gyro_y': sensor_data.gyro_data.gyro_y,
                    'gyro_z': sensor_data.gyro_data.gyro_z,
                    'accel_x': sensor_data.gyro_data.accel_x,
                    'accel_y': sensor_data.gyro_data.accel_y,
                    'accel_z': sensor_data.gyro_data.accel_z,
                    'battery_level': sensor_data.battery_level,
                    'signal_strength': sensor_data.signal_strength
                }
                writer.writerow(row)
        
        logger.info(f"샘플 저장 완료: {csv_file.name}")
        
    except Exception as e:
        logger.error(f"샘플 저장 실패: {str(e)}")


async def save_experiment_data():
    """실험 완료 시 통합 데이터 저장"""
    if not data_manager.current_experiment or not data_manager.collected_samples:
        return
    
    try:
        experiment = data_manager.current_experiment
        samples = data_manager.collected_samples
        
        # 통합 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        class_label = experiment['class_label']
        
        unified_file = data_manager.directories['unified'] / f"{class_label}_unified_{timestamp}.csv"
        
        import csv
        with open(unified_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'sample_id', 'timestamp', 'device_id', 'class_label', 'category',
                'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                'battery_level', 'signal_strength'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample_id, sensor_data in enumerate(samples):
                row = {
                    'sample_id': sample_id,
                    'timestamp': sensor_data.timestamp.isoformat(),
                    'device_id': sensor_data.device_id,
                    'class_label': class_label,
                    'category': experiment['category'],
                    'flex_1': sensor_data.flex_sensors.flex_1,
                    'flex_2': sensor_data.flex_sensors.flex_2,
                    'flex_3': sensor_data.flex_sensors.flex_3,
                    'flex_4': sensor_data.flex_sensors.flex_4,
                    'flex_5': sensor_data.flex_sensors.flex_5,
                    'gyro_x': sensor_data.gyro_data.gyro_x,
                    'gyro_y': sensor_data.gyro_data.gyro_y,
                    'gyro_z': sensor_data.gyro_data.gyro_z,
                    'accel_x': sensor_data.gyro_data.accel_x,
                    'accel_y': sensor_data.gyro_data.accel_y,
                    'accel_z': sensor_data.gyro_data.accel_z,
                    'battery_level': sensor_data.battery_level,
                    'signal_strength': sensor_data.signal_strength
                }
                writer.writerow(row)
        
        logger.info(f"통합 데이터 저장 완료: {unified_file.name}")
        
    except Exception as e:
        logger.error(f"통합 데이터 저장 실패: {str(e)}")


@app.get("/stats", response_model=ServerStats)
async def get_server_stats():
    """서버 통계 조회"""
    try:
        # 파일 통계
        raw_files = list(data_manager.directories['raw'].rglob("*.csv"))
        unified_files = list(data_manager.directories['unified'].glob("*.csv"))
        
        # 클래스 목록
        classes_collected = set()
        for file in unified_files:
            if "_unified_" in file.name:
                class_label = file.name.split("_unified_")[0]
                classes_collected.add(class_label)
        
        # 디스크 사용량
        disk_usage = {}
        for name, path in data_manager.directories.items():
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                disk_usage[name] = size
        
        return ServerStats(
            total_sessions=len(list(data_manager.directories['metadata'].glob("session_*.json"))),
            total_samples=len(raw_files),
            classes_collected=list(classes_collected),
            disk_usage=disk_usage,
            uptime="running"
        )
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/list")
async def list_experiments():
    """실험 목록 조회"""
    try:
        unified_files = list(data_manager.directories['unified'].glob("*_unified_*.csv"))
        experiments = []
        
        for file in unified_files:
            if "_unified_" in file.name:
                parts = file.name.replace(".csv", "").split("_unified_")
                class_label = parts[0]
                timestamp = parts[1]
                
                # 파일 정보
                stat = file.stat()
                experiments.append({
                    'class_label': class_label,
                    'timestamp': timestamp,
                    'file_size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'file_path': str(file.relative_to(Path.cwd()))
                })
        
        return {'experiments': experiments}
        
    except Exception as e:
        logger.error(f"실험 목록 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """파일 다운로드"""
    try:
        if file_type not in data_manager.directories:
            raise HTTPException(status_code=400, detail="잘못된 파일 타입")
        
        file_path = data_manager.directories[file_type] / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"파일 다운로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 새로운 통합 API 엔드포인트 =====

@app.post("/inference/predict")
async def predict_ksl(sensor_data_list: List[SensorData]):
    """
    실시간 KSL 예측 (KLP-SignGlove 통합 기능)
    """
    try:
        if not sensor_data_list:
            raise HTTPException(status_code=400, detail="센서 데이터가 비어있습니다")
        
        # 전처리
        processed_data = preprocessor.preprocess_sensor_sequence(
            sensor_data_list,
            apply_filter=True,
            apply_normalization=True,
            apply_smoothing=True
        )
        
        # 추론 (MockModel 사용)
        if hasattr(inference_engine.model, 'predict'):
            predicted_class, confidence = inference_engine.model.predict(processed_data)
        else:
            predicted_class, confidence = "ㄱ", 0.75  # 기본값
        
        # 안정성 체크
        is_stable, stability_score = inference_engine.stability_checker.add_prediction(
            predicted_class, confidence
        )
        
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_stable": is_stable,
            "stability_score": stability_score,
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(sensor_data_list)
        }
        
        logger.info(f"KSL 예측: {predicted_class} (신뢰도: {confidence:.2f}, 안정성: {is_stable})")
        return result
        
    except Exception as e:
        logger.error(f"KSL 예측 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/stream")
async def stream_inference(sensor_data: SensorData):
    """
    실시간 스트리밍 추론 (562 FPS 성능)
    """
    try:
        # 추론 엔진 시작 (아직 시작되지 않은 경우)
        if not inference_engine.is_running:
            inference_engine.start()
        
        # 센서 데이터 추가
        success = inference_engine.add_sensor_data(sensor_data)
        
        if not success:
            raise HTTPException(status_code=429, detail="처리 큐가 가득참")
        
        # 현재 안정된 예측 결과 반환
        stable_prediction = inference_engine.get_stable_prediction()
        
        result = {
            "queued": True,
            "queue_size": inference_engine.sensor_queue.qsize(),
            "stable_prediction": stable_prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"스트리밍 추론 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/metrics")
async def get_inference_metrics():
    """
    추론 엔진 성능 메트릭 조회
    """
    try:
        metrics = inference_engine.get_performance_metrics()
        
        # 성능 모니터에 메트릭 기록
        capture_inference_metrics(
            fps=metrics.fps,
            latency_ms=metrics.avg_latency_ms,
            stable_predictions=metrics.stable_predictions,
            total_predictions=metrics.total_predictions
        )
        
        return {
            "fps": metrics.fps,
            "avg_latency_ms": metrics.avg_latency_ms,
            "total_predictions": metrics.total_predictions,
            "stable_predictions": metrics.stable_predictions,
            "accuracy_rate": metrics.accuracy_rate,
            "uptime_seconds": metrics.uptime_seconds,
            "is_running": inference_engine.is_running,
            "queue_size": inference_engine.sensor_queue.qsize()
        }
        
    except Exception as e:
        logger.error(f"메트릭 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/speak")
async def speak_text(text: str, confidence: float = 1.0):
    """
    TTS 음성 출력 (한국어 KSL 지원)
    """
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다")
        
        # TTS 엔진 시작 (아직 시작되지 않은 경우)
        if not tts_engine.is_running:
            tts_engine.start()
        
        # 비동기 음성 출력
        speak_ksl_async(text, confidence)
        
        # KSL → 음성 텍스트 변환 결과 반환
        speech_text = tts_engine.convert_ksl_to_speech(text)
        
        result = {
            "original_text": text,
            "speech_text": speech_text,
            "confidence": confidence,
            "queued": True,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"TTS 요청: {text} → {speech_text}")
        return result
        
    except Exception as e:
        logger.error(f"TTS 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/test")
async def test_tts():
    """TTS 테스트"""
    try:
        # TTS 엔진 시작
        if not tts_engine.is_running:
            tts_engine.start()
        
        # 테스트 실행
        success = tts_engine.test_speech()
        
        return {
            "success": success,
            "platform": tts_engine.platform,
            "voice": tts_engine.config.voice,
            "enabled": tts_engine.config.enabled,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"TTS 테스트 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocessing/process")
async def preprocess_data(sensor_data_list: List[SensorData], 
                         apply_filter: bool = True,
                         apply_normalization: bool = True,
                         apply_smoothing: bool = True,
                         create_windows: bool = False):
    """
    센서 데이터 전처리 (KLP-SignGlove 전처리 파이프라인)
    """
    try:
        if not sensor_data_list:
            raise HTTPException(status_code=400, detail="센서 데이터가 비어있습니다")
        
        # 전처리 실행
        processed_data = preprocessor.preprocess_sensor_sequence(
            sensor_data_list,
            apply_filter=apply_filter,
            apply_normalization=apply_normalization,
            apply_smoothing=apply_smoothing,
            create_windows=create_windows
        )
        
        # numpy 배열을 리스트로 변환 (JSON 직렬화용)
        result = {
            "flex_data": processed_data['flex'].tolist(),
            "gyro_data": processed_data['gyro'].tolist(), 
            "accel_data": processed_data['accel'].tolist(),
            "sample_count": len(sensor_data_list),
            "processing_config": {
                "filter_applied": apply_filter,
                "normalization_applied": apply_normalization,
                "smoothing_applied": apply_smoothing,
                "windows_created": create_windows
            },
            "preprocessor_stats": preprocessor.get_preprocessing_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
        if 'windowed' in processed_data:
            result["windowed_data"] = processed_data['windowed'].tolist()
            result["window_shape"] = processed_data['windowed'].shape
        
        logger.info(f"전처리 완료: {len(sensor_data_list)}개 샘플")
        return result
        
    except Exception as e:
        logger.error(f"전처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/current")
async def get_current_performance():
    """현재 성능 메트릭 조회"""
    try:
        metrics = performance_monitor.get_current_metrics()
        return metrics
    except Exception as e:
        logger.error(f"성능 메트릭 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/summary")
async def get_performance_summary(duration_minutes: int = 60):
    """성능 요약 통계 조회"""
    try:
        summary = performance_monitor.get_performance_summary(duration_minutes)
        return summary
    except Exception as e:
        logger.error(f"성능 요약 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/performance/start")
async def start_performance_monitoring():
    """성능 모니터링 시작"""
    try:
        performance_monitor.start_monitoring()
        return {"status": "started", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"성능 모니터링 시작 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/performance/stop")
async def stop_performance_monitoring():
    """성능 모니터링 중지"""
    try:
        performance_monitor.stop_monitoring()
        return {"status": "stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"성능 모니터링 중지 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/report")
async def export_performance_report():
    """성능 보고서 내보내기"""
    try:
        report_file = performance_monitor.export_performance_report()
        return {
            "report_file": str(report_file),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"성능 보고서 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 서버 시작 함수
def start_server(host: str = "0.0.0.0", port: int = 8000):
    """서버 시작"""
    print("=" * 60)
    print("🚀 SignGlove 통합 데이터 수집 서버 시작")
    print("=" * 60)
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"📊 API 문서: http://{host}:{port}/docs")
    print(f"📁 데이터 경로: {data_manager.data_root}")
    print("")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server() 