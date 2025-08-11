#!/usr/bin/env python3
"""
SignGlove 실시간 추론 엔진
KLP-SignGlove의 562 FPS 고성능 추론 시스템과 5-window 안정성 체크 통합

포함 기능:
- 멀티스레드 실시간 추론 파이프라인
- 5-window 예측 일관성 체크
- 신뢰도 기반 출력 제어
- 성능 모니터링 (FPS, 지연시간)
"""

import asyncio
import threading
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Callable
import logging
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue

from .models.sensor_data import SensorData
from .preprocessing import SensorPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """추론 결과 클래스"""
    predicted_class: str
    confidence: float
    processing_time_ms: float
    timestamp: datetime
    is_stable: bool = False
    stability_score: float = 0.0


@dataclass
class PerformanceMetrics:
    """성능 메트릭 클래스"""
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    total_predictions: int = 0
    stable_predictions: int = 0
    accuracy_rate: float = 0.0
    uptime_seconds: float = 0.0


class StabilityChecker:
    """5-window 예측 안정성 체크 (KLP-SignGlove 기법)"""
    
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.7):
        """
        안정성 체커 초기화
        
        Args:
            window_size: 안정성 체크 윈도우 크기
            confidence_threshold: 신뢰도 임계값
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        
    def add_prediction(self, prediction: str, confidence: float) -> Tuple[bool, float]:
        """
        예측 결과 추가 및 안정성 체크
        
        Args:
            prediction: 예측된 클래스
            confidence: 예측 신뢰도
            
        Returns:
            is_stable: 안정성 여부
            stability_score: 안정성 점수 (0-1)
        """
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        if len(self.prediction_history) < self.window_size:
            return False, 0.0
        
        # 일관성 체크
        most_common_prediction = max(set(self.prediction_history), 
                                   key=list(self.prediction_history).count)
        consistency_count = list(self.prediction_history).count(most_common_prediction)
        consistency_ratio = consistency_count / self.window_size
        
        # 신뢰도 체크
        avg_confidence = np.mean(list(self.confidence_history))
        min_confidence = np.min(list(self.confidence_history))
        
        # 안정성 점수 계산
        stability_score = (consistency_ratio * 0.6 + 
                          avg_confidence * 0.3 + 
                          min_confidence * 0.1)
        
        # 안정성 판단
        is_stable = (consistency_ratio >= 0.6 and  # 60% 이상 일관성
                    avg_confidence >= self.confidence_threshold and
                    min_confidence >= 0.5)
        
        return is_stable, stability_score
    
    def get_stable_prediction(self) -> Optional[Tuple[str, float]]:
        """안정된 예측 결과 반환"""
        if len(self.prediction_history) < self.window_size:
            return None
            
        most_common = max(set(self.prediction_history), 
                         key=list(self.prediction_history).count)
        avg_confidence = np.mean(list(self.confidence_history))
        
        return most_common, avg_confidence


class MockInferenceModel:
    """추론 모델 모킹 클래스 (실제 모델 연동 전까지 사용)"""
    
    def __init__(self):
        """모킹 모델 초기화"""
        self.ksl_classes = [
            'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',  # 자음
            'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',  # 모음
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'  # 숫자
        ]
        self.model_loaded = True
        
    def predict(self, preprocessed_data: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        모킹 추론 실행
        
        Args:
            preprocessed_data: 전처리된 센서 데이터
            
        Returns:
            predicted_class, confidence
        """
        # 실제 모델 대신 플렉스 센서 값 기반 간단한 로직
        flex_data = preprocessed_data.get('flex', np.zeros((1, 5)))
        
        if len(flex_data.shape) == 2 and flex_data.shape[0] > 0:
            # 플렉스 센서 평균값 기반 클래스 선택
            avg_flex = np.mean(flex_data[-1])  # 최신 데이터 사용
            
            if avg_flex < 0.3:
                predicted_class = 'ㅁ'  # 주먹
                confidence = 0.85 + np.random.normal(0, 0.1)
            elif avg_flex > 0.7:
                predicted_class = 'ㅏ'  # 펼친 손
                confidence = 0.80 + np.random.normal(0, 0.1)
            else:
                predicted_class = 'ㄱ'  # 중간 상태
                confidence = 0.75 + np.random.normal(0, 0.1)
        else:
            # 랜덤 선택
            predicted_class = np.random.choice(self.ksl_classes)
            confidence = 0.6 + np.random.random() * 0.3
        
        # 신뢰도 범위 조정
        confidence = np.clip(confidence, 0.1, 0.99)
        
        return predicted_class, confidence


class RealtimeInferenceEngine:
    """실시간 추론 엔진 (KLP-SignGlove 기법)"""
    
    def __init__(self, 
                 model: Optional[object] = None,
                 max_fps: float = 562.0,
                 confidence_threshold: float = 0.7,
                 stability_window: int = 5):
        """
        추론 엔진 초기화
        
        Args:
            model: 추론 모델 (None일 경우 MockModel 사용)
            max_fps: 최대 FPS 제한
            confidence_threshold: 신뢰도 임계값
            stability_window: 안정성 체크 윈도우
        """
        self.model = model if model is not None else MockInferenceModel()
        self.preprocessor = SensorPreprocessor()
        self.stability_checker = StabilityChecker(stability_window, confidence_threshold)
        
        # 성능 설정
        self.max_fps = max_fps
        self.min_interval = 1.0 / max_fps  # 최소 처리 간격
        self.confidence_threshold = confidence_threshold
        
        # 상태 관리
        self.is_running = False
        self.sensor_queue = queue.Queue(maxsize=1000)
        self.result_callbacks: List[Callable] = []
        
        # 성능 모니터링
        self.metrics = PerformanceMetrics()
        self.start_time = None
        self.last_process_time = 0
        self.processing_times = deque(maxlen=100)
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"추론 엔진 초기화: 최대 {max_fps} FPS, 신뢰도 임계값 {confidence_threshold}")
    
    def add_result_callback(self, callback: Callable[[InferenceResult], None]):
        """결과 콜백 추가"""
        self.result_callbacks.append(callback)
    
    def add_sensor_data(self, sensor_data: SensorData) -> bool:
        """
        센서 데이터 추가 (비동기)
        
        Args:
            sensor_data: 센서 데이터
            
        Returns:
            success: 큐 추가 성공 여부
        """
        try:
            self.sensor_queue.put_nowait(sensor_data)
            return True
        except queue.Full:
            logger.warning("센서 데이터 큐 오버플로우")
            return False
    
    def _process_single_inference(self, sensor_data_window: List[SensorData]) -> InferenceResult:
        """
        단일 추론 처리
        
        Args:
            sensor_data_window: 센서 데이터 윈도우
            
        Returns:
            inference_result: 추론 결과
        """
        start_time = time.time()
        
        try:
            # 전처리
            processed_data = self.preprocessor.preprocess_sensor_sequence(
                sensor_data_window,
                apply_filter=True,
                apply_normalization=True,
                apply_smoothing=True,
                create_windows=False
            )
            
            # 추론
            predicted_class, confidence = self.model.predict(processed_data)
            
            # 안정성 체크
            is_stable, stability_score = self.stability_checker.add_prediction(
                predicted_class, confidence
            )
            
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_times.append(processing_time)
            
            result = InferenceResult(
                predicted_class=predicted_class,
                confidence=confidence,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                is_stable=is_stable,
                stability_score=stability_score
            )
            
            # 메트릭 업데이트
            self.metrics.total_predictions += 1
            if is_stable:
                self.metrics.stable_predictions += 1
            
            return result
            
        except Exception as e:
            logger.error(f"추론 처리 오류: {e}")
            return InferenceResult(
                predicted_class="ERROR",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now()
            )
    
    def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        current_time = time.time()
        
        if self.start_time:
            self.metrics.uptime_seconds = current_time - self.start_time
            
            # FPS 계산
            if self.metrics.uptime_seconds > 0:
                self.metrics.fps = self.metrics.total_predictions / self.metrics.uptime_seconds
            
            # 평균 지연시간 계산
            if self.processing_times:
                self.metrics.avg_latency_ms = np.mean(list(self.processing_times))
            
            # 안정성 비율 계산
            if self.metrics.total_predictions > 0:
                self.metrics.accuracy_rate = self.metrics.stable_predictions / self.metrics.total_predictions
    
    def _inference_worker(self):
        """추론 워커 스레드"""
        sensor_window = deque(maxlen=10)  # 10개 센서 데이터 윈도우
        
        while self.is_running:
            try:
                # FPS 제한
                current_time = time.time()
                time_since_last = current_time - self.last_process_time
                
                if time_since_last < self.min_interval:
                    sleep_time = self.min_interval - time_since_last
                    time.sleep(sleep_time)
                
                # 센서 데이터 수집
                try:
                    sensor_data = self.sensor_queue.get(timeout=0.1)
                    sensor_window.append(sensor_data)
                except queue.Empty:
                    continue
                
                # 최소 윈도우 크기 확보
                if len(sensor_window) < 3:
                    continue
                
                # 추론 실행
                result = self._process_single_inference(list(sensor_window))
                
                # 결과 콜백 호출
                for callback in self.result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"콜백 오류: {e}")
                
                self.last_process_time = time.time()
                
                # 메트릭 업데이트 (10번에 1번)
                if self.metrics.total_predictions % 10 == 0:
                    self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"추론 워커 오류: {e}")
                time.sleep(0.1)
    
    def start(self):
        """추론 엔진 시작"""
        if self.is_running:
            logger.warning("추론 엔진이 이미 실행 중입니다.")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.last_process_time = time.time()
        
        # 워커 스레드 시작
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("실시간 추론 엔진 시작")
    
    def stop(self):
        """추론 엔진 중지"""
        self.is_running = False
        
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=2.0)
        
        self.executor.shutdown(wait=True)
        logger.info("실시간 추론 엔진 중지")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """성능 메트릭 반환"""
        self._update_performance_metrics()
        return self.metrics
    
    def get_stable_prediction(self) -> Optional[Tuple[str, float]]:
        """현재 안정된 예측 결과 반환"""
        return self.stability_checker.get_stable_prediction()
    
    def reset_metrics(self):
        """성능 메트릭 리셋"""
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        self.processing_times.clear()
        logger.info("성능 메트릭 리셋")


# 전역 추론 엔진 인스턴스
_global_inference_engine: Optional[RealtimeInferenceEngine] = None


def get_inference_engine() -> RealtimeInferenceEngine:
    """전역 추론 엔진 인스턴스 반환"""
    global _global_inference_engine
    
    if _global_inference_engine is None:
        _global_inference_engine = RealtimeInferenceEngine()
    
    return _global_inference_engine


def start_inference_engine():
    """추론 엔진 시작"""
    engine = get_inference_engine()
    engine.start()


def stop_inference_engine():
    """추론 엔진 중지"""
    global _global_inference_engine
    
    if _global_inference_engine:
        _global_inference_engine.stop()
        _global_inference_engine = None
