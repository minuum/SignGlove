#!/usr/bin/env python3
"""
SignGlove 성능 모니터링 시스템
KLP-SignGlove의 성능 메트릭 및 모니터링 기능 통합

포함 기능:
- FPS/지연시간/안정성 실시간 모니터링
- 성능 히스토리 추적
- 알림 및 임계값 모니터링
- 성능 보고서 생성
"""

import time
import threading
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """성능 스냅샷 클래스"""
    timestamp: datetime
    fps: float
    latency_ms: float
    cpu_percent: float
    memory_mb: float
    stable_predictions: int
    total_predictions: int
    accuracy_rate: float
    system_load: float
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'fps': self.fps,
            'latency_ms': self.latency_ms,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'stable_predictions': self.stable_predictions,
            'total_predictions': self.total_predictions,
            'accuracy_rate': self.accuracy_rate,
            'system_load': self.system_load
        }


@dataclass
class PerformanceThresholds:
    """성능 임계값 설정"""
    min_fps: float = 30.0
    max_latency_ms: float = 100.0
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 1024.0
    min_accuracy_rate: float = 0.7
    
    def check_violations(self, snapshot: PerformanceSnapshot) -> List[str]:
        """임계값 위반 체크"""
        violations = []
        
        if snapshot.fps < self.min_fps:
            violations.append(f"FPS 저하: {snapshot.fps:.1f} < {self.min_fps}")
        
        if snapshot.latency_ms > self.max_latency_ms:
            violations.append(f"지연시간 초과: {snapshot.latency_ms:.1f}ms > {self.max_latency_ms}ms")
        
        if snapshot.cpu_percent > self.max_cpu_percent:
            violations.append(f"CPU 사용률 높음: {snapshot.cpu_percent:.1f}% > {self.max_cpu_percent}%")
        
        if snapshot.memory_mb > self.max_memory_mb:
            violations.append(f"메모리 사용량 높음: {snapshot.memory_mb:.1f}MB > {self.max_memory_mb}MB")
        
        if snapshot.accuracy_rate < self.min_accuracy_rate:
            violations.append(f"정확도 저하: {snapshot.accuracy_rate:.2f} < {self.min_accuracy_rate}")
        
        return violations


class PerformanceMonitor:
    """성능 모니터링 시스템 (KLP-SignGlove 기법)"""
    
    def __init__(self, 
                 history_size: int = 1000,
                 monitor_interval: float = 1.0,
                 report_dir: str = "data/performance"):
        """
        성능 모니터 초기화
        
        Args:
            history_size: 히스토리 보관 크기
            monitor_interval: 모니터링 간격 (초)
            report_dir: 보고서 저장 디렉토리
        """
        self.history_size = history_size
        self.monitor_interval = monitor_interval
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # 성능 히스토리
        self.performance_history: deque = deque(maxlen=history_size)
        self.fps_history: deque = deque(maxlen=100)
        self.latency_history: deque = deque(maxlen=100)
        
        # 임계값 설정
        self.thresholds = PerformanceThresholds()
        
        # 모니터링 상태
        self.is_monitoring = False
        self.start_time = None
        
        # 콜백 함수들
        self.alert_callbacks: List[Callable] = []
        
        # 시스템 모니터링
        self.process = psutil.Process()
        
        logger.info(f"성능 모니터 초기화: 히스토리 {history_size}, 간격 {monitor_interval}초")
    
    def add_alert_callback(self, callback: Callable[[List[str]], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def capture_performance_snapshot(self, 
                                   fps: float = 0.0,
                                   latency_ms: float = 0.0,
                                   stable_predictions: int = 0,
                                   total_predictions: int = 0) -> PerformanceSnapshot:
        """
        현재 성능 스냅샷 캡처
        
        Args:
            fps: 현재 FPS
            latency_ms: 현재 지연시간
            stable_predictions: 안정된 예측 수
            total_predictions: 총 예측 수
            
        Returns:
            snapshot: 성능 스냅샷
        """
        try:
            # 시스템 메트릭 수집
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # 시스템 로드 (macOS/Linux에서만 사용 가능)
            try:
                system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            except:
                system_load = 0.0
            
            # 정확도 계산
            accuracy_rate = stable_predictions / total_predictions if total_predictions > 0 else 0.0
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                fps=fps,
                latency_ms=latency_ms,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                stable_predictions=stable_predictions,
                total_predictions=total_predictions,
                accuracy_rate=accuracy_rate,
                system_load=system_load
            )
            
            # 히스토리에 추가
            self.performance_history.append(snapshot)
            self.fps_history.append(fps)
            self.latency_history.append(latency_ms)
            
            # 임계값 체크
            violations = self.thresholds.check_violations(snapshot)
            if violations:
                self._trigger_alerts(violations)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"성능 스냅샷 캡처 오류: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                fps=0.0, latency_ms=0.0, cpu_percent=0.0, memory_mb=0.0,
                stable_predictions=0, total_predictions=0, accuracy_rate=0.0, system_load=0.0
            )
    
    def _trigger_alerts(self, violations: List[str]):
        """알림 트리거"""
        for callback in self.alert_callbacks:
            try:
                callback(violations)
            except Exception as e:
                logger.error(f"알림 콜백 오류: {e}")
    
    def get_current_metrics(self) -> Dict:
        """현재 성능 메트릭 반환"""
        if not self.performance_history:
            return {}
        
        latest = self.performance_history[-1]
        
        # 최근 성능 통계
        recent_fps = list(self.fps_history)[-10:] if len(self.fps_history) >= 10 else list(self.fps_history)
        recent_latency = list(self.latency_history)[-10:] if len(self.latency_history) >= 10 else list(self.latency_history)
        
        return {
            'current': latest.to_dict(),
            'averages': {
                'fps_avg_10': sum(recent_fps) / len(recent_fps) if recent_fps else 0.0,
                'latency_avg_10': sum(recent_latency) / len(recent_latency) if recent_latency else 0.0,
            },
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'total_snapshots': len(self.performance_history)
        }
    
    def get_performance_summary(self, duration_minutes: int = 60) -> Dict:
        """성능 요약 통계"""
        if not self.performance_history:
            return {}
        
        # 지정된 시간 내의 데이터만 필터링
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_data = [s for s in self.performance_history if s.timestamp >= cutoff_time]
        
        if not recent_data:
            return {}
        
        # 통계 계산
        fps_values = [s.fps for s in recent_data]
        latency_values = [s.latency_ms for s in recent_data]
        cpu_values = [s.cpu_percent for s in recent_data]
        accuracy_values = [s.accuracy_rate for s in recent_data]
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_data),
            'fps': {
                'avg': sum(fps_values) / len(fps_values),
                'min': min(fps_values),
                'max': max(fps_values)
            },
            'latency_ms': {
                'avg': sum(latency_values) / len(latency_values),
                'min': min(latency_values),
                'max': max(latency_values)
            },
            'cpu_percent': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'accuracy_rate': {
                'avg': sum(accuracy_values) / len(accuracy_values),
                'min': min(accuracy_values),
                'max': max(accuracy_values)
            }
        }
    
    def export_performance_report(self, output_file: Optional[Path] = None) -> Path:
        """성능 보고서 내보내기"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.report_dir / f"performance_report_{timestamp}.json"
        
        # 보고서 데이터 구성
        report_data = {
            'report_timestamp': datetime.now().isoformat(),
            'monitoring_duration': {
                'start': self.start_time.isoformat() if self.start_time else None,
                'end': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            },
            'thresholds': {
                'min_fps': self.thresholds.min_fps,
                'max_latency_ms': self.thresholds.max_latency_ms,
                'max_cpu_percent': self.thresholds.max_cpu_percent,
                'max_memory_mb': self.thresholds.max_memory_mb,
                'min_accuracy_rate': self.thresholds.min_accuracy_rate
            },
            'summary_1h': self.get_performance_summary(60),
            'summary_24h': self.get_performance_summary(1440),
            'current_metrics': self.get_current_metrics(),
            'raw_data': [s.to_dict() for s in list(self.performance_history)[-100:]]  # 최근 100개만
        }
        
        # JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"성능 보고서 저장: {output_file}")
        return output_file
    
    def _monitor_worker(self):
        """모니터링 워커 스레드"""
        while self.is_monitoring:
            try:
                # 기본 성능 스냅샷 (외부 메트릭 없이)
                self.capture_performance_snapshot()
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"모니터링 워커 오류: {e}")
                time.sleep(1.0)
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            logger.warning("이미 모니터링 중입니다.")
            return
        
        self.is_monitoring = True
        self.start_time = datetime.now()
        
        # 워커 스레드 시작
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        logger.info("성능 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        
        # 최종 보고서 생성
        if self.performance_history:
            report_file = self.export_performance_report()
            logger.info(f"최종 성능 보고서: {report_file}")
        
        logger.info("성능 모니터링 중지")
    
    def update_thresholds(self, **kwargs):
        """임계값 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info(f"임계값 업데이트: {key} = {value}")
    
    def clear_history(self):
        """히스토리 초기화"""
        self.performance_history.clear()
        self.fps_history.clear()
        self.latency_history.clear()
        logger.info("성능 히스토리 초기화")


# 전역 성능 모니터 인스턴스
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """전역 성능 모니터 인스턴스 반환"""
    global _global_performance_monitor
    
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    
    return _global_performance_monitor


def start_performance_monitoring():
    """성능 모니터링 시작"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring():
    """성능 모니터링 중지"""
    global _global_performance_monitor
    
    if _global_performance_monitor:
        _global_performance_monitor.stop_monitoring()


def capture_inference_metrics(fps: float, latency_ms: float, stable_predictions: int, total_predictions: int):
    """추론 메트릭 캡처"""
    monitor = get_performance_monitor()
    return monitor.capture_performance_snapshot(fps, latency_ms, stable_predictions, total_predictions)
