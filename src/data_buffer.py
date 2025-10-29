#!/usr/bin/env python3
"""
SignGlove 데이터 버퍼 시스템
- 실시간 센서 데이터 버퍼링 및 관리
- 큐 기반 데이터 처리 및 스레드 안전성 보장
"""

import sys
import os
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import json
from pathlib import Path

# 상위 디렉토리의 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class SensorReading:
    """센서 읽기 데이터"""
    timestamp_ms: int
    recv_timestamp_ms: int
    
    # 센서 데이터
    flex1: int
    flex2: int
    flex3: int
    flex4: int
    flex5: int
    pitch: float
    roll: float
    yaw: float
    
    # 메타데이터
    sampling_hz: float = 0.0
    sequence_id: Optional[str] = None

@dataclass
class BufferStats:
    """버퍼 통계"""
    total_samples: int = 0
    dropped_samples: int = 0
    buffer_usage: float = 0.0
    avg_sampling_rate: float = 0.0
    last_sample_time: Optional[float] = None
    buffer_warnings: int = 0
    max_queue_usage: float = 0.0

class SignGloveDataBuffer:
    """SignGlove 데이터 버퍼"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 target_sampling_rate: float = 33.3,
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.95):
        
        self.max_size = max_size
        self.target_sampling_rate = target_sampling_rate
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        # 데이터 큐 (스레드 안전)
        self.data_queue = queue.Queue(maxsize=max_size)
        
        # 버퍼 통계
        self.stats = BufferStats()
        
        # 샘플링 레이트 제어
        self.sampling_rate_history = deque(maxlen=100)
        self.last_sample_time = None
        
        # 콜백 함수들
        self.callbacks = {
            'on_data_added': [],
            'on_buffer_full': [],
            'on_buffer_warning': [],
            'on_sampling_rate_change': []
        }
        
        # 스레드 제어
        self.running = False
        self.buffer_thread = None
        
        print(f"📊 데이터 버퍼 초기화 완료 (최대 크기: {max_size})")
    
    def add_data(self, reading: SensorReading) -> bool:
        """데이터 추가"""
        try:
            # 큐가 가득 찬 경우 처리
            if self.data_queue.full():
                # 오래된 데이터 제거
                try:
                    self.data_queue.get_nowait()
                    self.stats.dropped_samples += 1
                except queue.Empty:
                    pass
            
            # 새 데이터 추가
            self.data_queue.put_nowait(reading)
            
            # 통계 업데이트
            self._update_stats(reading)
            
            # 콜백 실행
            self._trigger_callbacks('on_data_added', reading)
            
            # 버퍼 상태 체크
            self._check_buffer_status()
            
            return True
            
        except queue.Full:
            self.stats.dropped_samples += 1
            return False
    
    def get_data(self, timeout: float = 0.1) -> Optional[SensorReading]:
        """데이터 가져오기"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_data_nowait(self) -> Optional[SensorReading]:
        """데이터 즉시 가져오기 (논블로킹)"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_sequence(self, length: int) -> List[SensorReading]:
        """시퀀스 데이터 가져오기"""
        sequence = []
        for _ in range(length):
            data = self.get_data_nowait()
            if data is None:
                break
            sequence.append(data)
        return sequence
    
    def get_latest_sequence(self, length: int) -> List[SensorReading]:
        """최신 시퀀스 데이터 가져오기"""
        # 모든 데이터를 임시로 가져와서 최신 것만 선택
        all_data = []
        while True:
            data = self.get_data_nowait()
            if data is None:
                break
            all_data.append(data)
        
        # 최신 length개만 반환
        return all_data[-length:] if len(all_data) >= length else all_data
    
    def clear(self):
        """버퍼 초기화"""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        # 통계 초기화
        self.stats = BufferStats()
        self.sampling_rate_history.clear()
        self.last_sample_time = None
        
        print("🗑️ 데이터 버퍼 초기화됨")
    
    def _update_stats(self, reading: SensorReading):
        """통계 업데이트"""
        current_time = time.time()
        
        # 샘플링 레이트 계산
        if self.last_sample_time is not None:
            dt = current_time - self.last_sample_time
            if dt > 0:
                current_rate = 1.0 / dt
                self.sampling_rate_history.append(current_rate)
                
                # 평균 샘플링 레이트 계산
                if self.sampling_rate_history:
                    self.stats.avg_sampling_rate = sum(self.sampling_rate_history) / len(self.sampling_rate_history)
        
        self.last_sample_time = current_time
        self.stats.total_samples += 1
        self.stats.last_sample_time = current_time
        
        # 버퍼 사용률 계산
        self.stats.buffer_usage = self.data_queue.qsize() / self.max_size
        self.stats.max_queue_usage = max(self.stats.max_queue_usage, self.stats.buffer_usage)
    
    def _check_buffer_status(self):
        """버퍼 상태 체크"""
        usage = self.stats.buffer_usage
        
        if usage >= self.critical_threshold:
            self.stats.buffer_warnings += 1
            self._trigger_callbacks('on_buffer_full', {'usage': usage})
        elif usage >= self.warning_threshold:
            self._trigger_callbacks('on_buffer_warning', {'usage': usage})
    
    def _trigger_callbacks(self, event: str, data: Any = None):
        """콜백 함수 실행"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"⚠️ 콜백 실행 오류 ({event}): {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """콜백 함수 등록"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            print(f"⚠️ 알 수 없는 이벤트: {event}")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            'total_samples': self.stats.total_samples,
            'dropped_samples': self.stats.dropped_samples,
            'buffer_usage': self.stats.buffer_usage,
            'avg_sampling_rate': self.stats.avg_sampling_rate,
            'max_queue_usage': self.stats.max_queue_usage,
            'buffer_warnings': self.stats.buffer_warnings,
            'queue_size': self.data_queue.qsize(),
            'max_size': self.max_size
        }
    
    def start_monitoring(self):
        """버퍼 모니터링 시작"""
        if self.running:
            return
        
        self.running = True
        self.buffer_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.buffer_thread.start()
        print("📊 버퍼 모니터링 시작됨")
    
    def stop_monitoring(self):
        """버퍼 모니터링 중지"""
        self.running = False
        if self.buffer_thread:
            self.buffer_thread.join(timeout=1.0)
        print("📊 버퍼 모니터링 중지됨")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.running:
            # 주기적으로 통계 출력
            if self.stats.total_samples > 0 and self.stats.total_samples % 100 == 0:
                stats = self.get_stats()
                print(f"📊 버퍼 상태: {stats['queue_size']}/{stats['max_size']} "
                      f"({stats['buffer_usage']*100:.1f}%) | "
                      f"샘플링 레이트: {stats['avg_sampling_rate']:.1f}Hz")
            
            time.sleep(1.0)  # 1초마다 체크
    
    def save_buffer_data(self, filepath: str, max_samples: int = 1000):
        """버퍼 데이터 저장"""
        data_list = []
        count = 0
        
        while count < max_samples:
            data = self.get_data_nowait()
            if data is None:
                break
            
            data_list.append(asdict(data))
            count += 1
        
        if data_list:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'total_samples': len(data_list),
                    'buffer_stats': self.get_stats(),
                    'data': data_list
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 버퍼 데이터 저장됨: {filepath} ({len(data_list)}개 샘플)")
            return True
        
        return False

class DataBufferManager:
    """데이터 버퍼 관리자"""
    
    def __init__(self):
        self.buffers: Dict[str, SignGloveDataBuffer] = {}
        self.default_buffer = None
        
        print("📊 데이터 버퍼 관리자 초기화됨")
    
    def create_buffer(self, 
                     name: str, 
                     max_size: int = 1000,
                     target_sampling_rate: float = 33.3) -> SignGloveDataBuffer:
        """새 버퍼 생성"""
        buffer = SignGloveDataBuffer(
            max_size=max_size,
            target_sampling_rate=target_sampling_rate
        )
        
        self.buffers[name] = buffer
        
        if self.default_buffer is None:
            self.default_buffer = buffer
        
        print(f"📊 버퍼 생성됨: {name} (최대 크기: {max_size})")
        return buffer
    
    def get_buffer(self, name: str) -> Optional[SignGloveDataBuffer]:
        """버퍼 가져오기"""
        return self.buffers.get(name)
    
    def get_default_buffer(self) -> Optional[SignGloveDataBuffer]:
        """기본 버퍼 가져오기"""
        return self.default_buffer
    
    def remove_buffer(self, name: str) -> bool:
        """버퍼 제거"""
        if name in self.buffers:
            buffer = self.buffers.pop(name)
            buffer.stop_monitoring()
            
            if self.default_buffer == buffer:
                self.default_buffer = list(self.buffers.values())[0] if self.buffers else None
            
            print(f"📊 버퍼 제거됨: {name}")
            return True
        
        return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """모든 버퍼 통계 반환"""
        return {name: buffer.get_stats() for name, buffer in self.buffers.items()}
    
    def start_all_monitoring(self):
        """모든 버퍼 모니터링 시작"""
        for buffer in self.buffers.values():
            buffer.start_monitoring()
    
    def stop_all_monitoring(self):
        """모든 버퍼 모니터링 중지"""
        for buffer in self.buffers.values():
            buffer.stop_monitoring()

def main():
    """테스트 함수"""
    print("SignGlove 데이터 버퍼 테스트")
    
    # 버퍼 생성
    buffer = SignGloveDataBuffer(max_size=100, target_sampling_rate=33.3)
    
    # 콜백 등록
    def on_warning(data):
        print(f"⚠️ 버퍼 경고: {data['usage']*100:.1f}% 사용")
    
    def on_full(data):
        print(f"🔴 버퍼 포화: {data['usage']*100:.1f}% 사용")
    
    buffer.register_callback('on_buffer_warning', on_warning)
    buffer.register_callback('on_buffer_full', on_full)
    
    # 모니터링 시작
    buffer.start_monitoring()
    
    # 테스트 데이터 생성 및 추가
    for i in range(150):  # 버퍼 크기보다 많은 데이터
        reading = SensorReading(
            timestamp_ms=int(time.time() * 1000),
            recv_timestamp_ms=int(time.time() * 1000),
            flex1=np.random.randint(200, 800),
            flex2=np.random.randint(200, 800),
            flex3=np.random.randint(200, 800),
            flex4=np.random.randint(200, 800),
            flex5=np.random.randint(200, 800),
            pitch=np.random.uniform(-180, 180),
            roll=np.random.uniform(-90, 90),
            yaw=np.random.uniform(-180, 180),
            sampling_hz=33.3
        )
        
        buffer.add_data(reading)
        time.sleep(0.03)  # 33.3Hz
    
    # 통계 출력
    stats = buffer.get_stats()
    print(f"\n📊 최종 통계: {stats}")
    
    # 모니터링 중지
    buffer.stop_monitoring()

if __name__ == "__main__":
    main()
