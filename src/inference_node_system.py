#!/usr/bin/env python3
"""
SignGlove 추론 노드 시스템
- ser.py의 키보드 입력 방식을 참고한 실시간 추론 시스템
- 키보드 제어로 추론 시작/중지, 모드 전환 등 가능
"""

import sys
import time
import threading
import numpy as np
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import os

# OS별 키보드 입력 모듈 임포트 (ser.py와 동일)
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

# 상위 디렉토리의 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from korean_composition_algorithm import KoreanComposition
from key_controlled_inference import AdvancedKeyController, InferenceState, InferenceResult

@dataclass
class InferenceNodeData:
    """추론 노드 데이터 구조"""
    timestamp_ms: int
    recv_timestamp_ms: int
    
    # 센서 데이터 (8개)
    flex1: int
    flex2: int
    flex3: int
    flex4: int
    flex5: int
    pitch: float
    roll: float
    yaw: float
    
    # 추론 결과
    predicted_class: Optional[str] = None
    confidence: float = 0.0
    probabilities: Optional[List[float]] = None
    
    # 메타데이터
    sampling_hz: float = 0.0
    node_id: str = "inference_node_001"

class InferenceNodeSystem:
    """SignGlove 추론 노드 시스템"""
    
    def __init__(self):
        print("🤖 SignGlove 추론 노드 시스템 초기화 중...")
        
        # 핵심 컴포넌트
        self.composer = KoreanComposition()
        self.key_controller = AdvancedKeyController(self.composer)
        
        # 추론 상태
        self.inference_active = False
        self.collecting_data = False
        self.current_mode = "idle"  # idle, collecting, inferring, composing
        
        # 데이터 버퍼
        self.data_queue: "queue.Queue[InferenceNodeData]" = queue.Queue(maxsize=1000)
        self.inference_buffer: List[InferenceNodeData] = []
        self.buffer_size = 80  # 2.4초 @ 33.3Hz
        
        # 통계
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'completed_words': 0,
            'completed_syllables': 0,
            'session_start': time.time(),
            'last_prediction_time': None,
        }
        
        # 설정
        self.settings = {
            'min_confidence': 0.7,
            'auto_complete_threshold': 0.8,
            'buffer_timeout': 5.0,  # 5초 후 자동 추론
            'realtime_display': True,
            'save_predictions': True,
        }
        
        # 한국어 클래스 정의 (ser.py와 동일)
        self.ksl_classes = {
            "consonants": ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"],
            "vowels": ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"],
        }
        self.all_classes = []
        for category in self.ksl_classes.values():
            self.all_classes.extend(category)
        
        # 모델 (Mock - 실제 모델로 교체 가능)
        self.model = None
        self.model_loaded = False
        
        # 파일 저장
        self.output_dir = Path("inference_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 키보드 핸들러 설정
        self.setup_keyboard_handlers()
        
        print("✅ 추론 노드 시스템 준비 완료!")
        self.show_usage_guide()
    
    def setup_keyboard_handlers(self):
        """키보드 핸들러 설정 (ser.py 방식)"""
        self.key_controller.add_custom_handler('space', self._handle_space, "데이터 수집 시작/중지")
        self.key_controller.add_custom_handler('enter', self._handle_enter, "추론 실행")
        self.key_controller.add_custom_handler('f1', self._toggle_inference, "추론 모드 토글")
        self.key_controller.add_custom_handler('f2', self._clear_buffer, "버퍼 초기화")
        self.key_controller.add_custom_handler('f3', self._show_stats, "통계 표시")
        self.key_controller.add_custom_handler('f4', self._toggle_realtime, "실시간 표시 토글")
        self.key_controller.add_custom_handler('f5', self._save_session, "세션 저장")
        self.key_controller.add_custom_handler('esc', self._handle_esc, "종료")
        self.key_controller.add_custom_handler('h', self._show_help, "도움말")
    
    def show_usage_guide(self):
        """사용법 가이드 표시 (ser.py 방식)"""
        print("\n" + "=" * 60)
        print("🤖 SignGlove 추론 노드 시스템")
        print("=" * 60)
        print("📋 조작 방법:")
        print("   SPACE: 데이터 수집 시작/중지")
        print("   ENTER: 추론 실행")
        print("   F1: 추론 모드 토글")
        print("   F2: 버퍼 초기화")
        print("   F3: 통계 표시")
        print("   F4: 실시간 표시 토글")
        print("   F5: 세션 저장")
        print("   H: 도움말")
        print("   ESC: 종료")
        print("")
        print("🎯 지원 클래스:")
        print("   자음 14개: ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
        print("   모음 10개: ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ")
        print("")
        print("💡 먼저 'SPACE' 키로 데이터 수집을 시작하세요!")
        print("=" * 60)
    
    def simulate_sensor_data(self) -> Optional[InferenceNodeData]:
        """센서 데이터 시뮬레이션 (실제 하드웨어 연동 시 교체)"""
        if not self.collecting_data:
            return None
        
        # 랜덤 센서 데이터 생성
        timestamp_ms = int(time.time() * 1000)
        
        # 플렉스 센서 데이터 (0-1023)
        flex_data = [
            np.random.randint(200, 800) for _ in range(5)
        ]
        
        # IMU 데이터 (오일러 각)
        imu_data = [
            np.random.uniform(-180, 180),  # pitch
            np.random.uniform(-90, 90),    # roll
            np.random.uniform(-180, 180),  # yaw
        ]
        
        return InferenceNodeData(
            timestamp_ms=timestamp_ms,
            recv_timestamp_ms=timestamp_ms,
            flex1=flex_data[0],
            flex2=flex_data[1],
            flex3=flex_data[2],
            flex4=flex_data[3],
            flex5=flex_data[4],
            pitch=imu_data[0],
            roll=imu_data[1],
            yaw=imu_data[2],
            sampling_hz=33.3
        )
    
    def mock_predict(self, sensor_data: np.ndarray) -> tuple:
        """Mock 추론 함수 (실제 모델로 교체)"""
        # 랜덤 예측 결과 생성
        class_idx = np.random.randint(0, len(self.all_classes))
        predicted_class = self.all_classes[class_idx]
        confidence = np.random.uniform(0.6, 0.95)
        
        # 확률 분포 생성
        probabilities = np.random.dirichlet(np.ones(len(self.all_classes)))
        probabilities[class_idx] = confidence  # 예측 클래스에 높은 확률
        
        return predicted_class, confidence, probabilities.tolist()
    
    def process_inference_data(self, data: InferenceNodeData):
        """추론 데이터 처리"""
        if not self.collecting_data:
            return
        
        # 센서 데이터를 numpy 배열로 변환
        sensor_array = np.array([
            data.flex1, data.flex2, data.flex3, data.flex4, data.flex5,
            data.pitch, data.roll, data.yaw
        ]).reshape(1, -1)
        
        # 추론 실행
        predicted_class, confidence, probabilities = self.mock_predict(sensor_array)
        
        # 결과 저장
        data.predicted_class = predicted_class
        data.confidence = confidence
        data.probabilities = probabilities
        
        # 버퍼에 추가
        self.inference_buffer.append(data)
        
        # 버퍼 크기 제한
        if len(self.inference_buffer) > self.buffer_size:
            self.inference_buffer.pop(0)
        
        # 실시간 표시
        if self.settings['realtime_display']:
            print(f"📊 {data.timestamp_ms}ms | "
                  f"예측: {predicted_class} ({confidence:.3f}) | "
                  f"버퍼: {len(self.inference_buffer)}/{self.buffer_size}")
        
        # 통계 업데이트
        self.stats['total_predictions'] += 1
        if confidence >= self.settings['min_confidence']:
            self.stats['successful_predictions'] += 1
        else:
            self.stats['failed_predictions'] += 1
        
        self.stats['last_prediction_time'] = time.time()
    
    def run_inference(self):
        """추론 실행"""
        if not self.inference_buffer:
            print("❌ 추론할 데이터가 없습니다. 먼저 데이터를 수집하세요.")
            return
        
        print(f"\n🔍 추론 실행 중... (버퍼 크기: {len(self.inference_buffer)})")
        
        # 최근 데이터로 추론
        recent_data = self.inference_buffer[-min(80, len(self.inference_buffer)):]
        
        # 센서 데이터 추출
        sensor_data = np.array([
            [d.flex1, d.flex2, d.flex3, d.flex4, d.flex5, d.pitch, d.roll, d.yaw]
            for d in recent_data
        ])
        
        # 최종 추론 (전체 시퀀스)
        predicted_class, confidence, probabilities = self.mock_predict(sensor_data.mean(axis=0))
        
        print(f"🎯 최종 예측: {predicted_class} (신뢰도: {confidence:.3f})")
        
        # 한글 조합에 추가
        if confidence >= self.settings['min_confidence']:
            result = self.composer.add_character(predicted_class)
            print(f"   조합 결과: {result.get('message', '처리됨')}")
            print(f"   현재 음절: '{result.get('current_syllable', '')}'")
            print(f"   현재 단어: '{result.get('current_word', '')}'")
            
            if result.get('can_complete', False):
                print("   ✅ 음절 완성 가능!")
        
        # 세션 저장
        if self.settings['save_predictions']:
            self._save_prediction(predicted_class, confidence, probabilities)
    
    def start_inference_loop(self):
        """추론 루프 시작"""
        print("\n🚀 추론 노드 시스템 시작...")
        
        try:
            # 키 리스너 시작 (별도 스레드)
            key_thread = threading.Thread(target=self.key_controller.start_inference, daemon=True)
            key_thread.start()
            
            # 메인 추론 루프
            while self.key_controller.running:
                # 센서 데이터 시뮬레이션
                if self.collecting_data:
                    data = self.simulate_sensor_data()
                    if data:
                        self.process_inference_data(data)
                
                # 자동 추론 (버퍼가 가득 찰 때)
                if (self.collecting_data and 
                    len(self.inference_buffer) >= self.buffer_size and
                    self.settings.get('auto_inference', False)):
                    self.run_inference()
                    self.inference_buffer.clear()
                
                time.sleep(0.03)  # 33.3Hz
                
        except KeyboardInterrupt:
            print("\n\n⏹️ 사용자에 의해 중단됨")
        finally:
            self.key_controller.stop_inference()
            print("🔚 추론 노드 시스템 종료")
    
    # 키보드 핸들러들
    def _handle_space(self):
        """SPACE 키: 데이터 수집 시작/중지"""
        self.collecting_data = not self.collecting_data
        status = "시작" if self.collecting_data else "중지"
        print(f"\n📡 데이터 수집 {status}")
        if self.collecting_data:
            print("   💡 ENTER를 눌러 추론을 실행하세요!")
    
    def _handle_enter(self):
        """ENTER 키: 추론 실행"""
        self.run_inference()
    
    def _toggle_inference(self):
        """F1 키: 추론 모드 토글"""
        self.inference_active = not self.inference_active
        status = "활성화" if self.inference_active else "비활성화"
        print(f"\n🔄 추론 모드 {status}")
    
    def _clear_buffer(self):
        """F2 키: 버퍼 초기화"""
        self.inference_buffer.clear()
        print("\n🗑️ 추론 버퍼 초기화됨")
    
    def _show_stats(self):
        """F3 키: 통계 표시"""
        runtime = time.time() - self.stats['session_start']
        success_rate = (self.stats['successful_predictions'] / 
                       max(1, self.stats['total_predictions']) * 100)
        
        print(f"\n📊 추론 노드 통계:")
        print(f"   실행 시간: {runtime:.1f}초")
        print(f"   총 예측: {self.stats['total_predictions']}개")
        print(f"   성공 예측: {self.stats['successful_predictions']}개")
        print(f"   실패 예측: {self.stats['failed_predictions']}개")
        print(f"   성공률: {success_rate:.1f}%")
        print(f"   완성된 단어: {self.stats['completed_words']}개")
        print(f"   완성된 음절: {self.stats['completed_syllables']}개")
        print(f"   현재 버퍼: {len(self.inference_buffer)}/{self.buffer_size}")
    
    def _toggle_realtime(self):
        """F4 키: 실시간 표시 토글"""
        self.settings['realtime_display'] = not self.settings['realtime_display']
        status = "활성화" if self.settings['realtime_display'] else "비활성화"
        print(f"\n🔄 실시간 표시 {status}")
    
    def _save_session(self):
        """F5 키: 세션 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"inference_session_{timestamp}.json"
        filepath = self.output_dir / filename
        
        session_data = {
            'timestamp': timestamp,
            'stats': self.stats,
            'settings': self.settings,
            'buffer_data': [asdict(d) for d in self.inference_buffer[-10:]],  # 최근 10개만
            'composition_state': self.composer.get_composition_state()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 세션 저장됨: {filepath}")
    
    def _handle_esc(self):
        """ESC 키: 종료"""
        print("\n👋 추론 노드 시스템 종료 요청")
        self.key_controller.stop_inference()
    
    def _show_help(self):
        """H 키: 도움말"""
        self.show_usage_guide()
    
    def _save_prediction(self, predicted_class: str, confidence: float, probabilities: List[float]):
        """예측 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"prediction_{timestamp}.json"
        filepath = self.output_dir / filename
        
        prediction_data = {
            'timestamp': timestamp,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'buffer_size': len(self.inference_buffer),
            'composition_state': self.composer.get_composition_state()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, indent=2, ensure_ascii=False)

def main():
    """메인 함수"""
    print("SignGlove 추론 노드 시스템을 시작합니다...")
    
    try:
        system = InferenceNodeSystem()
        system.start_inference_loop()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
