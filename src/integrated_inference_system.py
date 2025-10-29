#!/usr/bin/env python3
"""
SignGlove 통합 추론 시스템
- 추론 노드, 엔진, 데이터 버퍼를 통합한 완전한 시스템
- ser.py의 키보드 제어 방식을 적용한 실시간 추론
"""

import sys
import os
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

# 현재 디렉토리의 모듈 import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from korean_composition_algorithm import KoreanComposition
    from key_controlled_inference import AdvancedKeyController, InferenceState, InferenceResult
    from inference_engine import SignGloveInferenceEngine, ModelConfig, create_model_config
    from data_buffer import SignGloveDataBuffer, SensorReading, DataBufferManager
    from arduino_interface import SignGloveArduinoInterface, ArduinoConfig, ArduinoStatus
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    print(f"💡 현재 디렉토리: {current_dir}")
    print("💡 해결 방법: src 디렉토리에서 실행하거나 PYTHONPATH 설정")
    sys.exit(1)

# OS별 키보드 입력 모듈 임포트 (ser.py와 동일)
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

@dataclass
class SystemConfig:
    """시스템 설정"""
    model_type: str = 'bigru'
    model_path: Optional[str] = None
    buffer_size: int = 1000
    target_sampling_rate: float = 33.3
    min_confidence: float = 0.7
    auto_inference: bool = False
    realtime_display: bool = True
    save_predictions: bool = True
    
    # 아두이노 설정
    arduino_enabled: bool = True
    arduino_port: Optional[str] = None
    arduino_baudrate: int = 115200
    arduino_auto_detect: bool = True
    arduino_auto_reconnect: bool = True
    use_simulation: bool = False  # True면 시뮬레이션, False면 실제 아두이노

class IntegratedInferenceSystem:
    """SignGlove 통합 추론 시스템"""
    
    def __init__(self, config: SystemConfig):
        print("🚀 SignGlove 통합 추론 시스템 초기화 중...")
        
        self.config = config
        
        # 핵심 컴포넌트
        self.composer = KoreanComposition()
        self.key_controller = AdvancedKeyController(self.composer)
        self.buffer_manager = DataBufferManager()
        self.inference_engine = None
        self.arduino_interface = None
        
        # 시스템 상태
        self.running = False
        self.collecting_data = False
        self.inference_active = False
        self.initial_posture = None
        
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
        
        # 출력 디렉토리
        self.output_dir = Path("inference_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 초기화
        self._initialize_components()
        self._setup_keyboard_handlers()
        
        print("✅ 통합 추론 시스템 준비 완료!")
        self.show_usage_guide()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        # 데이터 버퍼 생성
        self.data_buffer = self.buffer_manager.create_buffer(
            name="main_buffer",
            max_size=self.config.buffer_size,
            target_sampling_rate=self.config.target_sampling_rate
        )
        
        # 추론 엔진 초기화
        model_config = create_model_config(
            model_type=self.config.model_type,
            model_path=self.config.model_path
        )
        self.inference_engine = SignGloveInferenceEngine(model_config)
        
        # 아두이노 인터페이스 초기화
        print("🔍 [DEBUG] 아두이노 인터페이스 초기화 확인")
        print(f"🔍 [DEBUG] arduino_enabled: {self.config.arduino_enabled}")
        print(f"🔍 [DEBUG] use_simulation: {self.config.use_simulation}")
        
        if self.config.arduino_enabled and not self.config.use_simulation:
            print("🔍 [DEBUG] 아두이노 인터페이스 생성 시작")
            arduino_config = ArduinoConfig(
                port=self.config.arduino_port,
                baudrate=self.config.arduino_baudrate,
                auto_detect=self.config.arduino_auto_detect,
                auto_reconnect=self.config.arduino_auto_reconnect
            )
            print(f"🔍 [DEBUG] 아두이노 설정: {arduino_config}")
            
            self.arduino_interface = SignGloveArduinoInterface(arduino_config)
            print("🔍 [DEBUG] 아두이노 인터페이스 생성 완료")
            
            # 아두이노 콜백 설정
            print("🔍 [DEBUG] 아두이노 콜백 설정")
            self.arduino_interface.register_callback('on_connected', self._on_arduino_connected)
            self.arduino_interface.register_callback('on_disconnected', self._on_arduino_disconnected)
            self.arduino_interface.register_callback('on_data_received', self._on_arduino_data_received)
            self.arduino_interface.register_callback('on_error', self._on_arduino_error)
            print("🔍 [DEBUG] 아두이노 콜백 설정 완료")
        else:
            print("🔍 [DEBUG] 아두이노 인터페이스 비활성화됨")
            print(f"🔍 [DEBUG] 이유: arduino_enabled={self.config.arduino_enabled}, use_simulation={self.config.use_simulation}")
        
        # 버퍼 콜백 설정
        self.data_buffer.register_callback('on_buffer_warning', self._on_buffer_warning)
        self.data_buffer.register_callback('on_buffer_full', self._on_buffer_full)
    
    def _setup_keyboard_handlers(self):
        """키보드 핸들러 설정 (ser.py 방식)"""
        # ser.py와 동일한 키 매핑
        self.key_controller.add_custom_handler('c', self._connect_arduino, "아두이노 연결")
        self.key_controller.add_custom_handler('n', self.start_collection, "새 수집 시작")
        self.key_controller.add_custom_handler('m', self.stop_collection, "수집 중지")
        self.key_controller.add_custom_handler('i', self.check_posture, "자세 확인")
        self.key_controller.add_custom_handler('s', self.set_posture, "자세 설정")
        self.key_controller.add_custom_handler('t', self.toggle_realtime, "실시간 표시 토글")
        self.key_controller.add_custom_handler('d', self.clear_buffers, "버퍼 초기화")
        self.key_controller.add_custom_handler('p', self.show_stats, "진행 상황 표시")
        self.key_controller.add_custom_handler('r', self.run_inference, "추론 실행")
        self.key_controller.add_custom_handler('w', self.complete_word, "단어 완성")
        self.key_controller.add_custom_handler('y', self.complete_syllable, "음절 완성")
        self.key_controller.add_custom_handler('a', self.show_arduino_status, "아두이노 상태")
        self.key_controller.add_custom_handler('q', self._handle_quit, "종료")
        self.key_controller.add_custom_handler('h', self.show_help, "도움말")
    
    def show_usage_guide(self):
        """사용법 가이드 표시 (ser.py 방식)"""
        print("\n" + "=" * 70)
        print("🚀 SignGlove 통합 추론 시스템")
        print("=" * 70)
        print("📋 조작 방법:")
        print("   C: 아두이노 연결")
        print("   N: 새 수집 시작")
        print("   M: 수집 중지")
        print("   I: 자세 확인")
        print("   S: 자세 설정")
        print("   T: 실시간 표시 토글")
        print("   D: 버퍼 초기화")
        print("   P: 진행 상황 표시")
        print("   R: 추론 실행")
        print("   W: 단어 완성")
        print("   Y: 음절 완성")
        print("   A: 아두이노 상태")
        print("   H: 도움말")
        print("   Q: 종료")
        print("")
        print("🎯 지원 클래스:")
        print("   자음 14개: ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
        print("   모음 10개: ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ")
        print("")
        print("💡 먼저 'SPACE' 키로 데이터 수집을 시작하세요!")
        print("=" * 70)
    
    def get_sensor_data(self) -> Optional[SensorReading]:
        """센서 데이터 가져오기 (아두이노 또는 시뮬레이션)"""
        if not self.collecting_data:
            return None
        
        # 아두이노에서 데이터 가져오기
        if self.arduino_interface and self.arduino_interface.status.connected:
            return self.arduino_interface.get_data_nowait()
        
        # 시뮬레이션 데이터 생성
        if self.config.use_simulation:
            return self._simulate_sensor_data()
        
        return None
    
    def _simulate_sensor_data(self) -> SensorReading:
        """센서 데이터 시뮬레이션"""
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
        
        return SensorReading(
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
    
    def process_sensor_data(self, reading: SensorReading):
        """센서 데이터 처리"""
        if not self.collecting_data:
            return
        
        # 데이터 버퍼에 추가
        success = self.data_buffer.add_data(reading)
        
        if not success:
            print("⚠️ 데이터 버퍼 포화 - 데이터 손실 발생")
        
        # 실시간 표시
        if self.config.realtime_display:
            print(f"📊 {reading.timestamp_ms}ms | "
                  f"Flex: {reading.flex1},{reading.flex2},{reading.flex3},{reading.flex4},{reading.flex5} | "
                  f"IMU: P:{reading.pitch:.1f},R:{reading.roll:.1f},Y:{reading.yaw:.1f}")
    
    def run_inference(self):
        """추론 실행"""
        # 최신 시퀀스 데이터 가져오기
        sequence_data = self.data_buffer.get_latest_sequence(80)
        
        if len(sequence_data) < 10:  # 최소 10개 샘플 필요
            print("❌ 추론할 데이터가 부족합니다. 더 많은 데이터를 수집하세요.")
            return
        
        print(f"\n🔍 추론 실행 중... (데이터: {len(sequence_data)}개 샘플)")
        
        # 센서 데이터를 numpy 배열로 변환
        sensor_data = np.array([
            [d.flex1, d.flex2, d.flex3, d.flex4, d.flex5, d.pitch, d.roll, d.yaw]
            for d in sequence_data
        ])
        
        # 추론 실행
        result = self.inference_engine.predict(sensor_data)
        
        print(f"🎯 예측 결과: {result.predicted_class} (신뢰도: {result.confidence:.3f})")
        print(f"⏱️ 처리 시간: {result.processing_time:.3f}초")
        
        # 신뢰도가 충분한 경우 한글 조합에 추가
        if result.confidence >= self.config.min_confidence:
            composition_result = self.composer.add_character(result.predicted_class)
            print(f"   조합 결과: {composition_result.get('message', '처리됨')}")
            print(f"   현재 음절: '{composition_result.get('current_syllable', '')}'")
            print(f"   현재 단어: '{composition_result.get('current_word', '')}'")
            
            if composition_result.get('can_complete', False):
                print("   ✅ 음절 완성 가능!")
        else:
            print(f"   ⚠️ 신뢰도 부족 ({result.confidence:.3f} < {self.config.min_confidence})")
        
        # 통계 업데이트
        self.stats['total_predictions'] += 1
        if result.confidence >= self.config.min_confidence:
            self.stats['successful_predictions'] += 1
        else:
            self.stats['failed_predictions'] += 1
        
        self.stats['last_prediction_time'] = time.time()
        
        # 예측 결과 저장
        if self.config.save_predictions:
            self._save_prediction(result)
    
    def start_system(self):
        """시스템 시작 (ser.py 방식)"""
        print("\n⏳ 키보드 입력 대기 중... (도움말은 위 참조)")
        
        try:
            # 버퍼 모니터링 시작
            self.data_buffer.start_monitoring()
            
            self.running = True
            
            # ser.py 방식의 메인 루프
            while self.running:
                # 키 입력 처리
                key = self.get_key()
                if key:
                    self.handle_key_input(key)
                
                # 센서 데이터 처리 (수집 중일 때만)
                if self.collecting_data:
                    sensor_data = self.get_sensor_data()
                    if sensor_data:
                        self.process_sensor_data(sensor_data)
                
                # 자동 추론 (설정된 경우)
                if (self.config.auto_inference and 
                    self.collecting_data and 
                    self.data_buffer.data_queue.qsize() >= 80):
                    self.run_inference()
                    # 추론 후 버퍼 일부 정리
                    for _ in range(40):  # 절반만 정리
                        self.data_buffer.get_data_nowait()
                
                time.sleep(0.01)  # ser.py와 동일한 간격
                
        except KeyboardInterrupt:
            if self.collecting_data:
                self.stop_collection()
            print("\n👋 프로그램을 종료합니다.")
        finally:
            self.stop_system()
    
    def stop_system(self):
        """시스템 중지"""
        self.running = False
        self.collecting_data = False
        self.data_buffer.stop_monitoring()
        if self.arduino_interface:
            self.arduino_interface.disconnect()
        print("🔚 SignGlove 통합 추론 시스템 종료")
    
    def get_key(self) -> str:
        """키 입력 받기 (ser.py 방식)"""
        if sys.platform == 'win32':
            if msvcrt.kbhit():
                try:
                    return msvcrt.getch().decode('utf-8').lower()
                except UnicodeDecodeError:
                    return ''
            return ""
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                import select
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch = sys.stdin.read(1)
                    return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ""
    
    def handle_key_input(self, key: str):
        """키 입력 처리 (ser.py 방식)"""
        if key == '\x03' or key == 'q':
            if self.collecting_data:
                self.stop_collection()
            print("\n👋 SignGlove 추론 시스템을 종료합니다.")
            self.running = False
        
        elif key == 'c':
            print("🔌 아두이노 연결 중...")
            if self._connect_arduino():
                print("✅ 연결 완료! 'N' 키로 수집을 시작하세요.")
            else:
                print("❌ 연결 실패. 아두이노와 케이블을 확인하세요.")
        
        elif key == 'n':
            if self.collecting_data:
                self.stop_collection()
            self.start_collection()
        
        elif key == 'm':
            if self.collecting_data:
                self.stop_collection()
            else:
                print("⚠️ 현재 수집 중이 아닙니다.")
        
        elif key == 'i':
            print("🧘 현재 자세가 초기 자세와 일치하는지 확인 중...")
            self.check_posture()
        
        elif key == 's':
            print("✨ 현재 자세를 초기 자세 기준으로 설정합니다...")
            self.set_posture()
        
        elif key == 't':
            self.config.realtime_display = not self.config.realtime_display
            if self.config.realtime_display:
                print("✅ 실시간 센서 값 출력이 활성화되었습니다.")
            else:
                print("❌ 실시간 센서 값 출력이 비활성화되었습니다.")
        
        elif key == 'd':
            self.clear_buffers()
        
        elif key == 'p':
            self.show_stats()
        
        elif key == 'r':
            self.run_inference()
        
        elif key == 'w':
            self.complete_word()
        
        elif key == 'y':
            self.complete_syllable()
        
        elif key == 'a':
            self.show_arduino_status()
        
        elif key == 'h':
            self.show_help()
        
        else:
            print(f"⚠️ 알 수 없는 키: {key.upper()}")
            print("💡 도움말: C(연결), N(새수집), M(중지), P(진행상황), Q(종료)")
    
    # ser.py 방식 키보드 핸들러들
    def start_collection(self):
        """N 키: 새 수집 시작"""
        self.collecting_data = True
        print("\n📡 데이터 수집 시작")
        print("💡 'M' 키로 수집을 중지하세요!")
    
    def stop_collection(self):
        """M 키: 수집 중지"""
        self.collecting_data = False
        print("\n⏹️ 데이터 수집 중지")
    
    def check_posture(self):
        """I 키: 자세 확인"""
        if not self.arduino_interface or not self.arduino_interface.status.connected:
            print("❌ 아두이노가 연결되지 않았습니다.")
            return
        
        # 최신 센서 데이터 가져오기
        reading = self.arduino_interface.get_data_nowait()
        if reading:
            print(f"📊 현재 센서 값:")
            print(f"   Flex: {reading.flex1},{reading.flex2},{reading.flex3},{reading.flex4},{reading.flex5}")
            print(f"   IMU: P:{reading.pitch:.1f},R:{reading.roll:.1f},Y:{reading.yaw:.1f}")
        else:
            print("⚠️ 센서 데이터가 없습니다.")
    
    def set_posture(self):
        """S 키: 자세 설정"""
        if not self.arduino_interface or not self.arduino_interface.status.connected:
            print("❌ 아두이노가 연결되지 않았습니다.")
            return
        
        # 최신 센서 데이터를 기준 자세로 설정
        reading = self.arduino_interface.get_data_nowait()
        if reading:
            self.initial_posture = reading
            print("✅ 현재 자세가 기준 자세로 설정되었습니다.")
            print(f"   기준값: Flex:{reading.flex1},{reading.flex2},{reading.flex3},{reading.flex4},{reading.flex5}")
        else:
            print("⚠️ 센서 데이터가 없습니다.")
    
    def clear_buffers(self):
        """D 키: 버퍼 초기화"""
        self.data_buffer.clear()
        print("🗑️ 데이터 버퍼 초기화됨")
    
    def show_stats(self):
        """P 키: 진행 상황 표시"""
        runtime = time.time() - self.stats['session_start']
        success_rate = (self.stats['successful_predictions'] / 
                       max(1, self.stats['total_predictions']) * 100)
        
        buffer_stats = self.data_buffer.get_stats()
        
        print(f"\n📊 진행 상황:")
        print(f"   실행 시간: {runtime:.1f}초")
        print(f"   총 예측: {self.stats['total_predictions']}개")
        print(f"   성공 예측: {self.stats['successful_predictions']}개")
        print(f"   성공률: {success_rate:.1f}%")
        print(f"   완성된 단어: {self.stats['completed_words']}개")
        print(f"   완성된 음절: {self.stats['completed_syllables']}개")
        print(f"   버퍼 사용률: {buffer_stats['buffer_usage']*100:.1f}%")
        print(f"   평균 샘플링 레이트: {buffer_stats['avg_sampling_rate']:.1f}Hz")
    
    def run_inference(self):
        """R 키: 추론 실행"""
        self.run_inference()
    
    def complete_word(self):
        """W 키: 단어 완성"""
        word = self.composer.complete_word()
        if word:
            print(f"\n🎉 단어 완성: '{word}'")
            self.stats['completed_words'] += 1
        else:
            print("\n❌ 완성할 단어가 없습니다")
    
    def complete_syllable(self):
        """Y 키: 음절 완성"""
        syllable = self.composer.complete_syllable()
        if syllable:
            print(f"\n✅ 음절 완성: '{syllable}'")
            self.stats['completed_syllables'] += 1
        else:
            print("\n❌ 완성할 음절이 없습니다")
    
    def show_arduino_status(self):
        """A 키: 아두이노 상태"""
        if not self.arduino_interface:
            print("❌ 아두이노 인터페이스가 초기화되지 않았습니다.")
            return
        
        status = self.arduino_interface.get_status()
        print(f"\n🔌 아두이노 상태:")
        print(f"   연결 상태: {'연결됨' if status['connected'] else '연결 안됨'}")
        print(f"   포트: {status['port'] or 'N/A'}")
        print(f"   총 샘플: {status['total_samples']:,}개")
        print(f"   오류 수: {status['error_count']}개")
        print(f"   연결 시도: {status['connection_attempts']}회")
        print(f"   큐 크기: {status['queue_size']}/{status['queue_maxsize']}")
        if status['last_error']:
            print(f"   마지막 오류: {status['last_error']}")
        if status['last_data_time']:
            last_data_age = time.time() - status['last_data_time']
            print(f"   마지막 데이터: {last_data_age:.1f}초 전")
    
    def show_help(self):
        """H 키: 도움말"""
        self.show_usage_guide()
    
    def toggle_realtime(self):
        """T 키: 실시간 표시 토글"""
        self.config.realtime_display = not self.config.realtime_display
        if self.config.realtime_display:
            print("✅ 실시간 센서 값 출력이 활성화되었습니다.")
        else:
            print("❌ 실시간 센서 값 출력이 비활성화되었습니다.")
    
    def _handle_quit(self):
        """Q 키: 종료"""
        print("\n👋 시스템 종료 요청")
        self.running = False
    
    
    def _on_buffer_warning(self, data):
        """버퍼 경고 콜백"""
        print(f"⚠️ 버퍼 경고: {data['usage']*100:.1f}% 사용")
    
    def _on_buffer_full(self, data):
        """버퍼 포화 콜백"""
        print(f"🔴 버퍼 포화: {data['usage']*100:.1f}% 사용 - 데이터 손실 위험!")
    
    def _connect_arduino(self):
        """C 키: 아두이노 연결"""
        print("🔍 [DEBUG] 아두이노 연결 요청")
        
        if not self.arduino_interface:
            print("❌ 아두이노 인터페이스가 초기화되지 않았습니다.")
            print("🔍 [DEBUG] 아두이노 인터페이스 상태: None")
            return
        
        print("🔍 [DEBUG] 아두이노 인터페이스 존재 확인")
        print(f"🔍 [DEBUG] 현재 연결 상태: {self.arduino_interface.status.connected}")
        
        if self.arduino_interface.status.connected:
            print("⚠️ 아두이노가 이미 연결되어 있습니다.")
            return
        
        print("🔌 아두이노 연결 중...")
        print("🔍 [DEBUG] connect() 메서드 호출")
        result = self.arduino_interface.connect()
        print(f"🔍 [DEBUG] connect() 결과: {result}")
        
        if result:
            print("✅ 아두이노 연결 성공!")
            print(f"🔍 [DEBUG] 최종 연결 상태: {self.arduino_interface.status.connected}")
        else:
            print("❌ 아두이노 연결 실패")
            print(f"🔍 [DEBUG] 연결 실패 후 상태: {self.arduino_interface.status.connected}")
            print(f"🔍 [DEBUG] 마지막 오류: {self.arduino_interface.status.last_error}")
    
    def _disconnect_arduino(self):
        """D 키: 아두이노 연결 해제"""
        if not self.arduino_interface:
            print("❌ 아두이노 인터페이스가 초기화되지 않았습니다.")
            return
        
        if not self.arduino_interface.status.connected:
            print("⚠️ 아두이노가 연결되어 있지 않습니다.")
            return
        
        print("🔌 아두이노 연결 해제 중...")
        self.arduino_interface.disconnect()
        print("✅ 아두이노 연결 해제됨")
    
    def _show_arduino_status(self):
        """F10 키: 아두이노 상태 표시"""
        if not self.arduino_interface:
            print("❌ 아두이노 인터페이스가 초기화되지 않았습니다.")
            return
        
        status = self.arduino_interface.get_status()
        print(f"\n🔌 아두이노 상태:")
        print(f"   연결 상태: {'연결됨' if status['connected'] else '연결 안됨'}")
        print(f"   포트: {status['port'] or 'N/A'}")
        print(f"   총 샘플: {status['total_samples']:,}개")
        print(f"   오류 수: {status['error_count']}개")
        print(f"   연결 시도: {status['connection_attempts']}회")
        print(f"   큐 크기: {status['queue_size']}/{status['queue_maxsize']}")
        if status['last_error']:
            print(f"   마지막 오류: {status['last_error']}")
        if status['last_data_time']:
            last_data_age = time.time() - status['last_data_time']
            print(f"   마지막 데이터: {last_data_age:.1f}초 전")
    
    def _on_arduino_connected(self, data):
        """아두이노 연결 콜백"""
        print(f"🎉 아두이노 연결됨: {data['port']}")
    
    def _on_arduino_disconnected(self, data):
        """아두이노 연결 해제 콜백"""
        print("🔌 아두이노 연결 해제됨")
    
    def _on_arduino_data_received(self, reading: SensorReading):
        """아두이노 데이터 수신 콜백"""
        if self.config.realtime_display:
            print(f"📊 아두이노 데이터: {reading.timestamp_ms}ms | "
                  f"Flex: {reading.flex1},{reading.flex2},{reading.flex3},{reading.flex4},{reading.flex5} | "
                  f"IMU: P:{reading.pitch:.1f},R:{reading.roll:.1f},Y:{reading.yaw:.1f}")
    
    def _on_arduino_error(self, data):
        """아두이노 오류 콜백"""
        print(f"❌ 아두이노 오류: {data['error']}")
    
    def _save_prediction(self, result):
        """예측 결과 저장"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"prediction_{timestamp}.json"
        filepath = self.output_dir / filename
        
        prediction_data = {
            'timestamp': timestamp,
            'predicted_class': result.predicted_class,
            'confidence': result.confidence,
            'probabilities': result.probabilities,
            'processing_time': result.processing_time,
            'model_type': result.model_type,
            'composition_state': self.composer.get_composition_state()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, indent=2, ensure_ascii=False)

def main():
    """메인 함수"""
    print("SignGlove 통합 추론 시스템을 시작합니다...")
    
    # 시스템 설정
    print("🔍 [DEBUG] 시스템 설정 생성")
    config = SystemConfig(
        model_type='bigru',
        model_path=None,  # 실제 모델 경로로 설정
        buffer_size=1000,
        target_sampling_rate=33.3,
        min_confidence=0.7,
        auto_inference=False,
        realtime_display=True,
        save_predictions=True,
        
        # 아두이노 설정
        arduino_enabled=True,
        arduino_port=None,  # 자동 탐지
        arduino_baudrate=115200,
        arduino_auto_detect=True,
        arduino_auto_reconnect=True,
        use_simulation=True  # 권한 문제로 인한 시뮬레이션 모드
    )
    
    print("🔍 [DEBUG] 시스템 설정 완료:")
    print(f"🔍 [DEBUG] arduino_enabled: {config.arduino_enabled}")
    print(f"🔍 [DEBUG] use_simulation: {config.use_simulation}")
    print(f"🔍 [DEBUG] arduino_port: {config.arduino_port}")
    print(f"🔍 [DEBUG] arduino_auto_detect: {config.arduino_auto_detect}")
    
    try:
        system = IntegratedInferenceSystem(config)
        system.start_system()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
