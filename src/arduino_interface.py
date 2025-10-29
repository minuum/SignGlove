#!/usr/bin/env python3
"""
SignGlove 아두이노 인터페이스
- ser.py의 아두이노 연결 방식을 참고한 실제 하드웨어 연동
- 시리얼 통신을 통한 센서 데이터 수신
"""

import sys
import os
import time
import serial
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json

# 현재 디렉토리의 모듈 import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_buffer import SensorReading

@dataclass
class ArduinoConfig:
    """아두이노 설정"""
    port: Optional[str] = None
    baudrate: int = 115200
    timeout: float = 1.0
    auto_detect: bool = True
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 2.0

@dataclass
class ArduinoStatus:
    """아두이노 상태"""
    connected: bool = False
    port: Optional[str] = None
    last_data_time: Optional[float] = None
    total_samples: int = 0
    error_count: int = 0
    connection_attempts: int = 0
    last_error: Optional[str] = None

class SignGloveArduinoInterface:
    """SignGlove 아두이노 인터페이스"""
    
    def __init__(self, config: ArduinoConfig):
        self.config = config
        self.status = ArduinoStatus()
        
        # 시리얼 통신
        self.serial_port: Optional[serial.Serial] = None
        self.serial_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 데이터 큐
        self.data_queue: "queue.Queue[SensorReading]" = queue.Queue(maxsize=1000)
        
        # 콜백 함수들
        self.callbacks = {
            'on_connected': [],
            'on_disconnected': [],
            'on_data_received': [],
            'on_error': []
        }
        
        # 통신 설정 (더 많은 패턴 추가)
        self.arduino_patterns = ['usbmodem', 'usbserial', 'ttyUSB', 'ttyACM', 'COM', 'ttyS', 'USB']
        
        print("🔌 SignGlove 아두이노 인터페이스 초기화됨")
    
    def register_callback(self, event: str, callback: Callable):
        """콜백 함수 등록"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            print(f"⚠️ 알 수 없는 이벤트: {event}")
    
    def _trigger_callbacks(self, event: str, data: Any = None):
        """콜백 함수 실행"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"⚠️ 콜백 실행 오류 ({event}): {e}")
    
    def find_arduino_port(self) -> Optional[str]:
        """아두이노 포트 자동 탐지 (ser.py 방식 + 개선된 맥 지원)"""
        print("🔍 [DEBUG] 포트 탐지 시작")
        
        try:
            import serial.tools.list_ports
            import platform
            
            # 1. 기본 시리얼 포트 탐색
            ports = serial.tools.list_ports.comports()
            print(f"🔍 [DEBUG] 발견된 포트 수: {len(ports)}")
            
            for i, port in enumerate(ports):
                port_name = port.device.lower()
                print(f"🔍 [DEBUG] 포트 {i+1}: {port.device} ({port.description})")
                print(f"🔍 [DEBUG] 포트명 소문자: {port_name}")
                print(f"🔍 [DEBUG] 패턴 매칭: {[p.lower() for p in self.arduino_patterns]}")
                
                if any(p.lower() in port_name for p in self.arduino_patterns):
                    # 권한 확인
                    try:
                        test_file = open(port.device, 'r')
                        test_file.close()
                        print(f"🔍 아두이노 포트 발견: {port.device} ({port.description})")
                        return port.device
                    except PermissionError:
                        print(f"🔍 [DEBUG] 포트 {port.device}는 권한이 없어서 건너뜀")
                        continue
                    except Exception as e:
                        print(f"🔍 [DEBUG] 포트 {port.device} 테스트 실패: {e}")
                        continue
                else:
                    print(f"🔍 [DEBUG] 포트 {port.device}는 아두이노 패턴과 일치하지 않음")
            
            # 2. macOS 전용 탐색 (더 포괄적)
            if platform.system() == "Darwin":
                print("🔍 [DEBUG] macOS 전용 탐색 시작")
                
                # cu.* 포트들 확인
                for i in range(1, 20):  # 더 많은 범위 확인
                    for prefix in ["cu.usbmodem", "cu.usbserial", "cu.SLAB_USBtoUART"]:
                        p = f"/dev/{prefix}{i}"
                        print(f"🔍 [DEBUG] 확인 중: {p}")
                        if Path(p).exists():
                            print(f"🔍 [DEBUG] macOS 포트 발견: {p}")
                            return p
                
                # tty.* 포트들도 확인
                for i in range(1, 20):
                    for prefix in ["tty.usbmodem", "tty.usbserial", "tty.SLAB_USBtoUART"]:
                        p = f"/dev/{prefix}{i}"
                        print(f"🔍 [DEBUG] 확인 중: {p}")
                        if Path(p).exists():
                            print(f"🔍 [DEBUG] macOS tty 포트 발견: {p}")
                            return p
                
                # 일반적인 아두이노 포트들 확인
                common_ports = [
                    "/dev/cu.usbmodem*",
                    "/dev/cu.usbserial*", 
                    "/dev/tty.usbmodem*",
                    "/dev/tty.usbserial*",
                    "/dev/cu.SLAB_USBtoUART*",
                    "/dev/tty.SLAB_USBtoUART*"
                ]
                
                import glob
                for pattern in common_ports:
                    matches = glob.glob(pattern)
                    for match in matches:
                        print(f"🔍 [DEBUG] glob 패턴 매치: {match}")
                        if Path(match).exists():
                            print(f"🔍 [DEBUG] macOS glob 포트 발견: {match}")
                            return match
            
            # 3. Linux 전용 추가 탐색
            elif platform.system() == "Linux":
                print("🔍 [DEBUG] Linux 전용 탐색 시작")
                
                # /dev/serial/by-id/ 확인
                serial_by_id = Path("/dev/serial/by-id")
                if serial_by_id.exists():
                    print("🔍 [DEBUG] /dev/serial/by-id 디렉토리 확인")
                    for link in serial_by_id.iterdir():
                        if link.is_symlink():
                            target = link.resolve()
                            print(f"🔍 [DEBUG] 심볼릭 링크: {link} -> {target}")
                            if any(p.lower() in str(target).lower() for p in ['arduino', 'usb', 'serial']):
                                print(f"🔍 [DEBUG] 아두이노 관련 링크 발견: {link} -> {target}")
                                return str(target)
                
                # /dev/serial/by-path/ 확인
                serial_by_path = Path("/dev/serial/by-path")
                if serial_by_path.exists():
                    print("🔍 [DEBUG] /dev/serial/by-path 디렉토리 확인")
                    for link in serial_by_path.iterdir():
                        if link.is_symlink():
                            target = link.resolve()
                            print(f"🔍 [DEBUG] 경로 링크: {link} -> {target}")
                            if any(p.lower() in str(target).lower() for p in ['usb', 'serial']):
                                print(f"🔍 [DEBUG] USB 관련 링크 발견: {link} -> {target}")
                                return str(target)
            
            print("🔍 [DEBUG] 아두이노 포트를 찾을 수 없음")
            print("💡 권한 문제 해결 방법:")
            print("   1. sudo usermod -a -G dialout $USER")
            print("   2. 로그아웃 후 다시 로그인")
            print("   3. 또는 sudo chmod 666 /dev/ttyS*")
            return None
            
        except Exception as e:
            print(f"❌ 포트 탐지 오류: {e}")
            print(f"🔍 [DEBUG] 포트 탐지 오류 상세: {type(e).__name__}: {e}")
            return None
    
    def connect(self, port: Optional[str] = None) -> bool:
        """아두이노 연결 (ser.py 방식)"""
        print("🔍 [DEBUG] 아두이노 연결 시작")
        
        try:
            # 기존 연결 종료
            if self.serial_port and self.serial_port.is_open:
                print("🔍 [DEBUG] 기존 연결 종료 중...")
                self.disconnect()
            
            # 포트 결정
            if port is None:
                print("🔍 [DEBUG] 포트 자동 탐지 시작")
                if self.config.auto_detect:
                    port = self.find_arduino_port()
                    print(f"🔍 [DEBUG] 포트 탐지 결과: {port}")
                    if not port:
                        print("❌ 아두이노 포트를 찾을 수 없습니다.")
                        return False
                else:
                    print("❌ 포트가 지정되지 않았습니다.")
                    return False
            else:
                print(f"🔍 [DEBUG] 지정된 포트 사용: {port}")
            
            print(f"🔌 {port}에 연결 중... (보드레이트: {self.config.baudrate})")
            
            # 시리얼 포트 열기
            print("🔍 [DEBUG] 시리얼 포트 열기 시도")
            self.serial_port = serial.Serial(
                port, 
                self.config.baudrate, 
                timeout=self.config.timeout
            )
            print("🔍 [DEBUG] 시리얼 포트 열기 성공")
            
            # 아두이노 리셋 대기
            print("🔍 [DEBUG] 아두이노 리셋 대기 (2초)")
            time.sleep(2)
            
            # 통신 테스트
            print("🔍 [DEBUG] 통신 테스트 시작")
            if not self._test_communication():
                print("❌ 아두이노 통신 테스트 실패")
                self.disconnect()
                return False
            print("🔍 [DEBUG] 통신 테스트 성공")
            
            # 상태 업데이트
            print("🔍 [DEBUG] 상태 업데이트 중")
            self.status.connected = True
            self.status.port = port
            self.status.connection_attempts += 1
            self.status.last_error = None
            
            print(f"✅ 아두이노 연결 성공: {port}")
            
            # 데이터 수신 스레드 시작
            print("🔍 [DEBUG] 데이터 수신 스레드 시작")
            self.start_data_reception()
            
            # 연결 콜백 실행
            print("🔍 [DEBUG] 연결 콜백 실행")
            self._trigger_callbacks('on_connected', {'port': port})
            
            print("🔍 [DEBUG] 아두이노 연결 완료")
            return True
            
        except Exception as e:
            error_msg = f"아두이노 연결 실패: {e}"
            print(f"❌ {error_msg}")
            print(f"🔍 [DEBUG] 연결 실패 상세: {type(e).__name__}: {e}")
            self.status.last_error = error_msg
            self.status.error_count += 1
            self._trigger_callbacks('on_error', {'error': error_msg})
            return False
    
    def disconnect(self):
        """아두이노 연결 해제"""
        try:
            # 데이터 수신 중지
            self.stop_data_reception()
            
            # 시리얼 포트 닫기
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                self.serial_port = None
            
            # 상태 업데이트
            self.status.connected = False
            self.status.port = None
            
            print("🔌 아두이노 연결 해제됨")
            
            # 연결 해제 콜백 실행
            self._trigger_callbacks('on_disconnected', {})
            
        except Exception as e:
            print(f"⚠️ 연결 해제 오류: {e}")
    
    def _test_communication(self) -> bool:
        """통신 테스트 (ser.py 방식)"""
        print("🔍 [DEBUG] 통신 테스트 시작")
        
        try:
            print("🔍 [DEBUG] 시리얼 버퍼 초기화")
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            print("🔍 [DEBUG] 'header' 명령 전송")
            self.serial_port.write(b"header\n")
            time.sleep(0.5)
            
            print("🔍 [DEBUG] 응답 대기 중...")
            for attempt in range(3):
                print(f"🔍 [DEBUG] 시도 {attempt + 1}/3")
                if self.serial_port.in_waiting > 0:
                    response = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    print(f"🔍 [DEBUG] 수신된 응답: '{response}'")
                    
                    if 'timestamp' in response.lower() and 'flex' in response.lower():
                        print(f"📋 헤더 확인: {response}")
                        print("🔍 [DEBUG] 통신 테스트 성공")
                        return True
                    else:
                        print(f"🔍 [DEBUG] 응답이 헤더 형식이 아님: '{response}'")
                else:
                    print(f"🔍 [DEBUG] 대기 중인 데이터 없음 (in_waiting: {self.serial_port.in_waiting})")
                
                time.sleep(0.3)
            
            print("🔍 [DEBUG] 통신 테스트 실패 - 유효한 응답 없음")
            return False
            
        except Exception as e:
            print(f"⚠️ 통신 테스트 오류: {e}")
            print(f"🔍 [DEBUG] 통신 테스트 오류 상세: {type(e).__name__}: {e}")
            return False
    
    def start_data_reception(self):
        """데이터 수신 시작"""
        if self.serial_thread and self.serial_thread.is_alive():
            self.stop_event.set()
            self.serial_thread.join(timeout=2)
        
        self.stop_event.clear()
        self.serial_thread = threading.Thread(target=self._data_reception_worker, daemon=True)
        self.serial_thread.start()
        print("📡 데이터 수신 스레드 시작됨")
    
    def stop_data_reception(self):
        """데이터 수신 중지"""
        self.stop_event.set()
        if self.serial_thread:
            self.serial_thread.join(timeout=2)
        print("📡 데이터 수신 스레드 중지됨")
    
    def _data_reception_worker(self):
        """데이터 수신 워커 (ser.py 방식)"""
        last_arduino_ms = None
        
        while not self.stop_event.is_set():
            try:
                if not self.serial_port or not self.serial_port.is_open:
                    break
                
                if self.serial_port.in_waiting > 0:
                    raw = self.serial_port.readline()
                    try:
                        line = raw.decode('utf-8', errors='ignore').strip()
                    except Exception:
                        continue
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    # CSV 파싱: timestamp,pitch,roll,yaw,accel_x,accel_y,accel_z,flex1..5
                    parts = line.split(',')
                    if len(parts) != 12:
                        continue
                    
                    try:
                        recv_time_ms = int(time.time() * 1000)
                        arduino_ts = int(float(parts[0]))
                        
                        # 샘플링 레이트 계산
                        sampling_hz = 0.0
                        if last_arduino_ms is not None:
                            dt_ms = max(1, arduino_ts - last_arduino_ms)
                            sampling_hz = 1000.0 / dt_ms
                        last_arduino_ms = arduino_ts
                        
                        # 센서 데이터 파싱
                        reading = SensorReading(
                            timestamp_ms=arduino_ts,
                            recv_timestamp_ms=recv_time_ms,
                            pitch=float(parts[1]),
                            roll=float(parts[2]),
                            yaw=float(parts[3]),
                            flex1=int(parts[7]),
                            flex2=int(parts[8]),
                            flex3=int(parts[9]),
                            flex4=int(parts[10]),
                            flex5=int(parts[11]),
                            sampling_hz=sampling_hz
                        )
                        
                        # 데이터 큐에 추가
                        if not self.data_queue.full():
                            self.data_queue.put(reading)
                            self.status.total_samples += 1
                            self.status.last_data_time = time.time()
                            
                            # 데이터 수신 콜백 실행
                            self._trigger_callbacks('on_data_received', reading)
                        else:
                            print("⚠️ 데이터 큐 포화 - 데이터 손실")
                    
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ 데이터 파싱 오류: {line} → {e}")
                        self.status.error_count += 1
                
                time.sleep(0.001)  # 1ms 대기
                
            except Exception as e:
                error_msg = f"데이터 수신 오류: {e}"
                print(f"❌ {error_msg}")
                self.status.last_error = error_msg
                self.status.error_count += 1
                self._trigger_callbacks('on_error', {'error': error_msg})
                break
    
    def get_data(self, timeout: float = 0.1) -> Optional[SensorReading]:
        """센서 데이터 가져오기"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_data_nowait(self) -> Optional[SensorReading]:
        """센서 데이터 즉시 가져오기 (논블로킹)"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_sequence(self, length: int) -> List[SensorReading]:
        """최신 시퀀스 데이터 가져오기"""
        sequence = []
        for _ in range(length):
            data = self.get_data_nowait()
            if data is None:
                break
            sequence.append(data)
        return sequence
    
    def clear_data_queue(self):
        """데이터 큐 초기화"""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        print("🗑️ 데이터 큐 초기화됨")
    
    def get_status(self) -> Dict[str, Any]:
        """아두이노 상태 반환"""
        return {
            'connected': self.status.connected,
            'port': self.status.port,
            'total_samples': self.status.total_samples,
            'error_count': self.status.error_count,
            'connection_attempts': self.status.connection_attempts,
            'last_error': self.status.last_error,
            'last_data_time': self.status.last_data_time,
            'queue_size': self.data_queue.qsize(),
            'queue_maxsize': self.data_queue.maxsize
        }
    
    def send_command(self, command: str) -> bool:
        """아두이노에 명령 전송"""
        if not self.status.connected or not self.serial_port:
            print("❌ 아두이노가 연결되지 않았습니다.")
            return False
        
        try:
            self.serial_port.write(f"{command}\n".encode())
            print(f"📤 명령 전송: {command}")
            return True
        except Exception as e:
            print(f"❌ 명령 전송 실패: {e}")
            return False
    
    def auto_reconnect(self):
        """자동 재연결"""
        if not self.config.auto_reconnect:
            return False
        
        if self.status.connection_attempts >= self.config.max_reconnect_attempts:
            print(f"❌ 최대 재연결 시도 횟수 초과 ({self.config.max_reconnect_attempts})")
            return False
        
        print(f"🔄 자동 재연결 시도 중... ({self.status.connection_attempts + 1}/{self.config.max_reconnect_attempts})")
        time.sleep(self.config.reconnect_delay)
        
        return self.connect()

def main():
    """테스트 함수"""
    print("SignGlove 아두이노 인터페이스 테스트")
    
    # 설정
    config = ArduinoConfig(
        auto_detect=True,
        auto_reconnect=True,
        max_reconnect_attempts=3
    )
    
    # 인터페이스 생성
    arduino = SignGloveArduinoInterface(config)
    
    # 콜백 등록
    def on_connected(data):
        print(f"🎉 아두이노 연결됨: {data['port']}")
    
    def on_disconnected(data):
        print("🔌 아두이노 연결 해제됨")
    
    def on_data_received(reading):
        print(f"📊 데이터 수신: {reading.timestamp_ms}ms | "
              f"Flex: {reading.flex1},{reading.flex2},{reading.flex3},{reading.flex4},{reading.flex5} | "
              f"IMU: P:{reading.pitch:.1f},R:{reading.roll:.1f},Y:{reading.yaw:.1f}")
    
    def on_error(data):
        print(f"❌ 오류: {data['error']}")
    
    arduino.register_callback('on_connected', on_connected)
    arduino.register_callback('on_disconnected', on_disconnected)
    arduino.register_callback('on_data_received', on_data_received)
    arduino.register_callback('on_error', on_error)
    
    # 연결 시도
    if arduino.connect():
        print("✅ 아두이노 연결 성공!")
        
        # 데이터 수신 테스트
        print("📡 데이터 수신 테스트 중...")
        for i in range(10):
            data = arduino.get_data(timeout=1.0)
            if data:
                print(f"   샘플 {i+1}: {data.timestamp_ms}ms")
            else:
                print(f"   샘플 {i+1}: 데이터 없음")
        
        # 상태 확인
        status = arduino.get_status()
        print(f"📊 아두이노 상태: {status}")
        
        # 연결 해제
        arduino.disconnect()
    else:
        print("❌ 아두이노 연결 실패")

if __name__ == "__main__":
    main()
