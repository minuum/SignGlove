#!/usr/bin/env python3
"""
SignGlove 데이터 클라이언트
노트북에서 실행되어 아두이노와 서버 사이의 데이터 전송을 담당

작성자: 양동건
역할: 아두이노 UART/WiFi 수신 → 서버 HTTP 전송
"""

import asyncio
import json
import time
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import requests
import serial
import serial.tools.list_ports

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SignGloveClient:
    """SignGlove 데이터 클라이언트"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """초기화"""
        self.server_url = server_url
        self.serial_conn = None
        self.is_connected = False
        self.current_session = None
        
        # 설정
        self.arduino_port = None
        self.baudrate = 115200
        self.timeout = 1.0
        
        logger.info("SignGlove 클라이언트 초기화 완료")
    
    def find_arduino_port(self) -> Optional[str]:
        """아두이노 포트 자동 검색"""
        logger.info("아두이노 포트 검색 중...")
        
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        
        for port in ports:
            # 아두이노 관련 키워드로 필터링
            port_info = str(port).lower()
            if any(keyword in port_info for keyword in ['arduino', 'ch340', 'cp210', 'ftdi', 'usb']):
                arduino_ports.append(port.device)
                logger.info(f"아두이노 포트 발견: {port.device} - {port.description}")
        
        if not arduino_ports:
            logger.error("아두이노 포트를 찾을 수 없습니다.")
            return None
        
        if len(arduino_ports) == 1:
            return arduino_ports[0]
        
        # 여러 포트가 있는 경우 사용자 선택
        print("\n여러 아두이노 포트가 발견되었습니다:")
        for i, port in enumerate(arduino_ports, 1):
            print(f"  {i}. {port}")
        
        while True:
            try:
                choice = int(input(f"포트를 선택하세요 (1-{len(arduino_ports)}): ")) - 1
                if 0 <= choice < len(arduino_ports):
                    return arduino_ports[choice]
                else:
                    print("올바른 번호를 입력하세요.")
            except ValueError:
                print("숫자를 입력하세요.")
    
    def connect_arduino(self, port: str = None) -> bool:
        """아두이노 연결"""
        try:
            if port is None:
                port = self.find_arduino_port()
                if port is None:
                    return False
            
            self.arduino_port = port
            
            logger.info(f"아두이노 연결 시도: {port}")
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # 연결 확인을 위한 대기
            time.sleep(2)
            
            # 테스트 데이터 읽기
            if self.test_arduino_communication():
                self.is_connected = True
                logger.info("아두이노 연결 성공!")
                return True
            else:
                logger.error("아두이노 통신 테스트 실패")
                return False
                
        except Exception as e:
            logger.error(f"아두이노 연결 실패: {e}")
            return False
    
    def test_arduino_communication(self) -> bool:
        """아두이노 통신 테스트"""
        try:
            # 몇 줄 읽어서 데이터 형식 확인
            for _ in range(5):
                line = self.serial_conn.readline().decode().strip()
                if line:
                    logger.info(f"테스트 데이터: {line[:100]}...")  # 처음 100자만
                    
                    # JSON 형식인지 확인
                    try:
                        json.loads(line)
                        logger.info("JSON 형식 데이터 수신 확인")
                        return True
                    except json.JSONDecodeError:
                        # CSV 형식일 수도 있음
                        if ',' in line and len(line.split(',')) > 10:
                            logger.info("CSV 형식 데이터 수신 확인")
                            return True
            
            logger.warning("올바른 형식의 데이터를 받지 못했습니다.")
            return False
            
        except Exception as e:
            logger.error(f"통신 테스트 오류: {e}")
            return False
    
    def start_experiment_session(self, class_label: str, **kwargs) -> bool:
        """실험 세션 시작"""
        try:
            session_data = {
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "performer_id": kwargs.get("performer_id", "donggeon"),
                "class_label": class_label,
                "category": self.get_class_category(class_label),
                "target_samples": kwargs.get("target_samples", 60),
                "duration_per_sample": kwargs.get("duration_per_sample", 5),
                "sampling_rate": kwargs.get("sampling_rate", 20),
                "storage_strategy": kwargs.get("storage_strategy", "both")
            }
            
            logger.info(f"실험 세션 시작 요청: {class_label}")
            response = requests.post(
                f"{self.server_url}/experiment/start",
                json=session_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.current_session = session_data
                logger.info(f"실험 세션 시작 성공: {result['session_id']}")
                return True
            else:
                logger.error(f"실험 세션 시작 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"실험 세션 시작 오류: {e}")
            return False
    
    def get_class_category(self, class_label: str) -> str:
        """클래스 카테고리 반환"""
        consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        if class_label in consonants:
            return "consonant"
        elif class_label in vowels:
            return "vowel"
        elif class_label in numbers:
            return "number"
        else:
            return "unknown"
    
    def read_arduino_data(self) -> Optional[Dict]:
        """아두이노에서 데이터 읽기"""
        try:
            if not self.serial_conn or not self.is_connected:
                return None
            
            line = self.serial_conn.readline().decode().strip()
            if not line:
                return None
            
            # JSON 형식 파싱 시도
            try:
                data = json.loads(line)
                return self.convert_to_server_format(data)
            except json.JSONDecodeError:
                # CSV 형식 파싱 시도
                return self.parse_csv_format(line)
                
        except Exception as e:
            logger.error(f"아두이노 데이터 읽기 오류: {e}")
            return None
    
    def parse_csv_format(self, line: str) -> Optional[Dict]:
        """CSV 형식 데이터 파싱"""
        try:
            parts = line.split(',')
            if len(parts) < 13:
                return None
            
            data = {
                "device_id": parts[1] if len(parts) > 1 else "ARDUINO_001",
                "timestamp": datetime.now().isoformat(),
                "flex_sensors": {
                    "flex_1": float(parts[2]),
                    "flex_2": float(parts[3]),
                    "flex_3": float(parts[4]),
                    "flex_4": float(parts[5]),
                    "flex_5": float(parts[6])
                },
                "gyro_data": {
                    "gyro_x": float(parts[7]),
                    "gyro_y": float(parts[8]),
                    "gyro_z": float(parts[9]),
                    "accel_x": float(parts[10]),
                    "accel_y": float(parts[11]),
                    "accel_z": float(parts[12])
                },
                "battery_level": float(parts[13]) if len(parts) > 13 else 95.0,
                "signal_strength": int(parts[14]) if len(parts) > 14 else -50
            }
            
            return data
            
        except Exception as e:
            logger.error(f"CSV 파싱 오류: {e}")
            return None
    
    def convert_to_server_format(self, arduino_data: Dict) -> Dict:
        """아두이노 데이터를 서버 형식으로 변환"""
        try:
            # 이미 올바른 형식인 경우
            if "flex_sensors" in arduino_data and "gyro_data" in arduino_data:
                return arduino_data
            
            # 단순 형식인 경우 변환
            converted = {
                "device_id": arduino_data.get("device_id", "ARDUINO_001"),
                "timestamp": arduino_data.get("timestamp", datetime.now().isoformat()),
                "flex_sensors": {
                    "flex_1": arduino_data.get("flex_1", 0),
                    "flex_2": arduino_data.get("flex_2", 0),
                    "flex_3": arduino_data.get("flex_3", 0),
                    "flex_4": arduino_data.get("flex_4", 0),
                    "flex_5": arduino_data.get("flex_5", 0)
                },
                "gyro_data": {
                    "gyro_x": arduino_data.get("gyro_x", 0),
                    "gyro_y": arduino_data.get("gyro_y", 0),
                    "gyro_z": arduino_data.get("gyro_z", 0),
                    "accel_x": arduino_data.get("accel_x", 0),
                    "accel_y": arduino_data.get("accel_y", 0),
                    "accel_z": arduino_data.get("accel_z", 0)
                },
                "battery_level": arduino_data.get("battery_level", 95.0),
                "signal_strength": arduino_data.get("signal_strength", -50)
            }
            
            return converted
            
        except Exception as e:
            logger.error(f"데이터 형식 변환 오류: {e}")
            return arduino_data
    
    def send_data_to_server(self, sensor_data: Dict) -> bool:
        """서버로 센서 데이터 전송"""
        try:
            response = requests.post(
                f"{self.server_url}/data/sensor",
                json=sensor_data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"데이터 전송 성공: {result.get('sample_id', 'N/A')}")
                return True
            else:
                logger.error(f"데이터 전송 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"서버 전송 오류: {e}")
            return False
    
    def complete_sample(self) -> bool:
        """샘플 완료 알림"""
        try:
            response = requests.post(
                f"{self.server_url}/sample/complete",
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"샘플 완료: {result.get('message', '')}")
                return True
            else:
                logger.error(f"샘플 완료 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"샘플 완료 오류: {e}")
            return False
    
    def stop_experiment(self) -> bool:
        """실험 세션 종료"""
        try:
            response = requests.post(
                f"{self.server_url}/experiment/stop",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"실험 종료: {result.get('message', '')}")
                self.current_session = None
                return True
            else:
                logger.error(f"실험 종료 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"실험 종료 오류: {e}")
            return False
    
    def get_server_status(self) -> Optional[Dict]:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"상태 확인 실패: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"상태 확인 오류: {e}")
            return None
    
    async def run_data_collection(self, class_label: str, duration_seconds: int = 300):
        """데이터 수집 실행"""
        logger.info(f"데이터 수집 시작: {class_label} ({duration_seconds}초)")
        
        try:
            # 1. 실험 세션 시작
            if not self.start_experiment_session(class_label):
                logger.error("실험 세션 시작 실패")
                return False
            
            # 2. 데이터 수집 루프
            start_time = time.time()
            sample_count = 0
            last_status_time = start_time
            
            while time.time() - start_time < duration_seconds:
                # 아두이노에서 데이터 읽기
                sensor_data = self.read_arduino_data()
                
                if sensor_data:
                    # 서버로 전송
                    if self.send_data_to_server(sensor_data):
                        sample_count += 1
                    
                    # 10초마다 상태 출력
                    current_time = time.time()
                    if current_time - last_status_time >= 10:
                        elapsed = current_time - start_time
                        rate = sample_count / elapsed if elapsed > 0 else 0
                        logger.info(f"진행률: {elapsed:.1f}s / {duration_seconds}s, "
                                  f"샘플: {sample_count}개, 속도: {rate:.1f}Hz")
                        last_status_time = current_time
                
                # 짧은 대기 (CPU 사용률 조절)
                await asyncio.sleep(0.01)
            
            # 3. 샘플 완료 처리
            self.complete_sample()
            
            # 4. 실험 종료
            self.stop_experiment()
            
            logger.info(f"데이터 수집 완료: {sample_count}개 샘플 수집")
            return True
            
        except Exception as e:
            logger.error(f"데이터 수집 중 오류: {e}")
            # 오류 발생 시 실험 종료
            try:
                self.stop_experiment()
            except:
                pass
            return False
    
    def disconnect(self):
        """연결 해제"""
        if self.serial_conn:
            self.serial_conn.close()
            self.is_connected = False
            logger.info("아두이노 연결 해제")


async def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 SignGlove 데이터 클라이언트 시작")
    print("=" * 60)
    
    client = SignGloveClient()
    
    try:
        # 1. 아두이노 연결
        if not client.connect_arduino():
            print("❌ 아두이노 연결 실패. 프로그램을 종료합니다.")
            return
        
        # 2. 서버 상태 확인
        status = client.get_server_status()
        if status:
            logger.info(f"서버 연결 확인: {status}")
        else:
            logger.warning("서버 상태 확인 실패. 서버가 실행 중인지 확인하세요.")
        
        # 3. 사용자 입력
        print("\n📊 데이터 수집 설정:")
        class_label = input("수집할 클래스 라벨 (예: ㄱ, ㅏ, 1): ").strip()
        
        try:
            duration = int(input("수집 시간 (초, 기본값 300): ") or "300")
        except ValueError:
            duration = 300
        
        # 4. 데이터 수집 실행
        await client.run_data_collection(class_label, duration)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main()) 