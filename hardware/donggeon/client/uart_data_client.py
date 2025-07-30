#!/usr/bin/env python3
"""
SignGlove UART 데이터 클라이언트
아두이노에서 시리얼 통신으로 데이터를 수신하고
기존 FastAPI 서버와 연동

작성자: 양동건 (미팅 내용 반영)
역할: 시리얼 수신 → FastAPI 클라이언트 브리지
"""

import serial
import serial.tools.list_ports
import csv
import json
import requests
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uart_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UARTDataClient:
    """UART 데이터 클라이언트 - 시리얼 수신 + FastAPI 연동"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """초기화"""
        self.server_url = server_url
        self.serial_conn = None
        self.is_running = False
        self.current_session = None
        
        # 시리얼 설정
        self.port = None
        self.baudrate = 115200
        self.timeout = 1.0
        
        # 데이터 저장
        self.csv_filename = f"uart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = None
        self.csv_writer = None
        
        # 통계
        self.received_count = 0
        self.last_stats_time = time.time()
        
        logger.info("UART 데이터 클라이언트 초기화 완료")
    
    def find_arduino_port(self) -> Optional[str]:
        """아두이노 포트 자동 검색"""
        logger.info("🔍 아두이노 포트 검색 중...")
        
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        
        for port in ports:
            port_info = str(port).lower()
            if any(keyword in port_info for keyword in ['arduino', 'ch340', 'cp210', 'ftdi', 'usb']):
                arduino_ports.append(port.device)
                logger.info(f"🔌 아두이노 포트 발견: {port.device} - {port.description}")
        
        if not arduino_ports:
            logger.error("❌ 아두이노 포트를 찾을 수 없습니다.")
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
            
            self.port = port
            
            logger.info(f"🔗 아두이노 연결 시도: {port}")
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # 연결 확인을 위한 대기
            time.sleep(2)
            
            # 테스트 데이터 읽기
            test_success = self.test_communication()
            
            if test_success:
                logger.info("✅ 아두이노 연결 성공!")
                return True
            else:
                logger.error("❌ 아두이노 통신 테스트 실패")
                return False
                
        except Exception as e:
            logger.error(f"아두이노 연결 실패: {e}")
            return False
    
    def test_communication(self) -> bool:
        """아두이노 통신 테스트"""
        try:
            # 몇 줄 읽어서 데이터 형식 확인
            for _ in range(10):
                line = self.serial_conn.readline().decode(errors='ignore').strip()
                if line and line.count(',') >= 8:  # timestamp,pitch,roll,yaw,flex1,flex2,flex3,flex4,flex5
                    logger.info(f"✅ 테스트 데이터 수신: {line[:50]}...")
                    return True
                time.sleep(0.1)
            
            logger.warning("⚠️ 올바른 형식의 데이터를 받지 못했습니다.")
            return False
            
        except Exception as e:
            logger.error(f"통신 테스트 오류: {e}")
            return False
    
    def start_data_collection(self):
        """데이터 수집 시작"""
        if not self.serial_conn:
            logger.error("❌ 아두이노가 연결되지 않았습니다.")
            return False
        
        # CSV 파일 준비
        self.setup_csv_file()
        
        self.is_running = True
        self.received_count = 0
        self.last_stats_time = time.time()
        
        logger.info("🚀 UART 데이터 수집 시작...")
        
        # 통계 출력 스레드
        stats_thread = threading.Thread(target=self.print_statistics, daemon=True)
        stats_thread.start()
        
        try:
            buffer = ""
            
            while self.is_running:
                try:
                    # 시리얼 데이터 읽기
                    if self.serial_conn.in_waiting > 0:
                        chunk = self.serial_conn.read(self.serial_conn.in_waiting).decode(errors='ignore')
                        buffer += chunk
                        
                        # 줄바꿈으로 구분된 데이터 처리
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            self.process_data_line(line.strip())
                    
                    time.sleep(0.01)  # CPU 사용률 조절
                    
                except Exception as e:
                    logger.error(f"데이터 읽기 오류: {e}")
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        finally:
            self.cleanup()
    
    def process_data_line(self, line: str):
        """받은 데이터 라인 처리"""
        if not line or line.startswith('=') or line.startswith('📊') or line.startswith('✅'):
            return
        
        # 헤더 라인 스킵
        if 'timestamp,pitch,roll,yaw,flex1,flex2,flex3,flex4,flex5' in line:
            return
        
        try:
            # CSV 형식 파싱: timestamp,pitch,roll,yaw,flex1,flex2,flex3,flex4,flex5
            parts = line.split(',')
            if len(parts) < 9:
                logger.debug(f"부족한 데이터 열: {line}")
                return
            
            # 데이터 파싱
            timestamp = int(parts[0])
            pitch, roll, yaw = float(parts[1]), float(parts[2]), float(parts[3])
            flex_values = [int(parts[i]) for i in range(4, 9)]
            
            # 데이터 구조화
            sensor_data = {
                "device_id": "ARDUINO_UART_001",
                "timestamp": datetime.now().isoformat(),
                "flex_sensors": {
                    "flex_1": flex_values[0],
                    "flex_2": flex_values[1],
                    "flex_3": flex_values[2],
                    "flex_4": flex_values[3],
                    "flex_5": flex_values[4]
                },
                "gyro_data": {
                    "gyro_x": roll,    # roll -> gyro_x
                    "gyro_y": pitch,   # pitch -> gyro_y
                    "gyro_z": yaw,     # yaw -> gyro_z
                    "accel_x": 0.0,    # UART 버전에는 가속도 없음
                    "accel_y": 0.0,
                    "accel_z": 0.0
                },
                "orientation": {
                    "pitch": pitch,
                    "roll": roll,
                    "yaw": yaw
                },
                "battery_level": 95.0,
                "signal_strength": -40
            }
            
            # CSV 저장
            self.save_to_csv(sensor_data)
            
            # 서버로 전송 (비동기)
            threading.Thread(
                target=self.send_to_server,
                args=(sensor_data,),
                daemon=True
            ).start()
            
            self.received_count += 1
            
            if self.received_count % 100 == 0:  # 100개마다 로그
                logger.debug(f"📊 처리 완료: {self.received_count}개, "
                           f"Flex=[{','.join(map(str, flex_values))}], "
                           f"Angles=[{pitch:.1f},{roll:.1f},{yaw:.1f}]")
            
        except Exception as e:
            logger.error(f"데이터 처리 오류: {e}, 원본: {line}")
    
    def setup_csv_file(self):
        """CSV 파일 설정"""
        try:
            self.csv_file = open(self.csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 헤더 작성
            header = [
                'timestamp', 'device_id',
                'pitch', 'roll', 'yaw',
                'flex1', 'flex2', 'flex3', 'flex4', 'flex5',
                'gyro_x', 'gyro_y', 'gyro_z'
            ]
            self.csv_writer.writerow(header)
            
            logger.info(f"📄 CSV 파일 생성: {self.csv_filename}")
            
        except Exception as e:
            logger.error(f"CSV 파일 설정 실패: {e}")
    
    def save_to_csv(self, sensor_data: Dict):
        """CSV 파일에 데이터 저장"""
        if not self.csv_writer:
            return
        
        try:
            row = [
                sensor_data['timestamp'],
                sensor_data['device_id'],
                sensor_data['orientation']['pitch'],
                sensor_data['orientation']['roll'],
                sensor_data['orientation']['yaw'],
                sensor_data['flex_sensors']['flex_1'],
                sensor_data['flex_sensors']['flex_2'],
                sensor_data['flex_sensors']['flex_3'],
                sensor_data['flex_sensors']['flex_4'],
                sensor_data['flex_sensors']['flex_5'],
                sensor_data['gyro_data']['gyro_x'],
                sensor_data['gyro_data']['gyro_y'],
                sensor_data['gyro_data']['gyro_z']
            ]
            
            self.csv_writer.writerow(row)
            
            # 주기적으로 파일 flush (데이터 손실 방지)
            if self.received_count % 50 == 0:
                self.csv_file.flush()
            
        except Exception as e:
            logger.error(f"CSV 저장 오류: {e}")
    
    def send_to_server(self, sensor_data: Dict):
        """FastAPI 서버로 데이터 전송"""
        try:
            response = requests.post(
                f"{self.server_url}/data/sensor",
                json=sensor_data,
                timeout=3  # UART는 빠른 데이터이므로 짧은 타임아웃
            )
            
            if response.status_code == 200:
                result = response.json()
                # 성공 로그는 너무 많으므로 생략
            else:
                logger.warning(f"⚠️ 서버 응답 오류: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            # 네트워크 오류는 WARNING으로 (서버가 꺼져있을 수 있음)
            pass
        except Exception as e:
            logger.error(f"서버 전송 오류: {e}")
    
    def print_statistics(self):
        """통계 정보 출력"""
        while self.is_running:
            time.sleep(10)  # 10초마다 출력
            
            current_time = time.time()
            elapsed = current_time - self.last_stats_time
            rate = self.received_count / elapsed if elapsed > 0 else 0
            
            logger.info(f"📊 수집 통계: {self.received_count}개 수신, "
                       f"평균 {rate:.1f}Hz, 파일: {self.csv_filename}")
    
    def start_experiment_session(self, class_label: str, **kwargs) -> bool:
        """실험 세션 시작"""
        try:
            session_data = {
                "session_id": f"uart_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "performer_id": kwargs.get("performer_id", "donggeon"),
                "class_label": class_label,
                "category": self.get_class_category(class_label),
                "target_samples": kwargs.get("target_samples", 3000),  # 50Hz * 60s = 3000
                "duration_per_sample": kwargs.get("duration_per_sample", 0.02),  # 20ms
                "sampling_rate": kwargs.get("sampling_rate", 50),  # 50Hz
                "storage_strategy": "both"
            }
            
            logger.info(f"🚀 실험 세션 시작: {class_label}")
            response = requests.post(
                f"{self.server_url}/experiment/start",
                json=session_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.current_session = session_data
                logger.info(f"✅ 실험 세션 시작 성공: {result['session_id']}")
                return True
            else:
                logger.error(f"❌ 실험 세션 시작 실패: {response.status_code}")
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
    
    def stop_experiment(self):
        """실험 세션 종료"""
        try:
            response = requests.post(f"{self.server_url}/experiment/stop", timeout=10)
            if response.status_code == 200:
                logger.info("✅ 실험 세션 종료")
                self.current_session = None
            else:
                logger.error(f"❌ 실험 종료 실패: {response.status_code}")
        except Exception as e:
            logger.error(f"실험 종료 오류: {e}")
    
    def stop(self):
        """클라이언트 중지"""
        logger.info("🛑 UART 클라이언트 중지 중...")
        self.is_running = False
        
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"📄 CSV 파일 저장 완료: {self.csv_filename}")
        
        if self.current_session:
            self.stop_experiment()


def main():
    """메인 함수"""
    print("=" * 60)
    print("📞 SignGlove UART 데이터 클라이언트")
    print("=" * 60)
    
    # 설정
    server_url = input("FastAPI 서버 URL (기본값 http://localhost:8000): ") or "http://localhost:8000"
    
    client = UARTDataClient(server_url=server_url)
    
    try:
        # 아두이노 연결
        if not client.connect_arduino():
            print("❌ 아두이노 연결 실패. 프로그램을 종료합니다.")
            return
        
        # 실험 세션 설정
        class_label = input("수집할 클래스 라벨 (예: ㄱ, ㅏ, 1): ").strip()
        if class_label:
            client.start_experiment_session(class_label)
        
        print("🚀 데이터 수집 시작... 종료하려면 Ctrl+C를 누르세요.")
        
        # 데이터 수집 시작 (블로킹)
        client.start_data_collection()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"프로그램 실행 오류: {e}")
    finally:
        client.stop()


if __name__ == "__main__":
    main() 