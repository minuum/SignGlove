#!/usr/bin/env python3
"""
SignGlove WiFi 데이터 클라이언트
아두이노 WiFi 클라이언트에서 TCP 소켓으로 데이터를 수신하고
기존 FastAPI 서버와 연동

작성자: 양동건 (미팅 내용 반영)
역할: TCP 서버 → FastAPI 클라이언트 브리지
"""

import socket
import csv
import json
import requests
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wifi_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WiFiDataClient:
    """WiFi 데이터 클라이언트 - TCP 서버 + FastAPI 연동"""
    
    def __init__(self, tcp_port: int = 5000, server_url: str = "http://localhost:8000"):
        """초기화"""
        self.tcp_port = tcp_port
        self.server_url = server_url
        self.is_running = False
        self.current_session = None
        
        # TCP 서버 설정
        self.tcp_socket = None
        
        # 데이터 저장
        self.csv_filename = f"wifi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = None
        self.csv_writer = None
        
        logger.info("WiFi 데이터 클라이언트 초기화 완료")
    
    def start_tcp_server(self):
        """TCP 서버 시작"""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.bind(('0.0.0.0', self.tcp_port))
            self.tcp_socket.listen(5)
            
            logger.info(f"📡 TCP 서버 시작: 포트 {self.tcp_port}")
            
            # CSV 파일 준비
            self.setup_csv_file()
            
            self.is_running = True
            
            while self.is_running:
                try:
                    conn, addr = self.tcp_socket.accept()
                    logger.info(f"🔗 연결됨: {addr}")
                    
                    # 각 연결을 별도 스레드에서 처리
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.is_running:
                        logger.error(f"소켓 오류: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"TCP 서버 시작 실패: {e}")
        finally:
            self.cleanup()
    
    def handle_client(self, conn, addr):
        """클라이언트 연결 처리"""
        try:
            buffer = b''
            conn.settimeout(10.0)  # 10초 타임아웃
            
            while self.is_running:
                try:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # 줄바꿈으로 구분된 데이터 처리
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        self.process_data_line(line.decode().strip())
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"데이터 수신 오류: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"클라이언트 처리 오류: {e}")
        finally:
            conn.close()
            logger.info(f"🔌 연결 종료: {addr}")
    
    def process_data_line(self, line: str):
        """받은 데이터 라인 처리"""
        if not line or line.startswith('==='):
            return
        
        try:
            # CSV 형식 파싱: timestamp,ax,ay,az,pitch,roll,yaw
            parts = line.split(',')
            if len(parts) < 7:
                logger.warning(f"잘못된 데이터 형식: {line}")
                return
            
            # 데이터 파싱
            timestamp = int(parts[0])
            ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
            pitch, roll, yaw = float(parts[4]), float(parts[5]), float(parts[6])
            
            # 데이터 구조화
            sensor_data = {
                "device_id": "ARDUINO_WIFI_001",
                "timestamp": datetime.now().isoformat(),
                "flex_sensors": {
                    "flex_1": 0.0,  # WiFi 버전에는 플렉스 센서 없음
                    "flex_2": 0.0,
                    "flex_3": 0.0,
                    "flex_4": 0.0,
                    "flex_5": 0.0
                },
                "gyro_data": {
                    "gyro_x": roll,    # roll -> gyro_x
                    "gyro_y": pitch,   # pitch -> gyro_y  
                    "gyro_z": yaw,     # yaw -> gyro_z
                    "accel_x": ax,
                    "accel_y": ay,
                    "accel_z": az
                },
                "orientation": {
                    "pitch": pitch,
                    "roll": roll,
                    "yaw": yaw
                },
                "battery_level": 95.0,
                "signal_strength": -50
            }
            
            # CSV 저장
            self.save_to_csv(sensor_data)
            
            # 서버로 전송 (비동기)
            threading.Thread(
                target=self.send_to_server,
                args=(sensor_data,),
                daemon=True
            ).start()
            
            logger.debug(f"📊 데이터 처리 완료: Pitch={pitch:.2f}°, Roll={roll:.2f}°, Yaw={yaw:.2f}°")
            
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
                'accel_x', 'accel_y', 'accel_z',
                'flex1', 'flex2', 'flex3', 'flex4', 'flex5'
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
                sensor_data['gyro_data']['accel_x'],
                sensor_data['gyro_data']['accel_y'],
                sensor_data['gyro_data']['accel_z'],
                sensor_data['flex_sensors']['flex_1'],
                sensor_data['flex_sensors']['flex_2'],
                sensor_data['flex_sensors']['flex_3'],
                sensor_data['flex_sensors']['flex_4'],
                sensor_data['flex_sensors']['flex_5']
            ]
            
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            
        except Exception as e:
            logger.error(f"CSV 저장 오류: {e}")
    
    def send_to_server(self, sensor_data: Dict):
        """FastAPI 서버로 데이터 전송"""
        try:
            response = requests.post(
                f"{self.server_url}/data/sensor",
                json=sensor_data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"✅ 서버 전송 성공: {result.get('sample_id', 'N/A')}")
            else:
                logger.warning(f"⚠️ 서버 응답 오류: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ 서버 연결 실패: {e}")
        except Exception as e:
            logger.error(f"서버 전송 오류: {e}")
    
    def start_experiment_session(self, class_label: str, **kwargs) -> bool:
        """실험 세션 시작"""
        try:
            session_data = {
                "session_id": f"wifi_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "performer_id": kwargs.get("performer_id", "donggeon"),
                "class_label": class_label,
                "category": self.get_class_category(class_label),
                "target_samples": kwargs.get("target_samples", 300),  # 5분 * 60초 = 300
                "duration_per_sample": kwargs.get("duration_per_sample", 1),
                "sampling_rate": kwargs.get("sampling_rate", 10),  # WiFi는 낮은 주파수
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
        logger.info("🛑 WiFi 클라이언트 중지 중...")
        self.is_running = False
        
        if self.tcp_socket:
            self.tcp_socket.close()
        
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
    print("📡 SignGlove WiFi 데이터 클라이언트")
    print("=" * 60)
    
    # 설정
    tcp_port = int(input("TCP 포트 (기본값 5000): ") or "5000")
    server_url = input("FastAPI 서버 URL (기본값 http://localhost:8000): ") or "http://localhost:8000"
    
    client = WiFiDataClient(tcp_port=tcp_port, server_url=server_url)
    
    try:
        # 실험 세션 설정
        class_label = input("수집할 클래스 라벨 (예: ㄱ, ㅏ, 1): ").strip()
        if class_label:
            client.start_experiment_session(class_label)
        
        print(f"🚀 TCP 서버 시작... 아두이노에서 {tcp_port} 포트로 연결하세요.")
        print("종료하려면 Ctrl+C를 누르세요.")
        
        # TCP 서버 시작 (블로킹)
        client.start_tcp_server()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"프로그램 실행 오류: {e}")
    finally:
        client.stop()


if __name__ == "__main__":
    main() 