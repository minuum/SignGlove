#!/usr/bin/env python3
"""
SignGlove 간단한 TCP 서버
미팅에서 제공된 코드를 기반으로 한 독립 실행 서버

작성자: 양동건 (미팅 내용 반영)
역할: 아두이노 WiFi 클라이언트에서 직접 데이터 수신 및 CSV 저장
"""

import socket
import csv
from datetime import datetime
import logging
import threading
import time
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTCPServer:
    """간단한 TCP 서버 - 아두이노 WiFi 클라이언트용"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        """초기화"""
        self.host = host
        self.port = port
        self.is_running = False
        self.server_socket = None
        
        # CSV 파일 설정
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = f"imu_data_{timestamp}.csv"
        self.csv_file = None
        self.csv_writer = None
        
        # 통계
        self.connection_count = 0
        self.data_count = 0
        
        logger.info(f"TCP 서버 초기화: {host}:{port}")
    
    def setup_csv_file(self):
        """CSV 파일 설정"""
        try:
            self.csv_file = open(self.csv_filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 헤더 작성 (미팅 코드와 동일)
            header = ['timestamp(ms)', 'ax(g)', 'ay(g)', 'az(g)', 'pitch(°)', 'roll(°)', 'yaw(°)']
            self.csv_writer.writerow(header)
            
            logger.info(f"📄 CSV 파일 생성: {self.csv_filename}")
            
        except Exception as e:
            logger.error(f"CSV 파일 설정 실패: {e}")
    
    def start_server(self):
        """서버 시작"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            logger.info(f"📡 서버 시작: {self.host}:{self.port}")
            
            # CSV 파일 준비
            self.setup_csv_file()
            
            # 통계 출력 스레드
            stats_thread = threading.Thread(target=self.print_statistics, daemon=True)
            stats_thread.start()
            
            self.is_running = True
            
            while self.is_running:
                try:
                    conn, addr = self.server_socket.accept()
                    self.connection_count += 1
                    
                    logger.info(f"🔗 연결 #{self.connection_count}: {addr}")
                    
                    # 각 연결을 별도 스레드에서 처리
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.is_running:
                        logger.error(f"소켓 오류: {e}")
                    break
                except KeyboardInterrupt:
                    logger.info("사용자에 의해 중단됨")
                    break
                    
        except Exception as e:
            logger.error(f"서버 시작 실패: {e}")
        finally:
            self.cleanup()
    
    def handle_client(self, conn, addr):
        """클라이언트 연결 처리 (미팅 코드 기반)"""
        try:
            buffer = b''
            conn.settimeout(30.0)  # 30초 타임아웃
            
            while True:
                try:
                    chunk = conn.recv(1024)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # 데이터가 완전히 받아졌는지 확인 (줄바꿈 기준)
                    if b'\n' in buffer:
                        data = buffer.decode('utf-8', errors='ignore').strip()
                        
                        if data:
                            self.process_data(data, addr)
                        
                        buffer = b''  # 버퍼 초기화
                    
                except socket.timeout:
                    logger.warning(f"⏰ 타임아웃: {addr}")
                    break
                except Exception as e:
                    logger.error(f"데이터 수신 오류: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"클라이언트 처리 오류: {e}")
        finally:
            conn.close()
            logger.info(f"🔌 연결 종료: {addr}")
    
    def process_data(self, data, addr):
        """데이터 처리 및 저장 (미팅 코드 기반)"""
        try:
            logger.debug(f"📥 수신 데이터: {repr(data)} from {addr}")
            
            # CSV 형식 파싱: timestamp,ax,ay,az,pitch,roll,yaw
            row = data.split(',')
            
            if len(row) == 7:
                # 데이터 검증
                try:
                    timestamp = int(row[0])
                    ax, ay, az = float(row[1]), float(row[2]), float(row[3])
                    pitch, roll, yaw = float(row[4]), float(row[5]), float(row[6])
                    
                    # CSV 파일에 저장
                    if self.csv_writer:
                        self.csv_writer.writerow(row)
                        self.csv_file.flush()  # 즉시 저장
                        
                        self.data_count += 1
                        
                        logger.info(f"✅ 데이터 저장: Pitch={pitch:.2f}°, Roll={roll:.2f}°, Yaw={yaw:.2f}°")
                    
                except ValueError as e:
                    logger.warning(f"⚠️ 데이터 형식 오류: {e}, 원본: {data}")
            else:
                logger.warning(f"⚠️ 잘못된 열 개수: {len(row)}개 (7개 필요), 데이터: {data}")
                
        except Exception as e:
            logger.error(f"데이터 처리 오류: {e}, 원본: {data}")
    
    def print_statistics(self):
        """통계 정보 출력"""
        start_time = time.time()
        
        while self.is_running:
            time.sleep(30)  # 30초마다 출력
            
            elapsed = time.time() - start_time
            rate = self.data_count / elapsed if elapsed > 0 else 0
            
            logger.info(f"📊 통계: 연결 {self.connection_count}회, "
                       f"데이터 {self.data_count}개, "
                       f"평균 {rate:.2f}Hz, "
                       f"파일: {self.csv_filename}")
    
    def stop_server(self):
        """서버 중지"""
        logger.info("🛑 서버 중지 중...")
        self.is_running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"📄 CSV 파일 저장 완료: {self.csv_filename}")
            
            # 파일 크기 출력
            try:
                file_size = Path(self.csv_filename).stat().st_size
                logger.info(f"파일 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            except:
                pass


def main():
    """메인 함수 (미팅 코드 기반)"""
    print("=" * 60)
    print("📡 SignGlove 간단한 TCP 서버")
    print("=" * 60)
    print("미팅에서 제공된 코드를 기반으로 제작")
    print("아두이노 WiFi 클라이언트와 직접 연동")
    print("=" * 60)
    
    # 설정
    host = input("서버 IP (기본값 0.0.0.0): ").strip() or '0.0.0.0'
    port = int(input("서버 포트 (기본값 5000): ").strip() or '5000')
    
    server = SimpleTCPServer(host=host, port=port)
    
    try:
        print(f"\n🚀 서버 시작... 아두이노에서 {host}:{port}로 연결하세요.")
        print("종료하려면 Ctrl+C를 누르세요.\n")
        
        # 서버 시작 (블로킹)
        server.start_server()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"서버 실행 오류: {e}")
    finally:
        server.stop_server()


if __name__ == "__main__":
    main() 