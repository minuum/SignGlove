#!/usr/bin/env python3
"""
CSV 데이터 수집기 (SignGlove_HW 통합)
SignGlove_HW 저장소의 CSV 중심 데이터 수집 기법 통합

포함 기능:
- CSV 형식 직접 저장
- 실시간 CSV 스트리밍
- 배치 CSV 처리
- 하드웨어 호환성 강화
"""

import csv
import time
import serial
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import queue

logger = logging.getLogger(__name__)


@dataclass
class CSVSensorReading:
    """CSV 센서 데이터 구조"""
    timestamp: float  # epoch seconds
    flex_1: int
    flex_2: int
    flex_3: int
    flex_4: int
    flex_5: int
    gyro_x: float
    gyro_y: float
    gyro_z: float
    accel_x: float
    accel_y: float
    accel_z: float
    battery: int
    signal_strength: int
    
    @classmethod
    def from_csv_line(cls, csv_line: str) -> 'CSVSensorReading':
        """CSV 라인으로부터 센서 데이터 생성"""
        try:
            parts = csv_line.strip().split(',')
            if len(parts) < 13:
                raise ValueError(f"CSV 형식 오류: {len(parts)}개 필드 (13개 필요)")
            
            return cls(
                timestamp=time.time(),
                flex_1=int(parts[0]),
                flex_2=int(parts[1]),
                flex_3=int(parts[2]),
                flex_4=int(parts[3]),
                flex_5=int(parts[4]),
                gyro_x=float(parts[5]),
                gyro_y=float(parts[6]),
                gyro_z=float(parts[7]),
                accel_x=float(parts[8]),
                accel_y=float(parts[9]),
                accel_z=float(parts[10]),
                battery=int(parts[11]),
                signal_strength=int(parts[12])
            )
        except (ValueError, IndexError) as e:
            logger.error(f"CSV 파싱 오류: {e}")
            raise
    
    def to_csv_row(self) -> List:
        """CSV 행으로 변환"""
        # 타임스탬프 포맷 확장: ISO8601, 밀리초
        from datetime import datetime
        ts_iso = datetime.fromtimestamp(self.timestamp).isoformat()
        ts_ms = int(self.timestamp * 1000)

        return [
            ts_iso,
            ts_ms,
            self.flex_1, self.flex_2, self.flex_3, self.flex_4, self.flex_5,
            self.gyro_x, self.gyro_y, self.gyro_z,
            self.accel_x, self.accel_y, self.accel_z,
            self.battery, self.signal_strength
        ]


class CSVDataCollector:
    """CSV 데이터 수집기 (SignGlove_HW 기법)"""
    
    def __init__(self, 
                 output_dir: str = "data/csv_collections",
                 buffer_size: int = 1000,
                 device_id: str = "CSV_COLLECTOR_001"):
        """
        CSV 수집기 초기화
        
        Args:
            output_dir: 출력 디렉토리
            buffer_size: 버퍼 크기
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self.data_buffer = queue.Queue(maxsize=buffer_size)
        self.is_collecting = False
        self.device_id = device_id
        
        # CSV 헤더
        self.csv_header = [
            'timestamp_iso', 'timestamp_ms',
            'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
            'gyro_x', 'gyro_y', 'gyro_z',
            'accel_x', 'accel_y', 'accel_z',
            'battery', 'signal_strength'
        ]
        
        # 통계 정보
        self.collection_stats = {
            'total_samples': 0,
            'start_time': None,
            'current_file': None,
            'samples_per_second': 0.0
        }
        
        logger.info(f"CSV 수집기 초기화: {self.output_dir}")
    
    def create_csv_file(self, gesture_label: str = None) -> Path:
        """
        새 CSV 파일 생성
        
        Args:
            gesture_label: 제스처 라벨 (옵션)
            
        Returns:
            csv_file_path: 생성된 CSV 파일 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if gesture_label:
            filename = f"{gesture_label}_{timestamp}.csv"
        else:
            filename = f"sensor_data_{timestamp}.csv"
        
        csv_path = self.output_dir / filename
        
        # 헤더 작성
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
        
        self.collection_stats['current_file'] = csv_path
        logger.info(f"새 CSV 파일 생성: {csv_path}")
        
        return csv_path
    
    def add_sensor_reading(self, reading: CSVSensorReading) -> bool:
        """
        센서 데이터 추가
        
        Args:
            reading: 센서 데이터
            
        Returns:
            success: 추가 성공 여부
        """
        try:
            self.data_buffer.put_nowait(reading)
            self.collection_stats['total_samples'] += 1
            return True
        except queue.Full:
            logger.warning("데이터 버퍼 오버플로우")
            return False
    
    def collect_from_serial(self, 
                           port: str, 
                           baudrate: int = 115200,
                           gesture_label: str = None,
                           duration: float = None) -> Path:
        """
        시리얼 포트에서 CSV 데이터 수집
        
        Args:
            port: 시리얼 포트
            baudrate: 보드레이트
            gesture_label: 제스처 라벨
            duration: 수집 시간 (초, None이면 무제한)
            
        Returns:
            csv_file_path: 저장된 CSV 파일 경로
        """
        csv_file = self.create_csv_file(gesture_label)
        
        try:
            # 시리얼 연결
            ser = serial.Serial(port, baudrate, timeout=1)
            logger.info(f"시리얼 연결: {port} @ {baudrate} baud")
            
            # 수집 시작
            self.is_collecting = True
            self.collection_stats['start_time'] = time.time()
            start_time = time.time()
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                while self.is_collecting:
                    # 시간 제한 체크
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    try:
                        # 시리얼 데이터 읽기
                        line = ser.readline().decode('utf-8').strip()
                        if not line:
                            continue
                        
                        # CSV 데이터 파싱
                        reading = CSVSensorReading.from_csv_line(line)
                        
                        # CSV 파일에 직접 쓰기
                        writer.writerow(reading.to_csv_row())
                        f.flush()  # 즉시 디스크에 쓰기
                        
                        # 버퍼에도 추가 (실시간 처리용)
                        self.add_sensor_reading(reading)
                        
                        # 통계 업데이트
                        if self.collection_stats['total_samples'] % 100 == 0:
                            self._update_collection_stats()
                        
                    except ValueError as e:
                        logger.warning(f"데이터 파싱 오류: {e}")
                        continue
                    except KeyboardInterrupt:
                        logger.info("사용자 중단")
                        break
            
            ser.close()
            
        except serial.SerialException as e:
            logger.error(f"시리얼 연결 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")
            raise
        finally:
            self.is_collecting = False
        
        self._finalize_collection(csv_file)
        return csv_file
    
    def collect_from_buffer(self, csv_file: Path = None) -> Path:
        """
        버퍼의 데이터를 CSV 파일로 저장
        
        Args:
            csv_file: CSV 파일 경로 (None이면 새로 생성)
            
        Returns:
            csv_file_path: 저장된 CSV 파일 경로
        """
        if csv_file is None:
            csv_file = self.create_csv_file()
        
        saved_count = 0
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 버퍼의 모든 데이터 저장
            while not self.data_buffer.empty():
                try:
                    reading = self.data_buffer.get_nowait()
                    writer.writerow(reading.to_csv_row())
                    saved_count += 1
                except queue.Empty:
                    break
        
        logger.info(f"버퍼 데이터 저장 완료: {saved_count}개 샘플 → {csv_file}")
        return csv_file
    
    def _update_collection_stats(self):
        """수집 통계 업데이트"""
        if self.collection_stats['start_time']:
            elapsed = time.time() - self.collection_stats['start_time']
            if elapsed > 0:
                self.collection_stats['samples_per_second'] = \
                    self.collection_stats['total_samples'] / elapsed
    
    def _finalize_collection(self, csv_file: Path):
        """수집 완료 처리"""
        self._update_collection_stats()
        
        # 통계 정보를 별도 파일로 저장
        stats_file = csv_file.with_suffix('.stats.json')
        import json
        
        final_stats = {
            **self.collection_stats,
            'csv_file': str(csv_file),
            'end_time': time.time(),
            'file_size_bytes': csv_file.stat().st_size if csv_file.exists() else 0
        }
        
        # timestamp는 JSON 직렬화 불가능하므로 문자열로 변환
        if final_stats.get('start_time'):
            final_stats['start_time_iso'] = datetime.fromtimestamp(
                final_stats['start_time']
            ).isoformat()
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"수집 완료: {final_stats['total_samples']}개 샘플, "
                   f"{final_stats['samples_per_second']:.1f} SPS")
    
    def load_csv_data(self, csv_file: Path) -> pd.DataFrame:
        """
        CSV 파일 로드
        
        Args:
            csv_file: CSV 파일 경로
            
        Returns:
            dataframe: 로드된 데이터프레임
        """
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"CSV 로드 완료: {len(df)}개 행, {csv_file}")
            return df
        except Exception as e:
            logger.error(f"CSV 로드 실패: {e}")
            raise
    
    def merge_csv_files(self, csv_files: List[Path], output_file: Path = None) -> Path:
        """
        여러 CSV 파일 병합
        
        Args:
            csv_files: 병합할 CSV 파일 목록
            output_file: 출력 파일 (None이면 자동 생성)
            
        Returns:
            merged_file: 병합된 파일 경로
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"merged_{timestamp}.csv"
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = self.load_csv_data(csv_file)
                all_data.append(df)
                logger.info(f"병합 대상 추가: {csv_file} ({len(df)}개 행)")
            except Exception as e:
                logger.error(f"파일 로드 실패, 스킵: {csv_file} - {e}")
                continue
        
        if not all_data:
            raise ValueError("병합할 데이터가 없습니다")
        
        # 데이터프레임 병합
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 타임스탬프 순 정렬
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        
        # 저장
        merged_df.to_csv(output_file, index=False)
        
        logger.info(f"CSV 병합 완료: {len(merged_df)}개 행 → {output_file}")
        return output_file
    
    def get_collection_stats(self) -> Dict:
        """수집 통계 반환"""
        self._update_collection_stats()
        return self.collection_stats.copy()
    
    def stop_collection(self):
        """수집 중지"""
        self.is_collecting = False
        logger.info("데이터 수집 중지")


class SerialPortDetector:
    """시리얼 포트 자동 탐지 (SignGlove_HW 기법)"""
    
    @staticmethod
    def find_arduino_ports() -> List[str]:
        """아두이노 포트 찾기"""
        import serial.tools.list_ports
        
        arduino_ports = []
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # 아두이노 식별자 확인
            if any(keyword in port.description.lower() for keyword in 
                   ['arduino', 'ch340', 'cp210', 'ftdi']):
                arduino_ports.append(port.device)
        
        return arduino_ports
    
    @staticmethod
    def test_port_connection(port: str, baudrate: int = 115200, timeout: float = 5.0) -> bool:
        """포트 연결 테스트"""
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            start_time = time.time()
            
            # 데이터 수신 대기
            while time.time() - start_time < timeout:
                line = ser.readline().decode('utf-8').strip()
                if line and ',' in line:  # CSV 형태 데이터 확인
                    ser.close()
                    return True
            
            ser.close()
            return False
            
        except Exception:
            return False


def main():
    """CSV 수집기 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SignGlove CSV 데이터 수집기")
    parser.add_argument("--port", type=str, help="시리얼 포트")
    parser.add_argument("--baudrate", type=int, default=115200, help="보드레이트")
    parser.add_argument("--duration", type=float, help="수집 시간 (초)")
    parser.add_argument("--gesture", type=str, help="제스처 라벨")
    parser.add_argument("--auto-detect", action="store_true", help="포트 자동 탐지")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    collector = CSVDataCollector()
    
    # 포트 자동 탐지
    if args.auto_detect or not args.port:
        print("🔍 아두이노 포트 탐지 중...")
        ports = SerialPortDetector.find_arduino_ports()
        
        if not ports:
            print("❌ 아두이노 포트를 찾을 수 없습니다.")
            return
        
        print(f"✅ 발견된 포트: {ports}")
        
        # 첫 번째 포트 테스트
        for port in ports:
            print(f"🧪 포트 테스트: {port}")
            if SerialPortDetector.test_port_connection(port):
                print(f"✅ 연결 성공: {port}")
                args.port = port
                break
        else:
            print("❌ 사용 가능한 포트가 없습니다.")
            return
    
    try:
        print(f"🚀 데이터 수집 시작...")
        print(f"   포트: {args.port}")
        print(f"   보드레이트: {args.baudrate}")
        print(f"   시간: {args.duration}초" if args.duration else "   시간: 무제한")
        print(f"   제스처: {args.gesture}" if args.gesture else "   제스처: 없음")
        
        csv_file = collector.collect_from_serial(
            port=args.port,
            baudrate=args.baudrate,
            gesture_label=args.gesture,
            duration=args.duration
        )
        
        print(f"✅ 수집 완료: {csv_file}")
        
        # 통계 출력
        stats = collector.get_collection_stats()
        print(f"📊 통계:")
        print(f"   총 샘플: {stats['total_samples']}")
        print(f"   초당 샘플: {stats['samples_per_second']:.1f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자 중단")
        collector.stop_collection()
    except Exception as e:
        print(f"❌ 오류: {e}")


if __name__ == "__main__":
    main()
