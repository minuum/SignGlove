#!/usr/bin/env python3
"""
SignGlove 고급 데이터 수집기
요구사항에 맞춘 개선된 버전:
1. 라벨 입력 (ㄱ/ㅏ/1)
2. 지정된 시간 + Hz 설정 가능  
3. 특정 파일명으로 overwrite 저장
4. 다음 라벨 측정 여부 확인

추가 기능:
- 실시간 센서 모니터링
- 데이터 품질 체크
- 세밀한 실험 제어
"""

import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import serial
import serial.tools.list_ports
from serial.serialutil import SerialException

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from server.ksl_classes import ksl_manager, KSLCategory
from server.data_storage import DataStorage
from server.models.sensor_data import SensorData, FlexSensorData, GyroData, SignGestureData, SignGestureType


class AdvancedDataCollector:
    """고급 데이터 수집기"""
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        """데이터 수집기 초기화"""
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.data_storage = DataStorage()
        self.is_collecting = False
        self.collected_data: List[SensorData] = []
        
        # 세션 정보
        self.session_id = f"advanced_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.performer_id = "test_performer"
        
        # 실험 설정
        self.sampling_rate_hz = 20  # 기본값
        self.output_directory = Path("data/experiments")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        print("🚀 SignGlove 고급 데이터 수집기 초기화 완료")
    
    def connect_arduino(self) -> bool:
        """아두이노 연결"""
        if not self.port:
            self.port = self.find_arduino_port()
            if not self.port:
                return False
        
        try:
            print(f"\n🔌 {self.port} 포트로 연결 중...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)
            
            if self.test_communication():
                print("✅ 아두이노 연결 성공!")
                return True
            else:
                print("❌ 아두이노 통신 테스트 실패")
                return False
                
        except SerialException as e:
            print(f"❌ 시리얼 연결 실패: {e}")
            return False
    
    def find_arduino_port(self) -> Optional[str]:
        """아두이노 포트 자동 감지"""
        print("\n🔍 아두이노 포트 검색 중...")
        
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        
        for port in ports:
            if any(keyword in str(port).lower() for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
                arduino_ports.append(port.device)
                print(f"   ✅ 발견: {port.device} - {port.description}")
        
        if not arduino_ports:
            print("   ❌ 아두이노를 찾을 수 없습니다.")
            return None
        
        if len(arduino_ports) == 1:
            return arduino_ports[0]
        
        # 여러 포트 선택
        print(f"\n📋 여러 포트가 발견되었습니다:")
        for i, port in enumerate(arduino_ports, 1):
            print(f"   {i}. {port}")
        
        while True:
            try:
                choice = int(input(f"포트를 선택하세요 (1-{len(arduino_ports)}): ")) - 1
                if 0 <= choice < len(arduino_ports):
                    return arduino_ports[choice]
                else:
                    print("❌ 잘못된 선택입니다.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
    
    def test_communication(self) -> bool:
        """아두이노 통신 테스트"""
        try:
            self.serial_conn.write(b"TEST\n")
            time.sleep(0.5)
            response = self.serial_conn.readline().decode().strip()
            return "OK" in response or len(response) > 0
        except Exception as e:
            print(f"통신 테스트 오류: {e}")
            return False
    
    def get_experiment_settings(self) -> Dict:
        """실험 설정 입력"""
        print("\n⚙️ 실험 설정")
        
        # 1. 라벨 입력
        label = self.get_label_input()
        
        # 2. 측정 시간 입력
        duration = self.get_measurement_duration()
        
        # 3. 샘플링 주파수 입력
        sampling_hz = self.get_sampling_rate()
        
        # 4. 파일명 설정
        filename = self.get_output_filename(label)
        
        # 5. 저장 모드 설정
        save_mode = self.get_save_mode()
        
        return {
            'label': label,
            'duration': duration,
            'sampling_hz': sampling_hz,
            'filename': filename,
            'save_mode': save_mode,
            'total_samples': int(duration * sampling_hz)
        }
    
    def get_label_input(self) -> str:
        """라벨 입력 (ㄱ/ㅏ/1 형태)"""
        print("\n🏷️ 측정할 라벨 입력")
        print("예시: ㄱ, ㅏ, 1, ㄴ, ㅓ, 2, ...")
        
        while True:
            label = input("라벨을 입력하세요: ").strip()
            
            if not label:
                print("❌ 라벨을 입력해주세요.")
                continue
            
            # 라벨 유효성 검사
            if self.validate_label(label):
                return label
            else:
                print("❌ 유효하지 않은 라벨입니다. 다시 입력해주세요.")
    
    def validate_label(self, label: str) -> bool:
        """라벨 유효성 검사"""
        # 한글 자음/모음 또는 숫자 확인
        consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        return label in consonants or label in vowels or label in numbers
    
    def get_measurement_duration(self) -> int:
        """측정 시간 입력"""
        while True:
            try:
                duration = int(input("\n⏱️ 측정 시간을 입력하세요 (1-60초): "))
                if 1 <= duration <= 60:
                    return duration
                else:
                    print("❌ 1-60초 사이의 값을 입력해주세요.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
    
    def get_sampling_rate(self) -> int:
        """샘플링 주파수 입력"""
        print("\n📊 샘플링 주파수 설정")
        print("   1. 10Hz (낮은 품질, 빠른 처리)")
        print("   2. 20Hz (표준 품질, 권장)")
        print("   3. 50Hz (높은 품질, 큰 용량)")
        print("   4. 100Hz (최고 품질, 매우 큰 용량)")
        print("   5. 직접 입력")
        
        while True:
            try:
                choice = input("선택 (1-5): ").strip()
                
                if choice == "1":
                    return 10
                elif choice == "2":
                    return 20
                elif choice == "3":
                    return 50
                elif choice == "4":
                    return 100
                elif choice == "5":
                    hz = int(input("샘플링 주파수를 직접 입력하세요 (1-200Hz): "))
                    if 1 <= hz <= 200:
                        return hz
                    else:
                        print("❌ 1-200Hz 사이의 값을 입력해주세요.")
                else:
                    print("❌ 1-5 중 하나를 선택해주세요.")
            except ValueError:
                print("❌ 올바른 값을 입력해주세요.")
    
    def get_output_filename(self, label: str) -> str:
        """출력 파일명 설정"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{label}_{timestamp}"
        
        print(f"\n📁 파일명 설정")
        print(f"기본 파일명: {default_name}")
        
        custom_name = input("사용자 정의 파일명 (Enter = 기본값 사용): ").strip()
        
        if custom_name:
            # 특수문자 제거
            custom_name = "".join(c for c in custom_name if c.isalnum() or c in "._-")
            return custom_name
        else:
            return default_name
    
    def get_save_mode(self) -> str:
        """저장 모드 설정"""
        print("\n💾 저장 모드 설정")
        print("   1. 덮어쓰기 (overwrite) - 기존 파일 삭제 후 새로 저장")
        print("   2. 추가하기 (append) - 기존 파일에 데이터 추가")
        
        while True:
            choice = input("선택 (1-2): ").strip()
            if choice == "1":
                return "overwrite"
            elif choice == "2":
                return "append"
            else:
                print("❌ 1 또는 2를 선택해주세요.")
    
    def read_sensor_data(self) -> Optional[SensorData]:
        """센서 데이터 읽기"""
        try:
            if not self.serial_conn or not self.serial_conn.is_open:
                return None
            
            line = self.serial_conn.readline().decode().strip()
            if not line:
                return None
            
            # JSON 또는 CSV 파싱
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                parts = line.split(',')
                if len(parts) >= 11:
                    data = {
                        'flex_1': float(parts[0]),
                        'flex_2': float(parts[1]),
                        'flex_3': float(parts[2]),
                        'flex_4': float(parts[3]),
                        'flex_5': float(parts[4]),
                        'gyro_x': float(parts[5]),
                        'gyro_y': float(parts[6]),
                        'gyro_z': float(parts[7]),
                        'accel_x': float(parts[8]),
                        'accel_y': float(parts[9]),
                        'accel_z': float(parts[10]),
                        'battery': float(parts[11]) if len(parts) > 11 else 100,
                        'signal': int(parts[12]) if len(parts) > 12 else -50
                    }
                else:
                    return None
            
            # SensorData 객체 생성
            sensor_data = SensorData(
                device_id="ADVANCED_ARDUINO_001",
                timestamp=datetime.now(),
                flex_sensors=FlexSensorData(
                    flex_1=data['flex_1'],
                    flex_2=data['flex_2'],
                    flex_3=data['flex_3'],
                    flex_4=data['flex_4'],
                    flex_5=data['flex_5']
                ),
                gyro_data=GyroData(
                    gyro_x=data['gyro_x'],
                    gyro_y=data['gyro_y'],
                    gyro_z=data['gyro_z'],
                    accel_x=data['accel_x'],
                    accel_y=data['accel_y'],
                    accel_z=data['accel_z']
                ),
                battery_level=float(data.get('battery', 100)),
                signal_strength=int(data.get('signal', -50))
            )
            
            return sensor_data
            
        except Exception as e:
            print(f"⚠️ 센서 데이터 읽기 오류: {e}")
            return None
    
    async def collect_data_advanced(self, settings: Dict) -> bool:
        """고급 데이터 수집"""
        label = settings['label']
        duration = settings['duration']
        sampling_hz = settings['sampling_hz']
        total_samples = settings['total_samples']
        
        print(f"\n🎯 라벨 '{label}' 측정 준비")
        print(f"   ⏱️ 측정 시간: {duration}초")
        print(f"   📊 샘플링: {sampling_hz}Hz")
        print(f"   📈 예상 샘플 수: {total_samples}개")
        print("   ⚠️ 손을 올바른 자세로 준비해주세요.")
        
        # 시작 확인
        input("\n준비가 되면 Enter를 눌러주세요...")
        
        # 정교한 카운트다운
        print(f"\n🚀 측정을 시작합니다!")
        for i in range(5, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("   🔴 측정 시작!")
        
        # 데이터 수집
        self.collected_data = []
        self.is_collecting = True
        start_time = time.time()
        sample_interval = 1.0 / sampling_hz
        
        try:
            sample_count = 0
            last_sample_time = start_time
            
            while time.time() - start_time < duration:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # 샘플링 타이밍 체크
                if current_time - last_sample_time >= sample_interval:
                    sensor_data = self.read_sensor_data()
                    if sensor_data:
                        self.collected_data.append(sensor_data)
                        sample_count += 1
                        last_sample_time = current_time
                
                # 실시간 진행률 표시
                progress = int((elapsed / duration) * 30)
                bar = "█" * progress + "░" * (30 - progress)
                remaining = duration - elapsed
                actual_hz = sample_count / elapsed if elapsed > 0 else 0
                
                print(f"\r   [{bar}] {elapsed:.1f}s/{duration}s | "
                      f"샘플: {sample_count}/{total_samples} | "
                      f"실제 Hz: {actual_hz:.1f} | "
                      f"남은시간: {remaining:.1f}s", end="", flush=True)
                
                # 짧은 대기
                await asyncio.sleep(0.001)  # 1ms 정밀도
            
            print(f"\n   ✅ 측정 완료!")
            print(f"   📊 수집된 데이터: {len(self.collected_data)}개")
            print(f"   📈 실제 샘플링: {len(self.collected_data)/duration:.1f}Hz")
            
            # 데이터 품질 체크
            quality_score = self.check_data_quality()
            print(f"   ⭐ 데이터 품질: {quality_score:.1f}%")
            
            if self.collected_data:
                await self.save_data_to_file(settings)
                return True
            else:
                print("   ❌ 수집된 데이터가 없습니다.")
                return False
                
        except KeyboardInterrupt:
            print("\n   ⏹️ 측정이 중단되었습니다.")
            return False
        
        finally:
            self.is_collecting = False
    
    def check_data_quality(self) -> float:
        """데이터 품질 체크"""
        if not self.collected_data:
            return 0.0
        
        quality_factors = []
        
        # 1. 데이터 완결성 (누락된 센서 값 체크)
        completeness = sum(1 for data in self.collected_data 
                          if all([data.flex_sensors.flex_1 is not None,
                                 data.gyro_data.gyro_x is not None])) / len(self.collected_data)
        quality_factors.append(completeness * 100)
        
        # 2. 신호 안정성 (큰 변화 없는지 체크)
        if len(self.collected_data) > 1:
            flex_variations = []
            for i in range(1, len(self.collected_data)):
                prev_flex = [self.collected_data[i-1].flex_sensors.flex_1,
                           self.collected_data[i-1].flex_sensors.flex_2]
                curr_flex = [self.collected_data[i].flex_sensors.flex_1,
                           self.collected_data[i].flex_sensors.flex_2]
                
                variation = sum(abs(a-b) for a, b in zip(prev_flex, curr_flex))
                flex_variations.append(variation)
            
            avg_variation = sum(flex_variations) / len(flex_variations)
            stability = max(0, 100 - avg_variation/10)  # 변화량이 클수록 안정성 낮음
            quality_factors.append(stability)
        
        # 3. 배터리 수준
        battery_levels = [data.battery_level for data in self.collected_data if data.battery_level]
        if battery_levels:
            avg_battery = sum(battery_levels) / len(battery_levels)
            battery_quality = min(100, avg_battery)
            quality_factors.append(battery_quality)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    async def save_data_to_file(self, settings: Dict):
        """데이터를 파일로 저장"""
        filename = settings['filename']
        save_mode = settings['save_mode']
        
        # 파일 경로 설정
        csv_file = self.output_directory / f"{filename}.csv"
        json_file = self.output_directory / f"{filename}.json"
        
        print(f"\n💾 데이터 저장 중...")
        
        try:
            # CSV 저장
            mode = 'w' if save_mode == 'overwrite' else 'a'
            file_exists = csv_file.exists()
            
            with open(csv_file, mode, newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'device_id', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                    'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                    'battery_level', 'signal_strength'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 헤더 쓰기 (새 파일이거나 overwrite 모드)
                if mode == 'w' or not file_exists:
                    writer.writeheader()
                
                # 데이터 쓰기
                for data in self.collected_data:
                    row = {
                        'timestamp': data.timestamp.isoformat(),
                        'device_id': data.device_id,
                        'flex_1': data.flex_sensors.flex_1,
                        'flex_2': data.flex_sensors.flex_2,
                        'flex_3': data.flex_sensors.flex_3,
                        'flex_4': data.flex_sensors.flex_4,
                        'flex_5': data.flex_sensors.flex_5,
                        'gyro_x': data.gyro_data.gyro_x,
                        'gyro_y': data.gyro_data.gyro_y,
                        'gyro_z': data.gyro_data.gyro_z,
                        'accel_x': data.gyro_data.accel_x,
                        'accel_y': data.gyro_data.accel_y,
                        'accel_z': data.gyro_data.accel_z,
                        'battery_level': data.battery_level,
                        'signal_strength': data.signal_strength
                    }
                    writer.writerow(row)
            
            # JSON 메타데이터 저장
            metadata = {
                'experiment_info': {
                    'label': settings['label'],
                    'duration': settings['duration'],
                    'sampling_hz': settings['sampling_hz'],
                    'total_samples': len(self.collected_data),
                    'actual_hz': len(self.collected_data) / settings['duration'],
                    'quality_score': self.check_data_quality(),
                    'performer_id': self.performer_id,
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'save_mode': save_mode
                },
                'data_summary': {
                    'sample_count': len(self.collected_data),
                    'duration_actual': (self.collected_data[-1].timestamp - self.collected_data[0].timestamp).total_seconds(),
                    'flex_ranges': self.get_sensor_ranges(),
                    'gyro_ranges': self.get_gyro_ranges()
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ CSV 저장: {csv_file}")
            print(f"   ✅ 메타데이터 저장: {json_file}")
            
        except Exception as e:
            print(f"   ❌ 저장 실패: {e}")
    
    def get_sensor_ranges(self) -> Dict:
        """플렉스 센서 범위 계산"""
        if not self.collected_data:
            return {}
        
        flex_values = {
            'flex_1': [d.flex_sensors.flex_1 for d in self.collected_data],
            'flex_2': [d.flex_sensors.flex_2 for d in self.collected_data],
            'flex_3': [d.flex_sensors.flex_3 for d in self.collected_data],
            'flex_4': [d.flex_sensors.flex_4 for d in self.collected_data],
            'flex_5': [d.flex_sensors.flex_5 for d in self.collected_data]
        }
        
        ranges = {}
        for sensor, values in flex_values.items():
            ranges[sensor] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'range': max(values) - min(values)
            }
        
        return ranges
    
    def get_gyro_ranges(self) -> Dict:
        """자이로 센서 범위 계산"""
        if not self.collected_data:
            return {}
        
        gyro_values = {
            'gyro_x': [d.gyro_data.gyro_x for d in self.collected_data],
            'gyro_y': [d.gyro_data.gyro_y for d in self.collected_data],
            'gyro_z': [d.gyro_data.gyro_z for d in self.collected_data],
            'accel_x': [d.gyro_data.accel_x for d in self.collected_data],
            'accel_y': [d.gyro_data.accel_y for d in self.collected_data],
            'accel_z': [d.gyro_data.accel_z for d in self.collected_data]
        }
        
        ranges = {}
        for sensor, values in gyro_values.items():
            ranges[sensor] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'range': max(values) - min(values)
            }
        
        return ranges
    
    def disconnect(self):
        """연결 해제"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("🔌 시리얼 연결 해제 완료")
    
    async def run(self):
        """메인 실행 루프"""
        try:
            # 초기화
            await self.data_storage.initialize()
            
            # 아두이노 연결
            if not self.connect_arduino():
                print("❌ 아두이노 연결에 실패했습니다. 프로그램을 종료합니다.")
                return
            
            print(f"\n🎯 고급 데이터 수집 세션 시작")
            print(f"👤 수행자: {self.performer_id}")
            print(f"📁 저장 경로: {self.output_directory}")
            
            experiment_count = 0
            
            while True:
                try:
                    print(f"\n{'='*60}")
                    print(f"🧪 실험 #{experiment_count + 1}")
                    print(f"{'='*60}")
                    
                    # 실험 설정 입력
                    settings = self.get_experiment_settings()
                    
                    # 설정 확인
                    print(f"\n📋 실험 설정 확인:")
                    print(f"   🏷️ 라벨: {settings['label']}")
                    print(f"   ⏱️ 시간: {settings['duration']}초")
                    print(f"   📊 샘플링: {settings['sampling_hz']}Hz")
                    print(f"   📁 파일명: {settings['filename']}")
                    print(f"   💾 저장모드: {settings['save_mode']}")
                    print(f"   📈 예상 샘플: {settings['total_samples']}개")
                    
                    confirm = input("\n이 설정으로 진행하시겠습니까? (y/n): ").strip().lower()
                    if confirm not in ['y', 'yes', '네', 'ㅇ']:
                        print("실험이 취소되었습니다.")
                        continue
                    
                    # 데이터 수집 실행
                    success = await self.collect_data_advanced(settings)
                    
                    if success:
                        experiment_count += 1
                        print(f"\n✅ 실험 #{experiment_count} 완료!")
                    else:
                        print(f"\n❌ 실험 실패")
                    
                    # 다음 실험 여부 확인
                    print(f"\n🔄 다음 라벨을 측정하시겠습니까?")
                    continue_choice = input("계속 (y/n): ").strip().lower()
                    
                    if continue_choice not in ['y', 'yes', '네', 'ㅇ']:
                        break
                        
                except KeyboardInterrupt:
                    break
            
            print(f"\n👋 데이터 수집을 종료합니다.")
            print(f"📊 총 {experiment_count}개 실험 완료")
            
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류 발생: {e}")
        
        finally:
            self.disconnect()
            await self.data_storage.cleanup()


def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 SignGlove 고급 데이터 수집기")
    print("=" * 60)
    print("📋 요구사항:")
    print("   1. 라벨 입력 (ㄱ/ㅏ/1)")
    print("   2. 시간 + Hz 설정")
    print("   3. 파일명 지정 + overwrite")
    print("   4. 다음 측정 여부 확인")
    print("")
    
    try:
        collector = AdvancedDataCollector()
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류: {e}")


if __name__ == "__main__":
    main() 