#!/usr/bin/env python3
"""
SignGlove USB 데이터 수집기
아두이노와 USB 시리얼 통신을 통해 센서 데이터를 수집하고 CSV로 저장하는 스크립트

사용법:
    python scripts/data_collector.py

기능:
    - 자음/모음/숫자 라벨 선택
    - 1-60초 카운트다운 측정
    - 실시간 센서 데이터 수집
    - CSV 자동 저장
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


class SerialDataCollector:
    """USB 시리얼 기반 데이터 수집기"""
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        """
        데이터 수집기 초기화
        
        Args:
            port: 시리얼 포트 (None일 경우 자동 감지)
            baudrate: 통신 속도
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.data_storage = DataStorage()
        self.is_collecting = False
        self.collected_data: List[SensorData] = []
        
        # 세션 정보
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.performer_id = "test_performer"  # TODO: 사용자 입력으로 변경
        
        print("🤖 SignGlove USB 데이터 수집기 초기화 완료")
    
    def find_arduino_port(self) -> Optional[str]:
        """아두이노 포트 자동 감지"""
        print("\n🔍 아두이노 포트 검색 중...")
        
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        
        for port in ports:
            # 아두이노 관련 키워드로 필터링
            if any(keyword in str(port).lower() for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
                arduino_ports.append(port.device)
                print(f"   ✅ 발견: {port.device} - {port.description}")
        
        if not arduino_ports:
            print("   ❌ 아두이노를 찾을 수 없습니다.")
            return None
        
        if len(arduino_ports) == 1:
            return arduino_ports[0]
        
        # 여러 포트가 있을 경우 사용자 선택
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
            
            # 연결 안정화를 위한 대기
            time.sleep(2)
            
            # 테스트 통신
            if self.test_communication():
                print("✅ 아두이노 연결 성공!")
                return True
            else:
                print("❌ 아두이노 통신 테스트 실패")
                return False
                
        except SerialException as e:
            print(f"❌ 시리얼 연결 실패: {e}")
            return False
    
    def test_communication(self) -> bool:
        """아두이노와의 통신 테스트"""
        try:
            # 테스트 명령 전송
            self.serial_conn.write(b"TEST\n")
            time.sleep(0.5)
            
            # 응답 대기
            response = self.serial_conn.readline().decode().strip()
            return "OK" in response or len(response) > 0
            
        except Exception as e:
            print(f"통신 테스트 오류: {e}")
            return False
    
    def read_sensor_data(self) -> Optional[SensorData]:
        """아두이노에서 센서 데이터 읽기"""
        try:
            if not self.serial_conn or not self.serial_conn.is_open:
                return None
            
            # 데이터 읽기
            line = self.serial_conn.readline().decode().strip()
            if not line:
                return None
            
            # JSON 파싱 시도
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # CSV 형태 파싱 시도
                parts = line.split(',')
                if len(parts) >= 11:  # 최소 필요한 데이터 개수
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
                        'battery': parts[11] if len(parts) > 11 else 100,
                        'signal': parts[12] if len(parts) > 12 else -50
                    }
                else:
                    return None
            
            # SensorData 객체 생성
            sensor_data = SensorData(
                device_id="USB_ARDUINO_001",
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
    
    def show_category_menu(self) -> KSLCategory:
        """카테고리 선택 메뉴"""
        print("\n📋 수어 카테고리를 선택하세요:")
        print("   1. 자음 (ㄱ, ㄴ, ㄷ, ...)")
        print("   2. 모음 (ㅏ, ㅓ, ㅗ, ...)")
        print("   3. 숫자 (0, 1, 2, ...)")
        
        while True:
            try:
                choice = input("선택 (1-3): ").strip()
                if choice == "1":
                    return KSLCategory.CONSONANT
                elif choice == "2":
                    return KSLCategory.VOWEL
                elif choice == "3":
                    return KSLCategory.NUMBER
                else:
                    print("❌ 1, 2, 3 중 하나를 선택해주세요.")
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                sys.exit(0)
    
    def show_available_labels(self, category: KSLCategory) -> List[str]:
        """선택된 카테고리의 사용 가능한 라벨 표시"""
        classes = ksl_manager.get_classes_by_category(category)
        labels = [cls.name for cls in classes]
        
        print(f"\n📝 {category.value} 카테고리의 사용 가능한 라벨:")
        
        # 라벨을 예쁘게 출력
        if category == KSLCategory.CONSONANT:
            print("   자음:", " ".join(labels))
        elif category == KSLCategory.VOWEL:
            print("   모음:", " ".join(labels))
        elif category == KSLCategory.NUMBER:
            print("   숫자:", " ".join(labels))
        
        return labels
    
    def get_label_input(self, available_labels: List[str]) -> str:
        """라벨 입력 받기"""
        while True:
            label = input(f"\n🏷️ 수집할 라벨을 입력하세요: ").strip()
            
            if not label:
                print("❌ 라벨을 입력해주세요.")
                continue
            
            if label in available_labels:
                return label
            else:
                print(f"❌ 사용 가능한 라벨이 아닙니다. 다음 중 선택하세요: {', '.join(available_labels)}")
    
    def get_measurement_duration(self) -> int:
        """측정 시간 입력 받기"""
        while True:
            try:
                duration = int(input("\n⏱️ 측정 시간을 입력하세요 (1-60초): "))
                if 1 <= duration <= 60:
                    return duration
                else:
                    print("❌ 1-60초 사이의 값을 입력해주세요.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                sys.exit(0)
    
    async def collect_data_with_countdown(self, label: str, duration: int, gesture_type: SignGestureType) -> bool:
        """카운트다운과 함께 데이터 수집"""
        print(f"\n🎯 라벨 '{label}' 수집 준비")
        print("⚠️ 손을 올바른 자세로 준비해주세요.")
        
        # 시작 확인
        input("준비가 되면 Enter를 눌러주세요...")
        
        print(f"\n🚀 {duration}초 후 측정이 시작됩니다!")
        
        # 카운트다운
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("   🔴 측정 시작!")
        
        # 데이터 수집 시작
        self.collected_data = []
        self.is_collecting = True
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 진행률 표시
                elapsed = time.time() - start_time
                progress = int((elapsed / duration) * 20)
                bar = "█" * progress + "░" * (20 - progress)
                remaining = duration - elapsed
                print(f"\r   [{bar}] {elapsed:.1f}s / {duration}s (남은 시간: {remaining:.1f}s)", end="", flush=True)
                
                # 센서 데이터 읽기
                sensor_data = self.read_sensor_data()
                if sensor_data:
                    self.collected_data.append(sensor_data)
                
                # 짧은 대기 (데이터 수집 주기 조절)
                await asyncio.sleep(0.05)  # 20Hz 정도
            
            print(f"\n   ✅ 측정 완료! {len(self.collected_data)}개 데이터 수집됨")
            
            # 수집된 데이터를 제스처 데이터로 저장
            if self.collected_data:
                await self.save_gesture_data(label, gesture_type, duration)
                return True
            else:
                print("   ❌ 수집된 데이터가 없습니다.")
                return False
                
        except KeyboardInterrupt:
            print("\n   ⏹️ 측정이 중단되었습니다.")
            return False
        
        finally:
            self.is_collecting = False
    
    async def save_gesture_data(self, label: str, gesture_type: SignGestureType, duration: int):
        """제스처 데이터 저장"""
        try:
            # 제스처 ID 생성
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            gesture_id = f"{label}_{timestamp_str}"
            
            # 제스처 데이터 객체 생성
            gesture_data = SignGestureData(
                gesture_id=gesture_id,
                gesture_label=label,
                gesture_type=gesture_type,
                sensor_sequence=self.collected_data,
                duration=duration,
                performer_id=self.performer_id,
                session_id=self.session_id,
                timestamp=datetime.now(),
                quality_score=0.85,  # TODO: 실제 품질 평가 알고리즘 구현
                notes=f"USB 시리얼 수집 - {len(self.collected_data)}개 샘플"
            )
            
            # 데이터 저장
            success = await self.data_storage.save_gesture_data(gesture_data)
            
            if success:
                print(f"   💾 데이터 저장 완료: {gesture_id}")
            else:
                print(f"   ❌ 데이터 저장 실패")
                
        except Exception as e:
            print(f"   ❌ 데이터 저장 중 오류: {e}")
    
    def disconnect(self):
        """시리얼 연결 해제"""
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
            
            print(f"\n🎯 세션 시작: {self.session_id}")
            print(f"👤 수행자: {self.performer_id}")
            
            while True:
                try:
                    # 카테고리 선택
                    category = self.show_category_menu()
                    
                    # 사용 가능한 라벨 표시
                    available_labels = self.show_available_labels(category)
                    
                    # 라벨 입력
                    label = self.get_label_input(available_labels)
                    
                    # 측정 시간 입력
                    duration = self.get_measurement_duration()
                    
                    # 제스처 타입 결정
                    gesture_type_map = {
                        KSLCategory.CONSONANT: SignGestureType.CONSONANT,
                        KSLCategory.VOWEL: SignGestureType.VOWEL,
                        KSLCategory.NUMBER: SignGestureType.NUMBER
                    }
                    gesture_type = gesture_type_map[category]
                    
                    # 데이터 수집 실행
                    success = await self.collect_data_with_countdown(label, duration, gesture_type)
                    
                    if success:
                        print("\n✅ 데이터 수집이 완료되었습니다!")
                    else:
                        print("\n❌ 데이터 수집에 실패했습니다.")
                    
                    # 계속할지 확인
                    print("\n🔄 다음 라벨을 측정하시겠습니까?")
                    continue_choice = input("계속 (y/n): ").strip().lower()
                    
                    if continue_choice not in ['y', 'yes', '네', 'ㅇ']:
                        break
                        
                except KeyboardInterrupt:
                    break
            
            print("\n👋 데이터 수집을 종료합니다.")
            
            # 통계 출력
            stats = await self.data_storage.get_statistics()
            print(f"\n📊 수집 통계:")
            print(f"   총 센서 레코드: {stats['total_sensor_records']}")
            print(f"   총 제스처 레코드: {stats['total_gesture_records']}")
            
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류 발생: {e}")
        
        finally:
            self.disconnect()
            await self.data_storage.cleanup()


def main():
    """메인 함수"""
    print("=" * 60)
    print("🤖 SignGlove USB 데이터 수집기")
    print("=" * 60)
    
    try:
        collector = SerialDataCollector()
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류: {e}")


if __name__ == "__main__":
    main() 