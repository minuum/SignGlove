#!/usr/bin/env python3
"""
SignGlove 데이터 수집기 테스트 및 시뮬레이션 스크립트
아두이노 하드웨어 없이도 데이터 수집 시스템을 테스트할 수 있는 시뮬레이션 모드 제공

사용법:
    python scripts/test_data_collector.py
"""

import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from server.ksl_classes import ksl_manager, KSLCategory
from server.data_storage import DataStorage
from server.models.sensor_data import SensorData, FlexSensorData, GyroData, SignGestureData, SignGestureType


class SimulatedArduino:
    """아두이노 시뮬레이터"""
    
    def __init__(self):
        """시뮬레이터 초기화"""
        self.is_open = True
        self.current_gesture = None
        self.noise_level = 0.1  # 노이즈 레벨
        
        # 기본 센서 값 (손을 편 상태)
        self.base_flex = [200, 200, 200, 200, 200]  # 낮은 값 = 펼친 상태
        self.base_gyro = [0.0, 0.0, 0.0]  # 정지 상태
        self.base_accel = [0.0, -9.8, 0.0]  # 중력만 작용
        
        # 제스처별 센서 패턴 정의
        self.gesture_patterns = {
            # 자음 패턴
            "ㄱ": {"flex": [800, 300, 300, 300, 300], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "ㄴ": {"flex": [300, 200, 200, 800, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "ㄷ": {"flex": [200, 200, 800, 800, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "ㄹ": {"flex": [300, 300, 300, 300, 800], "gyro": [5, 0, 0], "accel": [1, -9.8, 0]},
            "ㅁ": {"flex": [800, 800, 800, 800, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            
            # 모음 패턴
            "ㅏ": {"flex": [200, 200, 800, 800, 800], "gyro": [0, 0, 10], "accel": [0, -9.8, 0]},
            "ㅓ": {"flex": [200, 200, 800, 800, 800], "gyro": [0, 0, -10], "accel": [0, -9.8, 0]},
            "ㅗ": {"flex": [200, 800, 800, 800, 800], "gyro": [0, 15, 0], "accel": [0, -9.8, 0]},
            "ㅜ": {"flex": [200, 800, 800, 800, 800], "gyro": [0, -15, 0], "accel": [0, -9.8, 0]},
            "ㅣ": {"flex": [800, 200, 800, 800, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            
            # 숫자 패턴
            "0": {"flex": [800, 800, 800, 800, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "1": {"flex": [800, 200, 800, 800, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "2": {"flex": [800, 200, 200, 800, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "3": {"flex": [800, 200, 200, 200, 800], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "4": {"flex": [800, 200, 200, 200, 200], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
            "5": {"flex": [200, 200, 200, 200, 200], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]},
        }
    
    def set_gesture(self, gesture_label: str):
        """현재 제스처 설정"""
        self.current_gesture = gesture_label
    
    def write(self, data: bytes):
        """명령 수신 (시뮬레이션)"""
        pass
    
    def readline(self) -> bytes:
        """센서 데이터 생성 및 반환"""
        if not self.current_gesture:
            # 기본 상태 (손을 편 상태)
            flex_values = [self._add_noise(val) for val in self.base_flex]
            gyro_values = [self._add_noise(val) for val in self.base_gyro]
            accel_values = [self._add_noise(val) for val in self.base_accel]
        else:
            # 설정된 제스처 패턴
            pattern = self.gesture_patterns.get(self.current_gesture, {
                "flex": self.base_flex,
                "gyro": self.base_gyro, 
                "accel": self.base_accel
            })
            
            flex_values = [self._add_noise(val) for val in pattern["flex"]]
            gyro_values = [self._add_noise(val) for val in pattern["gyro"]]
            accel_values = [self._add_noise(val) for val in pattern["accel"]]
        
        # 배터리 및 신호 강도 시뮬레이션
        battery = random.uniform(80, 100)
        signal = random.randint(-60, -30)
        
        # CSV 형태로 데이터 생성
        data_line = f"{flex_values[0]:.1f},{flex_values[1]:.1f},{flex_values[2]:.1f},{flex_values[3]:.1f},{flex_values[4]:.1f}," \
                   f"{gyro_values[0]:.2f},{gyro_values[1]:.2f},{gyro_values[2]:.2f}," \
                   f"{accel_values[0]:.2f},{accel_values[1]:.2f},{accel_values[2]:.2f}," \
                   f"{battery:.1f},{signal}\n"
        
        return data_line.encode()
    
    def _add_noise(self, value: float) -> float:
        """센서 값에 노이즈 추가"""
        noise = random.uniform(-self.noise_level, self.noise_level) * abs(value) if value != 0 else random.uniform(-1, 1)
        return value + noise
    
    def close(self):
        """연결 종료"""
        self.is_open = False


class SimulatedDataCollector:
    """시뮬레이션 데이터 수집기"""
    
    def __init__(self):
        """시뮬레이터 초기화"""
        self.arduino_sim = SimulatedArduino()
        self.data_storage = DataStorage()
        self.is_collecting = False
        self.collected_data: List[SensorData] = []
        
        # 세션 정보
        self.session_id = f"sim_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.performer_id = "simulator_test"
        
        print("🎮 SignGlove 시뮬레이터 초기화 완료")
    
    def read_sensor_data(self) -> Optional[SensorData]:
        """시뮬레이션 센서 데이터 읽기"""
        try:
            # 시뮬레이션 데이터 읽기
            line = self.arduino_sim.readline().decode().strip()
            if not line:
                return None
            
            # CSV 파싱
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
                
                # SensorData 객체 생성
                sensor_data = SensorData(
                    device_id="SIMULATOR_001",
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
                    battery_level=data['battery'],
                    signal_strength=data['signal']
                )
                
                return sensor_data
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ 시뮬레이션 데이터 읽기 오류: {e}")
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
        """시뮬레이션 가능한 라벨 표시"""
        if category == KSLCategory.CONSONANT:
            labels = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ"]
        elif category == KSLCategory.VOWEL:
            labels = ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ"]
        elif category == KSLCategory.NUMBER:
            labels = ["0", "1", "2", "3", "4", "5"]
        else:
            labels = []
        
        print(f"\n📝 {category.value} 카테고리의 시뮬레이션 가능한 라벨:")
        print(f"   {' '.join(labels)}")
        print("   ⚠️ 시뮬레이션 모드에서는 제한된 라벨만 지원됩니다.")
        
        return labels
    
    def get_label_input(self, available_labels: List[str]) -> str:
        """라벨 입력 받기"""
        while True:
            label = input(f"\n🏷️ 시뮬레이션할 라벨을 입력하세요: ").strip()
            
            if not label:
                print("❌ 라벨을 입력해주세요.")
                continue
            
            if label in available_labels:
                return label
            else:
                print(f"❌ 시뮬레이션 가능한 라벨이 아닙니다. 다음 중 선택하세요: {', '.join(available_labels)}")
    
    def get_measurement_duration(self) -> int:
        """측정 시간 입력 받기"""
        while True:
            try:
                duration = int(input("\n⏱️ 시뮬레이션 시간을 입력하세요 (1-60초): "))
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
        """시뮬레이션 데이터 수집"""
        print(f"\n🎮 라벨 '{label}' 시뮬레이션 준비")
        print("⚠️ 시뮬레이션 모드에서 가상 센서 데이터를 생성합니다.")
        
        # 시작 확인
        input("시뮬레이션을 시작하려면 Enter를 눌러주세요...")
        
        # 아두이노 시뮬레이터에 제스처 설정
        self.arduino_sim.set_gesture(label)
        
        print(f"\n🚀 {duration}초간 시뮬레이션을 시작합니다!")
        
        # 카운트다운
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print("   🔴 시뮬레이션 시작!")
        
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
                
                # 시뮬레이션 센서 데이터 읽기
                sensor_data = self.read_sensor_data()
                if sensor_data:
                    self.collected_data.append(sensor_data)
                
                # 데이터 수집 주기 (약 20Hz)
                await asyncio.sleep(0.05)
            
            print(f"\n   ✅ 시뮬레이션 완료! {len(self.collected_data)}개 데이터 생성됨")
            
            # 수집된 데이터를 제스처 데이터로 저장
            if self.collected_data:
                await self.save_gesture_data(label, gesture_type, duration)
                return True
            else:
                print("   ❌ 생성된 데이터가 없습니다.")
                return False
                
        except KeyboardInterrupt:
            print("\n   ⏹️ 시뮬레이션이 중단되었습니다.")
            return False
        
        finally:
            self.is_collecting = False
    
    async def save_gesture_data(self, label: str, gesture_type: SignGestureType, duration: int):
        """시뮬레이션 제스처 데이터 저장"""
        try:
            # 제스처 ID 생성
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            gesture_id = f"SIM_{label}_{timestamp_str}"
            
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
                quality_score=0.95,  # 시뮬레이션은 높은 품질
                notes=f"시뮬레이션 데이터 - {len(self.collected_data)}개 샘플"
            )
            
            # 데이터 저장
            success = await self.data_storage.save_gesture_data(gesture_data)
            
            if success:
                print(f"   💾 시뮬레이션 데이터 저장 완료: {gesture_id}")
            else:
                print(f"   ❌ 데이터 저장 실패")
                
        except Exception as e:
            print(f"   ❌ 데이터 저장 중 오류: {e}")
    
    async def run(self):
        """시뮬레이터 메인 실행"""
        try:
            # 초기화
            await self.data_storage.initialize()
            
            print(f"\n🎮 시뮬레이션 세션 시작: {self.session_id}")
            print(f"👤 시뮬레이터: {self.performer_id}")
            
            while True:
                try:
                    # 카테고리 선택
                    category = self.show_category_menu()
                    
                    # 시뮬레이션 가능한 라벨 표시
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
                    
                    # 시뮬레이션 실행
                    success = await self.collect_data_with_countdown(label, duration, gesture_type)
                    
                    if success:
                        print("\n✅ 시뮬레이션이 완료되었습니다!")
                    else:
                        print("\n❌ 시뮬레이션에 실패했습니다.")
                    
                    # 계속할지 확인
                    print("\n🔄 다른 라벨을 시뮬레이션하시겠습니까?")
                    continue_choice = input("계속 (y/n): ").strip().lower()
                    
                    if continue_choice not in ['y', 'yes', '네', 'ㅇ']:
                        break
                        
                except KeyboardInterrupt:
                    break
            
            print("\n👋 시뮬레이션을 종료합니다.")
            
            # 통계 출력
            stats = await self.data_storage.get_statistics()
            print(f"\n📊 시뮬레이션 통계:")
            print(f"   총 센서 레코드: {stats['total_sensor_records']}")
            print(f"   총 제스처 레코드: {stats['total_gesture_records']}")
            
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류 발생: {e}")
        
        finally:
            self.arduino_sim.close()
            await self.data_storage.cleanup()


def main():
    """메인 함수"""
    print("=" * 60)
    print("🎮 SignGlove 시뮬레이션 테스트")
    print("=" * 60)
    print("⚠️ 이 프로그램은 아두이노 하드웨어 없이 데이터 수집 시스템을")
    print("   테스트하기 위한 시뮬레이션 모드입니다.")
    print("")
    
    try:
        simulator = SimulatedDataCollector()
        asyncio.run(simulator.run())
    except KeyboardInterrupt:
        print("\n👋 시뮬레이션이 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 시뮬레이션 실행 중 오류: {e}")


if __name__ == "__main__":
    main() 