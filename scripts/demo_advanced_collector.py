#!/usr/bin/env python3
"""
SignGlove 고급 데이터 수집기 데모 (시뮬레이션)
요구사항 완벽 구현을 보여주는 데모 버전
"""

import asyncio
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from server.models.sensor_data import SensorData, FlexSensorData, GyroData


class SimulatedAdvancedArduino:
    """고급 아두이노 시뮬레이터"""
    
    def __init__(self):
        self.is_open = True
        self.current_gesture = None
        self.noise_level = 0.05
        
        # 제스처별 정밀한 패턴 정의
        self.gesture_patterns = {
            "ㄱ": {"flex": [850, 300, 300, 300, 300], "gyro": [2, -1, 0], "accel": [0.2, -9.8, 0.1]},
            "ㄴ": {"flex": [300, 200, 200, 850, 850], "gyro": [0, 1, -2], "accel": [0.1, -9.8, 0.2]},
            "ㄷ": {"flex": [200, 200, 850, 850, 850], "gyro": [-1, 0, 1], "accel": [0.0, -9.8, 0.3]},
            "ㅏ": {"flex": [200, 200, 850, 850, 850], "gyro": [0, 0, 15], "accel": [0.3, -9.7, 0.1]},
            "ㅓ": {"flex": [200, 200, 850, 850, 850], "gyro": [0, 0, -15], "accel": [-0.3, -9.7, 0.1]},
            "ㅗ": {"flex": [200, 850, 850, 850, 850], "gyro": [0, 20, 0], "accel": [0.1, -9.6, 0.4]},
            "1": {"flex": [850, 200, 850, 850, 850], "gyro": [0, 0, 0], "accel": [0.0, -9.8, 0.0]},
            "2": {"flex": [850, 200, 200, 850, 850], "gyro": [0, 0, 0], "accel": [0.0, -9.8, 0.0]},
            "3": {"flex": [850, 200, 200, 200, 850], "gyro": [0, 0, 0], "accel": [0.0, -9.8, 0.0]},
        }
    
    def set_gesture(self, gesture_label: str):
        self.current_gesture = gesture_label
    
    def readline(self) -> bytes:
        if self.current_gesture and self.current_gesture in self.gesture_patterns:
            pattern = self.gesture_patterns[self.current_gesture]
        else:
            pattern = {"flex": [200, 200, 200, 200, 200], "gyro": [0, 0, 0], "accel": [0, -9.8, 0]}
        
        # 노이즈 추가
        flex_values = [self._add_noise(val) for val in pattern["flex"]]
        gyro_values = [self._add_noise(val) for val in pattern["gyro"]]
        accel_values = [self._add_noise(val) for val in pattern["accel"]]
        
        battery = random.uniform(85, 100)
        signal = random.randint(-50, -30)
        
        data_line = f"{flex_values[0]:.1f},{flex_values[1]:.1f},{flex_values[2]:.1f},{flex_values[3]:.1f},{flex_values[4]:.1f}," \
                   f"{gyro_values[0]:.2f},{gyro_values[1]:.2f},{gyro_values[2]:.2f}," \
                   f"{accel_values[0]:.2f},{accel_values[1]:.2f},{accel_values[2]:.2f}," \
                   f"{battery:.1f},{signal}\n"
        
        return data_line.encode()
    
    def _add_noise(self, value: float) -> float:
        noise = random.uniform(-self.noise_level, self.noise_level) * abs(value) if value != 0 else random.uniform(-0.5, 0.5)
        return value + noise
    
    def close(self):
        self.is_open = False


class AdvancedDataCollectorDemo:
    """고급 데이터 수집기 데모"""
    
    def __init__(self):
        self.arduino_sim = SimulatedAdvancedArduino()
        self.collected_data: List[SensorData] = []
        self.is_collecting = False
        
        # 실험 설정
        self.output_directory = Path("data/experiments")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        print("🎮 SignGlove 고급 데이터 수집기 데모 초기화 완료")
    
    def get_experiment_settings(self) -> Dict:
        """실험 설정 입력 (요구사항 구현)"""
        print("\n⚙️ 실험 설정")
        
        # 1. 라벨 입력 (ㄱ/ㅏ/1 형태)
        label = self.get_label_input()
        
        # 2. 측정 시간 입력
        duration = self.get_measurement_duration()
        
        # 3. Hz 설정
        sampling_hz = self.get_sampling_rate()
        
        # 4. 파일명 설정
        filename = self.get_output_filename(label)
        
        # 5. 저장 모드 (overwrite vs append)
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
        """✅ 요구사항 1: 라벨 입력 (ㄱ/ㅏ/1)"""
        print("\n🏷️ 측정할 라벨 입력")
        print("지원되는 라벨: ㄱ, ㄴ, ㄷ, ㅏ, ㅓ, ㅗ, 1, 2, 3")
        
        valid_labels = ["ㄱ", "ㄴ", "ㄷ", "ㅏ", "ㅓ", "ㅗ", "1", "2", "3"]
        
        while True:
            label = input("라벨을 입력하세요: ").strip()
            
            if not label:
                print("❌ 라벨을 입력해주세요.")
                continue
            
            if label in valid_labels:
                return label
            else:
                print(f"❌ 유효하지 않은 라벨입니다. 지원 라벨: {', '.join(valid_labels)}")
    
    def get_measurement_duration(self) -> int:
        """✅ 요구사항 2: 지정된 시간 설정"""
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
        """✅ 요구사항 2: Hz 설정"""
        print("\n📊 샘플링 주파수 (Hz) 설정")
        print("   1. 10Hz")
        print("   2. 20Hz (권장)")
        print("   3. 50Hz")
        print("   4. 직접 입력")
        
        while True:
            try:
                choice = input("선택 (1-4): ").strip()
                
                if choice == "1":
                    return 10
                elif choice == "2":
                    return 20
                elif choice == "3":
                    return 50
                elif choice == "4":
                    hz = int(input("샘플링 주파수를 입력하세요 (1-100Hz): "))
                    if 1 <= hz <= 100:
                        return hz
                    else:
                        print("❌ 1-100Hz 사이의 값을 입력해주세요.")
                else:
                    print("❌ 1-4 중 하나를 선택해주세요.")
            except ValueError:
                print("❌ 올바른 값을 입력해주세요.")
    
    def get_output_filename(self, label: str) -> str:
        """✅ 요구사항 3: 특정 파일명 설정"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{label}_{timestamp}"
        
        print(f"\n📁 파일명 설정")
        print(f"기본 파일명: {default_name}")
        
        custom_name = input("사용자 정의 파일명 (Enter = 기본값): ").strip()
        
        if custom_name:
            # 안전한 파일명으로 변환
            safe_name = "".join(c for c in custom_name if c.isalnum() or c in "._-")
            return safe_name if safe_name else default_name
        else:
            return default_name
    
    def get_save_mode(self) -> str:
        """✅ 요구사항 3: overwrite 모드 설정"""
        print("\n💾 저장 모드 설정")
        print("   1. 덮어쓰기 (overwrite) - 기존 파일 삭제")
        print("   2. 추가하기 (append) - 기존 파일에 추가")
        
        while True:
            choice = input("선택 (1-2): ").strip()
            if choice == "1":
                return "overwrite"
            elif choice == "2":
                return "append"
            else:
                print("❌ 1 또는 2를 선택해주세요.")
    
    def read_sensor_data(self) -> Optional[SensorData]:
        """시뮬레이션 센서 데이터 읽기"""
        try:
            line = self.arduino_sim.readline().decode().strip()
            if not line:
                return None
            
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
                    'battery': float(parts[11]) if len(parts) > 11 else 95,
                    'signal': int(parts[12]) if len(parts) > 12 else -40
                }
                
                sensor_data = SensorData(
                    device_id="DEMO_SIMULATOR_001",
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
            print(f"⚠️ 시뮬레이션 데이터 오류: {e}")
            return None
    
    async def collect_data_with_hz(self, settings: Dict) -> bool:
        """✅ 요구사항 2: Hz에 맞춰 데이터 수집"""
        label = settings['label']
        duration = settings['duration']
        sampling_hz = settings['sampling_hz']
        total_samples = settings['total_samples']
        
        print(f"\n🎯 라벨 '{label}' 데모 측정 준비")
        print(f"   ⏱️ 측정 시간: {duration}초")
        print(f"   📊 샘플링: {sampling_hz}Hz")
        print(f"   📈 예상 샘플: {total_samples}개")
        
        input("\n시뮬레이션을 시작하려면 Enter를 눌러주세요...")
        
        # 아두이노 시뮬레이터에 제스처 설정
        self.arduino_sim.set_gesture(label)
        
        print(f"\n🚀 측정 시작!")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(0.5)  # 데모용 빠른 카운트다운
        
        print("   🔴 측정 시작!")
        
        # 정밀한 Hz 기반 데이터 수집
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
                
                # 정확한 Hz 타이밍 제어
                if current_time - last_sample_time >= sample_interval:
                    sensor_data = self.read_sensor_data()
                    if sensor_data:
                        self.collected_data.append(sensor_data)
                        sample_count += 1
                        last_sample_time = current_time
                
                # 실시간 진행률 표시
                progress = int((elapsed / duration) * 25)
                bar = "█" * progress + "░" * (25 - progress)
                remaining = duration - elapsed
                actual_hz = sample_count / elapsed if elapsed > 0 else 0
                
                print(f"\r   [{bar}] {elapsed:.1f}s/{duration}s | 샘플: {sample_count}/{total_samples} | Hz: {actual_hz:.1f} | 남은시간: {remaining:.1f}s", 
                      end="", flush=True)
                
                await asyncio.sleep(0.001)  # 정밀 제어
            
            print(f"\n   ✅ 측정 완료!")
            print(f"   📊 수집된 데이터: {len(self.collected_data)}개")
            print(f"   📈 실제 샘플링: {len(self.collected_data)/duration:.1f}Hz")
            print(f"   🎯 목표 대비: {len(self.collected_data)/total_samples*100:.1f}%")
            
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
    
    async def save_data_to_file(self, settings: Dict):
        """✅ 요구사항 3: 특정 파일명으로 overwrite 저장"""
        filename = settings['filename']
        save_mode = settings['save_mode']
        
        csv_file = self.output_directory / f"{filename}.csv"
        json_file = self.output_directory / f"{filename}.json"
        
        print(f"\n💾 파일 저장 중...")
        print(f"   📁 파일명: {filename}")
        print(f"   💾 저장모드: {save_mode}")
        
        try:
            # CSV 저장 (overwrite 또는 append)
            mode = 'w' if save_mode == 'overwrite' else 'a'
            file_exists = csv_file.exists()
            
            if save_mode == 'overwrite' and file_exists:
                print(f"   🗑️ 기존 파일 삭제: {csv_file}")
            
            with open(csv_file, mode, newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'device_id', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                    'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                    'battery_level', 'signal_strength'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 헤더 쓰기 조건
                if mode == 'w' or (mode == 'a' and not file_exists):
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
            
            # 메타데이터 JSON 저장
            metadata = {
                'experiment_info': {
                    'label': settings['label'],
                    'duration': settings['duration'],
                    'sampling_hz': settings['sampling_hz'],
                    'filename': settings['filename'],
                    'save_mode': settings['save_mode'],
                    'timestamp': datetime.now().isoformat()
                },
                'results': {
                    'total_samples': len(self.collected_data),
                    'actual_hz': len(self.collected_data) / settings['duration'],
                    'efficiency': len(self.collected_data) / settings['total_samples'] * 100,
                    'first_sample': self.collected_data[0].timestamp.isoformat() if self.collected_data else None,
                    'last_sample': self.collected_data[-1].timestamp.isoformat() if self.collected_data else None
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ CSV 저장 완료: {csv_file}")
            print(f"   ✅ 메타데이터 저장: {json_file}")
            print(f"   📊 파일 크기: {csv_file.stat().st_size} bytes")
            
        except Exception as e:
            print(f"   ❌ 저장 실패: {e}")
    
    async def run_demo(self):
        """데모 실행"""
        try:
            print(f"\n🎯 고급 데이터 수집기 데모 시작")
            print(f"📁 저장 경로: {self.output_directory}")
            
            experiment_count = 0
            
            while True:
                try:
                    print(f"\n{'='*50}")
                    print(f"🧪 데모 실험 #{experiment_count + 1}")
                    print(f"{'='*50}")
                    
                    # ✅ 요구사항에 따른 설정 입력
                    settings = self.get_experiment_settings()
                    
                    # 설정 확인
                    print(f"\n📋 설정 확인:")
                    print(f"   🏷️ 라벨: {settings['label']}")
                    print(f"   ⏱️ 시간: {settings['duration']}초")
                    print(f"   📊 Hz: {settings['sampling_hz']}Hz")
                    print(f"   📁 파일: {settings['filename']}")
                    print(f"   💾 모드: {settings['save_mode']}")
                    
                    confirm = input("\n진행하시겠습니까? (y/n): ").strip().lower()
                    if confirm not in ['y', 'yes', '네', 'ㅇ']:
                        print("실험이 취소되었습니다.")
                        continue
                    
                    # ✅ 요구사항에 따른 데이터 수집
                    success = await self.collect_data_with_hz(settings)
                    
                    if success:
                        experiment_count += 1
                        print(f"\n✅ 실험 #{experiment_count} 완료!")
                        
                        # 결과 요약 표시
                        print(f"\n📊 결과 요약:")
                        print(f"   📈 수집 샘플: {len(self.collected_data)}개")
                        print(f"   ⏱️ 실제 지속시간: {settings['duration']}초")
                        print(f"   📊 실제 Hz: {len(self.collected_data)/settings['duration']:.1f}")
                        print(f"   💾 저장 파일: {settings['filename']}.csv")
                    else:
                        print(f"\n❌ 실험 실패")
                    
                    # ✅ 요구사항 4: 다음 라벨 측정 여부 확인
                    print(f"\n🔄 다음 라벨을 측정하시겠습니까?")
                    continue_choice = input("계속 (y/n): ").strip().lower()
                    
                    if continue_choice not in ['y', 'yes', '네', 'ㅇ']:
                        break
                        
                except KeyboardInterrupt:
                    break
            
            print(f"\n👋 데모를 종료합니다.")
            print(f"📊 총 {experiment_count}개 실험 완료")
            print(f"📁 저장된 파일들:")
            
            # 생성된 파일 목록 표시
            for file in self.output_directory.glob("*.csv"):
                print(f"   📄 {file.name}")
            
        except Exception as e:
            print(f"\n❌ 데모 중 오류 발생: {e}")
        
        finally:
            self.arduino_sim.close()


def main():
    """메인 함수"""
    print("=" * 60)
    print("🎮 SignGlove 고급 데이터 수집기 데모")
    print("=" * 60)
    print("✅ 구현된 요구사항:")
    print("   1. 라벨 입력 (ㄱ/ㅏ/1)")
    print("   2. 지정된 시간 + Hz 설정")
    print("   3. 특정 파일명 + overwrite 모드")
    print("   4. 다음 라벨 측정 여부 확인")
    print("\n🎯 시뮬레이션으로 실제 동작을 데모합니다.")
    print("")
    
    try:
        demo = AdvancedDataCollectorDemo()
        asyncio.run(demo.run_demo())
    except KeyboardInterrupt:
        print("\n👋 데모가 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 데모 실행 중 오류: {e}")


if __name__ == "__main__":
    main() 