#!/usr/bin/env python3
"""
SignGlove 통합 데이터 수집 시스템
딥러닝 베스트 프랙티스를 적용한 데이터 수집 및 관리 시스템

기능:
1. 클래스별 개별 저장
2. 통합 데이터셋 저장 
3. Train/Validation/Test 자동 분할
4. 메타데이터 및 통계 관리
5. 실험 추적 및 재현성
"""

import asyncio
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import serial
import serial.tools.list_ports
from serial.serialutil import SerialException

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from server.ksl_classes import ksl_manager, KSLCategory
from server.data_storage import DataStorage
from server.models.sensor_data import SensorData, FlexSensorData, GyroData, SignGestureData, SignGestureType


class UnifiedDataCollector:
    """통합 데이터 수집 시스템"""
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        """초기화"""
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.collected_data: List[SensorData] = []
        self.is_collecting = False
        
        # 베스트 프랙티스 디렉토리 구조 생성
        self.setup_directory_structure()
        
        # 실험 세션 관리
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_config = self.load_experiment_config()
        
        print("🚀 SignGlove 통합 데이터 수집 시스템 초기화 완료")
        print(f"📁 데이터 경로: {self.data_root}")
    
    def setup_directory_structure(self):
        """베스트 프랙티스 디렉토리 구조 생성"""
        self.project_root = Path(".")
        self.data_root = self.project_root / "data"
        
        # 베스트 프랙티스 디렉토리 구조
        self.directories = {
            'raw': self.data_root / "raw",                    # 원본 데이터
            'processed': self.data_root / "processed",        # 전처리된 데이터
            'interim': self.data_root / "interim",            # 임시 데이터
            'experiments': self.data_root / "experiments",    # 실험별 데이터
            'unified': self.data_root / "unified",            # 통합 데이터셋
            'splits': self.data_root / "splits",              # 학습용 분할 데이터
            'metadata': self.data_root / "metadata",          # 메타데이터
            'stats': self.data_root / "stats"                 # 통계 정보
        }
        
        # 디렉토리 생성
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 클래스별 하위 디렉토리 생성
        self.class_categories = ['consonant', 'vowel', 'number']
        for category in self.class_categories:
            (self.directories['raw'] / category).mkdir(exist_ok=True)
            (self.directories['processed'] / category).mkdir(exist_ok=True)
    
    def load_experiment_config(self) -> Dict:
        """실험 설정 로드"""
        config_file = self.directories['metadata'] / "experiment_config.json"
        
        default_config = {
            'target_classes': {
                'consonant': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'],
                'vowel': ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'],
                'number': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            },
            'target_samples_per_class': 60,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'sampling_rate': 20,
            'measurement_duration': 5
        }
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = default_config
            self.save_experiment_config(config)
        
        return config
    
    def save_experiment_config(self, config: Dict):
        """실험 설정 저장"""
        config_file = self.directories['metadata'] / "experiment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def connect_arduino(self) -> bool:
        """아두이노 연결 (기존 코드와 동일)"""
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
    
    def get_collection_mode(self) -> str:
        """데이터 수집 모드 선택"""
        print("\n📊 데이터 수집 모드 선택:")
        print("   1. 단일 클래스 수집 (개별 파일 저장)")
        print("   2. 다중 클래스 수집 (통합 데이터셋 구축)")
        print("   3. 전체 데이터셋 완성 (34개 클래스 모두)")
        
        while True:
            choice = input("선택 (1-3): ").strip()
            if choice == "1":
                return "single"
            elif choice == "2":
                return "multi"
            elif choice == "3":
                return "complete"
            else:
                print("❌ 1-3 중 하나를 선택해주세요.")
    
    def get_storage_strategy(self) -> str:
        """저장 전략 선택"""
        print("\n💾 저장 전략 선택:")
        print("   1. 클래스별 개별 저장만")
        print("   2. 통합 데이터셋 저장만") 
        print("   3. 둘 다 저장 (권장)")
        
        while True:
            choice = input("선택 (1-3): ").strip()
            if choice == "1":
                return "individual"
            elif choice == "2":
                return "unified"
            elif choice == "3":
                return "both"
            else:
                print("❌ 1-3 중 하나를 선택해주세요.")
    
    def get_class_selection(self) -> List[str]:
        """수집할 클래스 선택"""
        print("\n🏷️ 수집할 클래스 선택:")
        
        all_classes = []
        for category, classes in self.experiment_config['target_classes'].items():
            all_classes.extend(classes)
        
        print("전체 클래스 목록:")
        for i, cls in enumerate(all_classes, 1):
            print(f"   {i:2d}. {cls}")
        
        print("\n선택 방법:")
        print("   - 개별 선택: 1,3,5 (쉼표로 구분)")
        print("   - 범위 선택: 1-5")
        print("   - 전체 선택: all")
        print("   - 카테고리: consonant, vowel, number")
        
        while True:
            selection = input("클래스를 선택하세요: ").strip()
            
            try:
                if selection.lower() == "all":
                    return all_classes
                elif selection.lower() in ["consonant", "vowel", "number"]:
                    return self.experiment_config['target_classes'][selection.lower()]
                elif "-" in selection:
                    start, end = map(int, selection.split("-"))
                    return [all_classes[i-1] for i in range(start, end+1)]
                elif "," in selection:
                    indices = [int(x.strip()) for x in selection.split(",")]
                    return [all_classes[i-1] for i in indices]
                else:
                    idx = int(selection)
                    return [all_classes[idx-1]]
            except (ValueError, IndexError):
                print("❌ 잘못된 선택입니다. 다시 입력해주세요.")
    
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
                device_id="UNIFIED_ARDUINO_001",
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
    
    async def collect_single_class(self, class_label: str, samples_count: int, storage_strategy: str) -> bool:
        """단일 클래스 데이터 수집"""
        print(f"\n🎯 클래스 '{class_label}' 수집 시작")
        print(f"   목표 샘플 수: {samples_count}개")
        
        collected_samples = []
        
        for sample_idx in range(samples_count):
            print(f"\n📊 샘플 {sample_idx + 1}/{samples_count}")
            
            # 데이터 수집
            success = await self.collect_single_sample(class_label, sample_idx)
            
            if success and self.collected_data:
                collected_samples.extend(self.collected_data)
                
                # 개별 저장 (옵션)
                if storage_strategy in ["individual", "both"]:
                    await self.save_individual_sample(class_label, sample_idx, self.collected_data)
            
            # 진행률 표시
            progress = (sample_idx + 1) / samples_count * 100
            print(f"   진행률: {progress:.1f}% ({len(collected_samples)}개 샘플 수집됨)")
        
        # 통합 저장 (옵션)
        if storage_strategy in ["unified", "both"]:
            await self.save_unified_class_data(class_label, collected_samples)
        
        print(f"✅ 클래스 '{class_label}' 수집 완료: {len(collected_samples)}개 샘플")
        return True
    
    async def collect_single_sample(self, class_label: str, sample_idx: int) -> bool:
        """단일 샘플 수집"""
        duration = self.experiment_config['measurement_duration']
        sampling_hz = self.experiment_config['sampling_rate']
        
        print(f"   샘플 #{sample_idx + 1} 측정 준비...")
        input("   준비되면 Enter를 눌러주세요...")
        
        # 카운트다운
        for i in range(3, 0, -1):
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
                
                # 진행률 표시
                progress = int((elapsed / duration) * 20)
                bar = "█" * progress + "░" * (20 - progress)
                remaining = duration - elapsed
                
                print(f"\r   [{bar}] {elapsed:.1f}s/{duration}s | 샘플: {sample_count} | 남은시간: {remaining:.1f}s", 
                      end="", flush=True)
                
                await asyncio.sleep(0.001)
            
            print(f"\n   ✅ 측정 완료: {len(self.collected_data)}개 데이터")
            return len(self.collected_data) > 0
            
        except KeyboardInterrupt:
            print("\n   ⏹️ 측정이 중단되었습니다.")
            return False
        
        finally:
            self.is_collecting = False
    
    async def save_individual_sample(self, class_label: str, sample_idx: int, data: List[SensorData]):
        """개별 샘플 저장"""
        # 클래스 카테고리 결정
        category = self.get_class_category(class_label)
        
        # 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_label}_sample_{sample_idx:03d}_{timestamp}"
        
        raw_dir = self.directories['raw'] / category
        csv_file = raw_dir / f"{filename}.csv"
        json_file = raw_dir / f"{filename}.json"
        
        try:
            # CSV 저장
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'device_id', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                    'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                    'battery_level', 'signal_strength'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for sensor_data in data:
                    row = {
                        'timestamp': sensor_data.timestamp.isoformat(),
                        'device_id': sensor_data.device_id,
                        'flex_1': sensor_data.flex_sensors.flex_1,
                        'flex_2': sensor_data.flex_sensors.flex_2,
                        'flex_3': sensor_data.flex_sensors.flex_3,
                        'flex_4': sensor_data.flex_sensors.flex_4,
                        'flex_5': sensor_data.flex_sensors.flex_5,
                        'gyro_x': sensor_data.gyro_data.gyro_x,
                        'gyro_y': sensor_data.gyro_data.gyro_y,
                        'gyro_z': sensor_data.gyro_data.gyro_z,
                        'accel_x': sensor_data.gyro_data.accel_x,
                        'accel_y': sensor_data.gyro_data.accel_y,
                        'accel_z': sensor_data.gyro_data.accel_z,
                        'battery_level': sensor_data.battery_level,
                        'signal_strength': sensor_data.signal_strength
                    }
                    writer.writerow(row)
            
            # 메타데이터 저장
            metadata = {
                'class_label': class_label,
                'category': category,
                'sample_index': sample_idx,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'sample_count': len(data),
                'duration': self.experiment_config['measurement_duration'],
                'sampling_rate': self.experiment_config['sampling_rate']
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"   💾 개별 저장: {csv_file.name}")
            
        except Exception as e:
            print(f"   ❌ 개별 저장 실패: {e}")
    
    async def save_unified_class_data(self, class_label: str, all_samples: List[SensorData]):
        """클래스별 통합 데이터 저장"""
        category = self.get_class_category(class_label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 통합 CSV 파일
        unified_file = self.directories['unified'] / f"{class_label}_unified_{timestamp}.csv"
        
        try:
            with open(unified_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'sample_id', 'timestamp', 'device_id', 'class_label', 'category',
                    'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                    'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                    'battery_level', 'signal_strength'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                sample_id = 0
                for sensor_data in all_samples:
                    row = {
                        'sample_id': sample_id,
                        'timestamp': sensor_data.timestamp.isoformat(),
                        'device_id': sensor_data.device_id,
                        'class_label': class_label,
                        'category': category,
                        'flex_1': sensor_data.flex_sensors.flex_1,
                        'flex_2': sensor_data.flex_sensors.flex_2,
                        'flex_3': sensor_data.flex_sensors.flex_3,
                        'flex_4': sensor_data.flex_sensors.flex_4,
                        'flex_5': sensor_data.flex_sensors.flex_5,
                        'gyro_x': sensor_data.gyro_data.gyro_x,
                        'gyro_y': sensor_data.gyro_data.gyro_y,
                        'gyro_z': sensor_data.gyro_data.gyro_z,
                        'accel_x': sensor_data.gyro_data.accel_x,
                        'accel_y': sensor_data.gyro_data.accel_y,
                        'accel_z': sensor_data.gyro_data.accel_z,
                        'battery_level': sensor_data.battery_level,
                        'signal_strength': sensor_data.signal_strength
                    }
                    writer.writerow(row)
                    sample_id += 1
            
            print(f"   💾 통합 저장: {unified_file.name} ({len(all_samples)}개 샘플)")
            
        except Exception as e:
            print(f"   ❌ 통합 저장 실패: {e}")
    
    def get_class_category(self, class_label: str) -> str:
        """클래스의 카테고리 반환"""
        for category, classes in self.experiment_config['target_classes'].items():
            if class_label in classes:
                return category
        return "unknown"
    
    async def create_master_dataset(self):
        """마스터 데이터셋 생성 (전체 클래스 통합)"""
        print("\n🎯 마스터 데이터셋 생성 중...")
        
        # 모든 통합 파일 찾기
        unified_files = list(self.directories['unified'].glob("*_unified_*.csv"))
        
        if not unified_files:
            print("❌ 통합 데이터 파일이 없습니다.")
            return
        
        all_data = []
        class_stats = {}
        
        for file_path in unified_files:
            df = pd.read_csv(file_path)
            all_data.append(df)
            
            # 클래스별 통계
            for class_label in df['class_label'].unique():
                class_count = len(df[df['class_label'] == class_label])
                if class_label in class_stats:
                    class_stats[class_label] += class_count
                else:
                    class_stats[class_label] = class_count
        
        # 마스터 데이터셋 생성
        master_df = pd.concat(all_data, ignore_index=True)
        
        # 마스터 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_file = self.directories['unified'] / f"master_dataset_{timestamp}.csv"
        master_df.to_csv(master_file, index=False, encoding='utf-8')
        
        # Train/Val/Test 분할
        await self.create_train_val_test_splits(master_df, timestamp)
        
        # 통계 저장
        await self.save_dataset_statistics(class_stats, len(master_df), timestamp)
        
        print(f"✅ 마스터 데이터셋 생성 완료:")
        print(f"   📁 파일: {master_file.name}")
        print(f"   📊 총 샘플: {len(master_df)}개")
        print(f"   🏷️ 클래스 수: {len(class_stats)}개")
    
    async def create_train_val_test_splits(self, master_df: pd.DataFrame, timestamp: str):
        """Train/Validation/Test 분할"""
        print("\n🔀 데이터셋 분할 중...")
        
        train_ratio = self.experiment_config['train_ratio']
        val_ratio = self.experiment_config['val_ratio']
        test_ratio = self.experiment_config['test_ratio']
        
        # 클래스별 층화 분할
        train_data = []
        val_data = []
        test_data = []
        
        for class_label in master_df['class_label'].unique():
            class_data = master_df[master_df['class_label'] == class_label]
            
            # Train/Temp 분할
            train_df, temp_df = train_test_split(
                class_data, 
                test_size=(val_ratio + test_ratio),
                random_state=42,
                shuffle=True
            )
            
            # Val/Test 분할
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_ratio/(val_ratio + test_ratio),
                random_state=42,
                shuffle=True
            )
            
            train_data.append(train_df)
            val_data.append(val_df)
            test_data.append(test_df)
        
        # 분할된 데이터 결합
        train_final = pd.concat(train_data, ignore_index=True).sample(frac=1, random_state=42)
        val_final = pd.concat(val_data, ignore_index=True).sample(frac=1, random_state=42)
        test_final = pd.concat(test_data, ignore_index=True).sample(frac=1, random_state=42)
        
        # 파일 저장
        splits_dir = self.directories['splits']
        train_file = splits_dir / f"train_{timestamp}.csv"
        val_file = splits_dir / f"val_{timestamp}.csv"
        test_file = splits_dir / f"test_{timestamp}.csv"
        
        train_final.to_csv(train_file, index=False, encoding='utf-8')
        val_final.to_csv(val_file, index=False, encoding='utf-8')
        test_final.to_csv(test_file, index=False, encoding='utf-8')
        
        print(f"   ✅ Train: {len(train_final)}개 ({len(train_final)/len(master_df)*100:.1f}%)")
        print(f"   ✅ Val: {len(val_final)}개 ({len(val_final)/len(master_df)*100:.1f}%)")
        print(f"   ✅ Test: {len(test_final)}개 ({len(test_final)/len(master_df)*100:.1f}%)")
    
    async def save_dataset_statistics(self, class_stats: Dict, total_samples: int, timestamp: str):
        """데이터셋 통계 저장"""
        stats = {
            'timestamp': timestamp,
            'total_samples': total_samples,
            'total_classes': len(class_stats),
            'class_distribution': class_stats,
            'data_collection_config': self.experiment_config,
            'session_id': self.session_id,
            'directory_structure': {
                'raw': str(self.directories['raw']),
                'processed': str(self.directories['processed']),
                'unified': str(self.directories['unified']),
                'splits': str(self.directories['splits'])
            }
        }
        
        stats_file = self.directories['stats'] / f"dataset_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"   📊 통계 저장: {stats_file.name}")
    
    def disconnect(self):
        """연결 해제"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("🔌 시리얼 연결 해제 완료")
    
    async def run(self):
        """메인 실행"""
        try:
            print(f"\n🎯 SignGlove 통합 데이터 수집 시작")
            print(f"📁 데이터 구조:")
            for name, path in self.directories.items():
                print(f"   {name}: {path}")
            
            # 아두이노 연결
            if not self.connect_arduino():
                print("❌ 아두이노 연결에 실패했습니다.")
                return
            
            # 수집 모드 선택
            collection_mode = self.get_collection_mode()
            storage_strategy = self.get_storage_strategy()
            
            if collection_mode == "single":
                # 단일 클래스 수집
                classes = self.get_class_selection()
                samples_per_class = self.experiment_config['target_samples_per_class']
                
                for class_label in classes:
                    await self.collect_single_class(class_label, samples_per_class, storage_strategy)
            
            elif collection_mode == "multi":
                # 다중 클래스 수집
                classes = self.get_class_selection()
                samples_per_class = self.experiment_config['target_samples_per_class']
                
                for class_label in classes:
                    await self.collect_single_class(class_label, samples_per_class, storage_strategy)
                
                # 마스터 데이터셋 생성
                if storage_strategy in ["unified", "both"]:
                    await self.create_master_dataset()
            
            elif collection_mode == "complete":
                # 전체 데이터셋 완성
                all_classes = []
                for classes in self.experiment_config['target_classes'].values():
                    all_classes.extend(classes)
                
                print(f"\n🎯 전체 데이터셋 수집 시작: {len(all_classes)}개 클래스")
                
                for i, class_label in enumerate(all_classes, 1):
                    print(f"\n진행률: {i}/{len(all_classes)} ({i/len(all_classes)*100:.1f}%)")
                    await self.collect_single_class(
                        class_label, 
                        self.experiment_config['target_samples_per_class'], 
                        storage_strategy
                    )
                
                # 마스터 데이터셋 생성
                if storage_strategy in ["unified", "both"]:
                    await self.create_master_dataset()
            
            print("\n🎉 데이터 수집 완료!")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
        
        finally:
            self.disconnect()


def main():
    """메인 함수"""
    print("=" * 60)
    print("🚀 SignGlove 통합 데이터 수집 시스템")
    print("=" * 60)
    print("🎯 딥러닝 베스트 프랙티스 적용:")
    print("   - 클래스별 개별 저장")
    print("   - 통합 데이터셋 구축") 
    print("   - Train/Val/Test 자동 분할")
    print("   - 메타데이터 관리")
    print("")
    
    try:
        collector = UnifiedDataCollector()
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류: {e}")


if __name__ == "__main__":
    main() 