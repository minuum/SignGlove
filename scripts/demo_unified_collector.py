#!/usr/bin/env python3
"""
SignGlove 통합 데이터 수집 시스템 데모
베스트 프랙티스 적용된 데이터 구조 및 수집 방식 시연
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
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from server.models.sensor_data import SensorData, FlexSensorData, GyroData


class DemoUnifiedCollector:
    """통합 데이터 수집 시스템 데모"""
    
    def __init__(self):
        """초기화"""
        self.setup_directory_structure()
        self.experiment_config = self.get_demo_config()
        self.session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("🚀 SignGlove 통합 데이터 수집 시스템 데모 초기화 완료")
        print(f"📁 베스트 프랙티스 디렉토리 구조 생성됨")
    
    def setup_directory_structure(self):
        """베스트 프랙티스 디렉토리 구조 생성"""
        self.project_root = Path(".")
        self.data_root = self.project_root / "data"
        
        # 딥러닝 베스트 프랙티스 구조
        self.directories = {
            'raw': self.data_root / "raw",                    # 원본 데이터
            'processed': self.data_root / "processed",        # 전처리된 데이터
            'interim': self.data_root / "interim",            # 임시 데이터
            'unified': self.data_root / "unified",            # 통합 데이터셋
            'splits': self.data_root / "splits",              # 학습용 분할 데이터
            'metadata': self.data_root / "metadata",          # 메타데이터
            'stats': self.data_root / "stats"                 # 통계 정보
        }
        
        # 디렉토리 생성
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 클래스별 하위 디렉토리
        categories = ['consonant', 'vowel', 'number']
        for category in categories:
            (self.directories['raw'] / category).mkdir(exist_ok=True)
            (self.directories['processed'] / category).mkdir(exist_ok=True)
    
    def get_demo_config(self) -> Dict:
        """데모 설정"""
        return {
            'target_classes': {
                'consonant': ['ㄱ', 'ㄴ', 'ㄷ'],
                'vowel': ['ㅏ', 'ㅓ', 'ㅗ'],
                'number': ['1', '2', '3']
            },
            'target_samples_per_class': 5,  # 데모용 적은 샘플
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'sampling_rate': 20,
            'measurement_duration': 3  # 데모용 짧은 시간
        }
    
    def show_directory_structure(self):
        """디렉토리 구조 표시"""
        print("\n📁 생성된 베스트 프랙티스 디렉토리 구조:")
        print("data/")
        print("├── raw/                    # 원본 데이터")
        print("│   ├── consonant/          # 자음 클래스")
        print("│   ├── vowel/              # 모음 클래스") 
        print("│   └── number/             # 숫자 클래스")
        print("├── processed/              # 전처리된 데이터")
        print("├── interim/                # 임시 데이터")
        print("├── unified/                # 통합 데이터셋")
        print("├── splits/                 # Train/Val/Test 분할")
        print("├── metadata/               # 메타데이터")
        print("└── stats/                  # 통계 정보")
        
        # 실제 경로 확인
        print(f"\n실제 생성된 경로:")
        for name, path in self.directories.items():
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {name}: {path}")
    
    def get_collection_mode(self) -> str:
        """수집 모드 선택 (데모용)"""
        print("\n📊 데이터 수집 모드 선택:")
        print("   1. 단일 클래스 데모 (ㄱ)")
        print("   2. 다중 클래스 데모 (ㄱ, ㅏ, 1)")
        print("   3. 전체 데이터셋 데모 (9개 클래스)")
        
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
        print("   3. 둘 다 저장 (머신러닝 최적화)")
        
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
    
    def generate_demo_sensor_data(self, class_label: str, sample_count: int) -> List[SensorData]:
        """데모용 센서 데이터 생성"""
        # 클래스별 패턴 정의
        patterns = {
            'ㄱ': {'flex': [850, 300, 300, 300, 300], 'gyro': [2, -1, 0], 'accel': [0.2, -9.8, 0.1]},
            'ㄴ': {'flex': [300, 200, 200, 850, 850], 'gyro': [0, 1, -2], 'accel': [0.1, -9.8, 0.2]},
            'ㄷ': {'flex': [200, 200, 850, 850, 850], 'gyro': [-1, 0, 1], 'accel': [0.0, -9.8, 0.3]},
            'ㅏ': {'flex': [200, 200, 850, 850, 850], 'gyro': [0, 0, 15], 'accel': [0.3, -9.7, 0.1]},
            'ㅓ': {'flex': [200, 200, 850, 850, 850], 'gyro': [0, 0, -15], 'accel': [-0.3, -9.7, 0.1]},
            'ㅗ': {'flex': [200, 850, 850, 850, 850], 'gyro': [0, 20, 0], 'accel': [0.1, -9.6, 0.4]},
            '1': {'flex': [850, 200, 850, 850, 850], 'gyro': [0, 0, 0], 'accel': [0.0, -9.8, 0.0]},
            '2': {'flex': [850, 200, 200, 850, 850], 'gyro': [0, 0, 0], 'accel': [0.0, -9.8, 0.0]},
            '3': {'flex': [850, 200, 200, 200, 850], 'gyro': [0, 0, 0], 'accel': [0.0, -9.8, 0.0]},
        }
        
        pattern = patterns.get(class_label, patterns['ㄱ'])
        data_list = []
        
        for _ in range(sample_count):
            # 노이즈 추가
            flex_values = [val + random.uniform(-20, 20) for val in pattern['flex']]
            gyro_values = [val + random.uniform(-0.5, 0.5) for val in pattern['gyro']]
            accel_values = [val + random.uniform(-0.1, 0.1) for val in pattern['accel']]
            
            sensor_data = SensorData(
                device_id="DEMO_UNIFIED_001",
                timestamp=datetime.now(),
                flex_sensors=FlexSensorData(
                    flex_1=flex_values[0],
                    flex_2=flex_values[1],
                    flex_3=flex_values[2],
                    flex_4=flex_values[3],
                    flex_5=flex_values[4]
                ),
                gyro_data=GyroData(
                    gyro_x=gyro_values[0],
                    gyro_y=gyro_values[1],
                    gyro_z=gyro_values[2],
                    accel_x=accel_values[0],
                    accel_y=accel_values[1],
                    accel_z=accel_values[2]
                ),
                battery_level=random.uniform(85, 100),
                signal_strength=random.randint(-50, -30)
            )
            data_list.append(sensor_data)
        
        return data_list
    
    async def demo_single_class_collection(self, class_label: str, storage_strategy: str):
        """단일 클래스 수집 데모"""
        print(f"\n🎯 클래스 '{class_label}' 수집 데모")
        samples_count = self.experiment_config['target_samples_per_class']
        
        all_samples = []
        
        for sample_idx in range(samples_count):
            print(f"   샘플 {sample_idx + 1}/{samples_count} 생성 중...")
            
            # 데모 데이터 생성 (실제로는 센서에서 수집)
            sample_data = self.generate_demo_sensor_data(class_label, 20)  # 샘플당 20개 데이터포인트
            all_samples.extend(sample_data)
            
            # 개별 저장 (옵션)
            if storage_strategy in ["individual", "both"]:
                await self.save_individual_sample_demo(class_label, sample_idx, sample_data)
            
            time.sleep(0.1)  # 데모용 짧은 대기
        
        # 통합 저장 (옵션)
        if storage_strategy in ["unified", "both"]:
            await self.save_unified_class_data_demo(class_label, all_samples)
        
        print(f"✅ 클래스 '{class_label}' 수집 완료: {len(all_samples)}개 데이터포인트")
        return all_samples
    
    async def save_individual_sample_demo(self, class_label: str, sample_idx: int, data: List[SensorData]):
        """개별 샘플 저장 데모"""
        category = self.get_class_category(class_label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_label}_sample_{sample_idx:03d}_{timestamp}"
        
        raw_dir = self.directories['raw'] / category
        csv_file = raw_dir / f"{filename}.csv"
        
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
        
        print(f"     💾 개별 저장: {csv_file.name}")
    
    async def save_unified_class_data_demo(self, class_label: str, all_samples: List[SensorData]):
        """클래스별 통합 데이터 저장 데모"""
        category = self.get_class_category(class_label)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        unified_file = self.directories['unified'] / f"{class_label}_unified_{timestamp}.csv"
        
        with open(unified_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'sample_id', 'timestamp', 'device_id', 'class_label', 'category',
                'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                'battery_level', 'signal_strength'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample_id, sensor_data in enumerate(all_samples):
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
        
        print(f"     💾 통합 저장: {unified_file.name}")
    
    def get_class_category(self, class_label: str) -> str:
        """클래스 카테고리 반환"""
        for category, classes in self.experiment_config['target_classes'].items():
            if class_label in classes:
                return category
        return "unknown"
    
    async def create_master_dataset_demo(self):
        """마스터 데이터셋 생성 데모"""
        print("\n🎯 마스터 데이터셋 생성 데모...")
        
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
        await self.create_train_val_test_splits_demo(master_df, timestamp)
        
        # 통계 저장
        await self.save_dataset_statistics_demo(class_stats, len(master_df), timestamp)
        
        print(f"✅ 마스터 데이터셋 생성 완료:")
        print(f"   📁 파일: {master_file.name}")
        print(f"   📊 총 샘플: {len(master_df)}개")
        print(f"   🏷️ 클래스 수: {len(class_stats)}개")
        
        return master_df, class_stats
    
    async def create_train_val_test_splits_demo(self, master_df: pd.DataFrame, timestamp: str):
        """Train/Val/Test 분할 데모"""
        print("   🔀 데이터셋 분할 중...")
        
        from sklearn.model_selection import train_test_split
        
        train_ratio = self.experiment_config['train_ratio']
        val_ratio = self.experiment_config['val_ratio']
        test_ratio = self.experiment_config['test_ratio']
        
        # 클래스별 층화 분할
        train_data = []
        val_data = []
        test_data = []
        
        for class_label in master_df['class_label'].unique():
            class_data = master_df[master_df['class_label'] == class_label]
            
            if len(class_data) < 3:  # 최소 3개 샘플 필요
                print(f"     ⚠️ 클래스 '{class_label}': 샘플 부족 ({len(class_data)}개)")
                continue
            
            # Train/Temp 분할
            train_df, temp_df = train_test_split(
                class_data, 
                test_size=(val_ratio + test_ratio),
                random_state=42,
                shuffle=True
            )
            
            if len(temp_df) >= 2:
                # Val/Test 분할
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=test_ratio/(val_ratio + test_ratio),
                    random_state=42,
                    shuffle=True
                )
            else:
                val_df = temp_df.iloc[:1] if len(temp_df) >= 1 else pd.DataFrame()
                test_df = temp_df.iloc[1:] if len(temp_df) >= 2 else pd.DataFrame()
            
            train_data.append(train_df)
            val_data.append(val_df)
            test_data.append(test_df)
        
        # 분할된 데이터 결합
        train_final = pd.concat(train_data, ignore_index=True)
        val_final = pd.concat(val_data, ignore_index=True)
        test_final = pd.concat(test_data, ignore_index=True)
        
        # 파일 저장
        splits_dir = self.directories['splits']
        train_file = splits_dir / f"train_{timestamp}.csv"
        val_file = splits_dir / f"val_{timestamp}.csv"
        test_file = splits_dir / f"test_{timestamp}.csv"
        
        train_final.to_csv(train_file, index=False, encoding='utf-8')
        val_final.to_csv(val_file, index=False, encoding='utf-8')
        test_final.to_csv(test_file, index=False, encoding='utf-8')
        
        print(f"     ✅ Train: {len(train_final)}개 샘플 → {train_file.name}")
        print(f"     ✅ Val: {len(val_final)}개 샘플 → {val_file.name}")
        print(f"     ✅ Test: {len(test_final)}개 샘플 → {test_file.name}")
    
    async def save_dataset_statistics_demo(self, class_stats: Dict, total_samples: int, timestamp: str):
        """데이터셋 통계 저장 데모"""
        stats = {
            'timestamp': timestamp,
            'total_samples': total_samples,
            'total_classes': len(class_stats),
            'class_distribution': class_stats,
            'data_collection_config': self.experiment_config,
            'session_id': self.session_id,
            'directory_structure': {name: str(path) for name, path in self.directories.items()}
        }
        
        stats_file = self.directories['stats'] / f"dataset_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"     📊 통계 저장: {stats_file.name}")
    
    def show_final_results(self):
        """최종 결과 표시"""
        print("\n📊 최종 생성된 파일들:")
        
        for name, path in self.directories.items():
            files = list(path.glob("*"))
            if files:
                print(f"\n📁 {name}/ ({len(files)}개 파일)")
                for file in files[:5]:  # 최대 5개만 표시
                    size = file.stat().st_size if file.is_file() else 0
                    print(f"   📄 {file.name} ({size} bytes)")
                if len(files) > 5:
                    print(f"   ... 및 {len(files)-5}개 파일 더")
    
    async def run_demo(self):
        """데모 실행"""
        try:
            # 디렉토리 구조 표시
            self.show_directory_structure()
            
            # 수집 모드 선택
            collection_mode = self.get_collection_mode()
            storage_strategy = self.get_storage_strategy()
            
            print(f"\n🚀 데모 시작: {collection_mode} 모드, {storage_strategy} 저장")
            
            if collection_mode == "single":
                # 단일 클래스 데모
                await self.demo_single_class_collection("ㄱ", storage_strategy)
                
            elif collection_mode == "multi":
                # 다중 클래스 데모
                demo_classes = ["ㄱ", "ㅏ", "1"]
                for class_label in demo_classes:
                    await self.demo_single_class_collection(class_label, storage_strategy)
                
                # 마스터 데이터셋 생성
                if storage_strategy in ["unified", "both"]:
                    master_df, class_stats = await self.create_master_dataset_demo()
                    
            elif collection_mode == "complete":
                # 전체 데이터셋 데모
                all_classes = []
                for classes in self.experiment_config['target_classes'].values():
                    all_classes.extend(classes)
                
                print(f"\n🎯 전체 데이터셋 데모: {len(all_classes)}개 클래스")
                
                for i, class_label in enumerate(all_classes, 1):
                    print(f"\n진행률: {i}/{len(all_classes)} ({i/len(all_classes)*100:.1f}%)")
                    await self.demo_single_class_collection(class_label, storage_strategy)
                
                # 마스터 데이터셋 생성
                if storage_strategy in ["unified", "both"]:
                    master_df, class_stats = await self.create_master_dataset_demo()
            
            # 최종 결과 표시
            self.show_final_results()
            
            print(f"\n🎉 데모 완료! 베스트 프랙티스 데이터 구조가 생성되었습니다.")
            
        except Exception as e:
            print(f"\n❌ 데모 중 오류: {e}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("🎮 SignGlove 통합 데이터 수집 시스템 데모")
    print("=" * 60)
    print("🎯 딥러닝 베스트 프랙티스 구조:")
    print("   - 원본/전처리/통합 데이터 분리")
    print("   - Train/Val/Test 자동 분할")
    print("   - 클래스별 체계적 관리")
    print("   - 메타데이터 및 통계 자동 생성")
    print("")
    
    try:
        demo = DemoUnifiedCollector()
        asyncio.run(demo.run_demo())
    except KeyboardInterrupt:
        print("\n👋 데모가 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 데모 실행 중 오류: {e}")


if __name__ == "__main__":
    main() 