"""
SignGlove 데이터 저장 모듈
센서 데이터 및 제스처 데이터를 CSV 형태로 저장하는 모듈
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import logging
import pandas as pd
import threading
from collections import defaultdict

from .models.sensor_data import SensorData, SignGestureData, DataCollectionSession

logger = logging.getLogger(__name__)

class DataStorage:
    """데이터 저장 관리 클래스"""
    
    def __init__(self, base_path: str = "data"):
        """
        데이터 저장소 초기화
        
        Args:
            base_path: 데이터 저장 기본 경로
        """
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "raw"
        self.processed_data_path = self.base_path / "processed"
        self.backup_path = self.base_path / "backup"
        
        # 폴더 생성
        for path in [self.raw_data_path, self.processed_data_path, self.backup_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # 데이터 통계 추적
        self.stats = {
            "total_sensor_records": 0,
            "total_gesture_records": 0,
            "sessions": {},
            "devices": set(),
            "last_update": datetime.now()
        }
        
        # 스레드 안전성을 위한 락
        self.lock = threading.RLock()
        
        # 세션 관리
        self.active_sessions: Dict[str, DataCollectionSession] = {}
        
        logger.info(f"데이터 저장소 초기화 완료: {self.base_path}")
    
    async def initialize(self):
        """데이터 저장소 초기화"""
        await self._load_existing_stats()
        await self._create_daily_files()
        logger.info("데이터 저장소 초기화 완료")
    
    async def cleanup(self):
        """데이터 저장소 정리"""
        await self._save_stats()
        await self._backup_current_data()
        logger.info("데이터 저장소 정리 완료")
    
    async def save_sensor_data(self, sensor_data: SensorData) -> bool:
        """
        센서 데이터를 CSV 파일로 저장
        
        Args:
            sensor_data: 저장할 센서 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with self.lock:
                # 날짜별 파일 경로 생성
                date_str = sensor_data.timestamp.strftime("%Y%m%d")
                sensor_file = self.raw_data_path / f"sensor_data_{date_str}.csv"
                
                # CSV 헤더 정의
                fieldnames = [
                    'timestamp', 'device_id', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
                    'gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z',
                    'battery_level', 'signal_strength'
                ]
                
                # 파일 존재 여부 확인
                file_exists = sensor_file.exists()
                
                # CSV 파일에 데이터 쓰기
                with open(sensor_file, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # 헤더 쓰기 (파일이 새로 생성된 경우)
                    if not file_exists:
                        writer.writeheader()
                    
                    # 데이터 행 쓰기
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
                
                # 통계 업데이트
                self.stats["total_sensor_records"] += 1
                self.stats["devices"].add(sensor_data.device_id)
                self.stats["last_update"] = datetime.now()
                
                logger.debug(f"센서 데이터 저장 완료: {sensor_data.device_id}")
                return True
                
        except Exception as e:
            logger.error(f"센서 데이터 저장 실패: {str(e)}")
            return False
    
    async def save_gesture_data(self, gesture_data: SignGestureData) -> bool:
        """
        제스처 데이터를 CSV 및 JSON 파일로 저장
        
        Args:
            gesture_data: 저장할 제스처 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with self.lock:
                # 날짜별 파일 경로 생성
                date_str = gesture_data.timestamp.strftime("%Y%m%d")
                gesture_file = self.raw_data_path / f"gesture_data_{date_str}.csv"
                sequence_file = self.raw_data_path / f"gesture_sequences_{date_str}.json"
                
                # 제스처 메타데이터 CSV 저장
                await self._save_gesture_metadata(gesture_file, gesture_data)
                
                # 센서 시퀀스 JSON 저장
                await self._save_sensor_sequence(sequence_file, gesture_data)
                
                # 통계 업데이트
                self.stats["total_gesture_records"] += 1
                self.stats["last_update"] = datetime.now()
                
                # 세션 정보 업데이트
                if gesture_data.session_id in self.active_sessions:
                    self.active_sessions[gesture_data.session_id].total_gestures += 1
                
                logger.debug(f"제스처 데이터 저장 완료: {gesture_data.gesture_label}")
                return True
                
        except Exception as e:
            logger.error(f"제스처 데이터 저장 실패: {str(e)}")
            return False
    
    async def _save_gesture_metadata(self, file_path: Path, gesture_data: SignGestureData):
        """제스처 메타데이터 CSV 저장"""
        fieldnames = [
            'timestamp', 'gesture_id', 'gesture_label', 'gesture_type', 'duration',
            'performer_id', 'session_id', 'quality_score', 'sequence_length', 'notes'
        ]
        
        file_exists = file_path.exists()
        
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row = {
                'timestamp': gesture_data.timestamp.isoformat(),
                'gesture_id': gesture_data.gesture_id,
                'gesture_label': gesture_data.gesture_label,
                'gesture_type': gesture_data.gesture_type.value,
                'duration': gesture_data.duration,
                'performer_id': gesture_data.performer_id,
                'session_id': gesture_data.session_id,
                'quality_score': gesture_data.quality_score,
                'sequence_length': len(gesture_data.sensor_sequence),
                'notes': gesture_data.notes
            }
            
            writer.writerow(row)
    
    async def _save_sensor_sequence(self, file_path: Path, gesture_data: SignGestureData):
        """센서 시퀀스 JSON 저장"""
        # 기존 데이터 로드
        sequences = {}
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sequences = json.load(f)
            except:
                sequences = {}
        
        # 센서 시퀀스를 딕셔너리로 변환
        sequence_data = []
        for sensor_data in gesture_data.sensor_sequence:
            sequence_data.append({
                'timestamp': sensor_data.timestamp.isoformat(),
                'device_id': sensor_data.device_id,
                'flex_sensors': {
                    'flex_1': sensor_data.flex_sensors.flex_1,
                    'flex_2': sensor_data.flex_sensors.flex_2,
                    'flex_3': sensor_data.flex_sensors.flex_3,
                    'flex_4': sensor_data.flex_sensors.flex_4,
                    'flex_5': sensor_data.flex_sensors.flex_5,
                },
                'gyro_data': {
                    'gyro_x': sensor_data.gyro_data.gyro_x,
                    'gyro_y': sensor_data.gyro_data.gyro_y,
                    'gyro_z': sensor_data.gyro_data.gyro_z,
                    'accel_x': sensor_data.gyro_data.accel_x,
                    'accel_y': sensor_data.gyro_data.accel_y,
                    'accel_z': sensor_data.gyro_data.accel_z,
                },
                'battery_level': sensor_data.battery_level,
                'signal_strength': sensor_data.signal_strength
            })
        
        # 제스처 ID를 키로 사용하여 시퀀스 저장
        sequences[gesture_data.gesture_id] = {
            'gesture_label': gesture_data.gesture_label,
            'gesture_type': gesture_data.gesture_type.value,
            'duration': gesture_data.duration,
            'performer_id': gesture_data.performer_id,
            'session_id': gesture_data.session_id,
            'timestamp': gesture_data.timestamp.isoformat(),
            'sensor_sequence': sequence_data
        }
        
        # JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sequences, f, ensure_ascii=False, indent=2)
    
    async def get_total_records(self) -> int:
        """총 레코드 수 반환"""
        with self.lock:
            return self.stats["total_sensor_records"] + self.stats["total_gesture_records"]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """데이터 수집 통계 반환"""
        with self.lock:
            return {
                "total_sensor_records": self.stats["total_sensor_records"],
                "total_gesture_records": self.stats["total_gesture_records"],
                "total_devices": len(self.stats["devices"]),
                "active_sessions": len(self.active_sessions),
                "last_update": self.stats["last_update"].isoformat(),
                "storage_path": str(self.base_path),
                "disk_usage": await self._get_disk_usage()
            }
    
    async def get_data_summary(self, date: str = None) -> Dict[str, Any]:
        """특정 날짜의 데이터 요약 반환"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        try:
            sensor_file = self.raw_data_path / f"sensor_data_{date}.csv"
            gesture_file = self.raw_data_path / f"gesture_data_{date}.csv"
            
            summary = {
                "date": date,
                "sensor_records": 0,
                "gesture_records": 0,
                "devices": set(),
                "gesture_types": defaultdict(int),
                "sessions": set()
            }
            
            # 센서 데이터 요약
            if sensor_file.exists():
                df = pd.read_csv(sensor_file)
                summary["sensor_records"] = len(df)
                summary["devices"].update(df['device_id'].unique())
            
            # 제스처 데이터 요약
            if gesture_file.exists():
                df = pd.read_csv(gesture_file)
                summary["gesture_records"] = len(df)
                summary["sessions"].update(df['session_id'].unique())
                
                for gesture_type in df['gesture_type']:
                    summary["gesture_types"][gesture_type] += 1
            
            # set을 list로 변환 (JSON 직렬화 가능)
            summary["devices"] = list(summary["devices"])
            summary["sessions"] = list(summary["sessions"])
            summary["gesture_types"] = dict(summary["gesture_types"])
            
            return summary
            
        except Exception as e:
            logger.error(f"데이터 요약 생성 실패: {str(e)}")
            return {"error": str(e)}
    
    async def _load_existing_stats(self):
        """기존 통계 데이터 로드"""
        stats_file = self.base_path / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    saved_stats = json.load(f)
                    self.stats.update(saved_stats)
                    # set 복원
                    self.stats["devices"] = set(self.stats.get("devices", []))
                    # datetime 복원
                    self.stats["last_update"] = datetime.fromisoformat(
                        self.stats.get("last_update", datetime.now().isoformat())
                    )
                logger.info("기존 통계 데이터 로드 완료")
            except Exception as e:
                logger.warning(f"통계 데이터 로드 실패: {str(e)}")
    
    async def _save_stats(self):
        """통계 데이터 저장"""
        stats_file = self.base_path / "stats.json"
        try:
            # 저장 가능한 형태로 변환
            save_stats = self.stats.copy()
            save_stats["devices"] = list(save_stats["devices"])
            save_stats["last_update"] = save_stats["last_update"].isoformat()
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(save_stats, f, ensure_ascii=False, indent=2)
            logger.info("통계 데이터 저장 완료")
        except Exception as e:
            logger.error(f"통계 데이터 저장 실패: {str(e)}")
    
    async def _create_daily_files(self):
        """일일 파일 생성"""
        # 오늘 날짜로 필요한 파일들 미리 생성
        today = datetime.now().strftime("%Y%m%d")
        files_to_create = [
            f"sensor_data_{today}.csv",
            f"gesture_data_{today}.csv"
        ]
        
        for filename in files_to_create:
            file_path = self.raw_data_path / filename
            if not file_path.exists():
                file_path.touch()
        
        logger.info(f"일일 파일 생성 완료: {today}")
    
    async def _backup_current_data(self):
        """현재 데이터 백업"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # 오늘 데이터 파일들 백업
            today = datetime.now().strftime("%Y%m%d")
            files_to_backup = [
                f"sensor_data_{today}.csv",
                f"gesture_data_{today}.csv",
                f"gesture_sequences_{today}.json"
            ]
            
            for filename in files_to_backup:
                source = self.raw_data_path / filename
                if source.exists():
                    dest = backup_dir / filename
                    dest.write_bytes(source.read_bytes())
            
            logger.info(f"데이터 백업 완료: {backup_dir}")
            
        except Exception as e:
            logger.error(f"데이터 백업 실패: {str(e)}")
    
    async def _get_disk_usage(self) -> Dict[str, int]:
        """디스크 사용량 반환"""
        try:
            total_size = 0
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return {
                "total_bytes": total_size,
                "total_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"디스크 사용량 계산 실패: {str(e)}")
            return {"total_bytes": 0, "total_mb": 0} 