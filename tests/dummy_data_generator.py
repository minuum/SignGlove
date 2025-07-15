"""
SignGlove 더미 데이터 생성기
테스트용 센서 데이터 및 제스처 데이터를 생성하는 모듈
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import httpx
from faker import Faker
import numpy as np

# 프로젝트 모듈 임포트
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server.models.sensor_data import (
    SensorData, GyroData, FlexSensorData, 
    SignGestureData, SignGestureType
)

fake = Faker('ko_KR')

class DummyDataGenerator:
    """더미 데이터 생성기 클래스"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        더미 데이터 생성기 초기화
        
        Args:
            server_url: 서버 URL
        """
        self.server_url = server_url
        self.device_id = "DUMMY_DEVICE_001"
        self.performer_id = "test_performer"
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 한국어 수어 라벨 정의
        self.korean_vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        self.korean_consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        print(f"더미 데이터 생성기 초기화 완료")
        print(f"서버 URL: {server_url}")
        print(f"세션 ID: {self.session_id}")
    
    def generate_realistic_flex_data(self, gesture_type: str = "neutral") -> FlexSensorData:
        """
        실제적인 플렉스 센서 데이터 생성
        
        Args:
            gesture_type: 제스처 타입 (neutral, fist, open, etc.)
            
        Returns:
            FlexSensorData: 플렉스 센서 데이터
        """
        base_values = {
            "neutral": [400, 420, 410, 430, 440],  # 중립 자세
            "fist": [800, 850, 830, 820, 810],     # 주먹 쥔 자세
            "open": [200, 180, 190, 170, 160],     # 손 펼친 자세
            "pointing": [200, 800, 400, 400, 400], # 검지 가리키기
            "thumbs_up": [800, 400, 400, 400, 400] # 엄지 올리기
        }
        
        if gesture_type not in base_values:
            gesture_type = "neutral"
        
        base = base_values[gesture_type]
        
        # 노이즈와 개인차 추가
        flex_values = []
        for i, base_val in enumerate(base):
            noise = random.gauss(0, 20)  # 가우시안 노이즈
            drift = random.uniform(-10, 10)  # 센서 드리프트
            value = max(0, min(1023, base_val + noise + drift))
            flex_values.append(value)
        
        return FlexSensorData(
            flex_1=flex_values[0],
            flex_2=flex_values[1],
            flex_3=flex_values[2],
            flex_4=flex_values[3],
            flex_5=flex_values[4]
        )
    
    def generate_realistic_gyro_data(self, motion_type: str = "stable") -> GyroData:
        """
        실제적인 자이로 센서 데이터 생성
        
        Args:
            motion_type: 모션 타입 (stable, gesture, shake)
            
        Returns:
            GyroData: 자이로 센서 데이터
        """
        motion_patterns = {
            "stable": {
                "gyro_range": 2.0,
                "accel_base": [0, 0, 9.8],
                "accel_range": 0.5
            },
            "gesture": {
                "gyro_range": 15.0,
                "accel_base": [0, 0, 9.8],
                "accel_range": 2.0
            },
            "shake": {
                "gyro_range": 50.0,
                "accel_base": [0, 0, 9.8],
                "accel_range": 5.0
            }
        }
        
        if motion_type not in motion_patterns:
            motion_type = "stable"
        
        pattern = motion_patterns[motion_type]
        
        # 자이로 데이터 생성
        gyro_x = random.uniform(-pattern["gyro_range"], pattern["gyro_range"])
        gyro_y = random.uniform(-pattern["gyro_range"], pattern["gyro_range"])
        gyro_z = random.uniform(-pattern["gyro_range"], pattern["gyro_range"])
        
        # 가속도 데이터 생성 (중력 포함)
        accel_x = pattern["accel_base"][0] + random.uniform(-pattern["accel_range"], pattern["accel_range"])
        accel_y = pattern["accel_base"][1] + random.uniform(-pattern["accel_range"], pattern["accel_range"])
        accel_z = pattern["accel_base"][2] + random.uniform(-pattern["accel_range"], pattern["accel_range"])
        
        return GyroData(
            gyro_x=gyro_x,
            gyro_y=gyro_y,
            gyro_z=gyro_z,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z
        )
    
    def generate_sensor_data(self, gesture_type: str = "neutral", motion_type: str = "stable") -> SensorData:
        """
        센서 데이터 생성
        
        Args:
            gesture_type: 제스처 타입
            motion_type: 모션 타입
            
        Returns:
            SensorData: 센서 데이터
        """
        flex_data = self.generate_realistic_flex_data(gesture_type)
        gyro_data = self.generate_realistic_gyro_data(motion_type)
        
        return SensorData(
            device_id=self.device_id,
            timestamp=datetime.now(),
            flex_sensors=flex_data,
            gyro_data=gyro_data,
            battery_level=random.uniform(30, 100),
            signal_strength=random.randint(-80, -30)
        )
    
    def generate_gesture_sequence(self, gesture_label: str, gesture_type: SignGestureType, 
                                duration: float = 2.0, sample_rate: int = 20) -> List[SensorData]:
        """
        제스처 시퀀스 생성
        
        Args:
            gesture_label: 제스처 라벨
            gesture_type: 제스처 타입
            duration: 지속 시간
            sample_rate: 샘플링 레이트
            
        Returns:
            List[SensorData]: 센서 데이터 시퀀스
        """
        total_samples = int(duration * sample_rate)
        sequence = []
        
        # 제스처 단계별 진행
        for i in range(total_samples):
            progress = i / total_samples
            
            # 제스처 단계 결정
            if progress < 0.2:
                # 시작 단계 (중립 자세)
                gesture_phase = "neutral"
                motion_phase = "stable"
            elif progress < 0.4:
                # 전환 단계 (동작 시작)
                gesture_phase = "neutral"
                motion_phase = "gesture"
            elif progress < 0.8:
                # 제스처 단계 (목표 자세)
                gesture_phase = self._get_gesture_pattern(gesture_label)
                motion_phase = "gesture"
            else:
                # 종료 단계 (중립 복귀)
                gesture_phase = "neutral"
                motion_phase = "stable"
            
            # 시간 간격 계산
            timestamp = datetime.now() + timedelta(seconds=i / sample_rate)
            
            sensor_data = self.generate_sensor_data(gesture_phase, motion_phase)
            sensor_data.timestamp = timestamp
            sequence.append(sensor_data)
        
        return sequence
    
    def _get_gesture_pattern(self, gesture_label: str) -> str:
        """제스처 라벨에 따른 손 모양 패턴 반환"""
        patterns = {
            # 숫자
            '1': 'pointing',
            '2': 'open',
            '3': 'open',
            '4': 'open',
            '5': 'open',
            
            # 자음 (예시)
            'ㄱ': 'fist',
            'ㄴ': 'pointing',
            'ㄷ': 'fist',
            'ㄹ': 'open',
            'ㅁ': 'fist',
            
            # 모음 (예시)
            'ㅏ': 'thumbs_up',
            'ㅑ': 'open',
            'ㅓ': 'pointing',
            'ㅕ': 'open',
            'ㅗ': 'fist',
        }
        
        return patterns.get(gesture_label, 'neutral')
    
    def generate_gesture_data(self, gesture_label: str, gesture_type: SignGestureType) -> SignGestureData:
        """
        제스처 데이터 생성
        
        Args:
            gesture_label: 제스처 라벨
            gesture_type: 제스처 타입
            
        Returns:
            SignGestureData: 제스처 데이터
        """
        duration = random.uniform(1.0, 3.0)
        sensor_sequence = self.generate_gesture_sequence(gesture_label, gesture_type, duration)
        
        gesture_id = f"{gesture_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        return SignGestureData(
            gesture_id=gesture_id,
            gesture_label=gesture_label,
            gesture_type=gesture_type,
            sensor_sequence=sensor_sequence,
            duration=duration,
            performer_id=self.performer_id,
            session_id=self.session_id,
            timestamp=datetime.now(),
            quality_score=random.uniform(0.7, 1.0)
        )
    
    async def send_sensor_data(self, sensor_data: SensorData) -> bool:
        """
        센서 데이터를 서버로 전송
        
        Args:
            sensor_data: 센서 데이터
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/data/sensor",
                    json=sensor_data.model_dump(mode="json"),
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return True
                else:
                    print(f"센서 데이터 전송 실패: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            print(f"센서 데이터 전송 오류: {str(e)}")
            return False
    
    async def send_gesture_data(self, gesture_data: SignGestureData) -> bool:
        """
        제스처 데이터를 서버로 전송
        
        Args:
            gesture_data: 제스처 데이터
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/data/gesture",
                    json=gesture_data.model_dump(mode="json"),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return True
                else:
                    print(f"제스처 데이터 전송 실패: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            print(f"제스처 데이터 전송 오류: {str(e)}")
            return False
    
    async def run_sensor_simulation(self, duration: int = 60, sample_rate: int = 20):
        """
        센서 데이터 시뮬레이션 실행
        
        Args:
            duration: 시뮬레이션 지속 시간 (초)
            sample_rate: 샘플링 레이트 (Hz)
        """
        print(f"센서 데이터 시뮬레이션 시작: {duration}초, {sample_rate}Hz")
        
        total_samples = duration * sample_rate
        interval = 1.0 / sample_rate
        
        successful_sends = 0
        failed_sends = 0
        
        for i in range(total_samples):
            # 다양한 제스처 패턴 시뮬레이션
            gesture_types = ["neutral", "fist", "open", "pointing", "thumbs_up"]
            motion_types = ["stable", "gesture", "shake"]
            
            gesture_type = random.choice(gesture_types)
            motion_type = random.choice(motion_types)
            
            sensor_data = self.generate_sensor_data(gesture_type, motion_type)
            
            success = await self.send_sensor_data(sensor_data)
            if success:
                successful_sends += 1
            else:
                failed_sends += 1
            
            # 진행률 출력
            if i % (sample_rate * 10) == 0:  # 10초마다
                progress = (i / total_samples) * 100
                print(f"진행률: {progress:.1f}% - 성공: {successful_sends}, 실패: {failed_sends}")
            
            await asyncio.sleep(interval)
        
        print(f"센서 데이터 시뮬레이션 완료!")
        print(f"총 전송: {total_samples}, 성공: {successful_sends}, 실패: {failed_sends}")
    
    async def run_gesture_simulation(self, gesture_count: int = 20):
        """
        제스처 데이터 시뮬레이션 실행
        
        Args:
            gesture_count: 생성할 제스처 수
        """
        print(f"제스처 데이터 시뮬레이션 시작: {gesture_count}개 제스처")
        
        successful_sends = 0
        failed_sends = 0
        
        for i in range(gesture_count):
            # 랜덤 제스처 선택
            gesture_type = random.choice(list(SignGestureType))
            
            if gesture_type == SignGestureType.VOWEL:
                gesture_label = random.choice(self.korean_vowels)
            elif gesture_type == SignGestureType.CONSONANT:
                gesture_label = random.choice(self.korean_consonants)
            elif gesture_type == SignGestureType.NUMBER:
                gesture_label = random.choice(self.numbers)
            else:
                gesture_label = random.choice(self.korean_vowels)
            
            gesture_data = self.generate_gesture_data(gesture_label, gesture_type)
            
            success = await self.send_gesture_data(gesture_data)
            if success:
                successful_sends += 1
                print(f"[{i+1}/{gesture_count}] 제스처 '{gesture_label}' 전송 성공")
            else:
                failed_sends += 1
                print(f"[{i+1}/{gesture_count}] 제스처 '{gesture_label}' 전송 실패")
            
            # 제스처 간 간격
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        print(f"제스처 데이터 시뮬레이션 완료!")
        print(f"총 전송: {gesture_count}, 성공: {successful_sends}, 실패: {failed_sends}")

def main():
    """메인 함수"""
    generator = DummyDataGenerator()
    
    print("=== SignGlove 더미 데이터 생성기 ===")
    print("1. 센서 데이터 시뮬레이션")
    print("2. 제스처 데이터 시뮬레이션")
    print("3. 혼합 시뮬레이션")
    print("4. 단일 테스트 데이터 생성")
    
    choice = input("선택하세요 (1-4): ")
    
    if choice == "1":
        duration = int(input("지속 시간 (초, 기본값: 60): ") or "60")
        sample_rate = int(input("샘플링 레이트 (Hz, 기본값: 20): ") or "20")
        asyncio.run(generator.run_sensor_simulation(duration, sample_rate))
    
    elif choice == "2":
        gesture_count = int(input("제스처 수 (기본값: 20): ") or "20")
        asyncio.run(generator.run_gesture_simulation(gesture_count))
    
    elif choice == "3":
        print("혼합 시뮬레이션 시작...")
        asyncio.run(asyncio.gather(
            generator.run_sensor_simulation(30, 10),
            generator.run_gesture_simulation(10)
        ))
    
    elif choice == "4":
        # 단일 테스트 데이터 생성
        sensor_data = generator.generate_sensor_data()
        print("=== 생성된 센서 데이터 ===")
        print(json.dumps(sensor_data.model_dump(mode="json"), indent=2, default=str))
        
        gesture_data = generator.generate_gesture_data("ㅏ", SignGestureType.VOWEL)
        print("\n=== 생성된 제스처 데이터 ===")
        print(f"제스처 ID: {gesture_data.gesture_id}")
        print(f"제스처 라벨: {gesture_data.gesture_label}")
        print(f"제스처 타입: {gesture_data.gesture_type}")
        print(f"지속 시간: {gesture_data.duration:.2f}초")
        print(f"센서 시퀀스 길이: {len(gesture_data.sensor_sequence)}")
    
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main() 