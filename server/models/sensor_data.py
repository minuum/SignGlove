"""
SignGlove 센서 데이터 모델 정의
플렉스 센서 및 자이로 센서 데이터를 위한 Pydantic 모델
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
from enum import Enum

class SensorType(str, Enum):
    """센서 타입 열거형"""
    FLEX = "flex"
    GYRO = "gyro"
    ACCELEROMETER = "accelerometer"

class GyroData(BaseModel):
    """자이로 센서 데이터 모델 (6DOF)"""
    gyro_x: float = Field(..., description="자이로 X축 값 (°/s)")
    gyro_y: float = Field(..., description="자이로 Y축 값 (°/s)")
    gyro_z: float = Field(..., description="자이로 Z축 값 (°/s)")
    accel_x: float = Field(..., description="가속도 X축 값 (m/s²)")
    accel_y: float = Field(..., description="가속도 Y축 값 (m/s²)")
    accel_z: float = Field(..., description="가속도 Z축 값 (m/s²)")
    
    @validator('gyro_x', 'gyro_y', 'gyro_z')
    def validate_gyro_range(cls, v):
        """자이로 센서 값 범위 검증 (-250 ~ 250 °/s)"""
        if not -250 <= v <= 250:
            raise ValueError('자이로 센서 값은 -250 ~ 250 °/s 범위여야 합니다')
        return v
    
    @validator('accel_x', 'accel_y', 'accel_z')
    def validate_accel_range(cls, v):
        """가속도 센서 값 범위 검증 (-16 ~ 16 m/s²)"""
        if not -16 <= v <= 16:
            raise ValueError('가속도 센서 값은 -16 ~ 16 m/s² 범위여야 합니다')
        return v

class FlexSensorData(BaseModel):
    """플렉스 센서 데이터 모델"""
    flex_1: float = Field(..., description="엄지 플렉스 센서 값 (0-1023)")
    flex_2: float = Field(..., description="검지 플렉스 센서 값 (0-1023)")
    flex_3: float = Field(..., description="중지 플렉스 센서 값 (0-1023)")
    flex_4: float = Field(..., description="약지 플렉스 센서 값 (0-1023)")
    flex_5: float = Field(..., description="소지 플렉스 센서 값 (0-1023)")
    
    @validator('flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5')
    def validate_flex_range(cls, v):
        """플렉스 센서 값 범위 검증 (0-1023)"""
        if not 0 <= v <= 1023:
            raise ValueError('플렉스 센서 값은 0-1023 범위여야 합니다')
        return v

class SensorData(BaseModel):
    """통합 센서 데이터 모델"""
    device_id: str = Field(..., description="디바이스 고유 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="데이터 수집 시간")
    flex_sensors: FlexSensorData = Field(..., description="플렉스 센서 데이터")
    gyro_data: GyroData = Field(..., description="자이로 센서 데이터")
    battery_level: Optional[float] = Field(None, description="배터리 잔량 (%)")
    signal_strength: Optional[int] = Field(None, description="WiFi 신호 강도 (dBm)")
    
    @validator('battery_level')
    def validate_battery_level(cls, v):
        """배터리 잔량 검증"""
        if v is not None and not 0 <= v <= 100:
            raise ValueError('배터리 잔량은 0-100% 범위여야 합니다')
        return v
    
    @validator('signal_strength')
    def validate_signal_strength(cls, v):
        """WiFi 신호 강도 검증"""
        if v is not None and not -100 <= v <= 0:
            raise ValueError('WiFi 신호 강도는 -100 ~ 0 dBm 범위여야 합니다')
        return v

class SignGestureType(str, Enum):
    """수어 제스처 타입"""
    VOWEL = "vowel"          # 모음
    CONSONANT = "consonant"  # 자음
    NUMBER = "number"        # 숫자
    WORD = "word"           # 단어
    SENTENCE = "sentence"    # 문장

class SignGestureData(BaseModel):
    """수어 제스처 데이터 모델"""
    gesture_id: str = Field(..., description="제스처 고유 ID")
    gesture_label: str = Field(..., description="제스처 라벨 (예: 'ㄱ', 'ㅏ', '1')")
    gesture_type: SignGestureType = Field(..., description="제스처 타입")
    sensor_sequence: List[SensorData] = Field(..., description="센서 데이터 시퀀스")
    duration: float = Field(..., description="제스처 지속 시간 (초)")
    performer_id: str = Field(..., description="수행자 ID")
    session_id: str = Field(..., description="세션 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="기록 시간")
    quality_score: Optional[float] = Field(None, description="제스처 품질 점수 (0-1)")
    notes: Optional[str] = Field(None, description="추가 메모")
    
    @validator('duration')
    def validate_duration(cls, v):
        """제스처 지속 시간 검증"""
        if v <= 0:
            raise ValueError('제스처 지속 시간은 0보다 커야 합니다')
        if v > 30:  # 30초 이상은 비정상
            raise ValueError('제스처 지속 시간은 30초를 초과할 수 없습니다')
        return v
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        """품질 점수 검증"""
        if v is not None and not 0 <= v <= 1:
            raise ValueError('품질 점수는 0-1 범위여야 합니다')
        return v
    
    @validator('sensor_sequence')
    def validate_sensor_sequence(cls, v):
        """센서 데이터 시퀀스 검증"""
        if len(v) < 1:
            raise ValueError('센서 데이터 시퀀스는 최소 1개 이상이어야 합니다')
        if len(v) > 1000:  # 과도한 데이터 방지
            raise ValueError('센서 데이터 시퀀스는 1000개를 초과할 수 없습니다')
        return v

class DataCollectionSession(BaseModel):
    """데이터 수집 세션 모델"""
    session_id: str = Field(..., description="세션 고유 ID")
    performer_id: str = Field(..., description="수행자 ID")
    start_time: datetime = Field(..., description="세션 시작 시간")
    end_time: Optional[datetime] = Field(None, description="세션 종료 시간")
    total_gestures: int = Field(0, description="총 제스처 수")
    session_notes: Optional[str] = Field(None, description="세션 메모")
    device_info: Optional[str] = Field(None, description="디바이스 정보")

class CalibrationData(BaseModel):
    """센서 캘리브레이션 데이터 모델"""
    device_id: str = Field(..., description="디바이스 ID")
    calibration_type: str = Field(..., description="캘리브레이션 타입")
    flex_min_values: List[float] = Field(..., description="플렉스 센서 최소값")
    flex_max_values: List[float] = Field(..., description="플렉스 센서 최대값")
    gyro_offset: GyroData = Field(..., description="자이로 센서 오프셋")
    calibration_date: datetime = Field(default_factory=datetime.now, description="캘리브레이션 날짜")
    is_active: bool = Field(True, description="활성 상태")
    
    @validator('flex_min_values', 'flex_max_values')
    def validate_flex_calibration(cls, v):
        """플렉스 센서 캘리브레이션 값 검증"""
        if len(v) != 5:
            raise ValueError('플렉스 센서 캘리브레이션 값은 5개여야 합니다')
        return v 