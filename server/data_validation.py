"""
SignGlove 데이터 검증 모듈
센서 데이터 및 제스처 데이터의 유효성을 검증하는 모듈
"""

from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel
import numpy as np
import logging

from .models.sensor_data import SensorData, SignGestureData, FlexSensorData, GyroData

logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    """검증 결과 모델"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = []
    validation_score: float = 1.0  # 0-1 범위의 검증 점수

class DataValidator:
    """데이터 검증기 클래스"""
    
    def __init__(self):
        """검증기 초기화"""
        # 센서 데이터 정상 범위 설정
        self.flex_normal_range = (50, 950)  # 일반적인 플렉스 센서 동작 범위
        self.gyro_stability_threshold = 5.0  # 자이로 안정성 임계값
        self.accel_stability_threshold = 2.0  # 가속도 안정성 임계값
        self.max_data_age_minutes = 5  # 데이터 최대 허용 지연 시간
        
        # 제스처 데이터 검증 설정
        self.min_gesture_duration = 0.1  # 최소 제스처 지속 시간
        self.max_gesture_duration = 10.0  # 최대 제스처 지속 시간
        self.min_samples_per_gesture = 3  # 제스처당 최소 샘플 수
        
        logger.info("데이터 검증기 초기화 완료")
    
    def validate_sensor_data(self, sensor_data: SensorData) -> ValidationResult:
        """
        센서 데이터 검증
        
        Args:
            sensor_data: 검증할 센서 데이터
            
        Returns:
            ValidationResult: 검증 결과
        """
        warnings = []
        validation_score = 1.0
        
        try:
            # 1. 타임스탬프 검증
            timestamp_result = self._validate_timestamp(sensor_data.timestamp)
            if not timestamp_result.is_valid:
                return timestamp_result
            if timestamp_result.warnings:
                warnings.extend(timestamp_result.warnings)
                validation_score *= 0.9
            
            # 2. 플렉스 센서 데이터 검증
            flex_result = self._validate_flex_sensors(sensor_data.flex_sensors)
            if not flex_result.is_valid:
                return flex_result
            if flex_result.warnings:
                warnings.extend(flex_result.warnings)
                validation_score *= flex_result.validation_score
            
            # 3. 자이로 센서 데이터 검증
            gyro_result = self._validate_gyro_data(sensor_data.gyro_data)
            if not gyro_result.is_valid:
                return gyro_result
            if gyro_result.warnings:
                warnings.extend(gyro_result.warnings)
                validation_score *= gyro_result.validation_score
            
            # 4. 디바이스 상태 검증
            device_result = self._validate_device_status(sensor_data)
            if device_result.warnings:
                warnings.extend(device_result.warnings)
                validation_score *= 0.95
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                validation_score=validation_score
            )
            
        except Exception as e:
            logger.error(f"센서 데이터 검증 중 오류 발생: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_message=f"검증 프로세스 오류: {str(e)}"
            )
    
    def validate_gesture_data(self, gesture_data: SignGestureData) -> ValidationResult:
        """
        제스처 데이터 검증
        
        Args:
            gesture_data: 검증할 제스처 데이터
            
        Returns:
            ValidationResult: 검증 결과
        """
        warnings = []
        validation_score = 1.0
        
        try:
            # 1. 기본 제스처 정보 검증
            if not gesture_data.gesture_label.strip():
                return ValidationResult(
                    is_valid=False,
                    error_message="제스처 라벨이 비어있습니다"
                )
            
            # 2. 제스처 지속 시간 검증
            if not self.min_gesture_duration <= gesture_data.duration <= self.max_gesture_duration:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"제스처 지속 시간이 유효하지 않습니다: {gesture_data.duration}초"
                )
            
            # 3. 센서 데이터 시퀀스 검증
            if len(gesture_data.sensor_sequence) < self.min_samples_per_gesture:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"센서 데이터 샘플 수가 부족합니다: {len(gesture_data.sensor_sequence)}"
                )
            
            # 4. 각 센서 데이터 검증
            for i, sensor_data in enumerate(gesture_data.sensor_sequence):
                sensor_result = self.validate_sensor_data(sensor_data)
                if not sensor_result.is_valid:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"센서 데이터 {i}번째 샘플 오류: {sensor_result.error_message}"
                    )
                if sensor_result.warnings:
                    warnings.extend([f"샘플 {i}: {w}" for w in sensor_result.warnings])
                    validation_score *= 0.95
            
            # 5. 제스처 연속성 검증
            continuity_result = self._validate_gesture_continuity(gesture_data.sensor_sequence)
            if not continuity_result.is_valid:
                return continuity_result
            if continuity_result.warnings:
                warnings.extend(continuity_result.warnings)
                validation_score *= continuity_result.validation_score
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                validation_score=validation_score
            )
            
        except Exception as e:
            logger.error(f"제스처 데이터 검증 중 오류 발생: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_message=f"검증 프로세스 오류: {str(e)}"
            )
    
    def _validate_timestamp(self, timestamp: datetime) -> ValidationResult:
        """타임스탬프 검증"""
        current_time = datetime.now()
        time_diff = abs((current_time - timestamp).total_seconds())
        
        if time_diff > self.max_data_age_minutes * 60:
            return ValidationResult(
                is_valid=False,
                error_message=f"데이터가 너무 오래되었습니다: {time_diff:.1f}초"
            )
        
        warnings = []
        if time_diff > 10:  # 10초 이상 지연 시 경고
            warnings.append(f"데이터 지연 감지: {time_diff:.1f}초")
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings
        )
    
    def _validate_flex_sensors(self, flex_data: FlexSensorData) -> ValidationResult:
        """플렉스 센서 데이터 검증"""
        flex_values = [
            flex_data.flex_1, flex_data.flex_2, flex_data.flex_3,
            flex_data.flex_4, flex_data.flex_5
        ]
        
        warnings = []
        validation_score = 1.0
        
        # 센서 값이 정상 범위 내에 있는지 확인
        for i, value in enumerate(flex_values):
            if not self.flex_normal_range[0] <= value <= self.flex_normal_range[1]:
                warnings.append(f"플렉스 센서 {i+1}의 값이 정상 범위를 벗어남: {value}")
                validation_score *= 0.9
        
        # 모든 센서 값이 동일한 경우 (센서 오작동 가능성)
        if len(set(flex_values)) == 1:
            warnings.append("모든 플렉스 센서 값이 동일합니다 (센서 오작동 가능성)")
            validation_score *= 0.7
        
        # 센서 값의 변화량 검증
        flex_range = max(flex_values) - min(flex_values)
        if flex_range < 10:  # 변화량이 너무 작은 경우
            warnings.append(f"플렉스 센서 변화량이 작습니다: {flex_range}")
            validation_score *= 0.8
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            validation_score=validation_score
        )
    
    def _validate_gyro_data(self, gyro_data: GyroData) -> ValidationResult:
        """자이로 센서 데이터 검증"""
        warnings = []
        validation_score = 1.0
        
        # 자이로 센서 값의 안정성 확인
        gyro_values = [gyro_data.gyro_x, gyro_data.gyro_y, gyro_data.gyro_z]
        gyro_magnitude = np.linalg.norm(gyro_values)
        
        if gyro_magnitude > self.gyro_stability_threshold:
            warnings.append(f"자이로 센서 변화량이 큽니다: {gyro_magnitude:.2f}")
            validation_score *= 0.9
        
        # 가속도 센서 값의 안정성 확인
        accel_values = [gyro_data.accel_x, gyro_data.accel_y, gyro_data.accel_z]
        accel_magnitude = np.linalg.norm(accel_values)
        
        # 중력 가속도 고려 (약 9.8 m/s²)
        expected_gravity = 9.8
        accel_deviation = abs(accel_magnitude - expected_gravity)
        
        if accel_deviation > self.accel_stability_threshold:
            warnings.append(f"가속도 센서 값이 예상과 다릅니다: {accel_magnitude:.2f}")
            validation_score *= 0.9
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            validation_score=validation_score
        )
    
    def _validate_device_status(self, sensor_data: SensorData) -> ValidationResult:
        """디바이스 상태 검증"""
        warnings = []
        
        # 배터리 잔량 확인
        if sensor_data.battery_level is not None:
            if sensor_data.battery_level < 20:
                warnings.append(f"배터리 잔량이 부족합니다: {sensor_data.battery_level}%")
            elif sensor_data.battery_level < 50:
                warnings.append(f"배터리 잔량이 낮습니다: {sensor_data.battery_level}%")
        
        # WiFi 신호 강도 확인
        if sensor_data.signal_strength is not None:
            if sensor_data.signal_strength < -70:
                warnings.append(f"WiFi 신호가 약합니다: {sensor_data.signal_strength}dBm")
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings
        )
    
    def _validate_gesture_continuity(self, sensor_sequence: List[SensorData]) -> ValidationResult:
        """제스처 데이터의 연속성 검증"""
        warnings = []
        validation_score = 1.0
        
        if len(sensor_sequence) < 2:
            return ValidationResult(is_valid=True)
        
        # 타임스탬프 순서 확인
        timestamps = [data.timestamp for data in sensor_sequence]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                warnings.append(f"타임스탬프 순서가 잘못되었습니다: {i}번째 데이터")
                validation_score *= 0.8
        
        # 데이터 간격 확인
        time_intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_intervals.append(interval)
        
        if time_intervals:
            avg_interval = np.mean(time_intervals)
            std_interval = np.std(time_intervals)
            
            # 불규칙한 샘플링 감지
            if std_interval > avg_interval * 0.5:
                warnings.append(f"불규칙한 데이터 샘플링 감지: 평균 {avg_interval:.3f}초, 표준편차 {std_interval:.3f}초")
                validation_score *= 0.9
        
        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            validation_score=validation_score
        ) 