#!/usr/bin/env python3
"""
SignGlove 전처리 모듈
KLP-SignGlove의 전처리 기법을 통합 적용

포함 기능:
- Butterworth 저역통과 필터
- Min-Max/Z-score/Robust 정규화
- 시계열 윈도우 분할
- 노이즈 제거 및 평활화
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime

from .models.sensor_data import SensorData

logger = logging.getLogger(__name__)


class SensorPreprocessor:
    """센서 데이터 전처리 클래스"""
    
    def __init__(self, 
                 sampling_rate: float = 20.0,
                 filter_cutoff: float = 5.0,
                 window_size: int = 100):
        """
        전처리기 초기화
        
        Args:
            sampling_rate: 센서 샘플링 주파수 (Hz)
            filter_cutoff: Butterworth 필터 차단 주파수 (Hz)
            window_size: 윈도우 크기 (샘플 수)
        """
        self.sampling_rate = sampling_rate
        self.filter_cutoff = filter_cutoff
        self.window_size = window_size
        
        # Butterworth 필터 설계 (KLP-SignGlove 기법)
        nyquist = sampling_rate / 2
        normalized_cutoff = filter_cutoff / nyquist
        self.butter_b, self.butter_a = signal.butter(
            4, normalized_cutoff, btype='low', analog=False
        )
        
        # 정규화 스케일러들
        self.flex_scaler = MinMaxScaler(feature_range=(0, 1))
        self.gyro_scaler = StandardScaler()
        self.accel_scaler = RobustScaler()
        
        self.is_fitted = False
        logger.info(f"전처리기 초기화: SR={sampling_rate}Hz, 차단주파수={filter_cutoff}Hz")
    
    def extract_sensor_arrays(self, sensor_data_list: List[SensorData]) -> Dict[str, np.ndarray]:
        """센서 데이터 리스트를 numpy 배열로 변환"""
        n_samples = len(sensor_data_list)
        
        # 플렉스 센서 배열 (n_samples, 5)
        flex_array = np.zeros((n_samples, 5))
        # 자이로 센서 배열 (n_samples, 3)  
        gyro_array = np.zeros((n_samples, 3))
        # 가속도 센서 배열 (n_samples, 3)
        accel_array = np.zeros((n_samples, 3))
        # 타임스탬프 배열
        timestamps = []
        
        for i, data in enumerate(sensor_data_list):
            flex_array[i] = [
                data.flex_sensors.flex_1,
                data.flex_sensors.flex_2,
                data.flex_sensors.flex_3,
                data.flex_sensors.flex_4,
                data.flex_sensors.flex_5
            ]
            
            gyro_array[i] = [
                data.gyro_data.gyro_x,
                data.gyro_data.gyro_y,
                data.gyro_data.gyro_z
            ]
            
            accel_array[i] = [
                data.gyro_data.accel_x,
                data.gyro_data.accel_y,
                data.gyro_data.accel_z
            ]
            
            timestamps.append(data.timestamp)
        
        return {
            'flex': flex_array,
            'gyro': gyro_array,
            'accel': accel_array,
            'timestamps': timestamps
        }
    
    def apply_butterworth_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Butterworth 저역통과 필터 적용 (KLP-SignGlove 기법)
        
        Args:
            data: 입력 데이터 (n_samples, n_features)
            
        Returns:
            filtered_data: 필터링된 데이터
        """
        if len(data.shape) == 1:
            # 1차원 데이터
            return signal.filtfilt(self.butter_b, self.butter_a, data)
        else:
            # 다차원 데이터 - 각 축별로 필터링
            filtered = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered[:, i] = signal.filtfilt(self.butter_b, self.butter_a, data[:, i])
            return filtered
    
    def normalize_data(self, 
                      flex_data: np.ndarray,
                      gyro_data: np.ndarray, 
                      accel_data: np.ndarray,
                      fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        센서 데이터 정규화 (KLP-SignGlove 기법)
        
        Args:
            flex_data: 플렉스 센서 데이터
            gyro_data: 자이로 센서 데이터
            accel_data: 가속도 센서 데이터
            fit_scalers: 스케일러 훈련 여부
            
        Returns:
            normalized data tuple
        """
        if fit_scalers:
            # 스케일러 훈련
            normalized_flex = self.flex_scaler.fit_transform(flex_data)
            normalized_gyro = self.gyro_scaler.fit_transform(gyro_data)
            normalized_accel = self.accel_scaler.fit_transform(accel_data)
            self.is_fitted = True
            logger.info("정규화 스케일러 훈련 완료")
        else:
            # 기존 스케일러 사용
            if not self.is_fitted:
                logger.warning("스케일러가 훈련되지 않았습니다. 훈련 모드로 전환합니다.")
                return self.normalize_data(flex_data, gyro_data, accel_data, fit_scalers=True)
            
            normalized_flex = self.flex_scaler.transform(flex_data)
            normalized_gyro = self.gyro_scaler.transform(gyro_data)
            normalized_accel = self.accel_scaler.transform(accel_data)
        
        return normalized_flex, normalized_gyro, normalized_accel
    
    def create_sliding_windows(self, 
                              data: np.ndarray,
                              labels: Optional[List] = None,
                              stride: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        슬라이딩 윈도우 생성 (KLP-SignGlove 기법)
        
        Args:
            data: 입력 데이터 (n_samples, n_features)
            labels: 라벨 데이터 (옵션)
            stride: 윈도우 이동 간격
            
        Returns:
            windowed_data: (n_windows, window_size, n_features)
            windowed_labels: (n_windows,) - labels가 있는 경우
        """
        n_samples, n_features = data.shape
        
        if n_samples < self.window_size:
            logger.warning(f"데이터 길이({n_samples})가 윈도우 크기({self.window_size})보다 작습니다.")
            # 패딩 또는 전체 데이터 사용
            padded_data = np.pad(data, ((0, self.window_size - n_samples), (0, 0)), mode='edge')
            return padded_data.reshape(1, self.window_size, n_features), None
        
        # 윈도우 개수 계산
        n_windows = (n_samples - self.window_size) // stride + 1
        
        # 윈도우 데이터 생성
        windowed_data = np.zeros((n_windows, self.window_size, n_features))
        windowed_labels = None
        
        if labels is not None:
            windowed_labels = np.zeros(n_windows, dtype=object)
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + self.window_size
            
            windowed_data[i] = data[start_idx:end_idx]
            
            if labels is not None:
                # 윈도우 중앙 지점의 라벨 사용
                mid_idx = start_idx + self.window_size // 2
                windowed_labels[i] = labels[min(mid_idx, len(labels) - 1)]
        
        return windowed_data, windowed_labels
    
    def remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        이상치 제거 (Z-score 기반)
        
        Args:
            data: 입력 데이터
            threshold: Z-score 임계값
            
        Returns:
            cleaned_data: 이상치가 제거된 데이터
        """
        if len(data.shape) == 1:
            # 1차원 데이터
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return data[z_scores < threshold]
        else:
            # 다차원 데이터
            cleaned_data = data.copy()
            for i in range(data.shape[1]):
                column = data[:, i]
                z_scores = np.abs((column - np.mean(column)) / np.std(column))
                outlier_mask = z_scores >= threshold
                
                # 이상치를 중앙값으로 대체
                cleaned_data[outlier_mask, i] = np.median(column)
            
            return cleaned_data
    
    def smooth_data(self, data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        데이터 평활화 (이동평균)
        
        Args:
            data: 입력 데이터
            kernel_size: 평활화 커널 크기
            
        Returns:
            smoothed_data: 평활화된 데이터
        """
        if len(data.shape) == 1:
            # 1차원 데이터
            kernel = np.ones(kernel_size) / kernel_size
            return np.convolve(data, kernel, mode='same')
        else:
            # 다차원 데이터
            smoothed_data = np.zeros_like(data)
            kernel = np.ones(kernel_size) / kernel_size
            
            for i in range(data.shape[1]):
                smoothed_data[:, i] = np.convolve(data[:, i], kernel, mode='same')
            
            return smoothed_data
    
    def preprocess_sensor_sequence(self, 
                                 sensor_data_list: List[SensorData],
                                 apply_filter: bool = True,
                                 apply_normalization: bool = True,
                                 apply_smoothing: bool = True,
                                 create_windows: bool = False) -> Dict[str, np.ndarray]:
        """
        전체 전처리 파이프라인 실행
        
        Args:
            sensor_data_list: 센서 데이터 리스트
            apply_filter: Butterworth 필터 적용 여부
            apply_normalization: 정규화 적용 여부
            apply_smoothing: 평활화 적용 여부
            create_windows: 윈도우 생성 여부
            
        Returns:
            processed_data: 전처리된 데이터 딕셔너리
        """
        logger.info(f"전처리 시작: {len(sensor_data_list)}개 샘플")
        
        # 1. 센서 데이터 추출
        sensor_arrays = self.extract_sensor_arrays(sensor_data_list)
        
        flex_data = sensor_arrays['flex']
        gyro_data = sensor_arrays['gyro']
        accel_data = sensor_arrays['accel']
        
        # 2. 이상치 제거
        flex_data = self.remove_outliers(flex_data)
        gyro_data = self.remove_outliers(gyro_data)
        accel_data = self.remove_outliers(accel_data)
        
        # 3. Butterworth 필터 적용
        if apply_filter:
            flex_data = self.apply_butterworth_filter(flex_data)
            gyro_data = self.apply_butterworth_filter(gyro_data)
            accel_data = self.apply_butterworth_filter(accel_data)
            logger.info("Butterworth 필터 적용 완료")
        
        # 4. 평활화
        if apply_smoothing:
            flex_data = self.smooth_data(flex_data)
            gyro_data = self.smooth_data(gyro_data)
            accel_data = self.smooth_data(accel_data)
            logger.info("데이터 평활화 완료")
        
        # 5. 정규화
        if apply_normalization:
            flex_data, gyro_data, accel_data = self.normalize_data(
                flex_data, gyro_data, accel_data, fit_scalers=not self.is_fitted
            )
            logger.info("데이터 정규화 완료")
        
        # 6. 윈도우 생성 (옵션)
        processed_data = {
            'flex': flex_data,
            'gyro': gyro_data,
            'accel': accel_data,
            'timestamps': sensor_arrays['timestamps']
        }
        
        if create_windows:
            # 모든 센서 데이터를 합쳐서 윈도우 생성
            combined_data = np.concatenate([flex_data, gyro_data, accel_data], axis=1)
            windowed_data, _ = self.create_sliding_windows(combined_data)
            processed_data['windowed'] = windowed_data
            logger.info(f"윈도우 생성 완료: {windowed_data.shape}")
        
        logger.info("전처리 완료")
        return processed_data
    
    def get_preprocessing_stats(self) -> Dict[str, any]:
        """전처리 통계 정보 반환"""
        return {
            'sampling_rate': self.sampling_rate,
            'filter_cutoff': self.filter_cutoff,
            'window_size': self.window_size,
            'is_fitted': self.is_fitted,
            'flex_scaler_range': getattr(self.flex_scaler, 'data_range_', None),
            'gyro_scaler_mean': getattr(self.gyro_scaler, 'mean_', None),
            'accel_scaler_center': getattr(self.accel_scaler, 'center_', None)
        }


def create_preprocessor_from_config(config: Dict) -> SensorPreprocessor:
    """설정으로부터 전처리기 생성"""
    return SensorPreprocessor(
        sampling_rate=config.get('sampling_rate', 20.0),
        filter_cutoff=config.get('filter_cutoff', 5.0),
        window_size=config.get('window_size', 100)
    )
