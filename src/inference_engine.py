#!/usr/bin/env python3
"""
SignGlove 추론 엔진
- 실시간 센서 데이터 처리 및 모델 추론
- 다양한 모델 타입 지원 (BiGRU, CNN, MLP)
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

# 상위 디렉토리의 모듈 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ModelConfig:
    """모델 설정"""
    model_type: str  # 'bigru', 'cnn', 'mlp'
    input_size: int = 8
    hidden_size: int = 64
    num_layers: int = 2
    num_classes: int = 24
    dropout: float = 0.2
    model_path: Optional[str] = None

@dataclass
class InferenceResult:
    """추론 결과"""
    predicted_class: str
    confidence: float
    probabilities: List[float]
    processing_time: float
    model_type: str
    timestamp: float

class BiGRUModel(nn.Module):
    """BiGRU 모델 정의"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 128),  # bidirectional이므로 *2
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        # 마지막 타임스텝만 사용
        last_output = gru_out[:, -1, :]
        logits = self.classifier(last_output)
        return logits

class CNN1DModel(nn.Module):
    """1D CNN 모델 정의"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        conv_out = self.conv_layers(x)
        conv_out = conv_out.squeeze(-1)  # (batch_size, 256)
        logits = self.classifier(conv_out)
        return logits

class MLPModel(nn.Module):
    """MLP 모델 정의"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 시퀀스 길이를 고려한 입력 크기
        input_features = config.input_size * 80  # 80 time steps
        
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        logits = self.classifier(x)
        return logits

class SignGloveInferenceEngine:
    """SignGlove 추론 엔진"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = self._load_class_names()
        
        print(f"🤖 추론 엔진 초기화 중... (디바이스: {self.device})")
        
        # 모델 생성
        self._create_model()
        
        # 모델 로드
        if config.model_path and Path(config.model_path).exists():
            self.load_model(config.model_path)
        else:
            print("⚠️ 모델 파일이 없습니다. Mock 모드로 실행됩니다.")
            self.model_loaded = False
        
        print("✅ 추론 엔진 준비 완료!")
    
    def _load_class_names(self) -> List[str]:
        """클래스 이름 로드"""
        # 한국어 수어 클래스 정의
        consonants = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
        vowels = ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"]
        return consonants + vowels
    
    def _create_model(self):
        """모델 생성"""
        if self.config.model_type.lower() == 'bigru':
            self.model = BiGRUModel(self.config)
        elif self.config.model_type.lower() == 'cnn':
            self.model = CNN1DModel(self.config)
        elif self.config.model_type.lower() == 'mlp':
            self.model = MLPModel(self.config)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.config.model_type}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path: str) -> bool:
        """모델 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model_loaded = True
            print(f"✅ 모델 로드 완료: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.model_loaded = False
            return False
    
    def preprocess_data(self, sensor_data: np.ndarray) -> torch.Tensor:
        """센서 데이터 전처리"""
        # 정규화 (0-1 범위로)
        normalized_data = sensor_data.astype(np.float32)
        
        # 각 센서별 정규화
        # 플렉스 센서 (0-1023 -> 0-1)
        normalized_data[:, :5] = normalized_data[:, :5] / 1023.0
        
        # IMU 센서 (각도 정규화)
        normalized_data[:, 5:8] = (normalized_data[:, 5:8] + 180) / 360.0
        
        # 시퀀스 길이 맞추기 (패딩 또는 자르기)
        target_length = 80
        if len(normalized_data) < target_length:
            # 패딩
            pad_length = target_length - len(normalized_data)
            padding = np.zeros((pad_length, normalized_data.shape[1]))
            normalized_data = np.vstack([normalized_data, padding])
        elif len(normalized_data) > target_length:
            # 자르기 (마지막 80개 사용)
            normalized_data = normalized_data[-target_length:]
        
        # 배치 차원 추가
        tensor_data = torch.from_numpy(normalized_data).unsqueeze(0)  # (1, 80, 8)
        
        return tensor_data.to(self.device)
    
    def predict(self, sensor_data: np.ndarray) -> InferenceResult:
        """추론 실행"""
        start_time = time.time()
        
        if not self.model_loaded:
            # Mock 추론
            return self._mock_predict(sensor_data, start_time)
        
        try:
            # 데이터 전처리
            input_tensor = self.preprocess_data(sensor_data)
            
            # 추론 실행
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                prob_list = probabilities.squeeze().cpu().numpy().tolist()
            
            processing_time = time.time() - start_time
            
            return InferenceResult(
                predicted_class=predicted_class,
                confidence=confidence_score,
                probabilities=prob_list,
                processing_time=processing_time,
                model_type=self.config.model_type,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"❌ 추론 오류: {e}")
            return self._mock_predict(sensor_data, start_time)
    
    def _mock_predict(self, sensor_data: np.ndarray, start_time: float) -> InferenceResult:
        """Mock 추론 (모델이 없을 때)"""
        # 랜덤 예측
        class_idx = np.random.randint(0, len(self.class_names))
        predicted_class = self.class_names[class_idx]
        confidence = np.random.uniform(0.6, 0.95)
        
        # 확률 분포 생성
        probabilities = np.random.dirichlet(np.ones(len(self.class_names)))
        probabilities[class_idx] = confidence
        
        processing_time = time.time() - start_time
        
        return InferenceResult(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities.tolist(),
            processing_time=processing_time,
            model_type=f"{self.config.model_type}_mock",
            timestamp=time.time()
        )
    
    def batch_predict(self, sensor_data_list: List[np.ndarray]) -> List[InferenceResult]:
        """배치 추론"""
        results = []
        for sensor_data in sensor_data_list:
            result = self.predict(sensor_data)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.config.model_type,
            'input_size': self.config.input_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_classes': self.config.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_loaded': self.model_loaded,
            'device': str(self.device),
            'class_names': self.class_names
        }

def create_model_config(model_type: str, **kwargs) -> ModelConfig:
    """모델 설정 생성"""
    default_configs = {
        'bigru': {
            'input_size': 8,
            'hidden_size': 64,
            'num_layers': 2,
            'num_classes': 24,
            'dropout': 0.2
        },
        'cnn': {
            'input_size': 8,
            'hidden_size': 64,
            'num_layers': 3,
            'num_classes': 24,
            'dropout': 0.2
        },
        'mlp': {
            'input_size': 8,
            'hidden_size': 128,
            'num_layers': 4,
            'num_classes': 24,
            'dropout': 0.3
        }
    }
    
    config_dict = default_configs.get(model_type.lower(), default_configs['bigru'])
    config_dict.update(kwargs)
    
    return ModelConfig(
        model_type=model_type,
        **config_dict
    )

def main():
    """테스트 함수"""
    print("SignGlove 추론 엔진 테스트")
    
    # 모델 설정
    config = create_model_config('bigru')
    
    # 추론 엔진 생성
    engine = SignGloveInferenceEngine(config)
    
    # 테스트 데이터 생성
    test_data = np.random.randn(80, 8)  # 80 time steps, 8 features
    
    # 추론 실행
    result = engine.predict(test_data)
    
    print(f"예측 결과: {result.predicted_class}")
    print(f"신뢰도: {result.confidence:.3f}")
    print(f"처리 시간: {result.processing_time:.3f}초")
    
    # 모델 정보 출력
    model_info = engine.get_model_info()
    print(f"\n모델 정보: {model_info}")

if __name__ == "__main__":
    main()
