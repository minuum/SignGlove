# SignGlove 통합 수어 데이터 수집기

## 🎯 개요

한국어 수어 34개 클래스를 위한 통합 데이터 수집 시스템입니다. ROS2 스타일 키보드 인터페이스와 KLP-SignGlove 호환 형식을 제공합니다.

## 🔧 하드웨어 요구사항

- **Arduino Nano 33 IoT** (LSM6DS3 IMU 내장)
- **플렉스 센서 5개** (A0, A1, A2, A3, A6 핀 연결)
- **USB 시리얼 연결** (115200 baud)

## 📊 데이터 형식

### CSV 출력 (12필드)
```
timestamp(ms),pitch(°),roll(°),yaw(°),accel_x(g),accel_y(g),accel_z(g),flex1,flex2,flex3,flex4,flex5
```

### H5 저장 형식 (KLP-SignGlove 호환)
```python
episode_YYYYMMDD_HHMMSS_클래스명.h5
├── timestamps          # PC 수신 타임스탬프 (int64)
├── arduino_timestamps  # 아두이노 타임스탬프 (int64)
├── sampling_rates     # 실시간 Hz 측정 (float32)
├── sensor_data        # 메인 센서 데이터 [8채널: flex5 + orientation3]
└── sensors/
    ├── flex           # 플렉스 센서 [5채널] (0-1023)
    ├── orientation    # 자이로 오일러각 [3채널] (degree)
    └── acceleration   # 가속도 [3채널] (g)
```

## 🤟 한국어 수어 34개 클래스

### 자음 (14개) - 우선순위 1,2
```
ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ
```

### 모음 (10개) - 우선순위 2,3
```
ㅏ ㅑ ㅓ ㅕ ㅗ ㅛ ㅜ ㅠ ㅡ ㅣ
```

### 숫자 (10개) - 우선순위 3
```
0 1 2 3 4 5 6 7 8 9
```

## 🎮 사용법

### 1. 아두이노 펌웨어 업로드
```bash
# imu_flex_serial.ino를 Arduino IDE로 업로드
```

### 2. 통합 수집기 실행
```bash
python integration/signglove_unified_collector.py
```

### 3. 키보드 조작
```
C: 아두이노 연결/재연결
N: 새 에피소드 시작 (클래스 선택)
M: 현재 에피소드 종료
P: 진행 상황 확인
R: 진행률 재계산 (H5 파일 스캔)
Q: 프로그램 종료
```

### 4. 수집 프로세스
1. **'C' 키로 아두이노 연결**
2. **'N' 키로 클래스 선택** (1-34번)
3. **자연스러운 수어 동작 수행** (3-5초 권장)
4. **'M' 키로 에피소드 종료**
5. **반복**

## 📈 진행률 관리

### 수집 목표 (클래스별)
- **우선순위 1 (기본 자음)**: 100개 에피소드
- **우선순위 2 (기본 모음)**: 80개 에피소드
- **우선순위 3 (복합/숫자)**: 50-60개 에피소드

### 자동 진행률 추적
- `datasets/unified/collection_progress.json`
- H5 파일 자동 스캔
- 실시간 완료도 표시

## 🔄 데이터 저장 구조

```
datasets/
├── raw/                    # 원본 CSV 파일
├── processed/              # 전처리된 데이터
├── unified/                # 통합 H5 데이터셋
└── ksl_34classes/         # 클래스별 정리
    ├── consonants/        # 자음 데이터
    ├── vowels/           # 모음 데이터
    └── numbers/          # 숫자 데이터
```

## ⚡ 성능 특징

- **실시간 Hz 측정**: 아두이노 타임스탬프 기준
- **논블로킹 수집**: 큐 기반 데이터 처리
- **메모리 효율성**: 스트리밍 방식 저장
- **안정성**: 연결 끊김 자동 감지/복구

## 🔗 연동 시스템

### KLP-SignGlove 호환
- 동일한 8채널 센서 데이터 구조
- 562 FPS 실시간 추론 지원
- CNN+LSTM+Attention 모델 적용 가능

### SignGlove 메인 프로젝트 연동
- `SensorData` 모델 완벽 호환
- FastAPI 서버 직접 연결 가능
- 실시간 추론 + TTS 통합

## 🧪 테스트 및 검증

### 데이터 품질 확인
```bash
# 수집된 H5 파일 검증
python -c "
import h5py
with h5py.File('episode_20250101_120000_ㄱ.h5', 'r') as f:
    print(f'클래스: {f.attrs[\"class_name\"]}')
    print(f'샘플 수: {f.attrs[\"num_samples\"]}')
    print(f'평균 Hz: {f.attrs[\"avg_sampling_rate\"]:.1f}')
    print(f'센서 데이터 형태: {f[\"sensor_data\"].shape}')
"
```

### 실시간 모니터링
- 샘플링 레이트 실시간 표시
- 수집 품질 자동 체크
- 에피소드 길이 권장사항 제공

## 📝 데이터셋 메타데이터

각 H5 파일에는 다음 메타데이터가 포함됩니다:

```python
attrs = {
    'class_name': 'ㄱ',                    # KSL 클래스명
    'class_category': 'consonant',        # 카테고리
    'episode_duration': 3.2,              # 수집 시간 (초)
    'num_samples': 128,                   # 샘플 수
    'avg_sampling_rate': 40.0,            # 평균 Hz
    'device_id': 'SIGNGLOVE_UNIFIED_001', # 장치 ID
    'collection_date': '2025-01-26T15:30:00', # 수집 날짜
    'label': 'ㄱ',                        # ML 라벨
    'label_idx': 0                        # 라벨 인덱스 (0-33)
}
```

## 🎓 머신러닝 활용

### 모델 학습 준비
1. H5 파일들을 학습/검증/테스트 셋으로 분할
2. 센서 데이터 정규화 (MinMax, Z-score, Robust)
3. 시계열 윈도우 생성 (sliding window)
4. 데이터 증강 (noise, rotation, scaling)

### 추천 모델 아키텍처
- **CNN+LSTM+Attention** (KLP-SignGlove 방식)
- **Transformer** (시계열 처리)
- **XGBoost/RandomForest** (전통적 ML)

## 🔮 향후 계획

- [ ] 실시간 모델 학습 파이프라인
- [ ] 웹 기반 수집 인터페이스
- [ ] 다중 사용자 수집 시스템
- [ ] 자동 데이터 품질 평가
- [ ] 클라우드 저장소 연동

---

**🤟 SignGlove Project - Making Sign Language Accessible Through Technology**
