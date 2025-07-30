# SignGlove 데이터 수집기 사용 가이드

## 개요
SignGlove USB 데이터 수집기는 아두이노와 USB 시리얼 통신을 통해 센서 데이터를 실시간으로 수집하고 CSV 형태로 저장하는 도구입니다.

## 주요 기능
- 🔌 아두이노 USB 자동 연결
- 📋 자음/모음/숫자 카테고리 선택
- 🏷️ 수어 라벨 입력 시스템
- ⏱️ 1-60초 측정 시간 설정
- 🔄 카운트다운 측정 진행
- 💾 자동 CSV 저장
- 📊 실시간 데이터 통계

## 시스템 요구사항
- Python 3.9+
- Poetry (의존성 관리)
- 아두이노 하드웨어 (USB 연결)
- 센서 데이터 출력이 가능한 아두이노 펌웨어

## 설치 및 설정

### 1. 의존성 설치
```bash
# Poetry 환경에서 실행
poetry install

# 또는 pip로 직접 설치
pip install pyserial fastapi pydantic pandas numpy
```

### 2. 아두이노 연결
1. 아두이노를 USB 포트에 연결
2. 센서 데이터를 시리얼로 출력하는 펌웨어 업로드
3. 적절한 보드레이트 설정 (기본: 115200)

### 3. 데이터 출력 형식
아두이노에서 다음 형식으로 데이터를 출력해야 합니다:

#### CSV 형식 (권장)
```
flex1,flex2,flex3,flex4,flex5,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,battery,signal
512,345,678,234,789,-1.2,0.5,2.1,0.1,-9.8,0.3,85,-45
```

#### JSON 형식 (선택사항)
```json
{
  "flex_1": 512,
  "flex_2": 345,
  "flex_3": 678,
  "flex_4": 234,
  "flex_5": 789,
  "gyro_x": -1.2,
  "gyro_y": 0.5,
  "gyro_z": 2.1,
  "accel_x": 0.1,
  "accel_y": -9.8,
  "accel_z": 0.3,
  "battery": 85,
  "signal": -45
}
```

## 사용법

### 1. 기본 실행
```bash
# Poetry 환경에서
poetry run python scripts/data_collector.py

# 또는 직접 실행
python scripts/data_collector.py
```

### 2. 실행 과정

#### 단계 1: 연결
```
🔍 아두이노 포트 검색 중...
   ✅ 발견: /dev/ttyUSB0 - Arduino Uno
🔌 /dev/ttyUSB0 포트로 연결 중...
✅ 아두이노 연결 성공!
```

#### 단계 2: 카테고리 선택
```
📋 수어 카테고리를 선택하세요:
   1. 자음 (ㄱ, ㄴ, ㄷ, ...)
   2. 모음 (ㅏ, ㅓ, ㅗ, ...)
   3. 숫자 (0, 1, 2, ...)
선택 (1-3): 1
```

#### 단계 3: 라벨 선택
```
📝 consonant 카테고리의 사용 가능한 라벨:
   자음: ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ

🏷️ 수집할 라벨을 입력하세요: ㄱ
```

#### 단계 4: 측정 시간 설정
```
⏱️ 측정 시간을 입력하세요 (1-60초): 5
```

#### 단계 5: 데이터 수집
```
🎯 라벨 'ㄱ' 수집 준비
⚠️ 손을 올바른 자세로 준비해주세요.
준비가 되면 Enter를 눌러주세요...

🚀 5초 후 측정이 시작됩니다!
   3...
   2...
   1...
   🔴 측정 시작!
   [████████████████████] 5.0s / 5s (남은 시간: 0.0s)
   ✅ 측정 완료! 97개 데이터 수집됨
   💾 데이터 저장 완료: ㄱ_20250726_143052_123
```

#### 단계 6: 계속 여부
```
✅ 데이터 수집이 완료되었습니다!

🔄 다음 라벨을 측정하시겠습니까?
계속 (y/n): y
```

## 데이터 저장 위치

### 파일 구조
```
data/
├── raw/
│   ├── gesture_data_20250726.csv      # 제스처 메타데이터
│   ├── gesture_sequences_20250726.json # 센서 시퀀스 데이터
│   └── sensor_data_20250726.csv       # 개별 센서 데이터
├── backup/
│   └── backup_20250726_143052/        # 자동 백업
└── stats.json                         # 수집 통계
```

### CSV 파일 형식

#### gesture_data_YYYYMMDD.csv
```csv
timestamp,gesture_id,gesture_label,gesture_type,duration,performer_id,session_id,quality_score,sequence_length,notes
2025-07-26T14:30:52.123,ㄱ_20250726_143052_123,ㄱ,consonant,5.0,test_performer,session_20250726_143052,0.85,97,USB 시리얼 수집 - 97개 샘플
```

#### sensor_data_YYYYMMDD.csv
```csv
timestamp,device_id,flex_1,flex_2,flex_3,flex_4,flex_5,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,battery_level,signal_strength
2025-07-26T14:30:52.123,USB_ARDUINO_001,512,345,678,234,789,-1.2,0.5,2.1,0.1,-9.8,0.3,85,-45
```

## 트러블슈팅

### 1. 아두이노 연결 실패
```bash
❌ 아두이노를 찾을 수 없습니다.
```
**해결 방법:**
- USB 케이블 연결 확인
- 아두이노 전원 확인
- 다른 USB 포트 시도
- 드라이버 설치 확인 (CH340, CP210x 등)

### 2. 통신 테스트 실패
```bash
❌ 아두이노 통신 테스트 실패
```
**해결 방법:**
- 보드레이트 확인 (기본: 115200)
- 다른 시리얼 프로그램으로 테스트
- 아두이노 펌웨어 확인

### 3. 데이터 읽기 오류
```bash
⚠️ 센서 데이터 읽기 오류: ...
```
**해결 방법:**
- 데이터 출력 형식 확인
- 센서 연결 상태 점검
- 시리얼 버퍼 클리어

### 4. 권한 오류 (Linux/Mac)
```bash
❌ 시리얼 연결 실패: Permission denied
```
**해결 방법:**
```bash
# 사용자를 dialout 그룹에 추가
sudo usermod -a -G dialout $USER

# 또는 sudo로 실행
sudo python scripts/data_collector.py
```

## 고급 사용법

### 1. 포트 직접 지정
```python
# 스크립트 수정
collector = SerialDataCollector(port="/dev/ttyUSB0")
```

### 2. 보드레이트 변경
```python
# 스크립트 수정
collector = SerialDataCollector(baudrate=9600)
```

### 3. 수행자 ID 변경
```python
# 스크립트에서 performer_id 변경
self.performer_id = "your_name"
```

## 데이터 활용

### 1. 데이터 로드 예시
```python
import pandas as pd
import json

# 제스처 메타데이터 로드
gestures = pd.read_csv("data/raw/gesture_data_20250726.csv")

# 센서 시퀀스 로드
with open("data/raw/gesture_sequences_20250726.json", "r") as f:
    sequences = json.load(f)

# 특정 제스처의 센서 데이터
gesture_id = "ㄱ_20250726_143052_123"
sensor_sequence = sequences[gesture_id]["sensor_sequence"]
```

### 2. 통계 분석
```python
# 수집된 제스처별 통계
gesture_counts = gestures['gesture_label'].value_counts()
print(gesture_counts)

# 평균 지속 시간
avg_duration = gestures['duration'].mean()
print(f"평균 지속 시간: {avg_duration:.2f}초")
```

## 다음 단계
1. 수집된 데이터로 AI 모델 학습
2. 실시간 인식 시스템 구축
3. 정확도 평가 및 개선
4. 더 많은 수어 클래스 추가

## 문의 및 지원
- 문제 발생 시 GitHub Issues 등록
- 개발팀 이메일: signglove@example.com 