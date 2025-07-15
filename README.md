# SignGlove 프로젝트

수어 인식을 위한 스마트 글러브 시스템

## 프로젝트 개요
- **목표**: 플렉스 센서와 자이로 센서를 사용한 수어 인식 시스템 구축
- **기술 스택**: Python, FastAPI, Arduino, WiFi 통신, Poetry
- **마감일**: 2025.07.04(Fri)

## 팀원
- **이민우**: 데이터 수집 서버 개발
- **양동건**: 하드웨어 설계 및 제작
- **YUBEEN**: 수어 클래스 정의
- **정재연**: 수어 클래스 정의

## 시스템 구성
- **하드웨어**: 플렉스 센서 5개 + 자이로 센서(6DOF) + 아두이노/라즈베리파이
- **통신**: WiFi 기반 실시간 데이터 전송
- **서버**: FastAPI 기반 데이터 수집 및 저장
- **데이터**: CSV 형태로 수어 동작 데이터 저장

## 프로젝트 구조
```
SignGlove/
├── pyproject.toml            # Poetry 프로젝트 설정
├── .cursorrules              # 개발 가이드라인
├── PROJECT_PLAN.md           # 프로젝트 계획서
├── README.md                 # 프로젝트 설명서
├── hardware/                 # 하드웨어 관련 파일
│   ├── arduino/              # 아두이노 코드
│   └── circuit_diagrams/     # 회로도
├── server/                   # 서버 코드
│   ├── main.py              # FastAPI 서버
│   ├── data_storage.py      # 데이터 저장 모듈
│   ├── data_validation.py   # 데이터 검증 모듈
│   └── models/              # 데이터 모델
├── tests/                    # 테스트 파일
│   ├── dummy_data_generator.py  # 더미 데이터 생성기
│   └── scenarios/           # 테스트 시나리오
├── scripts/                  # 실행 스크립트
│   ├── start_server.py      # 서버 시작
│   ├── run_tests.py         # 테스트 실행
│   └── demo.py              # 전체 데모
├── data/                     # 데이터 파일
│   ├── raw/                 # 원시 데이터
│   ├── processed/           # 처리된 데이터
│   └── backup/              # 백업 데이터
└── docs/                     # 문서
```

## 🚀 빠른 시작

### 1. Poetry 환경 설정
```bash
# Poetry 설치 (이미 설치되어 있다면 생략)
curl -sSL https://install.python-poetry.org | python3 -

# 프로젝트 클론
git clone <repository-url>
cd SignGlove

# 의존성 설치 및 가상환경 생성
poetry install

# 가상환경 활성화
poetry shell
```

### 2. 전체 데모 실행 (추천)
```bash
# 서버 시작부터 테스트까지 자동화
python scripts/demo.py
```

### 3. 개별 실행
```bash
# 서버만 시작
python scripts/start_server.py

# 다른 터미널에서 테스트 실행
python scripts/run_tests.py
```

## 📋 테스트 시나리오

### 1. 기본 연결성 테스트
- 서버 헬스 체크
- 서버 상태 정보 조회
- 데이터 통계 조회

### 2. 센서 데이터 테스트
- 단일 센서 데이터 전송
- 연속 센서 데이터 전송
- 다양한 제스처 타입 테스트

### 3. 제스처 데이터 테스트
- 한국어 모음 제스처 (ㅏ, ㅓ, ㅗ, ㅜ, ㅡ)
- 한국어 자음 제스처 (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ)
- 숫자 제스처 (1-5)

### 4. 성능 테스트
- 고속 데이터 전송 (100건)
- 동시 전송 테스트
- 메모리 사용량 확인

### 5. 데이터 검증 테스트
- 정상 데이터 검증
- 경계값 테스트
- 타임스탬프 검증

### 6. 데이터 저장 테스트
- CSV 파일 생성 확인
- JSON 시퀀스 파일 확인
- 데이터 무결성 검증

## 🔧 개발 도구

### Poetry 명령어
```bash
# 의존성 추가
poetry add <package>

# 개발 의존성 추가
poetry add --group dev <package>

# 의존성 업데이트
poetry update

# 스크립트 실행
poetry run python scripts/demo.py
```

### 테스트 실행
```bash
# 전체 테스트
poetry run pytest

# 커버리지 포함 테스트
poetry run pytest --cov=server

# 더미 데이터 생성
poetry run python tests/dummy_data_generator.py
```

### 코드 포맷팅
```bash
# 코드 포맷팅
poetry run black .

# 코드 스타일 검사
poetry run flake8

# 타입 검사
poetry run mypy server/
```

## 🎯 더미 데이터 생성기

더미 데이터 생성기는 실제 하드웨어 없이도 시스템을 테스트할 수 있게 해줍니다.

### 사용법
```bash
python tests/dummy_data_generator.py
```

### 지원 기능
- **센서 데이터 시뮬레이션**: 실제적인 플렉스 센서 및 자이로 센서 데이터 생성
- **제스처 시퀀스 생성**: 한국어 모음/자음/숫자 제스처 패턴 시뮬레이션
- **실시간 전송**: HTTP를 통한 실시간 데이터 전송
- **다양한 시나리오**: 중립, 주먹, 펼침, 가리키기 등 다양한 손 모양 패턴

## 🖥️ API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 주요 엔드포인트
- `GET /`: 서버 상태 확인
- `GET /health`: 헬스 체크
- `POST /data/sensor`: 센서 데이터 수집
- `POST /data/gesture`: 제스처 데이터 수집
- `GET /data/stats`: 데이터 통계 조회

## 📊 데이터 형식

### 센서 데이터
```json
{
  "device_id": "SIGNGLOVE_001",
  "timestamp": "2024-01-03T10:30:00",
  "flex_sensors": {
    "flex_1": 400.5,
    "flex_2": 420.1,
    "flex_3": 410.8,
    "flex_4": 430.2,
    "flex_5": 440.7
  },
  "gyro_data": {
    "gyro_x": 1.2,
    "gyro_y": -0.5,
    "gyro_z": 0.8,
    "accel_x": 0.1,
    "accel_y": 0.2,
    "accel_z": 9.8
  },
  "battery_level": 85.5,
  "signal_strength": -45
}
```

### 제스처 데이터
```json
{
  "gesture_id": "ㅏ_20240103_103000_1234",
  "gesture_label": "ㅏ",
  "gesture_type": "vowel",
  "duration": 2.5,
  "performer_id": "user_001",
  "session_id": "session_20240103_103000",
  "sensor_sequence": [
    // 센서 데이터 배열
  ],
  "quality_score": 0.95
}
```

## 🔧 하드웨어 설정

### Arduino 설정
1. `hardware/arduino/sign_glove_client.ino` 파일 열기
2. WiFi 설정 수정:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. 서버 URL 설정:
   ```cpp
   const char* server_url = "http://YOUR_SERVER_IP:8000/data/sensor";
   ```

### 필요한 라이브러리
- WiFi
- HTTPClient
- ArduinoJson
- Wire (I2C)
- MPU6050

## 🐛 문제 해결

### 서버 시작 오류
```bash
# 포트 8000이 이미 사용 중인 경우
lsof -ti:8000 | xargs kill -9

# 의존성 문제
poetry install --no-cache
```

### 데이터 전송 오류
- WiFi 연결 상태 확인
- 서버 URL 및 포트 확인
- 방화벽 설정 확인

### 테스트 실패
```bash
# 서버가 실행 중인지 확인
curl http://localhost:8000/health

# 로그 확인
tail -f server.log
```

## 📈 성능 최적화

### 서버 성능
- 비동기 처리로 높은 동시성 지원
- 백그라운드 태스크로 데이터 저장
- 메모리 효율적인 CSV 스트리밍

### 데이터 처리
- 실시간 데이터 검증
- 배치 처리로 디스크 I/O 최적화
- 자동 백업 시스템

## 🤝 기여 방법

1. 이슈 생성 또는 기존 이슈 확인
2. 브랜치 생성: `git checkout -b feature/your-feature`
3. 개발 및 테스트
4. 커밋: `git commit -m "feat: 새로운 기능 추가"`
5. 푸시: `git push origin feature/your-feature`
6. Pull Request 생성

## 📝 개발 가이드라인

자세한 개발 규칙은 `.cursorrules` 파일을 참조하세요.

### 주요 규칙
- 모든 코드는 한국어 주석으로 작성
- 함수명과 변수명은 영어로, 설명은 한국어로
- 에러 처리는 반드시 포함
- 테스트 코드 작성 필수

## 📄 라이센스

MIT License

## 📞 연락처

- **이민우**: 프로젝트 리드 및 서버 개발
- **GitHub**: [Repository Link]
- **문의**: 팀 슬랙 또는 이메일

---

**SignGlove 프로젝트** - 수어 인식을 통한 소통의 새로운 가능성을 열어갑니다. 🤟 