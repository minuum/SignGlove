# SignGlove 프로젝트

수어 인식을 위한 스마트 글러브 시스템

## 프로젝트 개요
- **목표**: 플렉스 센서와 자이로 센서를 사용한 한국어 수어 인식 시스템 구축
- **기술 스택**: Python, FastAPI, Arduino, WiFi 통신, Poetry, KSL(Korean Sign Language)
- **마감일**: 2025.08.15(Fri) - 데이터셋 수집 시작 목표

## 시스템 아키텍처

```
[아두이노 글러브] → [WiFi] → [교수님 서버(미정)] → [FastAPI 서버] → [CSV/JSON 저장]
     ↓                                                    ↓
[플렉스 센서 5개]                                      [실시간 모니터링]
[자이로 센서 6DOF]                                     [데이터 분석]
```

## 팀원 및 담당 업무

| 팀원 | 담당 업무 | 진행 상황 | 주요 과제 |
|------|-----------|-----------|-----------|
| **이민우** | 데이터 수집 서버 개발 (FastAPI) | ✅ **완료** | 아두이노-서버 연동 준비 |
| **양동건** | 하드웨어 설계 및 제작, WiFi 통신 | ✅🔄 **75% 완료** | LSM6DS3 분석 완료, 납땜 연결 작업중 |
| **YUBEEN** | 한국어 수어 클래스 정의 | 🔄 **진행중** | 모음/자음/숫자 체계 확립 |
| **정재연** | 한국어 수어 클래스 정의, 데이터셋 설계 | 🔄 **진행중** | 데이터 수집 방식 설계 |

## 시스템 구성
- **하드웨어**: 플렉스 센서 5개 + 자이로 센서(6DOF) + 아두이노/라즈베리파이
- **통신**: WiFi 기반 실시간 데이터 전송 (아두이노 → 교수님 서버 → FastAPI)
- **서버**: FastAPI 기반 데이터 수집 및 저장 (완료)
- **데이터**: CSV/JSON 형태로 KSL 제스처 데이터 저장

## 프로젝트 구조
```
SignGlove/
├── pyproject.toml            # Poetry 프로젝트 설정 (양동건 스크립트 포함)
├── setup.py                  # 🆕 통합 환경 설정 스크립트
├── QUICKSTART.md             # 🆕 빠른 시작 가이드
├── PROJECT_PLAN.md           # 프로젝트 계획서
├── README.md                 # 프로젝트 설명서
├── scripts/                  # 🆕 환경 설정 스크립트
│   ├── setup_windows.bat     # Windows 환경 설정
│   ├── setup_macos.sh        # macOS 환경 설정
│   ├── setup_ubuntu.sh       # Ubuntu/Linux 환경 설정
│   └── setup_environment.py  # Python 범용 설정
├── hardware/                 # 하드웨어 관련 파일
│   ├── arduino/              # 기존 아두이노 코드
│   ├── donggeon/             # 🆕 양동건 팀원 코드
│   │   ├── arduino/          # Arduino 펌웨어 (WiFi/UART)
│   │   ├── client/           # Python 클라이언트 코드
│   │   ├── server/           # TCP 서버 코드
│   │   └── README.md         # 양동건 코드 사용법
│   └── circuit_diagrams/     # 회로도
├── server/                   # 서버 코드 (이민우)
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

### ⚡ 초고속 설정 (추천)
```bash
# 단 한 줄로 모든 환경 설정 완료!
git clone <repository-url>
cd SignGlove
python setup.py
```

### 🖥️ 플랫폼별 설정
```bash
# Windows
scripts\setup_windows.bat

# macOS
./scripts/setup_macos.sh

# Ubuntu/Linux
./scripts/setup_ubuntu.sh
```

### 🚀 실행
```bash
# Poetry 환경 활성화
poetry shell

# FastAPI 서버 시작
poetry run start-server

# 양동건 팀원 하드웨어 클라이언트
poetry run donggeon-uart        # UART 방식 (플렉스+IMU)
poetry run donggeon-wifi        # WiFi 방식 (IMU만)
poetry run donggeon-tcp-server  # 간단한 TCP 서버

# 기존 데모 실행
python scripts/demo.py
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

## 📚 관련 문서

### 기술 문서
- **[📊 프로젝트 완전 개요](PROJECT_OVERVIEW.md)** - 전체 시스템 아키텍처 및 팀 현황 🆕
- **[기술적 도전과제](TECHNICAL_CHALLENGES.md)** - 자이로 센서 모호함 해결 등 핵심 기술 이슈
- **[센서 퓨전 알고리즘 가이드](SENSOR_FUSION_GUIDE.md)** - Madgwick Filter 구현 및 방향 모호함 해결 🆕
- **[팀 역할 분담](TEAM_ROLES.md)** - 팀원별 상세 담당 업무 및 협업 방안
- **[빠른 시작 가이드](QUICKSTART.md)** - 프로젝트 설치 및 실행 방법
- **[우분투 서버 배포 가이드](UBUNTU_DEPLOYMENT.md)** - 우분투 프로덕션 서버 배포 방법 🆕

### 프로젝트 관리
- **현재 진행상황**: FastAPI 데이터 수집 서버 완성 ✅ (2025.07.15)
- **다음 우선순위**: 자이로 센서 모호함 해결, 하드웨어 제작
- **마일스톤**: 2025.08.15 데이터셋 수집 시작 목표

### 연락처
- **이민우**: 데이터 서버 관련 문의
- **양동건**: 하드웨어 및 센서 관련 문의  
- **YUBEEN/정재연**: 수어 클래스 정의 관련 문의

## 📄 라이센스

MIT License

## 📞 연락처

- **이민우**: 프로젝트 리드 및 서버 개발
- **GitHub**: [Repository Link]
- **문의**: 팀 슬랙 또는 이메일

---

**SignGlove 프로젝트** - 수어 인식을 통한 소통의 새로운 가능성을 열어갑니다. 🤟 