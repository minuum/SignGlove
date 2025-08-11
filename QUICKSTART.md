# SignGlove 빠른 시작 가이드 🚀

이 가이드는 SignGlove 프로젝트를 빠르게 설정하고 실행하는 방법을 안내합니다.

## 📋 사전 요구사항

- Python 3.9 이상
- Git
- Poetry (자동 설치됨)

## ⚡ 초고속 설정 (추천)

GitHub에서 클론한 후 단 한 줄로 모든 환경 설정이 완료됩니다!

```bash
git clone <repository-url>
cd SignGlove
python setup.py
```

## 🖥️ 플랫폼별 상세 설정

### Windows
```batch
git clone <repository-url>
cd SignGlove
scripts\setup_windows.bat
```

### macOS
```bash
git clone <repository-url>
cd SignGlove
./scripts/setup_macos.sh
```

### Ubuntu/Linux
```bash
git clone <repository-url>
cd SignGlove
./scripts/setup_ubuntu.sh
```

## 🚀 실행 방법

환경 설정 완료 후 다음 명령어들을 사용할 수 있습니다:

### 🖥️ 서버 시작
```bash
poetry shell                    # Poetry 환경 활성화
poetry run start-server         # FastAPI 서버 시작
```
브라우저에서 `http://localhost:8000`으로 API 문서 확인

### 🔗 서브모듈 초기화 (외부 저장소 사용 시)

```bash
# 처음 클론하는 경우 (추천)
git clone --recurse-submodules <repository-url>

# 이미 클론했다면
git submodule update --init --recursive

# 최신 원격 반영
git submodule update --remote --merge
```

### 🤖 양동건 팀원 하드웨어 클라이언트
```bash
# WiFi 방식 (Arduino Nano 33 IoT)
poetry run donggeon-wifi

# UART 방식 (Arduino + 플렉스 센서)
poetry run donggeon-uart

# 간단한 TCP 서버 (미팅 코드 호환)
poetry run donggeon-tcp-server
```

### 🧪 테스트 및 개발
```bash
poetry run pytest              # 테스트 실행
poetry run test-dummy          # 더미 데이터 생성
poetry run black .             # 코드 포맷팅
poetry run flake8              # 코드 린팅
```

## 🔧 고급 설정 옵션

### 빠른 설정 (Poetry만)
```bash
python setup.py --quick
```

### 양동건 스크립트 테스트만
```bash
python setup.py --test-donggeon
```

### Python 범용 스크립트만
```bash
python setup.py --python-only
```

## 📁 확장된 프로젝트 구조

```
SignGlove/
├── server/                     # FastAPI 서버 코드
├── hardware/                   # 하드웨어 관련 코드
│   └── donggeon/              # 양동건 팀원 코드
│       ├── arduino/           # Arduino 펌웨어
│       ├── client/            # Python 클라이언트
│       └── server/            # TCP 서버
├── scripts/                   # 환경 설정 스크립트
├── tests/                     # 테스트 코드
├── data/                      # 데이터 디렉토리
├── config/                    # 설정 파일
├── docs/                      # 문서
└── setup.py                   # 통합 환경 설정
```

## 🎯 사용 시나리오별 가이드

### 1️⃣ 서버 개발자 (이민우)
```bash
git clone <repo>
cd SignGlove
python setup.py
poetry shell
poetry run start-server
```

### 2️⃣ 하드웨어 개발자 (양동건)
```bash
git clone <repo>
cd SignGlove
python setup.py
poetry shell

# WiFi 방식
poetry run donggeon-wifi

# 또는 UART 방식
poetry run donggeon-uart
```

### 3️⃣ 전체 시스템 테스트
```bash
# 터미널 1: 서버 시작
poetry run start-server

# 터미널 2: 하드웨어 클라이언트
poetry run donggeon-uart

# 터미널 3: 모니터링
curl http://localhost:8000/status
```

## 📊 양동건 팀원 하드웨어 사용법

### WiFi 방식 (간편함)
- Arduino Nano 33 IoT 사용
- WiFi 네트워크를 통한 무선 데이터 전송
- 10Hz 샘플링
- IMU 센서만 (LSM6DS3)

### UART 방식 (고성능)
- Arduino + 플렉스 센서 5개
- USB 시리얼 통신
- 50Hz 고속 샘플링
- IMU + 플렉스 센서 통합

### 데이터 흐름
```
Arduino → Python 클라이언트 → FastAPI 서버 → Database
```

## 🔍 트러블슈팅

### Poetry 설치 문제
```bash
curl -sSL https://install.python-poetry.org | python3 -
# 또는
pip install poetry
```

### 시리얼 포트 권한 문제 (Linux)
```bash
sudo usermod -a -G dialout $USER
# 재로그인 필요
```

### Arduino 드라이버 문제 (Windows)
- CH340 드라이버 설치 필요
- Arduino IDE에서 드라이버 자동 설치

### 환경 재설정
```bash
poetry env remove python  # 가상환경 삭제
poetry install            # 재설치
```

## 📞 문제 해결 및 지원

### 일반적인 문제
- [기술적 도전과제](TECHNICAL_CHALLENGES.md)
- [팀 역할 및 연락처](TEAM_ROLES.md)

### 양동건 팀원 하드웨어 관련
- [하드웨어 문서](hardware/donggeon/README.md)
- 시리얼 포트 권한
- 아두이노 드라이버 설치

### 로그 확인
```bash
# 각 클라이언트별 로그 파일 생성됨
ls *.log
tail -f wifi_client.log      # WiFi 클라이언트 로그
tail -f uart_client.log      # UART 클라이언트 로그
```

## 📖 더 많은 정보

- [프로젝트 개요](PROJECT_OVERVIEW.md)
- [팀 역할](TEAM_ROLES.md)
- [양동건 하드웨어 가이드](hardware/donggeon/README.md)
- [센서 융합 가이드](SENSOR_FUSION_GUIDE.md)

---

**🎉 축하합니다! SignGlove 환경 설정이 완료되었습니다.**

이제 `poetry shell`로 환경을 활성화하고 원하는 컴포넌트를 실행해보세요!