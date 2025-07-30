#!/bin/bash
# SignGlove Ubuntu/Linux 환경 설정 스크립트

set -e  # 에러 발생 시 스크립트 중단

echo "========================================"
echo "SignGlove Ubuntu/Linux 환경 설정 시작"
echo "========================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 시스템 업데이트
echo -e "${BLUE}[INFO]${NC} 시스템 패키지 업데이트 중..."
sudo apt update

# Python 설치 확인
echo -e "${BLUE}[INFO]${NC} Python 설치 확인 중..."
if ! command -v python3 &> /dev/null; then
    echo -e "${BLUE}[INFO]${NC} Python3 설치 중..."
    sudo apt install -y python3 python3-pip python3-venv
else
    echo -e "${GREEN}[SUCCESS]${NC} Python3 이미 설치됨"
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}[SUCCESS]${NC} Python ${PYTHON_VERSION} 확인 완료"

# pip 업그레이드
echo -e "${BLUE}[INFO]${NC} pip 업그레이드 중..."
python3 -m pip install --upgrade pip

# 필수 시스템 패키지 설치
echo -e "${BLUE}[INFO]${NC} 시스템 의존성 설치 중..."
sudo apt install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    python3-dev

# 시리얼 통신을 위한 권한 설정
echo -e "${BLUE}[INFO]${NC} 시리얼 포트 권한 설정 중..."
sudo usermod -a -G dialout $USER
sudo usermod -a -G tty $USER

# Poetry 설치 확인
echo -e "${BLUE}[INFO]${NC} Poetry 설치 확인 중..."
if ! command -v poetry &> /dev/null; then
    echo -e "${BLUE}[INFO]${NC} Poetry를 설치하는 중..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # PATH에 Poetry 추가
    export PATH="$HOME/.local/bin:$PATH"
    
    # .bashrc에 PATH 추가
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    
    echo -e "${BLUE}[INFO]${NC} Poetry PATH가 ~/.bashrc에 추가되었습니다."
    echo -e "${YELLOW}[INFO]${NC} 새 터미널을 열거나 다음 명령어를 실행하세요:"
    echo "  source ~/.bashrc"
else
    echo -e "${GREEN}[SUCCESS]${NC} Poetry 설치 확인 완료"
fi

# Poetry 의존성 설치
echo -e "${BLUE}[INFO]${NC} Poetry 의존성 설치 중..."
poetry install

# Python 패키지 구조 확인
echo -e "${BLUE}[INFO]${NC} Python 패키지 구조 확인 중..."
if [ ! -f "hardware/__init__.py" ]; then
    echo -e "${YELLOW}[WARNING]${NC} hardware/__init__.py가 없습니다. 생성합니다."
    touch hardware/__init__.py
fi

# udev 규칙 설정 (Arduino 접근용)
echo -e "${BLUE}[INFO]${NC} Arduino udev 규칙 설정 중..."
sudo tee /etc/udev/rules.d/99-arduino.rules > /dev/null <<EOF
# Arduino Uno
SUBSYSTEMS=="usb", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0043", MODE:="0666", GROUP="dialout", SYMLINK+="arduino_uno"
# Arduino Nano 33 IoT
SUBSYSTEMS=="usb", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="8057", MODE:="0666", GROUP="dialout", SYMLINK+="arduino_nano33iot"
# CH340 (저가형 Arduino 호환 보드)
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE:="0666", GROUP="dialout", SYMLINK+="arduino_ch340"
# CP2102 (또 다른 USB-Serial 변환기)
SUBSYSTEMS=="usb", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0666", GROUP="dialout", SYMLINK+="arduino_cp2102"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

# 양동건 스크립트 테스트
echo -e "${BLUE}[INFO]${NC} 양동건 팀원 스크립트 확인 중..."

if poetry run python -c "import hardware.donggeon.client.wifi_data_client; print('WiFi 클라이언트 import 성공')" 2>/dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} WiFi 클라이언트 import 성공"
else
    echo -e "${YELLOW}[WARNING]${NC} WiFi 클라이언트 import 실패"
fi

if poetry run python -c "import hardware.donggeon.client.uart_data_client; print('UART 클라이언트 import 성공')" 2>/dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} UART 클라이언트 import 성공"
else
    echo -e "${YELLOW}[WARNING]${NC} UART 클라이언트 import 실패"
fi

# 포트 확인
echo -e "${BLUE}[INFO]${NC} 사용 가능한 시리얼 포트 확인 중..."
if ls /dev/ttyUSB* 2>/dev/null || ls /dev/ttyACM* 2>/dev/null; then
    echo -e "${GREEN}[SUCCESS]${NC} 시리얼 포트 발견:"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
else
    echo -e "${YELLOW}[WARNING]${NC} 현재 연결된 시리얼 포트가 없습니다."
    echo "Arduino를 USB로 연결한 후 다시 확인하세요."
fi

echo "========================================"
echo -e "${GREEN}Ubuntu/Linux 환경 설정 완료!${NC}"
echo "========================================"
echo ""
echo "⚠️  중요: 시리얼 포트 권한 적용을 위해 로그아웃 후 다시 로그인하세요."
echo ""
echo "사용법:"
echo "  1. WiFi 클라이언트: poetry run donggeon-wifi"
echo "  2. UART 클라이언트: poetry run donggeon-uart"
echo "  3. TCP 서버: poetry run donggeon-tcp-server"
echo "  4. FastAPI 서버: poetry run start-server"
echo ""
echo "Poetry 가상환경 활성화:"
echo "  poetry shell"
echo ""