#!/bin/bash
# SignGlove macOS 환경 설정 스크립트

set -e  # 에러 발생 시 스크립트 중단

echo "========================================"
echo "SignGlove macOS 환경 설정 시작"
echo "========================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Python 버전 확인
echo -e "${BLUE}[INFO]${NC} Python 버전 확인 중..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python3가 설치되지 않았습니다."
    echo "Homebrew로 Python을 설치하세요:"
    echo "  brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}[SUCCESS]${NC} Python ${PYTHON_VERSION} 확인 완료"

# Homebrew 확인 (선택사항)
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Homebrew가 설치되지 않았습니다."
    echo "Homebrew 설치를 권장합니다: https://brew.sh"
fi

# Poetry 설치 확인
echo -e "${BLUE}[INFO]${NC} Poetry 설치 확인 중..."
if ! command -v poetry &> /dev/null; then
    echo -e "${BLUE}[INFO]${NC} Poetry를 설치하는 중..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # PATH에 Poetry 추가
    export PATH="$HOME/.local/bin:$PATH"
    
    # .bashrc나 .zshrc에 PATH 추가
    if [[ "$SHELL" == *"zsh"* ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
        echo -e "${BLUE}[INFO]${NC} Poetry PATH가 ~/.zshrc에 추가되었습니다."
    else
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo -e "${BLUE}[INFO]${NC} Poetry PATH가 ~/.bashrc에 추가되었습니다."
    fi
    
    echo -e "${YELLOW}[INFO]${NC} 새 터미널을 열거나 다음 명령어를 실행하세요:"
    echo "  source ~/.zshrc  # zsh 사용자"
    echo "  source ~/.bashrc # bash 사용자"
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

# 권한 설정 (시리얼 포트 접근용)
echo -e "${BLUE}[INFO]${NC} 시리얼 포트 권한 확인 중..."
if groups | grep -q "dialout\|uucp"; then
    echo -e "${GREEN}[SUCCESS]${NC} 시리얼 포트 접근 권한 확인 완료"
else
    echo -e "${YELLOW}[WARNING]${NC} 시리얼 포트 접근을 위해 관리자 권한이 필요할 수 있습니다."
fi

echo "========================================"
echo -e "${GREEN}macOS 환경 설정 완료!${NC}"
echo "========================================"
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