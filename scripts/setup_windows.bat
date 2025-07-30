@echo off
REM SignGlove Windows 환경 설정 스크립트
echo ========================================
echo SignGlove Windows 환경 설정 시작
echo ========================================

REM Python 버전 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python이 설치되지 않았습니다.
    echo Python 3.9 이상을 설치해주세요: https://python.org
    pause
    exit /b 1
)

echo [INFO] Python 버전 확인 완료

REM Poetry 설치 확인
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Poetry를 설치하는 중...
    curl -sSL https://install.python-poetry.org | python -
    REM PATH에 Poetry 추가
    set PATH=%APPDATA%\Python\Scripts;%PATH%
    echo [INFO] 시스템을 재시작하거나 새 터미널을 열어 Poetry를 사용하세요.
) else (
    echo [INFO] Poetry 설치 확인 완료
)

REM Poetry 의존성 설치
echo [INFO] Poetry 의존성 설치 중...
poetry install

REM Poetry 가상환경 활성화 안내
echo [INFO] 다음 명령어로 Poetry 가상환경을 활성화하세요:
echo poetry shell

REM 양동건 스크립트 테스트
echo [INFO] 양동건 팀원 스크립트 확인 중...
poetry run python -c "import hardware.donggeon.client.wifi_data_client; print('WiFi 클라이언트 import 성공')"
if %errorlevel% neq 0 (
    echo [WARNING] WiFi 클라이언트 import 실패
) else (
    echo [SUCCESS] WiFi 클라이언트 import 성공
)

poetry run python -c "import hardware.donggeon.client.uart_data_client; print('UART 클라이언트 import 성공')"
if %errorlevel% neq 0 (
    echo [WARNING] UART 클라이언트 import 실패
) else (
    echo [SUCCESS] UART 클라이언트 import 성공
)

echo ========================================
echo Windows 환경 설정 완료!
echo ========================================
echo.
echo 사용법:
echo   1. WiFi 클라이언트: poetry run donggeon-wifi
echo   2. UART 클라이언트: poetry run donggeon-uart
echo   3. TCP 서버: poetry run donggeon-tcp-server
echo   4. FastAPI 서버: poetry run start-server
echo.
pause