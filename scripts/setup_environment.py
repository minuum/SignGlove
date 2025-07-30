#!/usr/bin/env python3
"""
SignGlove 환경 설정 스크립트 (범용)
모든 플랫폼에서 사용 가능한 Python 기반 환경 설정

작성자: 이민우 & 양동건
역할: Poetry 환경 설정, 의존성 설치, 패키지 검증
"""

import os
import sys
import subprocess
import platform
import importlib
from pathlib import Path


class Colors:
    """ANSI 색상 코드"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'  # 종료
    BOLD = '\033[1m'


def print_colored(message: str, color: str = Colors.WHITE):
    """색상이 있는 메시지 출력"""
    if platform.system() == "Windows":
        # Windows에서는 색상 코드가 제대로 작동하지 않을 수 있음
        print(message)
    else:
        print(f"{color}{message}{Colors.ENDC}")


def run_command(command: str, check: bool = True) -> tuple:
    """명령어 실행"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""


def check_python_version():
    """Python 버전 확인"""
    print_colored("[INFO] Python 버전 확인 중...", Colors.BLUE)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_colored("[ERROR] Python 3.9 이상이 필요합니다.", Colors.RED)
        print_colored(f"현재 버전: {version.major}.{version.minor}.{version.micro}", Colors.RED)
        return False
    
    print_colored(f"[SUCCESS] Python {version.major}.{version.minor}.{version.micro} 확인 완료", Colors.GREEN)
    return True


def check_poetry():
    """Poetry 설치 확인"""
    print_colored("[INFO] Poetry 설치 확인 중...", Colors.BLUE)
    
    success, stdout, stderr = run_command("poetry --version", check=False)
    
    if success:
        print_colored(f"[SUCCESS] Poetry 설치 확인 완료: {stdout}", Colors.GREEN)
        return True
    else:
        print_colored("[WARNING] Poetry가 설치되지 않았습니다.", Colors.YELLOW)
        return install_poetry()


def install_poetry():
    """Poetry 설치"""
    print_colored("[INFO] Poetry 설치 중...", Colors.BLUE)
    
    # Poetry 설치 URL
    if platform.system() == "Windows":
        install_cmd = "curl -sSL https://install.python-poetry.org | python -"
    else:
        install_cmd = "curl -sSL https://install.python-poetry.org | python3 -"
    
    success, stdout, stderr = run_command(install_cmd, check=False)
    
    if success:
        print_colored("[SUCCESS] Poetry 설치 완료", Colors.GREEN)
        print_colored("[INFO] 새 터미널을 열거나 PATH를 업데이트하세요.", Colors.YELLOW)
        return True
    else:
        print_colored(f"[ERROR] Poetry 설치 실패: {stderr}", Colors.RED)
        return False


def install_dependencies():
    """Poetry 의존성 설치"""
    print_colored("[INFO] Poetry 의존성 설치 중...", Colors.BLUE)
    
    success, stdout, stderr = run_command("poetry install")
    
    if success:
        print_colored("[SUCCESS] 의존성 설치 완료", Colors.GREEN)
        return True
    else:
        print_colored(f"[ERROR] 의존성 설치 실패: {stderr}", Colors.RED)
        return False


def create_init_files():
    """__init__.py 파일 생성"""
    print_colored("[INFO] Python 패키지 구조 확인 중...", Colors.BLUE)
    
    init_files = [
        "hardware/__init__.py",
        "hardware/donggeon/__init__.py",
        "hardware/donggeon/client/__init__.py",
        "hardware/donggeon/server/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.write_text("# SignGlove Package\n")
            print_colored(f"[INFO] 생성됨: {init_file}", Colors.CYAN)


def test_imports():
    """Python 패키지 import 테스트"""
    print_colored("[INFO] 양동건 팀원 스크립트 import 테스트 중...", Colors.BLUE)
    
    test_modules = [
        ("hardware.donggeon.client.wifi_data_client", "WiFi 클라이언트"),
        ("hardware.donggeon.client.uart_data_client", "UART 클라이언트"),
        ("hardware.donggeon.server.simple_tcp_server", "TCP 서버")
    ]
    
    for module_name, display_name in test_modules:
        try:
            # Poetry 환경에서 import 테스트
            cmd = f'poetry run python -c "import {module_name}; print(\'{display_name} import 성공\')"'
            success, stdout, stderr = run_command(cmd, check=False)
            
            if success:
                print_colored(f"[SUCCESS] {display_name} import 성공", Colors.GREEN)
            else:
                print_colored(f"[WARNING] {display_name} import 실패: {stderr}", Colors.YELLOW)
        except Exception as e:
            print_colored(f"[WARNING] {display_name} 테스트 중 오류: {e}", Colors.YELLOW)


def check_serial_ports():
    """시리얼 포트 확인"""
    print_colored("[INFO] 시리얼 포트 확인 중...", Colors.BLUE)
    
    try:
        cmd = 'poetry run python -c "import serial.tools.list_ports; ports = list(serial.tools.list_ports.comports()); print(f\\"발견된 포트: {len(ports)}개\\"); [print(f\\"  - {port.device}: {port.description}\\") for port in ports]"'
        success, stdout, stderr = run_command(cmd, check=False)
        
        if success:
            print_colored("[INFO] 시리얼 포트 상태:", Colors.CYAN)
            print(stdout)
        else:
            print_colored("[WARNING] 시리얼 포트 확인 실패", Colors.YELLOW)
    except Exception as e:
        print_colored(f"[WARNING] 시리얼 포트 확인 중 오류: {e}", Colors.YELLOW)


def print_usage_info():
    """사용법 안내"""
    print_colored("\n" + "="*50, Colors.MAGENTA)
    print_colored("환경 설정 완료! 🚀", Colors.GREEN + Colors.BOLD)
    print_colored("="*50, Colors.MAGENTA)
    
    print_colored("\n📋 사용 가능한 명령어:", Colors.CYAN)
    commands = [
        ("poetry run donggeon-wifi", "WiFi 데이터 클라이언트 실행"),
        ("poetry run donggeon-uart", "UART 데이터 클라이언트 실행"),
        ("poetry run donggeon-tcp-server", "간단한 TCP 서버 실행"),
        ("poetry run start-server", "FastAPI 서버 실행"),
        ("poetry shell", "Poetry 가상환경 활성화")
    ]
    
    for cmd, desc in commands:
        print_colored(f"  {cmd:<30} - {desc}", Colors.WHITE)
    
    print_colored("\n🔧 개발 명령어:", Colors.CYAN)
    dev_commands = [
        ("poetry run pytest", "테스트 실행"),
        ("poetry run black .", "코드 포맷팅"),
        ("poetry run flake8", "코드 린팅")
    ]
    
    for cmd, desc in dev_commands:
        print_colored(f"  {cmd:<30} - {desc}", Colors.WHITE)
    
    print_colored(f"\n📁 현재 디렉토리: {os.getcwd()}", Colors.BLUE)
    print_colored(f"🐍 Python 버전: {platform.python_version()}", Colors.BLUE)


def main():
    """메인 함수"""
    print_colored("="*60, Colors.MAGENTA)
    print_colored("SignGlove 환경 설정 스크립트", Colors.GREEN + Colors.BOLD)
    print_colored("="*60, Colors.MAGENTA)
    print_colored(f"플랫폼: {platform.system()} {platform.release()}", Colors.CYAN)
    print_colored(f"아키텍처: {platform.machine()}", Colors.CYAN)
    print("")
    
    # 1. Python 버전 확인
    if not check_python_version():
        sys.exit(1)
    
    # 2. Poetry 확인
    if not check_poetry():
        print_colored("[ERROR] Poetry 설치에 실패했습니다.", Colors.RED)
        sys.exit(1)
    
    # 3. __init__.py 파일 생성
    create_init_files()
    
    # 4. 의존성 설치
    if not install_dependencies():
        sys.exit(1)
    
    # 5. import 테스트
    test_imports()
    
    # 6. 시리얼 포트 확인
    check_serial_ports()
    
    # 7. 사용법 안내
    print_usage_info()


if __name__ == "__main__":
    main()