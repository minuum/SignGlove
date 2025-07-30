#!/usr/bin/env python3
"""
SignGlove 통합 환경 설정 스크립트
GitHub clone 후 첫 실행용 스크립트

사용법:
  python setup.py                    # 기본 설정
  python setup.py --quick           # 빠른 설정 (최소한의 확인)
  python setup.py --test-donggeon   # 양동건 스크립트 테스트만

작성자: 이민우 & 양동건
"""

import os
import sys
import argparse
import platform
import subprocess
from pathlib import Path


def print_banner():
    """시작 배너 출력"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                   🤖 SignGlove Project                       ║
    ║                  환경 설정 자동화 스크립트                      ║
    ║                                                              ║
    ║  개발자: 이민우 (서버) & 양동건 (하드웨어)                       ║
    ║  목표: 수어 인식 스마트 글러브 시스템                            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"🖥️  플랫폼: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"📁 디렉토리: {os.getcwd()}")
    print("")


def detect_platform():
    """플랫폼 감지"""
    system = platform.system().lower()
    
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        # Linux 배포판 구분
        try:
            with open("/etc/os-release", "r") as f:
                content = f.read().lower()
                if "ubuntu" in content or "debian" in content:
                    return "ubuntu"
                else:
                    return "linux"
        except:
            return "linux"
    else:
        return "unknown"


def run_platform_setup(platform_name: str):
    """플랫폼별 설정 스크립트 실행"""
    scripts_dir = Path("scripts")
    
    if platform_name == "windows":
        script_path = scripts_dir / "setup_windows.bat"
        if script_path.exists():
            print("🚀 Windows 전용 설정 스크립트 실행 중...")
            os.system(str(script_path))
        else:
            print("❌ Windows 설정 스크립트를 찾을 수 없습니다.")
            
    elif platform_name == "macos":
        script_path = scripts_dir / "setup_macos.sh"
        if script_path.exists():
            print("🚀 macOS 전용 설정 스크립트 실행 중...")
            os.system(f"chmod +x {script_path} && {script_path}")
        else:
            print("❌ macOS 설정 스크립트를 찾을 수 없습니다.")
            
    elif platform_name in ["ubuntu", "linux"]:
        script_path = scripts_dir / "setup_ubuntu.sh"
        if script_path.exists():
            print("🚀 Ubuntu/Linux 전용 설정 스크립트 실행 중...")
            os.system(f"chmod +x {script_path} && {script_path}")
        else:
            print("❌ Ubuntu/Linux 설정 스크립트를 찾을 수 없습니다.")
    
    else:
        print(f"⚠️ 지원하지 않는 플랫폼: {platform_name}")
        print("범용 Python 스크립트를 실행합니다.")


def run_python_setup():
    """Python 범용 설정 스크립트 실행"""
    python_script = Path("scripts") / "setup_environment.py"
    
    if python_script.exists():
        print("🐍 Python 범용 설정 스크립트 실행 중...")
        try:
            subprocess.run([sys.executable, str(python_script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Python 설정 스크립트 실행 실패: {e}")
    else:
        print("❌ Python 설정 스크립트를 찾을 수 없습니다.")


def quick_setup():
    """빠른 설정 (최소한의 확인만)"""
    print("⚡ 빠른 설정 모드")
    print("Poetry 의존성만 설치합니다...")
    
    try:
        subprocess.run(["poetry", "install"], check=True)
        print("✅ 의존성 설치 완료!")
        
        print("\n📋 사용 가능한 명령어:")
        print("  poetry run donggeon-wifi      - WiFi 클라이언트")
        print("  poetry run donggeon-uart      - UART 클라이언트")
        print("  poetry run start-server       - FastAPI 서버")
        print("  poetry shell                  - Poetry 환경 활성화")
        
    except subprocess.CalledProcessError:
        print("❌ Poetry가 설치되지 않았습니다.")
        print("전체 설정을 실행하세요: python setup.py")
    except FileNotFoundError:
        print("❌ Poetry를 찾을 수 없습니다.")
        print("전체 설정을 실행하세요: python setup.py")


def test_donggeon_scripts():
    """양동건 스크립트만 테스트"""
    print("🧪 양동건 팀원 스크립트 테스트")
    
    test_commands = [
        ("poetry run python -c \"import hardware.donggeon.client.wifi_data_client\"", "WiFi 클라이언트"),
        ("poetry run python -c \"import hardware.donggeon.client.uart_data_client\"", "UART 클라이언트"),
        ("poetry run python -c \"import hardware.donggeon.server.simple_tcp_server\"", "TCP 서버")
    ]
    
    for cmd, name in test_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {name} - OK")
            else:
                print(f"❌ {name} - 실패: {result.stderr}")
        except Exception as e:
            print(f"❌ {name} - 오류: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="SignGlove 환경 설정")
    parser.add_argument("--quick", action="store_true", help="빠른 설정 (Poetry만)")
    parser.add_argument("--test-donggeon", action="store_true", help="양동건 스크립트 테스트만")
    parser.add_argument("--python-only", action="store_true", help="Python 스크립트만 실행")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.quick:
        quick_setup()
        return
    
    if args.test_donggeon:
        test_donggeon_scripts()
        return
    
    if args.python_only:
        run_python_setup()
        return
    
    # 기본 설정 프로세스
    platform_name = detect_platform()
    print(f"🔍 감지된 플랫폼: {platform_name}")
    
    if platform_name != "unknown":
        # 1. 플랫폼별 설정 실행
        run_platform_setup(platform_name)
        print("")
        
        # 2. Python 범용 설정 실행 (추가 검증용)
        print("🔧 추가 검증을 위한 Python 스크립트 실행...")
        run_python_setup()
    else:
        # 알 수 없는 플랫폼인 경우 Python 스크립트만 실행
        print("⚠️ 알 수 없는 플랫폼입니다. Python 범용 설정만 실행합니다.")
        run_python_setup()
    
    print("\n" + "="*60)
    print("🎉 SignGlove 환경 설정 완료!")
    print("="*60)
    print("")
    print("다음 단계:")
    print("1. poetry shell                    # Poetry 환경 활성화")
    print("2. poetry run start-server         # FastAPI 서버 시작")
    print("3. poetry run donggeon-uart        # 양동건 UART 클라이언트 실행")
    print("")
    print("📖 자세한 사용법은 README.md를 참조하세요.")


if __name__ == "__main__":
    main()